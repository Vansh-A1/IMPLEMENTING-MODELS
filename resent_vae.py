import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS         = 80
LATENT_CH      = 16
LR             = 1e-4

# ---- KL / capacity schedule -------------------------------------------------
# Instead of a near-zero fixed KL weight (which turns this into an autoencoder
# with a decorative KL term), we use CAPACITY ANNEALING (Burgess et al. 2018).
# We target an explicit information budget C (in nats, summed over the whole
# latent tensor) that grows linearly from ~0 to C_MAX over C_WARMUP_EPOCHS,
# and penalize the *distance* of KL from that budget rather than penalizing
# KL directly. This gives you direct, interpretable control over how much
# information the latent is allowed to carry, instead of hoping a weight
# constant happens to land in the right regime.
KL_BASE_WEIGHT = 1.0        # weight on the capacity-distance penalty (gamma in the paper)
C_MAX          = 1024.0     # target nats budget at full warmup (tune: latent has 16*8*8=1024 dims)
C_WARMUP_EPOCHS = 40         # epochs to linearly ramp capacity 0 -> C_MAX
FREE_BITS      = 0.05       # small residual per-dim floor, mostly a safety net now that
                             # capacity annealing is doing the real work

MSE_W          = 0.5
PERCEPTUAL_W   = 1.0
GRAD_CLIP      = 1.0
ACCUM_STEPS    = 4           # gradient accumulation -> effective batch size 2*4=8 at 1024^2

LOGVAR_MIN     = -6.0        # tightened from -10: prevents near-deterministic per-dim collapse
LOGVAR_MAX     = 6.0

OUTPUT_DIR     = "/data/projectwork/HR_IMAGES/training_output"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class VAE_DATASET(Dataset):
    def __init__(self, HR_DIR, transform):
        self.HR          = HR_DIR
        self.transform   = transform
        self.image_names = sorted([
            f for f in os.listdir(self.HR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.HR, self.image_names[index])
        image      = Image.open(image_path).convert("RGB")
        return self.transform(image)


train_transform = transforms.Compose([
    transforms.RandomCrop((1024, 1024)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.CenterCrop((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_path      = "/data/projectwork/HR_IMAGES/train"
validation_path = "/data/projectwork/HR_IMAGES/validation"

train_dataset      = VAE_DATASET(train_path,      train_transform)
validation_dataset = VAE_DATASET(validation_path, val_transform)

train_dataloader = DataLoader(
    train_dataset,      batch_size=2, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=2, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


def make_enc_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
    layers = [ResBlock(in_ch, out_ch, stride=2)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
        )
        self.res = ResBlock(out_ch, out_ch, stride=1, num_groups=num_groups)

    def forward(self, x):
        return self.res(self.up(x))


# ══════════════════════════════════════════════════════════════════════════════
# Encoder
# ══════════════════════════════════════════════════════════════════════════════
class ResNetEncoder(nn.Module):
    def __init__(self, latent_ch: int = 16, blocks_per_stage=(2, 2, 2, 2, 2, 2)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.stage1 = make_enc_stage( 64,  64, blocks_per_stage[0])
        self.stage2 = make_enc_stage( 64, 128, blocks_per_stage[1])
        self.stage3 = make_enc_stage(128, 256, blocks_per_stage[2])
        self.stage4 = make_enc_stage(256, 512, blocks_per_stage[3])
        self.stage5 = make_enc_stage(512, 512, blocks_per_stage[4])
        self.stage6 = make_enc_stage(512, 512, blocks_per_stage[5])

        self.final_norm = nn.GroupNorm(8, 512)
        self.final_act  = nn.SiLU()

        self.conv_mu     = nn.Conv2d(512, latent_ch, kernel_size=1)
        self.conv_logvar = nn.Conv2d(512, latent_ch, kernel_size=1)

        # zero-init logvar -> starts outputting unit Gaussian, stable early training
        nn.init.zeros_(self.conv_logvar.weight)
        nn.init.zeros_(self.conv_logvar.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.final_act(self.final_norm(x))
        mu     = self.conv_mu(x)
        logvar = self.conv_logvar(x).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


# ══════════════════════════════════════════════════════════════════════════════
# Decoder
# (Deliberately NO encoder-decoder skip connections. This is a design choice,
#  not an omission: skips let the decoder bypass the latent bottleneck, which
#  directly works against learning a meaningful, information-carrying latent.
#  Expect lower PSNR than a skip-connected version -- that's the trade-off
#  you're explicitly making by prioritizing distribution learning.)
# ══════════════════════════════════════════════════════════════════════════════
class ResNetDecoder(nn.Module):
    def __init__(self, latent_ch: int = 16):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(latent_ch, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
        )
        self.up1 = UpsampleBlock(512, 512)
        self.up2 = UpsampleBlock(512, 512)
        self.up3 = UpsampleBlock(512, 256)
        self.up4 = UpsampleBlock(256, 128)
        self.up5 = UpsampleBlock(128,  64)
        self.up6 = UpsampleBlock( 64,  64)
        self.up7 = UpsampleBlock( 64,  32)

        self.head = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16,  3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.proj(z)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        x = self.up5(x); x = self.up6(x); x = self.up7(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# VAE
# ══════════════════════════════════════════════════════════════════════════════
class ResNetVAE(nn.Module):
    def __init__(self, latent_ch: int = 16):
        super().__init__()
        self.encoder = ResNetEncoder(latent_ch)
        self.decoder = ResNetDecoder(latent_ch)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        # Tied explicitly to module training mode rather than autograd context.
        # Deterministic (mean) reconstruction is used at eval time; stochastic
        # sampling is used during training. If you want to evaluate this model
        # as a *generative* model (sample z ~ N(0,I) and decode) rather than as
        # a reconstruction model (encode -> mean -> decode), do that separately
        # -- these measure very different things and should not be conflated.
        if not training:
            return mu
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar, self.training)
        recon      = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, latent_ch: int, spatial: int, device):
        """Pure generative sampling: draw z ~ N(0, I) and decode. This is the
        real test of whether the latent has learned a usable prior-matched
        distribution, independent of reconstruction quality."""
        z = torch.randn(n, latent_ch, spatial, spatial, device=device)
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════════════
# VGG Perceptual Loss (multi-crop, since a single global 256x256 downsample
# from 1024x1024 throws away most fine-detail supervision perceptual loss is
# meant to provide)
# ══════════════════════════════════════════════════════════════════════════════
class VGGPerceptualLoss(nn.Module):
    LAYER_IDS     = {3, 8, 15, 22}
    LAYER_WEIGHTS = {3: 0.5, 8: 1.0, 15: 1.5, 22: 2.0}
    CROP          = 256
    NUM_CROPS     = 3   # random crops per image per forward pass, in addition to a global resize

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        layers = []
        for i, layer in enumerate(vgg):
            layers.append(layer)
            if i == max(self.LAYER_IDS):
                break
        self.vgg    = nn.Sequential(*layers)
        self._feats = {}

        for i, layer in enumerate(self.vgg):
            if i in self.LAYER_IDS:
                layer.register_forward_hook(self._make_hook(i))

        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _make_hook(self, idx):
        def hook(_, __, output):
            self._feats[idx] = output
        return hook

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def _vgg_forward(self, x):
        self._feats.clear()
        self.vgg(x)
        return dict(self._feats)

    def _feature_loss(self, feats_r, feats_t):
        return sum(
            self.LAYER_WEIGHTS[k] * F.l1_loss(feats_r[k], feats_t[k])
            for k in self.LAYER_IDS
        ) / sum(self.LAYER_WEIGHTS.values())

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = recon.shape
        losses = []

        # 1) global downsampled view -- captures coarse structure/color
        r_global = F.interpolate(self._normalize(recon), size=(self.CROP, self.CROP),
                                  mode="bilinear", align_corners=False)
        t_global = F.interpolate(self._normalize(target.detach()), size=(self.CROP, self.CROP),
                                  mode="bilinear", align_corners=False)
        fr = self._vgg_forward(r_global)
        ft = self._vgg_forward(t_global)
        losses.append(self._feature_loss(fr, ft))

        # 2) random full-resolution crops -- captures fine texture detail
        if H > self.CROP and W > self.CROP:
            for _ in range(self.NUM_CROPS):
                top  = torch.randint(0, H - self.CROP + 1, (1,)).item()
                left = torch.randint(0, W - self.CROP + 1, (1,)).item()
                r_c = self._normalize(recon[:, :, top:top+self.CROP, left:left+self.CROP])
                t_c = self._normalize(target[:, :, top:top+self.CROP, left:left+self.CROP].detach())
                fr = self._vgg_forward(r_c)
                ft = self._vgg_forward(t_c)
                losses.append(self._feature_loss(fr, ft))

        return sum(losses) / len(losses)


# ══════════════════════════════════════════════════════════════════════════════
# Loss — Capacity-Annealed KL (Burgess et al., "Understanding disentangling
# in beta-VAE", 2018) with a small residual free-bits floor.
#
# Standard beta-VAE penalizes KL directly: L = Recon + beta * KL
# This makes it hard to reason about *how much* information the latent
# actually carries -- a fixed beta doesn't tell you the resulting nats budget.
#
# Capacity annealing instead targets an explicit budget C (in nats) and
# penalizes the *distance* from that budget:
#     L = Recon + gamma * |KL - C|
# C is annealed from 0 up to C_MAX over training. Early on the model is
# forced to use very little capacity (near-autoencoder-with-noise regime,
# stable), and capacity is gradually released, letting the encoder decide
# how to allocate it across dimensions as more room becomes available.
# This gives interpretable, direct control over the information bottleneck
# instead of an opaque weight constant.
# ══════════════════════════════════════════════════════════════════════════════
def capacity_for_epoch(epoch: int) -> float:
    frac = min(1.0, epoch / C_WARMUP_EPOCHS)
    return C_MAX * frac


def vae_loss(recon, target, mu, logvar, perceptual_fn, capacity, free_bits=FREE_BITS):
    mse_loss  = F.mse_loss(recon, target, reduction="mean")
    perc_loss = perceptual_fn(recon, target)

    # KL per spatial location and channel: shape (B, C, H, W)
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

    # small residual free-bits floor as a safety net against total collapse
    # in any individual dimension, on top of capacity annealing
    kl_floored = torch.clamp(kl_per_dim, min=free_bits)

    # total KL in nats, summed over latent dims, averaged over batch
    B = kl_floored.shape[0]
    kl_sum_per_sample = kl_floored.view(B, -1).sum(dim=1)   # (B,)
    kl_total = kl_sum_per_sample.mean()

    # capacity-distance penalty
    kl_capacity_loss = KL_BASE_WEIGHT * (kl_total - capacity).abs()

    # raw (unfloored) KL, for diagnostics -- shows true collapse if it happens
    kl_raw = kl_per_dim.view(B, -1).sum(dim=1).mean()

    active_units = (kl_per_dim.detach().mean(0) > 0.1).float().mean().item()

    total = MSE_W * mse_loss + PERCEPTUAL_W * perc_loss + kl_capacity_loss

    return total, mse_loss, perc_loss, kl_capacity_loss, kl_total, kl_raw, active_units


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
def latent_diagnostics(mu: torch.Tensor, logvar: torch.Tensor) -> dict:
    with torch.no_grad():
        kl_per_dim   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        active_units = (kl_per_dim.mean(0) > 0.1).float().mean().item()

        # aggregate posterior stats: how far mu drifts from N(0,1) on average
        # (large mu_abs_mean / non-unit variance across the batch signals the
        # aggregate posterior q(z) is drifting away from the prior p(z) --
        # exactly the failure mode that hurts sampling quality)
        mu_flat = mu.detach().reshape(-1)
        aggregate_var = mu_flat.var(unbiased=False).item()

        return {
            "mu_max":         mu.max().item(),
            "mu_min":         mu.min().item(),
            "mu_abs_mean":    mu.abs().mean().item(),
            "lv_max":         logvar.max().item(),
            "lv_min":         logvar.min().item(),
            "lv_mean":        logvar.mean().item(),
            "explv_max":      logvar.exp().max().item(),
            "kl_raw_mean":    kl_per_dim.mean().item(),
            "active_units":   active_units,
            "aggregate_mu_var": aggregate_var,   # want this near 1.0 for a sample-able prior
        }


def accumulate_diag(acc: dict, d: dict):
    for k, v in d.items():
        acc[k] = acc.get(k, 0.0) + v


def average_diag(acc: dict, n: int) -> dict:
    return {k: v / n for k, v in acc.items()}


def log_scalars(writer, prefix, d, step):
    for k, v in d.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_train_step(epoch, epochs, step, loss, mse, perc, kl_cap_loss, kl_total, kl_raw, capacity, active, diag):
    print(
        f"  Ep {epoch:>3}/{epochs}  step {step:>4} │ "
        f"loss={loss:.4f}  mse={mse:.4f}  perc={perc:.4f}  "
        f"kl_cap_loss={kl_cap_loss:.4f}  kl_total={kl_total:.2f}  kl_raw={kl_raw:.2f}  "
        f"target_C={capacity:.1f} │ "
        f"active={active*100:.1f}%  "
        f"mu_abs={diag['mu_abs_mean']:.4f}  agg_mu_var={diag['aggregate_mu_var']:.4f}  "
        f"lv_mean={diag['lv_mean']:.4f}"
    )


def print_epoch_summary(tag, epoch, epochs, losses, diag, extra=""):
    print(
        f"\n{'─'*110}\n"
        f"  [{tag}] Ep {epoch:>3}/{epochs} │ "
        f"loss={losses['total']:.4f}  mse={losses['mse']:.4f}  "
        f"perc={losses['perc']:.4f}  kl_cap_loss={losses['kl_cap_loss']:.4f}  "
        f"kl_total={losses['kl_total']:.2f}  kl_raw={losses['kl_raw']:.2f}{extra}\n"
        f"  Latent │ "
        f"active={diag['active_units']*100:.1f}%  "
        f"agg_mu_var={diag['aggregate_mu_var']:.4f}  "   # most important number for generation quality
        f"mu_abs={diag['mu_abs_mean']:.4f}  "
        f"mu=[{diag['mu_min']:.3f},{diag['mu_max']:.3f}]  "
        f"lv_mean={diag['lv_mean']:.4f}  "
        f"lv=[{diag['lv_min']:.3f},{diag['lv_max']:.3f}]  "
        f"exp(lv)_max={diag['explv_max']:.3f}\n"
        f"{'─'*110}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    runs_dir = os.path.join(OUTPUT_DIR, "runs")
    log_path = os.path.join(OUTPUT_DIR, "train.log")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    log_file = open(log_path, "a")
    def log(msg):
        print(msg); log_file.write(msg + "\n"); log_file.flush()

    log(f"Training on      : {device}")
    log(f"Latent shape     : (B, {LATENT_CH}, 8, 8) = {LATENT_CH * 8 * 8} dims")
    log(f"Capacity target  : 0 -> {C_MAX} nats over {C_WARMUP_EPOCHS} epochs (capacity annealing)")
    log(f"Free bits floor  : {FREE_BITS} nats/dim (residual safety net)")
    log(f"Grad accumulation: {ACCUM_STEPS} steps (effective batch = {2*ACCUM_STEPS})")
    log(f"Output dir       : {OUTPUT_DIR}\n")

    model     = ResNetVAE(LATENT_CH).to(device)
    perc_fn   = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler    = GradScaler("cuda")
    writer    = SummaryWriter(log_dir=runs_dir)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, EPOCHS + 1):
        capacity = capacity_for_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = t_mse = t_perc = t_klcap = t_kltotal = t_kl_raw = t_gnorm = t_active = 0.0
        train_diag_acc: dict = {}
        num_train = 0

        optimizer.zero_grad(set_to_none=True)
        for step, x in enumerate(train_dataloader, 1):
            x = x.to(device, non_blocking=True)

            with autocast("cuda"):
                recon, mu, logvar = model(x)
                loss, mse, perc, kl_cap_loss, kl_total, kl_raw, active = vae_loss(
                    recon, x, mu, logvar, perc_fn, capacity
                )
                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if step % ACCUM_STEPS == 0 or step == len(train_dataloader):
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                t_gnorm += grad_norm.item()

            t_loss    += loss.item()
            t_mse     += mse.item()
            t_perc    += perc.item()
            t_klcap   += kl_cap_loss.item()
            t_kltotal += kl_total.item()
            t_kl_raw  += kl_raw.item()
            t_active  += active
            global_step += 1
            num_train   += 1

            diag = latent_diagnostics(mu, logvar)
            accumulate_diag(train_diag_acc, diag)

            writer.add_scalar("step/loss",         loss.item(),         global_step)
            writer.add_scalar("step/mse",          mse.item(),          global_step)
            writer.add_scalar("step/perc",         perc.item(),         global_step)
            writer.add_scalar("step/kl_cap_loss",  kl_cap_loss.item(),  global_step)
            writer.add_scalar("step/kl_total",     kl_total.item(),     global_step)
            writer.add_scalar("step/kl_raw",       kl_raw.item(),       global_step)
            writer.add_scalar("step/active_units", active,              global_step)
            writer.add_scalar("step/aggregate_mu_var", diag["aggregate_mu_var"], global_step)

            if step % 50 == 0:
                print_train_step(epoch, EPOCHS, step, loss.item(), mse.item(), perc.item(),
                                  kl_cap_loss.item(), kl_total.item(), kl_raw.item(),
                                  capacity, active, diag)

        n = len(train_dataloader)
        n_accum_steps = max(1, n // ACCUM_STEPS)
        avg_train = {
            "total":       t_loss    / n,
            "mse":         t_mse     / n,
            "perc":        t_perc    / n,
            "kl_cap_loss": t_klcap   / n,
            "kl_total":    t_kltotal / n,
            "kl_raw":      t_kl_raw  / n,
        }
        avg_gnorm      = t_gnorm  / n_accum_steps
        avg_active     = t_active / n
        avg_train_diag = average_diag(train_diag_acc, num_train)

        log_scalars(writer, "train/loss",  avg_train,      epoch)
        log_scalars(writer, "train/diag",  avg_train_diag, epoch)
        writer.add_scalar("train/grad_norm",   avg_gnorm,  epoch)
        writer.add_scalar("train/active_units",avg_active, epoch)
        writer.add_scalar("train/capacity_target", capacity, epoch)
        writer.add_scalar("train/lr",          scheduler.get_last_lr()[0], epoch)
        if torch.cuda.is_available():
            writer.add_scalar("gpu/mem_allocated_MB",
                              torch.cuda.memory_allocated(device) / 1e6, epoch)
            writer.add_scalar("gpu/mem_reserved_MB",
                              torch.cuda.memory_reserved(device)  / 1e6, epoch)

        print_epoch_summary(
            "TRAIN", epoch, EPOCHS, avg_train, avg_train_diag,
            extra=f"  active={avg_active*100:.1f}%  grad_norm={avg_gnorm:.4f}  "
                  f"capacity_target={capacity:.1f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        v_loss = v_mse = v_perc = v_klcap = v_kltotal = v_kl_raw = v_active = 0.0
        val_diag_acc: dict = {}
        num_val = 0

        with torch.no_grad():
            for x in validation_dataloader:
                x = x.to(device, non_blocking=True)
                with autocast("cuda"):
                    recon, mu, logvar = model(x)
                    loss, mse, perc, kl_cap_loss, kl_total, kl_raw, active = vae_loss(
                        recon, x, mu, logvar, perc_fn, capacity
                    )
                v_loss    += loss.item()
                v_mse     += mse.item()
                v_perc    += perc.item()
                v_klcap   += kl_cap_loss.item()
                v_kltotal += kl_total.item()
                v_kl_raw  += kl_raw.item()
                v_active  += active
                num_val   += 1

                diag = latent_diagnostics(mu, logvar)
                accumulate_diag(val_diag_acc, diag)

                orig_01  = ((x     + 1) / 2).clamp(0, 1)
                recon_01 = ((recon + 1) / 2).clamp(0, 1)
                psnr_metric.update(recon_01, orig_01)
                ssim_metric.update(recon_01, orig_01)

        nv = len(validation_dataloader)
        avg_val = {
            "total":       v_loss    / nv,
            "mse":         v_mse     / nv,
            "perc":        v_perc    / nv,
            "kl_cap_loss": v_klcap   / nv,
            "kl_total":    v_kltotal / nv,
            "kl_raw":      v_kl_raw  / nv,
        }
        avg_val_active = v_active / num_val
        avg_val_diag   = average_diag(val_diag_acc, num_val)
        avg_psnr       = psnr_metric.compute().item()
        avg_ssim       = ssim_metric.compute().item()

        log_scalars(writer, "val/loss",  avg_val,      epoch)
        log_scalars(writer, "val/diag",  avg_val_diag, epoch)
        writer.add_scalar("val/active_units", avg_val_active, epoch)
        writer.add_scalar("val/psnr",         avg_psnr,       epoch)
        writer.add_scalar("val/ssim",         avg_ssim,       epoch)

        print_epoch_summary(
            "VAL", epoch, EPOCHS, avg_val, avg_val_diag,
            extra=f"  active={avg_val_active*100:.1f}%  "
                  f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}"
        )
        log(
            f"  [VAL] Ep {epoch:>3}/{EPOCHS} │ "
            f"loss={avg_val['total']:.4f}  mse={avg_val['mse']:.4f}  "
            f"perc={avg_val['perc']:.4f}  kl_cap_loss={avg_val['kl_cap_loss']:.4f}  "
            f"kl_total={avg_val['kl_total']:.2f}  kl_raw={avg_val['kl_raw']:.2f}  "
            f"active={avg_val_active*100:.1f}%  "
            f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}  "
            f"agg_mu_var={avg_val_diag['aggregate_mu_var']:.4f}  "
            f"mu_abs={avg_val_diag['mu_abs_mean']:.4f}  "
            f"lv_mean={avg_val_diag['lv_mean']:.4f}"
        )

        # ── Generation sanity check every 5 epochs ──────────────────────────
        # Decodes pure N(0,I) noise -- the real test of whether the latent has
        # learned something sample-able, independent of reconstruction quality.
        if epoch % 5 == 0 or epoch == EPOCHS:
            with torch.no_grad():
                samples = model.sample(n=4, latent_ch=LATENT_CH, spatial=8, device=device)
                samples_01 = ((samples + 1) / 2).clamp(0, 1)
                writer.add_images("generation/prior_samples", samples_01, epoch)

        scheduler.step()

        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "val_loss":  avg_val["total"],
            "psnr":      avg_psnr,
            "ssim":      avg_ssim,
            "latent_ch": LATENT_CH,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pth"))

        if avg_val["total"] < best_val_loss:
            best_val_loss = avg_val["total"]
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            log(f"  ✓ Best saved  (val={best_val_loss:.4f}  "
                f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f})")

    writer.close()
    log_file.close()
    print("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Resume
# ══════════════════════════════════════════════════════════════════════════════
def resume(ckpt_path: str):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(ckpt_path, map_location=device)
    latent_ch = ckpt.get("latent_ch", LATENT_CH)

    model     = ResNetVAE(latent_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler    = GradScaler("cuda")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    print(f"Resumed from epoch {ckpt['epoch']}  "
          f"val={ckpt['val_loss']:.4f}  "
          f"PSNR={ckpt.get('psnr', float('nan')):.2f}dB  "
          f"SSIM={ckpt.get('ssim', float('nan')):.4f}")
    return model, optimizer, scheduler, scaler, ckpt["epoch"]


if __name__ == "__main__":
    train()

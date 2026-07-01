import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS            = 80
LATENT_CH         = 16
LR                = 1e-4

KL_WEIGHT         = 1e-2     # final beta once warmup completes (standard beta-VAE weight)
KL_WARMUP_EPOCHS  = 20       # linear ramp of beta from 0 -> KL_WEIGHT

# ── Free bits ────────────────────────────────────────────────────────────────
# Applied per CHANNEL GROUP (16 groups), NOT per individual scalar latent
# position. The latent is (16, 8, 8) = 1024 scalars; the previous implementation
# floored each of the 1024 positions independently at 0.5 nats, giving an
# inescapable 1024*0.5=512-nat floor that dominated the loss and had zero
# gradient for ~85% of the latent (torch.clamp's gradient is 0 below the
# floor). Here each channel's KL is SUMMED over its 8x8 spatial extent and
# averaged over the batch before the floor is applied -- this keeps it on the
# same "total nats" scale as the real ELBO KL term (kl_true_mean below), and
# only requires 16 groups (not 1024 scalars) to individually clear the bar.
# 2.0 nats/channel * 16 channels = 32-nat total floor. At the fully warmed-up
# kl_weight=0.01 that contributes 0.32 to the loss -- comparable to, not 5-6x
# larger than, MSE+perceptual (~0.9). Tune this value if you want a stronger
# or weaker regularizer; keep it well below the natural scale of MSE+perceptual
# once multiplied by KL_WEIGHT.
FREE_BITS_PER_CHANNEL = 2.0

MSE_W             = 0.5
PERCEPTUAL_W      = 1.0
GRAD_CLIP         = 1.0
ACCUM_STEPS       = 4        # gradient accumulation -> effective batch size 2*4=8 at 1024^2

LOGVAR_MIN        = -6.0
LOGVAR_MAX        = 6.0

WARMUP_FRACTION   = 0.05     # fraction of TOTAL optimizer steps spent on LR warmup.
                              # Fix: previously a hard-coded 500-step warmup assumed a
                              # much larger dataset. With this dataset's actual step
                              # count per epoch, 500 steps took ~50 of 80 epochs to
                              # complete -- most of training ran at a tiny fraction of
                              # the target LR. Deriving it as a % of total steps makes
                              # it correct regardless of dataset size.

OUTPUT_DIR        = "/home/projectwork/Deep_learning/VANSH_WORK/Internship/VAE/VAE_1_MSE/resnet_vae"


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

# ── Dynamic warmup / cosine schedule lengths ──────────────────────────────────
# Computed from the ACTUAL dataloader length instead of hard-coded, so the LR
# schedule is correct regardless of dataset size (see WARMUP_FRACTION comment).
STEPS_PER_EPOCH       = max(1, len(train_dataloader) // ACCUM_STEPS)
TOTAL_OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS          = max(1, int(WARMUP_FRACTION * TOTAL_OPTIMIZER_STEPS))
COSINE_STEPS          = max(1, TOTAL_OPTIMIZER_STEPS - WARMUP_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    """zero_init controls whether conv2's weights start at zero (near-identity
    block at init). Only the LAST block in each stage is zero-init'd, so most
    residual paths carry real gradient from step 1."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, num_groups: int = 8,
                 zero_init: bool = True):
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
        self._init_weights(zero_init)

    def _init_weights(self, zero_init):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if zero_init:
            nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


def make_enc_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
    layers = [ResBlock(in_ch, out_ch, stride=2, zero_init=False)]
    for i in range(1, num_blocks):
        is_last = (i == num_blocks - 1)
        layers.append(ResBlock(out_ch, out_ch, stride=1, zero_init=is_last))
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
        self.res = ResBlock(out_ch, out_ch, stride=1, num_groups=num_groups, zero_init=False)

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
# Decoder — no skip connections (kept as-is)
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
        z = torch.randn(n, latent_ch, spatial, spatial, device=device)
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════════════
# VGG Perceptual Loss
# ══════════════════════════════════════════════════════════════════════════════
class VGGPerceptualLoss(nn.Module):
    LAYER_IDS     = {3, 8, 15, 22}
    LAYER_WEIGHTS = {3: 0.5, 8: 1.0, 15: 1.5, 22: 2.0}
    CROP          = 256
    NUM_CROPS     = 3

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

        r_global = F.interpolate(self._normalize(recon), size=(self.CROP, self.CROP),
                                  mode="bilinear", align_corners=False)
        t_global = F.interpolate(self._normalize(target.detach()), size=(self.CROP, self.CROP),
                                  mode="bilinear", align_corners=False)
        fr = self._vgg_forward(r_global)
        ft = self._vgg_forward(t_global)
        losses.append(self._feature_loss(fr, ft))

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
# KL divergence + Free Bits (fixed: grouped by channel, correctly scaled)
# ══════════════════════════════════════════════════════════════════════════════
def compute_kl(mu: torch.Tensor, logvar: torch.Tensor):
    """
    mu, logvar: (B, C, H, W)

    Returns:
        kl_for_backward : scalar used in the loss. Free bits applied per
                           CHANNEL GROUP (16 groups), not per individual
                           scalar latent position.
        kl_true_mean    : scalar, the TRUE unfloored per-sample total KL,
                           averaged over the batch. This is the honest,
                           standard VAE KL term -- nothing is floored here.
                           Used for logging/diagnostics only, so printed
                           numbers always reflect real training progress.
        kl_per_channel  : (C,) true per-channel KL (summed over spatial,
                           averaged over batch) -- used for the
                           active-channel diagnostic.
    """
    kl_per_element = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())   # (B,C,H,W), >= 0

    kl_true_mean = kl_per_element.sum(dim=(1, 2, 3)).mean()

    kl_per_channel_persample = kl_per_element.sum(dim=(2, 3))           # (B, C) -- sum over spatial
    kl_per_channel = kl_per_channel_persample.mean(dim=0)                # (C,)   -- mean over batch
    kl_per_channel_floored = torch.clamp(kl_per_channel, min=FREE_BITS_PER_CHANNEL)

    kl_for_backward = kl_per_channel_floored.sum()

    return kl_for_backward, kl_true_mean, kl_per_channel.detach()


def kl_weight_for_epoch(epoch: int) -> float:
    if KL_WARMUP_EPOCHS <= 0:
        return KL_WEIGHT
    frac = min(1.0, epoch / KL_WARMUP_EPOCHS)
    return KL_WEIGHT * frac


def vae_loss(recon, target, mu, logvar, perceptual_fn, kl_weight):
    mse_loss  = F.mse_loss(recon, target, reduction="mean")
    perc_loss = perceptual_fn(recon, target)

    kl_for_backward, kl_true_mean, kl_per_channel = compute_kl(mu, logvar)

    total = MSE_W * mse_loss + PERCEPTUAL_W * perc_loss + kl_weight * kl_for_backward

    active_units = (kl_per_channel > 0.1).float().mean().item()   # fraction of 16 channels "in use"

    return total, mse_loss, perc_loss, kl_true_mean, active_units


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
def latent_diagnostics(mu: torch.Tensor, logvar: torch.Tensor) -> dict:
    with torch.no_grad():
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_channel = kl_per_element.sum(dim=(2, 3)).mean(dim=0)     # (C,), matches compute_kl
        active_units   = (kl_per_channel > 0.1).float().mean().item()

        return {
            "mu_max":       mu.max().item(),
            "mu_min":       mu.min().item(),
            "mu_abs_mean":  mu.abs().mean().item(),
            "lv_max":       logvar.max().item(),
            "lv_min":       logvar.min().item(),
            "lv_mean":      logvar.mean().item(),
            "explv_max":    logvar.exp().max().item(),
            "kl_mean":      kl_per_element.mean().item(),
            "active_units": active_units,
        }


def accumulate_diag(acc: dict, d: dict):
    for k, v in d.items():
        acc[k] = acc.get(k, 0.0) + v


def average_diag(acc: dict, n: int) -> dict:
    return {k: v / n for k, v in acc.items()}


def print_train_step(epoch, epochs, step, loss, mse, perc, kl, kl_w, active, diag, lr):
    print(
        f"  Ep {epoch:>3}/{epochs}  step {step:>4} │ "
        f"loss={loss:.4f}  mse={mse:.4f}  perc={perc:.4f}  "
        f"kl={kl:.4f}  kl_w={kl_w:.6f}  lr={lr:.2e} │ "
        f"active={active*100:.1f}%  "
        f"mu_abs={diag['mu_abs_mean']:.4f}  "
        f"mu=[{diag['mu_min']:.3f},{diag['mu_max']:.3f}]  "
        f"lv_mean={diag['lv_mean']:.4f}  "
        f"lv=[{diag['lv_min']:.3f},{diag['lv_max']:.3f}]"
    )


def print_epoch_summary(tag, epoch, epochs, losses, diag, extra=""):
    print(
        f"\n{'─'*100}\n"
        f"  [{tag}] Ep {epoch:>3}/{epochs} │ "
        f"loss={losses['total']:.4f}  mse={losses['mse']:.4f}  "
        f"perc={losses['perc']:.4f}  kl={losses['kl']:.4f}{extra}\n"
        f"  Latent │ active={diag['active_units']*100:.1f}%  "
        f"mu_abs={diag['mu_abs_mean']:.4f}  "
        f"mu=[{diag['mu_min']:.3f},{diag['mu_max']:.3f}]  "
        f"lv_mean={diag['lv_mean']:.4f}  "
        f"lv=[{diag['lv_min']:.3f},{diag['lv_max']:.3f}]\n"
        f"{'─'*100}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer / Scheduler (shared by train() and resume() for consistency)
# ══════════════════════════════════════════════════════════════════════════════
def build_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    # T_max = remaining steps AFTER warmup (fix: previously used the full step
    # count, so SequentialLR only ever let cosine run through a fraction of
    # its own schedule and never reached eta_min).
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_STEPS, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_STEPS]
    )
    return optimizer, scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    log_path = os.path.join(OUTPUT_DIR, "train.log")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file = open(log_path, "a")
    def log(msg):
        print(msg); log_file.write(msg + "\n"); log_file.flush()

    log(f"Training on      : {device}")
    log(f"Latent shape     : (B, {LATENT_CH}, 8, 8) = {LATENT_CH * 8 * 8} dims")
    log(f"KL weight (final): {KL_WEIGHT}, linear warmup over {KL_WARMUP_EPOCHS} epochs")
    log(f"Free bits floor  : {FREE_BITS_PER_CHANNEL} nats/channel x {LATENT_CH} channels "
        f"= {FREE_BITS_PER_CHANNEL * LATENT_CH:.1f} nats total (grouped, collapse prevention)")
    log(f"LR warmup        : {WARMUP_STEPS} optimizer steps "
        f"({WARMUP_FRACTION*100:.0f}% of {TOTAL_OPTIMIZER_STEPS} total), then cosine annealing")
    log(f"Loss             : MSE + VGG Perceptual + beta * KL  (standard ELBO only)")
    log(f"Output dir       : {OUTPUT_DIR}\n")

    model     = ResNetVAE(LATENT_CH).to(device)
    perc_fn   = VGGPerceptualLoss().to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model)

    with torch.no_grad():
        sample_x = next(iter(train_dataloader))
        log(f"Dataset sanity check: batch mean={sample_x.mean().item():.4f} "
            f"std={sample_x.std().item():.4f} shape={tuple(sample_x.shape)}")

    scaler = GradScaler("cuda")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    best_val_loss = float("inf")
    optimizer_step = 0

    for epoch in range(1, EPOCHS + 1):
        kl_w = kl_weight_for_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = t_mse = t_perc = t_kl = t_gnorm = t_active = 0.0
        train_diag_acc: dict = {}
        num_train = 0
        n = len(train_dataloader)

        optimizer.zero_grad(set_to_none=True)
        for step, x in enumerate(train_dataloader, 1):
            x = x.to(device, non_blocking=True)

            # Fix: scale by the TRUE size of the current accumulation window,
            # not a fixed ACCUM_STEPS -- the final window of an epoch may be
            # smaller if n isn't divisible by ACCUM_STEPS.
            group_start = ((step - 1) // ACCUM_STEPS) * ACCUM_STEPS + 1
            group_size  = min(ACCUM_STEPS, n - group_start + 1)

            with autocast("cuda"):
                recon, mu, logvar = model(x)
                loss, mse, perc, kl, active = vae_loss(recon, x, mu, logvar, perc_fn, kl_w)
                loss_scaled = loss / group_size

            scaler.scale(loss_scaled).backward()

            at_boundary = (step % ACCUM_STEPS == 0) or (step == n)
            if at_boundary:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                # Fix: only advance the LR scheduler if AMP actually took the
                # step (skips the step silently on inf/nan grads).
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    optimizer_step += 1

                t_gnorm += grad_norm.item()

            t_loss   += loss.item()
            t_mse    += mse.item()
            t_perc   += perc.item()
            t_kl     += kl.item()
            t_active += active
            num_train += 1

            diag = latent_diagnostics(mu, logvar)
            accumulate_diag(train_diag_acc, diag)

            if step % 50 == 0:
                print_train_step(epoch, EPOCHS, step, loss.item(), mse.item(),
                                  perc.item(), kl.item(), kl_w, active, diag,
                                  optimizer.param_groups[0]["lr"])

        n_accum_steps = max(1, n // ACCUM_STEPS)
        avg_train = {
            "total": t_loss / n, "mse": t_mse / n, "perc": t_perc / n, "kl": t_kl / n,
        }
        avg_gnorm      = t_gnorm  / n_accum_steps
        avg_active     = t_active / n
        avg_train_diag = average_diag(train_diag_acc, num_train)

        print_epoch_summary(
            "TRAIN", epoch, EPOCHS, avg_train, avg_train_diag,
            extra=f"  active={avg_active*100:.1f}%  grad_norm={avg_gnorm:.4f}  "
                  f"kl_w={kl_w:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()

        v_loss = v_mse = v_perc = v_kl = v_active = 0.0
        val_diag_acc: dict = {}
        num_val = 0

        with torch.no_grad():
            for x in validation_dataloader:
                x = x.to(device, non_blocking=True)
                with autocast("cuda"):
                    recon, mu, logvar = model(x)
                    loss, mse, perc, kl, active = vae_loss(recon, x, mu, logvar, perc_fn, kl_w)
                v_loss   += loss.item()
                v_mse    += mse.item()
                v_perc   += perc.item()
                v_kl     += kl.item()
                v_active += active
                num_val  += 1

                diag = latent_diagnostics(mu, logvar)
                accumulate_diag(val_diag_acc, diag)

                orig_01  = ((x     + 1) / 2).clamp(0, 1)
                recon_01 = ((recon + 1) / 2).clamp(0, 1)
                psnr_metric.update(recon_01, orig_01)
                ssim_metric.update(recon_01, orig_01)

        nv = len(validation_dataloader)
        avg_val = {
            "total": v_loss / nv, "mse": v_mse / nv, "perc": v_perc / nv, "kl": v_kl / nv,
        }
        avg_val_active = v_active / num_val
        avg_val_diag   = average_diag(val_diag_acc, num_val)
        avg_psnr       = psnr_metric.compute().item()
        avg_ssim       = ssim_metric.compute().item()

        print_epoch_summary(
            "VAL", epoch, EPOCHS, avg_val, avg_val_diag,
            extra=f"  active={avg_val_active*100:.1f}%  PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}"
        )
        log(
            f"  [VAL] Ep {epoch:>3}/{EPOCHS} │ "
            f"loss={avg_val['total']:.4f}  mse={avg_val['mse']:.4f}  "
            f"perc={avg_val['perc']:.4f}  kl={avg_val['kl']:.4f}  "
            f"active={avg_val_active*100:.1f}%  "
            f"PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.4f}  "
            f"mu_abs={avg_val_diag['mu_abs_mean']:.4f}  "
            f"lv_mean={avg_val_diag['lv_mean']:.4f}"
        )

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

    log_file.close()
    print("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Resume
# ══════════════════════════════════════════════════════════════════════════════
def resume(ckpt_path: str):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(ckpt_path, map_location=device)
    latent_ch = ckpt.get("latent_ch", LATENT_CH)

    model = ResNetVAE(latent_ch).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    scaler = GradScaler("cuda")

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

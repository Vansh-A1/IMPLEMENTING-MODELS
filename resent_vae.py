import atexit
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class Dataset_VAE(Dataset):
    def __init__(self, HR, transform=None):
        self.HR_DIR = HR
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(self.HR_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.HR_DIR, self.image_names[index])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


train_transform = transforms.Compose([
    transforms.RandomCrop((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.CenterCrop((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

validation_path = "/data/projectwork/HR_IMAGES/validation"
dataset_path    = "/data/projectwork/HR_IMAGES/train"

validation_dataset = Dataset_VAE(HR=validation_path, transform=val_transform)
dataset            = Dataset_VAE(HR=dataset_path,    transform=train_transform)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=6,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# FIX (Category A): Replace BatchNorm2d with GroupNorm throughout.
#
# Root cause of the validation KL spikes:
#   BatchNorm in eval() uses running statistics accumulated during training
#   with batch_size=1. Single-image running stats are noisy and do not
#   generalise to validation batches of 6, causing mis-normalised features,
#   which causes fc_mean to output large values, which causes mean.pow(2)
#   in the KL formula to explode — and under fp16 autocast, "large" becomes
#   inf at float16 saturation (~65504), producing the extreme logged spikes.
#
# GroupNorm normalises within each sample using that sample's own spatial
# statistics. It has no running statistics and behaves identically in train
# and eval mode. This eliminates the train/eval discrepancy entirely.
#
# GroupNorm(num_groups=8, num_channels=C) is valid for all channel counts
# used here (32, 64, 128, 256, 512) since all are divisible by 8.

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )
        self.fc_mean   = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        x      = self.conv_block(x)
        mean   = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        # logvar clamp: guards exp(logvar) from overflow in both fp32 and fp16
        logvar = torch.clamp(logvar, min=-10, max=10)

        # FIX (Category A): clamp mean to prevent mean.pow(2) overflow in fp16.
        # Float16 saturates at ~65504; mean values beyond ~256 cause mean^2 = inf.
        # The KL penalty (alpha=0.001) is too weak in early training to prevent
        # mean from drifting large, especially with noisy normalisation features.
        # This clamp is a hard safety net; it does not alter learned behaviour
        # when training is healthy (mean rarely exceeds ±10 at convergence).
        mean = torch.clamp(mean, min=-100, max=100)

        z = self.reparameterize(mean, logvar)
        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.deconv_block(x)
        return x


def VAE_LOSS(target, output, mean, logvar, alpha=0.001, beta=1.0):
    recon_loss = F.mse_loss(output, target, reduction="mean")
    kl_loss    = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = beta * recon_loss + alpha * kl_loss
    return total_loss, recon_loss, kl_loss


writer = SummaryWriter(log_dir="runs/VAE_Experiment_1")
atexit.register(writer.close)

Epochs = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

encoder = Encoder(latent_dim=256).to(device)
decoder = Decoder(latent_dim=256).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)

scaler    = torch.amp.GradScaler("cuda")
best_loss = float('inf')

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

total_params = (
    sum(p.numel() for p in encoder.parameters())
    +
    sum(p.numel() for p in decoder.parameters())
)

print(f"Device         : {device}")
print(f"Dataset size   : {len(dataset)}")
print(f"Batch size     : {data_loader.batch_size}")
print(f"Total params   : {total_params:,}")

fixed_batch = next(iter(validation_loader)).to(device)

for epoch in range(Epochs):

    encoder.train()
    decoder.train()

    total_loss      = 0
    total_recon     = 0
    total_kl        = 0
    total_grad_norm = 0

    # Per-epoch diagnostic accumulators (training)
    train_mean_max_accum  = 0.0
    train_mean_min_accum  = 0.0
    train_lv_max_accum    = 0.0
    train_lv_min_accum    = 0.0
    train_lv_mean_accum   = 0.0
    train_explv_max_accum = 0.0
    num_train_batches     = 0

    for hr_batches in data_loader:

        hr_batches = hr_batches.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda"):
            z, mean, logvar = encoder(hr_batches)
            output_image    = decoder(z)
            loss, recon_loss, kl_loss = VAE_LOSS(hr_batches, output_image, mean, logvar)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(decoder.parameters()),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss      += loss.item()
        total_recon     += recon_loss.item()
        total_kl        += kl_loss.item()
        total_grad_norm += grad_norm.item()

        # Accumulate diagnostics (detach to avoid holding graph)
        with torch.no_grad():
            train_mean_max_accum  += mean.detach().max().item()
            train_mean_min_accum  += mean.detach().min().item()
            train_lv_max_accum    += logvar.detach().max().item()
            train_lv_min_accum    += logvar.detach().min().item()
            train_lv_mean_accum   += logvar.detach().mean().item()
            train_explv_max_accum += logvar.detach().exp().max().item()
        num_train_batches += 1

    avg_loss      = total_loss      / len(data_loader)
    avg_recon     = total_recon     / len(data_loader)
    avg_kl        = total_kl        / len(data_loader)
    avg_grad_norm = total_grad_norm / len(data_loader)

    # Average train diagnostics
    avg_train_mean_max  = train_mean_max_accum  / num_train_batches
    avg_train_mean_min  = train_mean_min_accum  / num_train_batches
    avg_train_lv_max    = train_lv_max_accum    / num_train_batches
    avg_train_lv_min    = train_lv_min_accum    / num_train_batches
    avg_train_lv_mean   = train_lv_mean_accum   / num_train_batches
    avg_train_explv_max = train_explv_max_accum / num_train_batches

    try:
        writer.add_scalar("Gradient_Norm", avg_grad_norm, epoch)
    except Exception as e:
        print("TensorBoard logging failed:", e)

    if torch.cuda.is_available():
        try:
            writer.add_scalar("GPU/Memory_Allocated_MB", torch.cuda.memory_allocated(device) / 1e6, epoch)
            writer.add_scalar("GPU/Memory_Reserved_MB",  torch.cuda.memory_reserved(device)  / 1e6, epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

    encoder.eval()
    decoder.eval()

    val_loss  = 0
    val_recon = 0
    val_kl    = 0

    all_latents = []
    all_labels  = []
    all_means   = []
    all_logvars = []

    psnr_metric.reset()
    ssim_metric.reset()

    # Per-epoch diagnostic accumulators (validation)
    val_mean_max_accum  = 0.0
    val_mean_min_accum  = 0.0
    val_lv_max_accum    = 0.0
    val_lv_min_accum    = 0.0
    val_lv_mean_accum   = 0.0
    val_explv_max_accum = 0.0
    num_val_batches     = 0

    with torch.no_grad():
        for batch_idx, validation_batches in enumerate(validation_loader):

            validation_batches = validation_batches.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda"):
                z1, mean1, logvar1 = encoder(validation_batches)
                output_image1      = decoder(z1)
                loss1, recon_loss1, kl_loss1 = VAE_LOSS(
                    validation_batches, output_image1, mean1, logvar1
                )

            val_loss  += loss1.item()
            val_recon += recon_loss1.item()
            val_kl    += kl_loss1.item()

            all_latents.append(z1.cpu())
            all_labels.extend([0] * z1.size(0))
            all_means.append(mean1.cpu())
            all_logvars.append(logvar1.cpu())

            orig_denorm  = ((validation_batches + 1) / 2).clamp(0, 1)
            recon_denorm = ((output_image1       + 1) / 2).clamp(0, 1)
            psnr_metric.update(recon_denorm, orig_denorm)
            ssim_metric.update(recon_denorm, orig_denorm)

            # Accumulate validation diagnostics
            val_mean_max_accum  += mean1.max().item()
            val_mean_min_accum  += mean1.min().item()
            val_lv_max_accum    += logvar1.max().item()
            val_lv_min_accum    += logvar1.min().item()
            val_lv_mean_accum   += logvar1.mean().item()
            val_explv_max_accum += logvar1.exp().max().item()
            num_val_batches     += 1

    avg_val_loss  = val_loss  / len(validation_loader)
    avg_val_recon = val_recon / len(validation_loader)
    avg_val_kl    = val_kl    / len(validation_loader)
    avg_psnr      = psnr_metric.compute()
    avg_ssim      = ssim_metric.compute()

    # Average validation diagnostics
    avg_val_mean_max  = val_mean_max_accum  / num_val_batches
    avg_val_mean_min  = val_mean_min_accum  / num_val_batches
    avg_val_lv_max    = val_lv_max_accum    / num_val_batches
    avg_val_lv_min    = val_lv_min_accum    / num_val_batches
    avg_val_lv_mean   = val_lv_mean_accum   / num_val_batches
    avg_val_explv_max = val_explv_max_accum / num_val_batches

    # Log scalars
    try:
        writer.add_scalar("Loss/Train",                avg_loss,      epoch)
        writer.add_scalar("Loss/Validation",           avg_val_loss,  epoch)
        writer.add_scalar("Reconstruction/Train",      avg_recon,     epoch)
        writer.add_scalar("Reconstruction/Validation", avg_val_recon, epoch)
        writer.add_scalar("KL/Train",                  avg_kl,        epoch)
        writer.add_scalar("KL/Validation",             avg_val_kl,    epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Metrics/PSNR",  float(avg_psnr), epoch)
        writer.add_scalar("Metrics/SSIM",  float(avg_ssim), epoch)
    except Exception as e:
        print("TensorBoard logging failed:", e)

    # Log train diagnostics
    try:
        writer.add_scalar("Train_Diagnostics/mean_max",    avg_train_mean_max,  epoch)
        writer.add_scalar("Train_Diagnostics/mean_min",    avg_train_mean_min,  epoch)
        writer.add_scalar("Train_Diagnostics/logvar_max",  avg_train_lv_max,    epoch)
        writer.add_scalar("Train_Diagnostics/logvar_min",  avg_train_lv_min,    epoch)
        writer.add_scalar("Train_Diagnostics/logvar_mean", avg_train_lv_mean,   epoch)
        writer.add_scalar("Train_Diagnostics/exp_logvar_max", avg_train_explv_max, epoch)
    except Exception as e:
        print("TensorBoard logging failed:", e)

    # Log validation diagnostics
    try:
        writer.add_scalar("Val_Diagnostics/mean_max",    avg_val_mean_max,  epoch)
        writer.add_scalar("Val_Diagnostics/mean_min",    avg_val_mean_min,  epoch)
        writer.add_scalar("Val_Diagnostics/logvar_max",  avg_val_lv_max,    epoch)
        writer.add_scalar("Val_Diagnostics/logvar_min",  avg_val_lv_min,    epoch)
        writer.add_scalar("Val_Diagnostics/logvar_mean", avg_val_lv_mean,   epoch)
        writer.add_scalar("Val_Diagnostics/exp_logvar_max", avg_val_explv_max, epoch)
    except Exception as e:
        print("TensorBoard logging failed:", e)

    if (epoch + 1) % 5 == 0:

        try:
            for name, param in encoder.named_parameters():
                writer.add_histogram(f"Encoder/{name}", param.detach(), epoch)
            for name, param in decoder.named_parameters():
                writer.add_histogram(f"Decoder/{name}", param.detach(), epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

        try:
            for name, param in encoder.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Encoder_Gradients/{name}", param.grad.detach(), epoch)
            for name, param in decoder.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Decoder_Gradients/{name}", param.grad.detach(), epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

        try:
            all_means_cat   = torch.cat(all_means,   dim=0)
            all_logvars_cat = torch.cat(all_logvars, dim=0)
            writer.add_histogram("Latent/Mean",   all_means_cat,   epoch)
            writer.add_histogram("Latent/LogVar", all_logvars_cat, epoch)
            writer.add_scalar("Posterior/Mean_abs",    all_means_cat.abs().mean().item(), epoch)
            writer.add_scalar("Posterior/LogVar_mean", all_logvars_cat.mean().item(),     epoch)
            writer.add_scalar("Posterior/Mean_std",    all_means_cat.std().item(),        epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    z_fixed, _, _ = encoder(fixed_batch)
                    recon_fixed   = decoder(z_fixed)

                originals       = ((fixed_batch[:8] + 1) / 2).clamp(0, 1)
                reconstructions = ((recon_fixed[:8] + 1) / 2).clamp(0, 1)
                combined        = torch.cat([originals, reconstructions], dim=0)
                grid            = torchvision.utils.make_grid(combined, nrow=8)
                writer.add_image("Original_vs_Reconstruction", grid, epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    random_z   = torch.randn(16, 256).to(device)
                    random_out = decoder(random_z)
                random_out  = ((random_out + 1) / 2).clamp(0, 1)
                grid_random = torchvision.utils.make_grid(random_out, nrow=4)
                writer.add_image("Random_Samples", grid_random, epoch)
        except Exception as e:
            print("TensorBoard logging failed:", e)

    if (epoch + 1) % 10 == 0:
        try:
            all_latents_tensor = (
                torch.cat(all_latents, dim=0)
                .detach()
                .cpu()
            )[:1000]
            writer.add_embedding(
                all_latents_tensor,
                global_step=epoch,
                tag="Latent_Space"
            )
        except Exception as e:
            print("TensorBoard logging failed:", e)

    try:
        writer.flush()
        print("TensorBoard data flushed")
    except Exception as e:
        print("TensorBoard logging failed:", e)

    print(
        f"Epoch [{epoch+1}/{Epochs}] | "
        f"Train Loss: {avg_loss:.6f} | "
        f"Train Recon: {avg_recon:.6f} | "
        f"Train KL: {avg_kl:.6f} || "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Val Recon: {avg_val_recon:.6f} | "
        f"Val KL: {avg_val_kl:.6f} | "
        f"PSNR: {float(avg_psnr):.4f} | "
        f"SSIM: {float(avg_ssim):.4f}"
    )
    print(
        f"[Train Diag] "
        f"mean: [{avg_train_mean_min:.3f}, {avg_train_mean_max:.3f}] | "
        f"logvar: [{avg_train_lv_min:.3f}, {avg_train_lv_max:.3f}] "
        f"(mean={avg_train_lv_mean:.3f}) | "
        f"exp(logvar).max: {avg_train_explv_max:.3f}"
    )
    print(
        f"[Val   Diag] "
        f"mean: [{avg_val_mean_min:.3f}, {avg_val_mean_max:.3f}] | "
        f"logvar: [{avg_val_lv_min:.3f}, {avg_val_lv_max:.3f}] "
        f"(mean={avg_val_lv_mean:.3f}) | "
        f"exp(logvar).max: {avg_val_explv_max:.3f}"
    )

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(
            {
                "epoch"                          : epoch + 1,
                "encoder_state_dict"             : encoder.state_dict(),
                "decoder_state_dict"             : decoder.state_dict(),
                "optimizer_state_dict"           : optimizer.state_dict(),
                "best_validation_loss"           : best_loss,
                "train_loss"                     : avg_loss,
                "validation_loss"                : avg_val_loss,
                "train_reconstruction_loss"      : avg_recon,
                "validation_reconstruction_loss" : avg_val_recon,
                "train_kl_loss"                  : avg_kl,
                "validation_kl_loss"             : avg_val_kl
            },
            "vae_best.pth"
        )
        print(
            f"Saved best model at epoch "
            f"{epoch+1} with validation loss "
            f"{best_loss:.6f}"
        )

try:
    writer.flush()
    writer.close()
except Exception as e:
    print("Writer close failed:", e)

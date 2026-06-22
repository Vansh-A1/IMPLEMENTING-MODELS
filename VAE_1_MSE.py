import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#
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

transform1 = transforms.Compose([
    transforms.RandomCrop((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = "/data/projectwork/HR_IMAGES"
dataset      = Dataset_VAE(HR=dataset_path, transform=transform1)
data_loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )
        self.fc_mean   = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def reparameterize(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std    = torch.exp(0.5 * logvar)
        eps    = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        x      = self.conv_block(x)
        mean   = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z      = self.reparameterize(mean, logvar)
        return z, mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
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




Epochs = 80

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


torch.backends.cudnn.benchmark = True

encoder = Encoder(latent_dim=256).to(device)
decoder = Decoder(latent_dim=256).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)


scaler = torch.cuda.amp.GradScaler()

best_loss = float('inf')

total_params = (
    sum(p.numel() for p in encoder.parameters())
    +
    sum(p.numel() for p in decoder.parameters())
)

print(f"Device         : {device}")
print(f"Dataset size   : {len(dataset)}")
print(f"Batch size     : {data_loader.batch_size}")
print(f"Total params   : {total_params:,}")

for epoch in range(Epochs):

    encoder.train()
    decoder.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    for hr_batches in data_loader:

       
        hr_batches = hr_batches.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

    
        with torch.cuda.amp.autocast():

            z, mean, logvar = encoder(
                hr_batches
            )

            output_image = decoder(
                z
            )

            loss, recon_loss, kl_loss = VAE_LOSS(
                hr_batches,
                output_image,
                mean,
                logvar
            )

        
        scaler.scale(loss).backward()

        
        scaler.unscale_(optimizer)

        
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) +
            list(decoder.parameters()),
            max_norm=1.0
        )

       
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_recon = total_recon / len(data_loader)
    avg_kl = total_kl / len(data_loader)

    print(
        f"Epoch [{epoch+1}/{Epochs}] | "
        f"Loss: {avg_loss:.6f} | "
        f"Recon: {avg_recon:.6f} | "
        f"KL: {avg_kl:.6f}"
    )

   
    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "reconstruction_loss": avg_recon,
                "kl_loss": avg_kl
            },
            "vae_best.pth"
        )

        print(
            f"Saved best model "
            f"at epoch {epoch+1} "
            f"with loss {best_loss:.6f}"
        )


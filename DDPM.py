import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import math

# =========================================================
# Config
# =========================================================
epochs = 50
num_timesteps = 1500
learning_rate = 1e-3
base_channel = 64
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Data
# =========================================================
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform1
)

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# =========================================================
# Forward diffusion schedule
# =========================================================
betas = torch.linspace(1e-4, 0.02, num_timesteps)
alphas = 1 - betas
alphas_bars = torch.cumprod(alphas, dim=0).to(device)


def forward_diffusion(x_0, t):
    """
    x_0: [B,1,28,28]
    t:   [B]  (per-sample timestep)
    Returns x_t and the actual noise used, both [B,1,28,28]
    """
    noise = torch.randn_like(x_0)
    alpha_bar_t = alphas_bars[t].view(-1, 1, 1, 1)

    x_t = (
        torch.sqrt(alpha_bar_t) * x_0
        + torch.sqrt(1 - alpha_bar_t) * noise
    )
    return x_t, noise


# =========================================================
# Sinusoidal timestep embedding + MLP
# =========================================================
def timestep_embedding(t, embedding_dim):
    half_dim = embedding_dim // 2
    frequency = torch.exp(
        -math.log(10000)
        * torch.arange(half_dim, device=t.device)
        / half_dim
    )
    angles = t[:, None].float() * frequency[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, time_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        t_emb = timestep_embedding(t, self.embedding_dim)
        return self.mlp(t_emb)


# =========================================================
# U-Net building blocks
# =========================================================
class UNetHead(nn.Module):
    """Input conv: [B,in_channels,28,28] -> [B,base_channels,28,28]"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownsampleBlock(nn.Module):
    """Halves spatial resolution: kernel=4, stride=2, padding=1 -> exact /2"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Doubles spatial resolution: nearest-neighbor upsample + 3x3 conv (stride 1, padding 1).
    A 3x3/stride1/pad1 conv preserves spatial size exactly, so 7->14->28 lines up.
    (A 4x4/stride1/pad1 conv, as in the original code, would shrink 14x14 -> 13x13 - that was a bug.)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    Pure feature-transform block: GroupNorm -> SiLU -> Conv -> +time -> GroupNorm -> SiLU -> Conv -> +residual.
    Does NOT change spatial resolution.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.silu = nn.SiLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.skip_conv(x)

        h = self.norm1(x)
        h = self.silu(h)
        h = self.conv1(h)

        # timestep conditioning: [B, out_channels] -> [B, out_channels, 1, 1]
        t = self.time_mlp(time_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)

        return h + residual


class DownBlock(nn.Module):
    """
    ResidualBlock -> ResidualBlock -> save skip (pre-downsample) -> Downsample
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.downsample = DownsampleBlock(out_channels)

    def forward(self, x, time_emb):
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsample -> concat skip (channel dim) -> ResidualBlock -> ResidualBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.upsample = UpsampleBlock(in_channels)
        self.res1 = ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        return x


# =========================================================
# U-Net
# =========================================================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(embedding_dim=time_emb_dim, time_dim=time_emb_dim)

        self.head = UNetHead(in_channels, base_channels)                              # 28x28, ->64ch

        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)            # 28x28 ->14x14, 64ch
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)        # 14x14 ->7x7,  128ch

        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 7x7, 256ch

        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, time_emb_dim)  # ->14x14, 128ch
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels, time_emb_dim)          # ->28x28, 64ch

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        time_emb = self.time_embedding(t)

        x = self.head(x)                        # [B,64,28,28]

        x, skip1 = self.down1(x, time_emb)       # skip1: [B,64,28,28]  x: [B,64,14,14]
        x, skip2 = self.down2(x, time_emb)       # skip2: [B,128,14,14] x: [B,128,7,7]

        x = self.bottleneck(x, time_emb)         # [B,256,7,7]

        x = self.up1(x, skip2, time_emb)         # [B,128,14,14]
        x = self.up2(x, skip1, time_emb)         # [B,64,28,28]

        out = self.final_conv(x)                 # [B,1,28,28]
        return out


# =========================================================
# Sanity tests (run BEFORE training)
# =========================================================
def run_sanity_tests():
    print("=" * 60)
    print("SANITY TESTS")
    print("=" * 60)

    time_emb_dim = 128
    time_embed = TimeEmbedding(time_emb_dim, time_emb_dim).to(device)

    # --- ResidualBlock ---
    rb = ResidualBlock(64, 128, time_emb_dim).to(device)
    x = torch.randn(4, 64, 14, 14, device=device)
    t = torch.randint(0, num_timesteps, (4,), device=device)
    te = time_embed(t)
    out = rb(x, te)
    print(f"ResidualBlock:  in {tuple(x.shape)} -> out {tuple(out.shape)}")
    assert out.shape == (4, 128, 14, 14)

    # --- DownBlock ---
    db = DownBlock(64, 128, time_emb_dim).to(device)
    x = torch.randn(4, 64, 28, 28, device=device)
    down_out, skip = db(x, te)
    print(f"DownBlock:      in {tuple(x.shape)} -> down {tuple(down_out.shape)}, skip {tuple(skip.shape)}")
    assert down_out.shape == (4, 128, 14, 14)
    assert skip.shape == (4, 128, 28, 28)

    # --- UpBlock ---
    ub = UpBlock(128, 64, 64, time_emb_dim).to(device)
    x = torch.randn(4, 128, 14, 14, device=device)
    skip = torch.randn(4, 64, 28, 28, device=device)
    up_out = ub(x, skip, te)
    print(f"UpBlock:        in {tuple(x.shape)}, skip {tuple(skip.shape)} -> out {tuple(up_out.shape)}")
    assert up_out.shape == (4, 64, 28, 28)

    # --- Full U-Net ---
    model = UNet(in_channels=1, out_channels=1, base_channels=base_channel, time_emb_dim=time_emb_dim).to(device)
    x = torch.randn(8, 1, 28, 28, device=device)
    t = torch.randint(0, num_timesteps, (8,), device=device)
    out = model(x, t)
    print(f"Full UNet:      in {tuple(x.shape)} -> out {tuple(out.shape)}")
    assert out.shape == (8, 1, 28, 28), "UNet output shape mismatch!"

    print("All sanity tests passed.")
    print("=" * 60)


# =========================================================
# Visualize forward diffusion (unchanged concept, just cleaned up)
# =========================================================
def visualize_forward_diffusion():
    timesteps_to_show = [0, 375, 750, 1125, 1499]
    x, _ = next(iter(dataloader))
    x = x.to(device)

    plt.figure(figsize=(15, 3))
    for i, t_value in enumerate(timesteps_to_show):
        t = torch.tensor([t_value], device=device)
        x_t, _ = forward_diffusion(x[0:1], t)
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_t[0].cpu().squeeze(), cmap="gray")
        plt.title(f"t = {t_value}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("forward_diffusion_preview.png")
    plt.close()
    print("Saved forward_diffusion_preview.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    run_sanity_tests()
    visualize_forward_diffusion()

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channel,
        time_emb_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # NOTE: AMP/GradScaler intentionally removed for this first debugging pass
    # (see item 21) - plain FP32 makes shape/architecture errors easier to see.
    # Re-add autocast + GradScaler once you've confirmed correctness.

    for epoch in range(epochs):
        running_loss = 0.0
        for xo, _ in dataloader:
            xo = xo.to(device)
            t = torch.randint(0, num_timesteps, (xo.shape[0],), device=device).long()

            xt, noise = forward_diffusion(xo, t)
            predicted_noise = model(xt, t)
            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

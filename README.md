# Variational Autoencoder (VAE) From Scratch in PyTorch

## Overview

This repository contains a complete implementation of a **Variational Autoencoder (VAE)** built from scratch using PyTorch. The model is trained on high-resolution RGB images and incorporates several modern training techniques to improve stability and efficiency.

---

## Features

* Encoder-Decoder architecture
* Reparameterization Trick
* KL Divergence + Reconstruction Loss
* Mixed Precision Training (AMP)
* Gradient Clipping
* Batch Normalization
* LeakyReLU activations
* Checkpoint Saving
* GPU Support
* Optimized Data Loading

---

## Architecture

### Encoder

The encoder consists of several convolutional layers followed by two fully connected layers:

* Mean layer (μ)
* Log Variance layer (log σ²)

Latent vector:

```python
z = μ + σϵ
```

where:

```python
σ = exp(0.5 × logvar)
ϵ ~ N(0,1)
```

---

### Decoder

The decoder reconstructs the image using:

* Fully connected layer
* Transposed convolution layers
* Tanh activation at the output

---

## Loss Function

The total VAE loss is:

```python
Loss = β × Reconstruction Loss + α × KL Divergence
```

### Reconstruction Loss

Mean Squared Error (MSE):

```python
F.mse_loss(output, target)
```

### KL Divergence

```python
-0.5 * torch.sum(
    1 + logvar
    - mean.pow(2)
    - logvar.exp(),
    dim=1
).mean()
```

Default weights:

* α = 0.001
* β = 1.0

---

## Training Techniques

### Automatic Mixed Precision (AMP)

Uses:

```python
torch.cuda.amp.autocast()
torch.cuda.amp.GradScaler()
```

Benefits:

* Faster training
* Lower GPU memory usage
* Stable FP16 training

---

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    parameters,
    max_norm=1.0
)
```

Prevents exploding gradients.

---

### CuDNN Benchmark

```python
torch.backends.cudnn.benchmark = True
```

Allows PyTorch to select the fastest convolution algorithm for fixed image sizes.

---

### Optimized Data Loading

```python
DataLoader(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

Features:

* Parallel data loading
* Faster CPU → GPU transfer
* Improved GPU utilization

---

## Dataset

Images are loaded using a custom PyTorch Dataset class.

Supported image formats:

* PNG
* JPG
* JPEG
* TIFF

Images are:

1. Randomly cropped to:

```python
1024 × 1024
```

2. Converted to tensors.

3. Normalized:

```python
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

---

## Hyperparameters

| Parameter                 | Value          |
| ------------------------- | -------------- |
| Epochs                    | 80             |
| Latent Dimension          | 256            |
| Learning Rate             | 1e-4           |
| Batch Size                | 32             |
| Optimizer                 | Adam           |
| Activation                | LeakyReLU(0.2) |
| Reconstruction Loss       | MSE            |
| KL Weight (α)             | 0.001          |
| Reconstruction Weight (β) | 1.0            |

---

## Dependencies

Install required packages:

```bash
pip install torch torchvision pillow
```

---

## Training

Run:

```bash
python train.py
```

During training, the script prints:

```text
Epoch [1/80] | Loss: ... | Recon: ... | KL: ...
```

---

## Model Checkpoint

The best model is automatically saved as:

```text
vae_best.pth
```

Saved information:

* Epoch number
* Encoder weights
* Decoder weights
* Optimizer state
* Best loss
* Reconstruction loss
* KL loss

---

## Repository Structure

```text
.
├── train.py
├── README.md
├── vae_best.pth
└── dataset/
```

---

## Future Improvements

* β-VAE
* Conditional VAE
* VQ-VAE
* Attention-based VAE
* Perceptual Loss
* SSIM Loss
* Residual Blocks
* U-Net Decoder

---

## Author

**Vansh Hosh**

School of Artificial Intelligence
Bennett University

---

## License

This project is open-source and available under the MIT License.

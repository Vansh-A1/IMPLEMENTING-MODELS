# Variational Autoencoder (VAE) from Scratch using PyTorch

A complete implementation of a **Variational Autoencoder (VAE)** in **PyTorch** designed for learning compact latent representations of high-resolution images.

This repository contains everything required to train, validate, monitor, and evaluate a convolutional VAE, including:

- Complete Encoder–Decoder architecture
- Reparameterization Trick
- KL Divergence Regularization
- Mixed Precision (AMP) Training
- TensorBoard Visualization
- PSNR & SSIM Evaluation
- Gradient Clipping
- Automatic Checkpoint Saving
- Extensive Latent Space Diagnostics

---

## Overview

The goal of this project is to learn a probabilistic latent representation of high-resolution images using a Variational Autoencoder.

Unlike a traditional Autoencoder, the encoder learns the parameters of a probability distribution rather than a deterministic latent vector. During training, latent vectors are sampled using the Reparameterization Trick, allowing the model to generate new images while maintaining a smooth latent space.

---

## Features

- Convolutional Encoder
- Convolutional Decoder
- Variational Latent Space
- Reparameterization Trick
- Group Normalization
- Mixed Precision Training (FP16)
- Automatic Gradient Scaling
- Gradient Clipping
- TensorBoard Logging
- Model Checkpointing
- Latent Space Embedding Visualization
- Random Image Generation
- Reconstruction Quality Metrics
- Stable KL Divergence Training

---

## Model Architecture

### Encoder

Input Image

```
1024 × 1024 × 3
```

The encoder progressively downsamples the image using convolution layers.

```
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Conv → GroupNorm → LeakyReLU
        ↓
Flatten
        ↓
FC(mean)
FC(log variance)
```

Output

- Mean (μ)
- Log Variance (log σ²)

---

### Latent Sampling

The latent vector is sampled using

```
z = μ + σ × ε
```

where

```
σ = exp(logσ² / 2)

ε ~ N(0,1)
```

This allows gradients to propagate through the stochastic sampling operation.

---

### Decoder

The decoder reconstructs the original image using transposed convolutions.

```
Latent Vector
      ↓
Fully Connected
      ↓
Reshape
      ↓
ConvTranspose
      ↓
ConvTranspose
      ↓
ConvTranspose
      ↓
ConvTranspose
      ↓
ConvTranspose
      ↓
ConvTranspose
      ↓
Tanh
```

Output

```
1024 × 1024 RGB Image
```

---

## Loss Function

The objective combines reconstruction quality with latent regularization.

### Reconstruction Loss

Mean Squared Error (MSE)

```
L_recon = ||x - x̂||²
```

### KL Divergence

```
KL = -0.5 Σ(1 + logσ² − μ² − exp(logσ²))
```

### Total Loss

```
Loss = β × Reconstruction + α × KL
```

Current hyperparameters

```
alpha = 0.001
beta = 1.0
```

---

## Stability Improvements

Several modifications were implemented to improve training stability.

### Group Normalization

Batch Normalization was replaced by Group Normalization because:

- independent of batch size
- stable for batch size = 1
- identical behaviour during training and validation

---

### Mean Clamping

```
mean = clamp(mean, -100, 100)
```

Prevents overflow during

```
mean²
```

especially under mixed precision.

---

### Log Variance Clamping

```
logvar = clamp(logvar, -10, 10)
```

Prevents

```
exp(logvar)
```

from exploding.

---

### Gradient Clipping

```
max_norm = 1.0
```

prevents unstable parameter updates.

---

## Dataset

Training images are loaded directly from folders.

Supported formats

- PNG
- JPG
- JPEG
- TIFF

Image preprocessing includes

- Random Crop (training)
- Center Crop (validation)
- Normalization
- Tensor conversion

---

## Metrics

The model evaluates reconstruction quality using

- PSNR
- SSIM

These are computed on every validation epoch.

---

## TensorBoard Logging

The repository logs a large number of statistics including

### Losses

- Training Loss
- Validation Loss
- Reconstruction Loss
- KL Loss

### Metrics

- PSNR
- SSIM

### GPU

- Memory Usage
- Reserved Memory

### Diagnostics

- Mean Distribution
- Log Variance Distribution
- exp(logvar)
- Gradient Norm
- Latent Histograms

### Images

- Original vs Reconstruction
- Randomly Generated Samples

### Latent Space

Embedding visualization of learned latent vectors.

---

## Mixed Precision Training

Training uses

```
torch.autocast()
```

and

```
GradScaler
```

to

- reduce memory usage
- speed up training
- maintain numerical stability

---

## Checkpointing

The best model is automatically saved whenever validation loss improves.

Saved checkpoint contains

- Encoder weights
- Decoder weights
- Optimizer state
- Current epoch
- Training loss
- Validation loss
- Reconstruction loss
- KL loss

---

## Training Pipeline

```
Load Dataset
      ↓
DataLoader
      ↓
Encoder
      ↓
Latent Distribution
      ↓
Reparameterization
      ↓
Decoder
      ↓
Reconstruction
      ↓
Loss Computation
      ↓
Backpropagation
      ↓
Gradient Clipping
      ↓
Optimizer Update
      ↓
Validation
      ↓
TensorBoard Logging
      ↓
Save Best Model
```

---

## Repository Structure

```
.
├── train.py
├── model.py
├── runs/
├── checkpoints/
├── vae_best.pth
└── README.md
```

---

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- TorchMetrics
- Pillow
- TensorBoard

Install dependencies

```bash
pip install torch torchvision torchmetrics pillow tensorboard
```

---

## Running

```bash
python train.py
```

TensorBoard

```bash
tensorboard --logdir=runs
```

---

## Output

The training process produces

- Model checkpoints
- TensorBoard logs
- Image reconstructions
- Random image generations
- Latent space embeddings
- PSNR/SSIM metrics

---

## Learning Outcomes

This project demonstrates practical implementation of

- Variational Autoencoders
- Representation Learning
- Latent Variable Models
- Mixed Precision Training
- Probabilistic Deep Learning
- Reconstruction-based Generative Models
- TensorBoard Experiment Tracking
- Stable Deep Learning Training Techniques

---

## Future Improvements

- β-VAE
- Conditional VAE
- Vector Quantized VAE (VQ-VAE)
- Hierarchical VAE
- Perceptual Loss
- Adversarial VAE
- Diffusion Prior
- Latent Diffusion Models

---

## License

This project is intended for educational and research purposes.

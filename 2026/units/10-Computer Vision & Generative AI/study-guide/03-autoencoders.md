# Autoencoders

**Prerequisites**: Neural networks, loss functions (MSE), PyTorch nn.Module
**USAAIO Relevance**: Foundation for VAEs and latent diffusion. Understanding bottleneck compression, latent space structure, and reconstruction loss is prerequisite to all generative models.

---

## Discovery

### The Core Question

> Can a neural network learn to *compress* data into a compact representation and then *reconstruct* it, discovering meaningful structure in the process?

An autoencoder learns the identity function $f(x) \approx x$ — but with a bottleneck that forces it to find a compressed representation. This seemingly simple task leads to powerful unsupervised learning.

### Historical Context

- **Rumelhart, Hinton, Williams (1986)**: Introduced autoencoders as a way to learn efficient representations through backpropagation.
- **Hinton & Salakhutdinov (2006)**: Showed deep autoencoders could learn better representations than PCA for dimensionality reduction.
- Autoencoders became the starting point for modern generative models: VAEs (2013) and latent diffusion (2022) both build on this architecture.

### Socratic Warm-Up

1. If the latent dimension equals the input dimension, what happens? Does the autoencoder learn anything useful?
2. How is an autoencoder's bottleneck related to PCA? (Hint: linear autoencoder.)
3. If the reconstruction is perfect, does the latent space necessarily have useful structure?

### Misconception Traps

- **"Autoencoders are generative models."** — Standard autoencoders are NOT generative. You can't sample new data from the latent space because there's no constraint on its distribution. VAEs fix this.
- **"Smaller bottleneck = better."** — Too small loses important information; too large allows the identity shortcut. The right size depends on the data's intrinsic dimensionality.
- **"Autoencoders always learn PCA."** — Only *linear* autoencoders with MSE loss learn PCA subspaces. Nonlinear autoencoders learn more complex manifolds.

---

## Intuition

### The Hourglass Architecture

```
Input x                    Reconstructed x̂
(784,)                     (784,)
  │                           ▲
  ▼                           │
[Linear 784→256]          [Linear 256→784]
[ReLU]                    [Sigmoid]
  │                           ▲
  ▼                           │
[Linear 256→64]           [Linear 64→256]
[ReLU]                    [ReLU]
  │                           ▲
  ▼                           │
[Linear 64→2]   ──────→  [Linear 2→64]
    z (latent)               [ReLU]
   (2,)
```

The network must squeeze all information through the narrow bottleneck $z$.

### What the Bottleneck Learns

```
Input space (784-D for MNIST):     Latent space (2-D):
┌─────────────────────┐            ┌────────────┐
│ 0 0 0 0 1 1 ...     │            │  *3  *8    │
│ 7 7 7 2 2 2 ...     │  Encoder → │  *0    *7  │
│ Each digit = 784 pix │            │    *1  *9  │
└─────────────────────┘            └────────────┘
                                    Digits cluster!
```

### Why Reconstruction Loss Works

The encoder must choose WHICH information to keep (limited by bottleneck size). To minimize reconstruction error, it keeps the information that varies most across the dataset — which is usually the semantically meaningful information.

---

## Math

### Architecture

**Encoder**: $z = f_\phi(x)$ where $f_\phi: \mathbb{R}^D \to \mathbb{R}^d$ and $d \ll D$

**Decoder**: $\hat{x} = g_\theta(z)$ where $g_\theta: \mathbb{R}^d \to \mathbb{R}^D$

### Loss Function

**MSE reconstruction loss** (for continuous data):
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n} \|x_i - g_\theta(f_\phi(x_i))\|^2$$

**BCE reconstruction loss** (for binary/normalized data, e.g., MNIST):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{D} \left[x_{ij}\log\hat{x}_{ij} + (1-x_{ij})\log(1-\hat{x}_{ij})\right]$$

### Linear Autoencoder = PCA

For a single-layer linear autoencoder:
$$\hat{x} = W_2 W_1 x, \quad W_1 \in \mathbb{R}^{d \times D}, \quad W_2 \in \mathbb{R}^{D \times d}$$

Minimizing $\|x - W_2 W_1 x\|^2$ yields the same subspace as the top $d$ principal components of the data covariance matrix.

### Capacity and Regularization

To prevent the autoencoder from learning the identity:
- **Undercomplete**: $d < D$ (bottleneck is smaller than input) — standard approach
- **Sparse**: Add $L_1$ penalty on activations: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \sum_j |z_j|$
- **Denoising**: Corrupt input $\tilde{x} = x + \epsilon$, reconstruct clean $x$: $\mathcal{L} = \|x - g_\theta(f_\phi(\tilde{x}))\|^2$

---

## Code

### Autoencoder for MNIST

```python
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# Training
model = Autoencoder(latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    for x_batch, _ in dataloader:
        x_flat = x_batch.view(-1, 784)
        x_hat, z = model(x_flat)
        loss = loss_fn(x_hat, x_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Convolutional Autoencoder

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # (B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B,64,7,7)
            nn.ReLU(),
            nn.Flatten(),                                 # (B,64*7*7)
            nn.Linear(64 * 7 * 7, latent_dim),           # (B,latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
```

### Visualizing the Latent Space

```python
# Encode all test data
model.eval()
all_z, all_y = [], []
with torch.no_grad():
    for x, y in test_loader:
        _, z = model(x.view(-1, 784))
        all_z.append(z)
        all_y.append(y)

z = torch.cat(all_z).numpy()
y = torch.cat(all_y).numpy()

import matplotlib.pyplot as plt
plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', s=1, alpha=0.5)
plt.colorbar()
plt.title("Autoencoder Latent Space (2D)")
plt.show()
```

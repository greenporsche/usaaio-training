# Generative Adversarial Networks (GANs)

**Prerequisites**: Neural networks, loss functions, optimization, probability distributions
**USAAIO Relevance**: GAN minimax objective, training dynamics analysis, and mode collapse are tested conceptually. Generator/discriminator implementation from scratch is a common coding question.

---

## Discovery

### The Core Question

> Instead of explicitly modeling $p(x)$ (like VAEs), can two networks *compete* against each other to produce realistic data? One network generates fakes; the other tries to detect them. Through this adversarial game, the generator learns to produce data indistinguishable from real.

### Historical Context

- **Goodfellow et al. (2014)**: "Generative Adversarial Nets" — introduced the GAN framework. Goodfellow reportedly conceived the idea during a discussion at a bar.
- **DCGAN** (Radford et al., 2015): Showed that CNNs work well as GAN architectures, establishing architectural guidelines.
- **WGAN** (Arjovsky et al., 2017): Replaced JS divergence with Wasserstein distance, improving training stability.
- **StyleGAN** (Karras et al., 2019): Generated photorealistic faces at 1024x1024.
- GANs revolutionized image generation but have been largely superseded by diffusion models for quality.

### Socratic Warm-Up

1. In the minimax game, what happens if the discriminator becomes *perfect* early in training?
2. Why is $\log(1 - D(G(z)))$ problematic at the start of training? (Hint: gradient magnitude.)
3. If the generator only produces one realistic-looking face, what goes wrong?

### Misconception Traps

- **"The discriminator should be much stronger than the generator."** — If D is too strong, G's gradients vanish. They need to be balanced.
- **"Mode collapse means the GAN failed."** — It means G found a shortcut: produce one good sample that always fools D. The loss might look fine!
- **"GANs learn p(x) explicitly."** — GANs are *implicit* generative models. They learn to sample from p(x) without ever computing p(x) directly.

---

## Intuition

### The Counterfeiter and the Detective

```
Generator G (counterfeiter):        Discriminator D (detective):
Random noise z ──→ G(z) ──→ fake    Real data x ──→ D(x) → 1 (real)
                                    Fake G(z)   ──→ D(G(z)) → 0 (fake)

Training:
- D improves at telling real from fake
- G improves at fooling D
- At equilibrium: D(x) = 0.5 for all x (can't tell the difference)
```

### Training Dynamics

```
Iteration 1:      G produces noise. D easily separates.
   Real: ████████  D=0.99
   Fake: ░░░░░░░░  D=0.01

Iteration 100:    G produces blurry images. D still wins.
   Real: ████████  D=0.85
   Fake: ▓▓▓▓▓▓▓▓  D=0.15

Iteration 1000:   G produces good images. D struggles.
   Real: ████████  D=0.55
   Fake: ▓▓▓▓▓▓▓▓  D=0.45

Convergence:      D can't tell the difference.
   Real: ████████  D=0.50
   Fake: ████████  D=0.50
```

### Mode Collapse

```
True distribution p(x):          Mode collapse:
     ╱╲    ╱╲    ╱╲              Only generates from one mode
    ╱  ╲  ╱  ╲  ╱  ╲
   ╱    ╲╱    ╲╱    ╲                    ╱╲
  ╱                  ╲                  ╱  ╲
 ╱                    ╲                ╱    ╲
(three modes of data)               (all samples here)
```

---

## Math

### Minimax Objective

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Discriminator objective** (maximize):
$$\max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

This is binary cross-entropy: D should output 1 for real, 0 for fake.

**Generator objective** (minimize):
$$\min_G \; \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Optimal Discriminator

For fixed G, the optimal D is:
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

When $p_g = p_{\text{data}}$: $D^*(x) = 1/2$ for all $x$.

### Global Optimum

Substituting $D^*$ back into $V$:
$$V(D^*, G) = -\log 4 + 2 \cdot D_{JS}(p_{\text{data}} \| p_g)$$

where $D_{JS}$ is the Jensen-Shannon divergence. The minimum is $-\log 4$ when $p_g = p_{\text{data}}$.

### Non-Saturating Loss

The original G loss $\log(1 - D(G(z)))$ has vanishing gradients when D is strong (D(G(z)) ≈ 0).

**Non-saturating alternative**: Instead of minimizing $\log(1-D(G(z)))$, maximize $\log D(G(z))$:
$$\max_G \; \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

Same fixed point, but much stronger gradients early in training.

### Wasserstein GAN (WGAN)

Replace JS divergence with Wasserstein-1 (Earth Mover's) distance:
$$\min_G \max_{D \in \mathcal{D}_1} \; \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

where $\mathcal{D}_1$ is the set of 1-Lipschitz functions. D is called a "critic" (no sigmoid, outputs a real number).

Lipschitz constraint enforced by:
- **Weight clipping**: Clamp weights to $[-c, c]$ after each update
- **Gradient penalty** (WGAN-GP): $\lambda \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$

---

## Code

### Simple GAN for MNIST

```python
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
```

### Training Loop

```python
latent_dim = 100
G = Generator(latent_dim)
D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_batch, _ in dataloader:
        batch_size = real_batch.size(0)
        real = real_batch.view(batch_size, -1)  # flatten
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, latent_dim)
        fake = G(z).detach()

        d_real = D(real)
        d_fake = D(fake)
        loss_D = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, latent_dim)
        fake = G(z)
        d_fake = D(fake)
        loss_G = criterion(d_fake, real_labels)  # non-saturating: fool D

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

### Generating Samples

```python
G.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    samples = G(z).view(-1, 1, 28, 28)
    # samples are in [-1, 1], rescale to [0, 1] for visualization
    samples = (samples + 1) / 2
```

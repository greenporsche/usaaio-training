# Denoising Diffusion Probabilistic Models (DDPM)

**Prerequisites**: Gaussian distributions, KL divergence, neural networks, noise/signal concepts
**USAAIO Relevance**: CRITICAL. Diffusion models are a major Round 2 topic. Forward process derivation, noise schedule computation, training objective, and sampling algorithm are all testable.

---

## Discovery

### The Core Question

> What if instead of learning to generate images directly, we learn to *gradually remove noise*? Start with pure Gaussian noise and iteratively denoise it, one small step at a time, until a clean image emerges.

This is the key insight of diffusion models: generation as iterative denoising.

### Historical Context

- **Sohl-Dickstein et al. (2015)**: First proposed diffusion models, inspired by non-equilibrium thermodynamics.
- **Ho, Jain, Abbeel (2020)**: "Denoising Diffusion Probabilistic Models" (DDPM) — made diffusion practical and showed image quality comparable to GANs.
- **Song et al. (2020)**: Score-based generative models — connected diffusion to score matching.
- **Dhariwal & Nichol (2021)**: "Diffusion Models Beat GANs on Image Synthesis" — established diffusion as SOTA.
- Now the backbone of Stable Diffusion, DALL-E 2, Midjourney, etc.

### Socratic Warm-Up

1. If you add a tiny bit of noise to an image, can you learn to remove it? What about a lot of noise?
2. Why $T=1000$ steps instead of going from clean to pure noise in one step?
3. If you know the exact noise $\epsilon$ that was added, how would you recover the clean image?

### Misconception Traps

- **"The model generates images from scratch."** — It starts from pure noise and *denoises* step by step. The model only learns one thing: predict the noise.
- **"Forward process is learned."** — NO. The forward process is fixed (predetermined noise schedule). Only the reverse process is learned.
- **"Each denoising step removes all noise."** — Each step removes only a small amount of noise. It takes $T$ steps to go from pure noise to clean image.

---

## Intuition

### Forward Process: Gradually Adding Noise

```
Clean image          Slightly noisy       More noisy          Pure noise
    x_0         →       x_1          →      x_2     → ... →    x_T
 ┌────────┐        ┌────────┐          ┌────────┐         ┌────────┐
 │  ☺     │        │  ☺~    │          │  ~☺~~  │         │ ~~~~~~ │
 │        │ +noise │   ~    │  +noise  │ ~~~~ ~ │ +noise  │ ~~~~~~ │
 │  ____  │ ────→  │  __~_  │  ────→   │ ~__~~~ │ ────→   │ ~~~~~~ │
 └────────┘        └────────┘          └────────┘         └────────┘

β_1 (small)         β_2 (slightly more)                    β_T (large)
```

### Reverse Process: Learned Denoising

```
Pure noise          Less noisy           Clearer             Clean image
    x_T         →      x_{T-1}     →      x_1      → ... →    x_0
 ┌────────┐        ┌────────┐          ┌────────┐         ┌────────┐
 │ ~~~~~~ │        │ ~~~~ ~ │          │  ☺~    │         │  ☺     │
 │ ~~~~~~ │ -noise │ ~__~~~ │  -noise  │   ~    │ -noise  │        │
 │ ~~~~~~ │ ────→  │ ~~~~~  │  ────→   │  __~_  │ ────→   │  ____  │
 └────────┘        └────────┘          └────────┘         └────────┘

Each step: model predicts ε_θ(x_t, t) — the noise to remove
```

### Why Many Small Steps?

```
One giant denoising step:          Many small denoising steps:
  Pure noise → ??? → Image           Pure noise → ... → Image
  (Too hard! Huge gap)               (Each step is easy!)

Analogy: Walking down a staircase
  - Jumping from top to bottom: dangerous
  - One step at a time: easy
```

---

## Math

### Forward Process

At each step, add Gaussian noise according to schedule $\beta_1, \beta_2, \ldots, \beta_T$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)$$

This means: $x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t$ where $\epsilon_t \sim \mathcal{N}(0, I)$.

### Closed-Form Forward (Skip to Any $t$)

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

By recursive substitution:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1 - \bar{\alpha}_t) I)$$

**Proof sketch**:
- $x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$
- $x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2 = \sqrt{\alpha_2\alpha_1} x_0 + \sqrt{1-\alpha_2\alpha_1}\bar{\epsilon}$
- By induction: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$

(Uses the property that $\sqrt{a}\epsilon_1 + \sqrt{b}\epsilon_2$ where $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0,I)$ has distribution $\mathcal{N}(0, (a+b)I)$.)

This is crucial: we can jump directly from $x_0$ to any $x_t$ without iterating through all intermediate steps:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Noise Schedule

Typical linear schedule (DDPM):
$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

with $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$.

As $t$ increases: $\bar{\alpha}_t \to 0$, so $x_T \approx \epsilon \sim \mathcal{N}(0, I)$.

### Reverse Process

The true reverse $q(x_{t-1}|x_t)$ is intractable, but we can approximate it:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

### Training Objective

The variational bound leads to a KL divergence at each timestep. Ho et al. (2020) showed this simplifies to:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t \sim U[1,T],\; x_0,\; \epsilon \sim \mathcal{N}(0,I)}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.

**In words**: Sample a random timestep $t$ and noise $\epsilon$. Compute the noisy image $x_t$. Train the network to predict the noise $\epsilon$ from $x_t$ and $t$.

### Sampling (Reverse Process)

Given trained $\epsilon_\theta$, generate images:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ for $t > 1$ and $z = 0$ for $t = 1$.

The mean is derived from Bayes' rule on $q(x_{t-1}|x_t, x_0)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

---

## Code

### Noise Schedule

```python
import torch

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

T = 1000
betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
```

### Forward Process (Add Noise)

```python
def forward_diffusion(x_0, t, alpha_bars):
    """
    x_0: (B, C, H, W) clean images
    t: (B,) timestep indices
    Returns: x_t, epsilon
    """
    epsilon = torch.randn_like(x_0)
    alpha_bar_t = alpha_bars[t][:, None, None, None]  # (B, 1, 1, 1)

    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
    return x_t, epsilon
```

### Simple Noise-Predicting Network

```python
class SimpleNoisePredictor(nn.Module):
    """Simplified UNet-like architecture for noise prediction."""
    def __init__(self, channels=1, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Conv2d(channels + time_emb_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, x_t, t):
        # Embed timestep
        t_emb = self.time_mlp(t.float().unsqueeze(-1))        # (B, time_emb_dim)
        t_emb = t_emb[:, :, None, None].expand(-1, -1, x_t.shape[2], x_t.shape[3])
        x = torch.cat([x_t, t_emb], dim=1)
        return self.net(x)
```

### Training Loop

```python
model = SimpleNoisePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for x_0, _ in dataloader:
        # Sample random timesteps
        t = torch.randint(0, T, (x_0.shape[0],))

        # Forward diffusion
        x_t, epsilon = forward_diffusion(x_0, t, alpha_bars)

        # Predict noise
        epsilon_pred = model(x_t, t)

        # Simple loss
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Sampling (Generation)

```python
@torch.no_grad()
def sample(model, shape, T, betas, alphas, alpha_bars):
    x = torch.randn(shape)  # Start from pure noise

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long)

        epsilon_pred = model(x, t_batch)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
        )

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean

    return x

# Generate 16 images
samples = sample(model, (16, 1, 28, 28), T, betas, alphas, alpha_bars)
```

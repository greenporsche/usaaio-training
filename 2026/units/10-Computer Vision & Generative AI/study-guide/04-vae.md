# Variational Autoencoder (VAE)

**Prerequisites**: Autoencoders, probability (KL divergence, Gaussian), calculus (chain rule), PyTorch
**USAAIO Relevance**: CRITICAL. ELBO derivation, reparameterization trick, and KL computation are classic exam questions. VAEs are also the foundation for latent diffusion (Stable Diffusion).

---

## Discovery

### The Core Question

> Standard autoencoders learn a compressed representation, but you can't generate new data by sampling from the latent space — there's no guarantee that random points in latent space decode to anything meaningful. How do you make the latent space *smooth and structured* so you can sample from it?

The answer: force the latent distribution to be close to a known distribution (standard Gaussian), then sample from that distribution to generate new data.

### Historical Context

- **Kingma & Welling (2013)**: "Auto-Encoding Variational Bayes" introduced VAEs, combining variational inference with neural networks.
- **Rezende, Mohamed, Wierstra (2014)**: Independently proposed a similar framework.
- Key insight: the reparameterization trick enables backpropagation through stochastic sampling.
- VAEs later became the encoder component of Stable Diffusion (Rombach et al., 2022).

### Socratic Warm-Up

1. If you sample a random vector $z$ and pass it through a standard autoencoder's decoder, why might the output be garbage?
2. What does KL divergence between two Gaussians measure geometrically?
3. Why can't you just backpropagate through the operation $z \sim \mathcal{N}(\mu, \sigma^2)$?

### Misconception Traps

- **"VAE = autoencoder + KL loss."** — The VAE is fundamentally a probabilistic model performing variational inference. The encoder approximates the posterior, the decoder defines the likelihood. The architecture happens to look like an autoencoder.
- **"The reparameterization trick is a training hack."** — It's mathematically principled: it separates the deterministic parameters from the stochastic noise, enabling gradient estimation.
- **"Stronger KL regularization = better generation."** — Too much KL weight causes "posterior collapse" where the encoder ignores the input and $q(z|x) \approx p(z)$ for all $x$.

---

## Intuition

### From Autoencoder to VAE

```
Autoencoder:                      VAE:
x → [Encoder] → z → [Decoder]    x → [Encoder] → (μ, σ) → z=μ+σε → [Decoder]
     deterministic point              distribution → sample

Latent space:                     Latent space:
  * * *                             ●●●
   * *  (scattered clusters)        ●●● (smooth, overlapping)
  * * *                             ●●●

Can't interpolate!                 Can interpolate smoothly!
```

### Why the KL Term Matters

```
Without KL:                       With KL:
Encoder puts each class           Encoder puts everything
in a tiny, distant cluster        near N(0,I)

   *0*                              0 0
              *1*                  1 1 0
                    *2*          2 1 0 0
                                 2 2 1

Gap between clusters =            Smooth transitions =
garbage when sampled              meaningful samples everywhere
```

### The Reparameterization Trick

```
NOT differentiable:              Differentiable:
μ, σ → [SAMPLE z ~ N(μ,σ²)] → z    μ, σ → z = μ + σ * ε
         ↑                                      ↑
    Can't backprop through           ε ~ N(0,1) is constant w.r.t. parameters
    random sampling                  Gradients flow through μ and σ!
```

---

## Math

### The Generative Model

We assume a latent variable model:
$$p(x) = \int p(x|z) p(z) \, dz$$

where $p(z) = \mathcal{N}(0, I)$ is the prior and $p_\theta(x|z)$ is the decoder (likelihood).

**Problem**: Computing $p(x) = \int p_\theta(x|z)p(z)dz$ is intractable (requires integrating over all possible $z$).

**Solution**: Introduce a variational approximation $q_\phi(z|x)$ to the true posterior $p(z|x)$.

### ELBO Derivation (Step by Step)

**Step 1**: Start with the log-likelihood we want to maximize:
$$\log p(x)$$

**Step 2**: Introduce the variational distribution $q_\phi(z|x)$:
$$\log p(x) = \log \int p(x, z) \, dz = \log \int \frac{p(x, z)}{q_\phi(z|x)} q_\phi(z|x) \, dz$$

**Step 3**: Apply Jensen's inequality ($\log$ is concave):
$$\log p(x) \geq \int q_\phi(z|x) \log \frac{p(x, z)}{q_\phi(z|x)} \, dz = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(x, z)}{q_\phi(z|x)}\right]$$

This lower bound is the **Evidence Lower Bound (ELBO)**.

**Step 4**: Decompose the ELBO:
$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(z)}{q_\phi(z|x)}\right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

**Step 5**: Verify the gap between $\log p(x)$ and ELBO:
$$\log p(x) = \text{ELBO} + D_{KL}(q_\phi(z|x) \| p(z|x))$$

Since KL $\geq 0$, we have ELBO $\leq \log p(x)$. Maximizing ELBO simultaneously:
- Makes $q_\phi(z|x)$ close to the true posterior $p(z|x)$
- Maximizes the marginal likelihood $\log p(x)$

### Gaussian VAE

**Encoder**: $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$

The encoder neural network outputs $\mu \in \mathbb{R}^d$ and $\log \sigma^2 \in \mathbb{R}^d$.

**Prior**: $p(z) = \mathcal{N}(0, I)$

**KL divergence (closed form)**:

For two $d$-dimensional Gaussians $q = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p = \mathcal{N}(0, I)$:

$$D_{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**Derivation of KL**:
$$D_{KL} = \mathbb{E}_q\left[\log \frac{q(z)}{p(z)}\right] = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]$$

$$\mathbb{E}_q[\log q(z)] = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_j \log \sigma_j^2 - \frac{d}{2}$$

$$\mathbb{E}_q[\log p(z)] = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2)$$

$$D_{KL} = -\frac{1}{2}\sum_j (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

### Reparameterization Trick

**Problem**: $z \sim q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ — sampling is not differentiable w.r.t. $\phi$.

**Solution**: Write $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$.

Now $z$ is a deterministic function of $\mu, \sigma, \epsilon$, and gradients flow through $\mu$ and $\sigma$:

$$\frac{\partial z}{\partial \mu} = I, \quad \frac{\partial z}{\partial \sigma} = \text{diag}(\epsilon)$$

### Full VAE Loss

$$\mathcal{L}(\theta, \phi; x) = -\text{ELBO} = \underbrace{\|x - \hat{x}\|^2}_{\text{or } -\log p_\theta(x|z)} + \underbrace{\frac{1}{2}\sum_{j=1}^{d}(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)}_{\text{KL divergence}}$$

Minimize this loss to train both encoder ($\phi$) and decoder ($\theta$) jointly.

---

## Code

### VAE in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)      # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)         # epsilon ~ N(0, I)
        return mu + std * eps               # z = mu + sigma * epsilon

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    # Reconstruction loss (BCE)
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + kl
```

### Training Loop

```python
model = VAE(input_dim=784, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for x_batch, _ in dataloader:
        x = x_batch.view(-1, 784)
        x_hat, mu, logvar = model(x)
        loss = vae_loss(x, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Generating New Samples

```python
model.eval()
with torch.no_grad():
    # Sample from prior
    z = torch.randn(16, 2)            # 16 samples from N(0, I)
    generated = model.decode(z)        # decode to image space
    generated = generated.view(-1, 1, 28, 28)
```

### Latent Space Interpolation

```python
# Interpolate between two points in latent space
z1 = torch.tensor([[-2.0, -2.0]])
z2 = torch.tensor([[2.0, 2.0]])
alphas = torch.linspace(0, 1, 10).unsqueeze(1)  # (10, 1)
z_interp = z1 * (1 - alphas) + z2 * alphas       # (10, 2)

with torch.no_grad():
    images = model.decode(z_interp).view(-1, 1, 28, 28)
```

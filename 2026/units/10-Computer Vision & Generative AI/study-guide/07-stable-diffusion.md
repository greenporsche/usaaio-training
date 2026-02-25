# Stable Diffusion (Latent Diffusion Models)

**Prerequisites**: DDPM (forward/reverse process, noise prediction), autoencoders, cross-attention
**USAAIO Relevance**: Understanding latent diffusion, cross-attention conditioning, and classifier-free guidance shows mastery of how modern generative AI systems work. Conceptual questions about the pipeline are likely.

---

## Discovery

### The Core Question

> DDPM works in pixel space — for a 512x512x3 image, the model processes 786,432-dimensional vectors at every denoising step. Can we do diffusion in a *compressed* latent space instead, making it orders of magnitude more efficient?

### Historical Context

- **Rombach, Blattmann, Lorenz, Esser, Ommer (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" — introduced Latent Diffusion Models (LDMs).
- Key insight: train a separate autoencoder to compress images, then run the entire diffusion process in the autoencoder's latent space.
- Stable Diffusion = Latent Diffusion + CLIP text encoder + large-scale training.
- Made high-resolution image generation feasible on consumer GPUs.

### Socratic Warm-Up

1. If DDPM works in 512x512x3 pixel space, how many floating-point operations does each UNet forward pass require? What if we work in 64x64x4 latent space instead?
2. How do you condition a diffusion model on a text prompt? Where does the text information enter the UNet?
3. What happens if you increase the classifier-free guidance scale to a very large value?

### Misconception Traps

- **"Stable Diffusion generates images directly."** — It generates latent codes, which are then decoded by a separate pretrained decoder to produce images.
- **"The text encoder is trained jointly."** — CLIP's text encoder is frozen during Stable Diffusion training.
- **"Classifier-free guidance requires a separate classifier."** — It does NOT use a classifier. It contrasts conditional vs. unconditional noise predictions.

---

## Intuition

### The Three-Stage Architecture

```
                         Text Prompt
                             │
                             ▼
                     ┌──────────────┐
                     │ CLIP Text    │
                     │ Encoder      │
                     │ (frozen)     │
                     └──────┬───────┘
                            │ text embeddings
                            ▼
Noise z_T ──→ ┌─────────────────────────┐ ──→ Clean z_0
              │   UNet (Denoising)       │
              │   with cross-attention   │
              │   for text conditioning  │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Decoder (frozen)       │ ──→ Image x
              │   from pretrained AE     │
              └─────────────────────────┘
```

### Why Latent Space?

```
Pixel space diffusion:               Latent space diffusion:
┌───────────────┐                    ┌───────────┐
│ 512 x 512 x 3│                    │ 64 x 64 x4│
│ = 786,432 dim │                    │ = 16,384   │
│ SLOW per step │                    │ 48x smaller│
└───────────────┘                    └───────────┘

Autoencoder compresses 48x while preserving perceptual quality!
```

### Cross-Attention for Conditioning

In each UNet block, text information is injected via cross-attention:

```
UNet features (spatial):           Text embeddings:
h ∈ R^(HW × D_model)              c ∈ R^(L × D_text)

Q = h · W_Q                       K = c · W_K
                                   V = c · W_V

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Result: each spatial position attends to relevant text tokens
```

### Classifier-Free Guidance

```
Unconditional prediction:     Conditional prediction:
ε_θ(x_t, t, ∅)              ε_θ(x_t, t, c)
   (no text prompt)              (with text prompt)

Guided prediction:
ε̃ = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
                       ↑
                  guidance scale s

s = 1.0:  normal conditional generation
s = 7.5:  typical (strong adherence to prompt)
s > 15:   over-saturated, artifact-prone
```

---

## Math

### Latent Diffusion Framework

**Stage 1: Autoencoder** (trained separately)
- Encoder: $z = \mathcal{E}(x)$ compresses image $x \in \mathbb{R}^{H \times W \times 3}$ to $z \in \mathbb{R}^{h \times w \times c}$
- Decoder: $\hat{x} = \mathcal{D}(z)$ reconstructs from latent
- Typically $H/h = W/w = 8$ (8x downsampling), $c = 4$

**Stage 2: Diffusion in latent space** (same DDPM math, but on $z$ instead of $x$)

Forward: $z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$

Training: $\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$

where $c$ is the conditioning signal (text embedding).

### Cross-Attention Mechanism

In UNet block $l$ with spatial features $h_l \in \mathbb{R}^{n \times d}$ and text embeddings $c \in \mathbb{R}^{m \times d_c}$:

$$Q = h_l W_Q, \quad K = c W_K, \quad V = c W_V$$

$$\text{CrossAttn}(h_l, c) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This allows each spatial location in the image to attend to relevant words in the prompt.

### Classifier-Free Guidance

During training, randomly drop the conditioning (replace $c$ with $\emptyset$) with some probability (e.g., 10%).

During sampling:
$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot \left(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)\right)$$

where $s$ is the guidance scale. This amplifies the "direction" from unconditional to conditional.

Equivalently: $\tilde{\epsilon} = (1-s) \cdot \epsilon_{\text{uncond}} + s \cdot \epsilon_{\text{cond}}$

### Full Stable Diffusion Pipeline

```
Generation:
1. Encode text: c = CLIP_text("a photo of a cat")
2. Sample noise: z_T ~ N(0, I)  in latent space (64x64x4)
3. For t = T, T-1, ..., 1:
   a. Compute ε_uncond = ε_θ(z_t, t, ∅)
   b. Compute ε_cond   = ε_θ(z_t, t, c)
   c. Apply guidance: ε̃ = ε_uncond + s·(ε_cond - ε_uncond)
   d. Denoise step: z_{t-1} = denoise(z_t, ε̃, t)
4. Decode image: x = D(z_0)
```

---

## Code

### Conceptual Latent Diffusion Pipeline

```python
class LatentDiffusion:
    def __init__(self, autoencoder, unet, text_encoder, T=1000):
        self.ae = autoencoder    # pretrained, frozen
        self.unet = unet         # trainable
        self.text_enc = text_encoder  # pretrained (CLIP), frozen
        self.T = T
        self.betas = linear_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def train_step(self, images, text_prompts):
        # Encode images to latent space
        with torch.no_grad():
            z_0 = self.ae.encode(images)  # (B, 4, 64, 64)
            c = self.text_enc(text_prompts)  # (B, L, D)

        # Sample timesteps and noise
        t = torch.randint(0, self.T, (z_0.shape[0],))
        eps = torch.randn_like(z_0)

        # Forward diffusion in latent space
        alpha_bar_t = self.alpha_bars[t][:, None, None, None]
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * eps

        # Predict noise (conditioned on text)
        eps_pred = self.unet(z_t, t, c)

        return nn.functional.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, text_prompt, guidance_scale=7.5):
        c = self.text_enc(text_prompt)
        z = torch.randn(1, 4, 64, 64)

        for t in reversed(range(self.T)):
            t_batch = torch.tensor([t])

            # Classifier-free guidance
            eps_uncond = self.unet(z, t_batch, None)
            eps_cond = self.unet(z, t_batch, c)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Denoise step
            z = self._denoise_step(z, eps, t)

        # Decode to pixel space
        image = self.ae.decode(z)
        return image
```

### Cross-Attention Block

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, d_context, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_context, d_model)
        self.W_V = nn.Linear(d_context, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        """
        x: (B, N, D) spatial features
        context: (B, M, D_ctx) text embeddings
        """
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(context).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(context).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)
```

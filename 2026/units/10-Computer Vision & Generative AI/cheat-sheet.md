# Computer Vision & Generative AI — Cheat Sheet

> Quick reference for USAAIO 2026 | AI 520

---

## Object Detection

| Concept | Formula / Detail |
|---|---|
| **IoU** | $\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$ |
| **Anchor boxes** | Pre-defined bounding boxes at each spatial position; network predicts offsets $(\Delta x, \Delta y, \Delta w, \Delta h)$ |
| **NMS** | 1. Sort by confidence. 2. Take highest, suppress all with IoU > threshold. 3. Repeat. |
| **mAP** | Mean of AP across classes; AP = area under precision-recall curve |
| **Precision** | $\text{Precision} = \frac{TP}{TP + FP}$ (at given IoU threshold) |
| **Recall** | $\text{Recall} = \frac{TP}{TP + FN}$ |

**IoU computation (axis-aligned boxes)**:
```
x1 = max(A.x1, B.x1);  y1 = max(A.y1, B.y1)
x2 = min(A.x2, B.x2);  y2 = min(A.y2, B.y2)
intersection = max(0, x2-x1) * max(0, y2-y1)
union = area_A + area_B - intersection
IoU = intersection / union
```

---

## UNet

| Component | Detail |
|---|---|
| **Encoder** | Repeated (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool 2x2); channels double each level |
| **Decoder** | Upsample (ConvTranspose2d or bilinear) → concat skip → Conv → BN → ReLU → Conv → BN → ReLU |
| **Skip connections** | Concatenate encoder features with decoder features (preserves spatial detail) |
| **Output** | 1x1 Conv → num_classes channels |
| **Loss** | Cross-entropy or Dice loss for segmentation |

```
Encoder:        [64] → [128] → [256] → [512]
Bottleneck:     [1024]
Decoder:        [512] → [256] → [128] → [64]
Skip:           enc_64 ──────────────────→ dec_64
                enc_128 ─────────────→ dec_128
                enc_256 ──────→ dec_256
                enc_512 → dec_512
```

**Dice loss**: $\mathcal{L}_{\text{Dice}} = 1 - \frac{2|P \cap G|}{|P| + |G|} = 1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$

---

## Autoencoder

| Component | Formula |
|---|---|
| **Encoder** | $z = f_\phi(x)$ maps input to latent code |
| **Decoder** | $\hat{x} = g_\theta(z)$ reconstructs from latent code |
| **Loss** | $\mathcal{L} = \|x - \hat{x}\|^2$ (MSE) or BCE for normalized inputs |
| **Bottleneck** | $\dim(z) \ll \dim(x)$ forces compression |

```
x ──→ [Encoder] ──→ z (latent) ──→ [Decoder] ──→ x̂
           compress            reconstruct
```

---

## Variational Autoencoder (VAE)

| Component | Formula |
|---|---|
| **Encoder** | $q_\phi(z\|x)$: outputs $\mu, \log\sigma^2$ |
| **Reparameterization** | $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ |
| **Decoder** | $p_\theta(x\|z)$: reconstructs from sampled $z$ |
| **ELBO** | $\mathcal{L} = \mathbb{E}_{q(z\|x)}[\log p(x\|z)] - D_{KL}(q(z\|x) \| p(z))$ |
| **KL (Gaussian)** | $D_{KL} = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ |
| **Prior** | $p(z) = \mathcal{N}(0, I)$ |

**ELBO derivation**:
$$\log p(x) = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] + D_{KL}(q(z|x) \| p(z|x)) \geq \text{ELBO}$$

**Why reparameterization?** Sampling $z \sim q(z|x)$ is not differentiable w.r.t. $\phi$. Reparameterization moves stochasticity to $\epsilon$, enabling backprop through $\mu$ and $\sigma$.

---

## Generative Adversarial Networks (GANs)

| Component | Formula |
|---|---|
| **Minimax** | $\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$ |
| **D objective** | Maximize $\log D(x) + \log(1 - D(G(z)))$ |
| **G objective** | Minimize $\log(1 - D(G(z)))$ or maximize $\log D(G(z))$ (non-saturating) |
| **Mode collapse** | G produces limited variety; D can exploit this |
| **WGAN** | Uses Wasserstein distance; critic (not classifier); weight clipping or gradient penalty |

**Training loop**:
```
for each batch:
    1. Train D: maximize log D(x_real) + log(1 - D(G(z)))
    2. Train G: maximize log D(G(z))   [non-saturating]
```

---

## Diffusion Models (DDPM)

**Forward process** (add noise):
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I)$$

**Closed-form forward** (jump to any $t$):
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I)$$
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

**Reverse process** (denoise):
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t),\; \sigma_t^2 I)$$

**Training objective** (simplified):
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**Noise schedule**: $\beta_1 < \beta_2 < \cdots < \beta_T$ (linear from $10^{-4}$ to $0.02$)

**Sampling**:
```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
```

---

## Stable Diffusion (Latent Diffusion)

| Component | Detail |
|---|---|
| **Key idea** | Diffuse in latent space of pretrained autoencoder, not pixel space |
| **Autoencoder** | $z = \mathcal{E}(x)$, $\hat{x} = \mathcal{D}(z)$; compresses e.g. 512x512x3 → 64x64x4 |
| **UNet** | Predicts noise in latent space; uses cross-attention for conditioning |
| **Cross-attention** | $Q$ from UNet features, $K, V$ from text encoder output |
| **Classifier-free guidance** | $\tilde{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$; $s > 1$ strengthens conditioning |

---

## CLIP (Contrastive Language-Image Pre-training)

| Component | Detail |
|---|---|
| **Architecture** | Image encoder (ViT or ResNet) + Text encoder (Transformer) |
| **Image encoder** | $v_i = \text{proj}_v(\text{ViT}(I_i))$ |
| **Text encoder** | $t_j = \text{proj}_t(\text{Transformer}(T_j))$ |
| **Similarity** | $s_{ij} = \frac{v_i \cdot t_j}{\|v_i\| \|t_j\|} = \cos(v_i, t_j)$ |
| **Temperature** | $\text{logits}_{ij} = s_{ij} / \tau$ (learnable $\tau$) |
| **InfoNCE (image→text)** | $\mathcal{L}_i^{i2t} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_{k=1}^{N} \exp(s_{ik}/\tau)}$ |
| **InfoNCE (text→image)** | $\mathcal{L}_j^{t2i} = -\log \frac{\exp(s_{jj}/\tau)}{\sum_{k=1}^{N} \exp(s_{kj}/\tau)}$ |
| **Symmetric loss** | $\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}(\mathcal{L}_i^{i2t} + \mathcal{L}_i^{t2i})$ |

**Zero-shot classification**:
```
1. Encode image: v = image_encoder(image)
2. Encode class prompts: t_k = text_encoder("a photo of a {class_k}")
3. Compute cosine similarities: s_k = cos(v, t_k)
4. Predict: argmax_k s_k
```

**CLIP similarity matrix** (batch of N image-text pairs):

```
          text_1  text_2  ...  text_N
image_1  [pos]    neg     ...  neg
image_2   neg    [pos]    ...  neg
  ...     ...     ...     ...  ...
image_N   neg     neg     ... [pos]
```

Diagonal = positive pairs. Off-diagonal = negatives.

---

## Vision Transformer (ViT)

| Step | Operation | Shape |
|---|---|---|
| 1. Patch embedding | Split image into $P \times P$ patches, flatten, linear project | $(B, N, D)$ where $N = (H/P)(W/P)$ |
| 2. Class token | Prepend learnable `[CLS]` token | $(B, N+1, D)$ |
| 3. Position embedding | Add learnable position embeddings | $(B, N+1, D)$ |
| 4. Transformer encoder | $L$ layers of multi-head self-attention + FFN | $(B, N+1, D)$ |
| 5. Classification | MLP head on `[CLS]` token | $(B, C)$ |

**Patch embedding**: For 224x224 image with P=16: $N = (224/16)^2 = 196$ patches

---

## Adversarial Attacks

| Attack | Formula |
|---|---|
| **FGSM** | $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y))$ |
| **PGD** | $x^{(k+1)} = \Pi_{B_\epsilon(x)}\left[x^{(k)} + \alpha \cdot \text{sign}(\nabla_{x^{(k)}} \mathcal{L})\right]$ |
| **Perturbation budget** | $\|x_{adv} - x\|_\infty \leq \epsilon$ |
| **Adversarial training** | $\min_\theta \max_{\delta: \|\delta\| \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y)$ |

**FGSM steps**:
```
1. Forward pass: loss = L(f(x), y)
2. Backward pass: grad = d(loss)/dx
3. Perturb: x_adv = x + eps * sign(grad)
4. Clamp: x_adv = clamp(x_adv, 0, 1)
```

**PGD** = Iterative FGSM with projection back onto $\ell_\infty$ ball after each step.

---

## Quick Recipes

```
IoU:              intersection / union of two boxes
NMS:              sort by conf → take best → suppress high-IoU → repeat
UNet:             encoder (down) → bottleneck → decoder (up) + skip connections
Autoencoder:      encode → bottleneck → decode → MSE loss
VAE:              encode → (mu, logvar) → reparameterize → decode → recon + KL loss
GAN:              train D on real/fake → train G to fool D → alternate
DDPM forward:     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
DDPM train:       sample t, eps → compute x_t → predict eps_theta → MSE(eps, eps_theta)
DDPM sample:      x_T ~ N(0,I) → denoise step by step → x_0
Stable Diffusion: encode to latent → diffuse in latent → denoise → decode to pixel
CLIP:             encode image & text → cosine sim → InfoNCE loss
CLIP zero-shot:   encode image + class prompts → argmax cosine sim
ViT:              patchify → embed → [CLS] + pos → transformer → classify
FGSM:             x_adv = x + eps * sign(grad_x L)
PGD:              iterate FGSM + project onto eps-ball
```

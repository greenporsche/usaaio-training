# CLIP (Contrastive Language-Image Pre-training)

**Prerequisites**: Cosine similarity, softmax/cross-entropy, Transformers, ViT, contrastive learning
**USAAIO Relevance**: THE MOST CRITICAL TOPIC FOR ROUND 2. The 2025 Round 2 Problem 3 (100 points) was entirely about CLIP. Master every detail in this guide.

---

## Discovery

### The Core Question

> Can we train a single model that understands both images and text, learning to match them in a shared embedding space? And can this model then classify images into *any* category — even ones it has never been trained on — just by comparing image embeddings with text descriptions?

### Historical Context

- **Radford et al. (2021)**: "Learning Transferable Visual Models From Natural Language Supervision" — introduced CLIP at OpenAI.
- Trained on 400 million image-text pairs scraped from the internet.
- Key innovation: instead of training on fixed categories, learn a general image-text similarity function.
- Zero-shot ImageNet accuracy of 76.2% — competitive with supervised ResNet-50, without seeing any ImageNet training data.
- CLIP became the text encoder and training paradigm for Stable Diffusion, DALL-E 2, and many other multimodal models.

### Socratic Warm-Up

1. In a batch of $N$ image-text pairs, how many positive pairs and how many negative pairs are there?
2. What role does the temperature parameter $\tau$ play? What happens as $\tau \to 0$ vs $\tau \to \infty$?
3. Why is cosine similarity used instead of dot product? What would go wrong with raw dot products?
4. For zero-shot classification with 1000 classes, how many text embeddings do you need?

### Misconception Traps

- **"CLIP classifies images."** — CLIP computes *similarity* between image and text embeddings. Classification is done by finding the text prompt most similar to the image.
- **"The image and text encoders share weights."** — They are completely separate networks (different architectures). They only share the embedding *space* through learned projection heads.
- **"CLIP is trained with labeled data."** — CLIP uses image-text pairs from the internet — no manual labels. The text IS the supervision signal.
- **"Temperature is a hyperparameter."** — In CLIP, $\tau$ is a *learnable* parameter (log-parameterized).

---

## Intuition

### Dual Encoder Architecture

```
Image                           Text
"a photo of a dog"             "a photo of a dog"
┌─────────┐                    ┌──────────────────┐
│  🖼️     │                    │ "a photo of a    │
│ (image) │                    │  dog"            │
└────┬────┘                    └────────┬─────────┘
     │                                  │
     ▼                                  ▼
┌─────────────┐               ┌─────────────────┐
│ Image       │               │ Text Encoder    │
│ Encoder     │               │ (Transformer)   │
│ (ViT/ResNet)│               │                 │
└─────┬───────┘               └────────┬────────┘
      │                                │
      ▼                                ▼
┌─────────┐                    ┌─────────┐
│ Project │                    │ Project │
│ to D dim│                    │ to D dim│
└────┬────┘                    └────┬────┘
     │                              │
     ▼                              ▼
   v_i ∈ R^D                     t_j ∈ R^D
     │                              │
     └──────────┬───────────────────┘
                │
         cosine_sim(v_i, t_j)
```

### The Contrastive Learning Idea

In a batch of $N$ image-text pairs, there are $N$ positive (matching) pairs on the diagonal and $N^2 - N$ negative (mismatched) pairs:

```
           t_1    t_2    t_3    t_4
    v_1  [ POS    neg    neg    neg  ]
    v_2  [ neg    POS    neg    neg  ]
    v_3  [ neg    neg    POS    neg  ]
    v_4  [ neg    neg    neg    POS  ]

Similarity matrix S: S_ij = cos(v_i, t_j) / τ

Goal: make diagonal entries (POS) large, off-diagonal (neg) small
→ This is just cross-entropy with labels = [0, 1, 2, 3]!
```

### Temperature Scaling

```
τ = 1.0 (high temperature):     τ = 0.01 (low temperature):
Softmax is smooth                Softmax is sharp

[0.35, 0.25, 0.20, 0.20]       [0.97, 0.01, 0.01, 0.01]
  ^--- correct class               ^--- correct class

High τ: model is lenient          Low τ: model is strict
        (many things look similar)        (must be very precise match)
```

$\tau$ controls the "sharpness" of the distribution. Smaller $\tau$ makes the model focus harder on the exact match.

### Zero-Shot Classification

```
Image of a dog:                Text prompts:
┌─────────┐                    "a photo of a cat"  → t_1
│  🐕     │  → v              "a photo of a dog"  → t_2
└─────────┘                    "a photo of a bird" → t_3

Cosine similarities:
cos(v, t_1) = 0.15  (cat)
cos(v, t_2) = 0.92  (dog)  ← highest!
cos(v, t_3) = 0.08  (bird)

Prediction: "dog" (argmax similarity)
```

No training on these specific classes needed! The model generalizes through language.

---

## Math

### Setup

Given a batch of $N$ image-text pairs $\{(I_i, T_i)\}_{i=1}^N$:

**Image encoder**: $v_i = \frac{W_v \cdot \text{ViT}(I_i)}{\|W_v \cdot \text{ViT}(I_i)\|}$ (L2-normalized)

**Text encoder**: $t_j = \frac{W_t \cdot \text{Transformer}(T_j)}{\|W_t \cdot \text{Transformer}(T_j)\|}$ (L2-normalized)

where $W_v, W_t$ are learned linear projection matrices.

### Cosine Similarity Matrix

$$S_{ij} = v_i^T t_j = \cos(v_i, t_j)$$

Since $v_i$ and $t_j$ are L2-normalized, dot product equals cosine similarity.

**Scaled logits**: $L_{ij} = S_{ij} / \tau$ where $\tau$ is the (learnable) temperature.

### InfoNCE Loss (Image-to-Text)

For image $i$, the positive text is $t_i$ and all other texts $t_{k \neq i}$ are negatives:

$$\mathcal{L}_i^{i2t} = -\log \frac{\exp(S_{ii}/\tau)}{\sum_{k=1}^{N} \exp(S_{ik}/\tau)}$$

This is exactly cross-entropy loss where the "correct class" is index $i$.

### InfoNCE Loss (Text-to-Image)

Symmetrically, for text $j$:

$$\mathcal{L}_j^{t2i} = -\log \frac{\exp(S_{jj}/\tau)}{\sum_{k=1}^{N} \exp(S_{kj}/\tau)}$$

### Symmetric CLIP Loss

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2N}\sum_{i=1}^{N}\left(\mathcal{L}_i^{i2t} + \mathcal{L}_i^{t2i}\right)$$

**Equivalently**, using PyTorch cross-entropy:
```python
labels = torch.arange(N)
loss_i2t = F.cross_entropy(logits, labels)      # rows
loss_t2i = F.cross_entropy(logits.T, labels)     # columns
loss = (loss_i2t + loss_t2i) / 2
```

### Temperature Parameter

In CLIP, $\tau$ is parameterized as $\tau = \exp(\log \tau)$ where $\log \tau$ is a learnable scalar, initialized to $\log(1/0.07) \approx 2.66$.

The gradient pushes $\tau$ to a value that makes the loss neither too easy (high $\tau$, everything looks similar) nor too hard (low $\tau$, gradients vanish for non-matching pairs).

### Connection to InfoNCE / NT-Xent

The loss is also known as:
- **InfoNCE** (van den Oord et al., 2018) — information noise-contrastive estimation
- **NT-Xent** (Chen et al., 2020, SimCLR) — normalized temperature-scaled cross-entropy

General form:
$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

### Zero-Shot Classification

Given an image $I$ and $K$ class names $\{c_1, \ldots, c_K\}$:

1. Compute image embedding: $v = \text{ImageEncoder}(I)$
2. Create text prompts: $T_k = \text{"a photo of a } c_k\text{"}$
3. Compute text embeddings: $t_k = \text{TextEncoder}(T_k)$ for $k = 1, \ldots, K$
4. Compute similarities: $s_k = \cos(v, t_k)$
5. Predict: $\hat{y} = \arg\max_k s_k$

**With softmax probabilities**:
$$P(y = k | I) = \frac{\exp(s_k / \tau)}{\sum_{j=1}^{K} \exp(s_j / \tau)}$$

---

## Code

### CLIP-style Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim, temperature_init=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Projection heads to shared embedding space
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim, bias=False)

        # Learnable temperature (log-parameterized)
        self.log_temperature = nn.Parameter(torch.tensor(float(-torch.log(torch.tensor(temperature_init)))))

    def encode_image(self, images):
        features = self.image_encoder(images)              # (B, img_dim)
        projected = self.image_proj(features)               # (B, embed_dim)
        return F.normalize(projected, dim=-1)               # L2 normalize

    def encode_text(self, texts):
        features = self.text_encoder(texts)                 # (B, txt_dim)
        projected = self.text_proj(features)                # (B, embed_dim)
        return F.normalize(projected, dim=-1)               # L2 normalize

    def forward(self, images, texts):
        image_embeds = self.encode_image(images)            # (N, D)
        text_embeds = self.encode_text(texts)               # (N, D)

        # Cosine similarity matrix scaled by temperature
        temperature = torch.exp(self.log_temperature)
        logits = (image_embeds @ text_embeds.T) / temperature  # (N, N)

        return logits
```

### CLIP Loss

```python
def clip_loss(logits):
    """
    logits: (N, N) similarity matrix scaled by temperature
    Diagonal entries are positive pairs.
    """
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)

    # Image-to-text loss (each row is a distribution over texts)
    loss_i2t = F.cross_entropy(logits, labels)

    # Text-to-image loss (each column is a distribution over images)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

### Training Loop

```python
model = CLIPModel(image_encoder, text_encoder, embed_dim=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for epoch in range(num_epochs):
    for images, texts in dataloader:
        logits = model(images, texts)
        loss = clip_loss(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Zero-Shot Classification

```python
@torch.no_grad()
def zero_shot_classify(model, image, class_names, prompt_template="a photo of a {}"):
    """
    image: single image tensor (C, H, W)
    class_names: list of K class name strings
    Returns: predicted class index
    """
    # Encode image
    image_embed = model.encode_image(image.unsqueeze(0))  # (1, D)

    # Encode all class prompts
    prompts = [prompt_template.format(name) for name in class_names]
    text_embeds = model.encode_text(prompts)               # (K, D)

    # Cosine similarities
    similarities = (image_embed @ text_embeds.T).squeeze(0)  # (K,)

    # Predict
    predicted_idx = similarities.argmax().item()
    return class_names[predicted_idx], similarities
```

### Manual InfoNCE Computation (Exam Style)

```python
def infonce_loss_manual(image_embeds, text_embeds, temperature):
    """
    Compute InfoNCE loss step by step (for understanding).
    image_embeds: (N, D) L2-normalized
    text_embeds: (N, D) L2-normalized
    temperature: scalar
    """
    N = image_embeds.shape[0]

    # Step 1: Cosine similarity matrix
    sim_matrix = image_embeds @ text_embeds.T      # (N, N)

    # Step 2: Scale by temperature
    logits = sim_matrix / temperature               # (N, N)

    # Step 3: Labels — diagonal is positive
    labels = torch.arange(N)                        # [0, 1, 2, ..., N-1]

    # Step 4: Image-to-text loss
    # For each image i, compute softmax over texts
    loss_i2t = 0.0
    for i in range(N):
        numerator = torch.exp(logits[i, i])         # positive pair
        denominator = torch.sum(torch.exp(logits[i, :]))  # all pairs
        loss_i2t += -torch.log(numerator / denominator)
    loss_i2t /= N

    # Step 5: Text-to-image loss
    loss_t2i = 0.0
    for j in range(N):
        numerator = torch.exp(logits[j, j])
        denominator = torch.sum(torch.exp(logits[:, j]))
        loss_t2i += -torch.log(numerator / denominator)
    loss_t2i /= N

    # Step 6: Symmetric loss
    return (loss_i2t + loss_t2i) / 2
```

### Numeric Example

```python
# Batch of 3 image-text pairs
# After L2 normalization:
v = torch.tensor([[1.0, 0.0],    # image 0
                   [0.0, 1.0],    # image 1
                   [0.707, 0.707]])  # image 2

t = torch.tensor([[0.95, 0.05],   # text 0 (matches image 0)
                   [0.05, 0.95],   # text 1 (matches image 1)
                   [0.6, 0.8]])    # text 2 (matches image 2)

# Normalize
v = F.normalize(v, dim=-1)
t = F.normalize(t, dim=-1)

tau = 0.07
sim = v @ t.T           # (3, 3) cosine similarity matrix
logits = sim / tau       # scaled logits

# Labels = [0, 1, 2] (diagonal)
labels = torch.arange(3)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

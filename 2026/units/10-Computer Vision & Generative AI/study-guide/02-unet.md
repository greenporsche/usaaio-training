# UNet

**Prerequisites**: CNNs, pooling, transposed convolution, skip connections
**USAAIO Relevance**: UNet is the backbone of diffusion model denoisers. Understanding its encoder-decoder structure with skip connections is essential for Stable Diffusion and DDPM problems.

---

## Discovery

### The Core Question

> How do you produce a dense, pixel-level output (segmentation map) from an image, preserving both high-level semantics and fine spatial detail?

Standard CNNs downsample aggressively to capture global context, losing spatial resolution. UNet solves this with a symmetric encoder-decoder architecture connected by skip connections.

### Historical Context

- **Ronneberger, Fischer, Brox (2015)**: Introduced UNet for biomedical image segmentation at MICCAI.
- Named "U-Net" because the architecture diagram looks like the letter U.
- Key innovation: skip connections that concatenate encoder features with decoder features, enabling the decoder to recover fine-grained spatial information lost during downsampling.
- Later adopted as the denoising network in diffusion models (DDPM, Stable Diffusion).

### Socratic Warm-Up

1. Why does a standard CNN classifier lose spatial information? (Hint: pooling layers.)
2. If you upsample a low-resolution feature map, what information is missing? How could you recover it?
3. Why concatenation of skip connections rather than addition? What's the tradeoff?

### Misconception Traps

- **"Skip connections in UNet are the same as in ResNet."** — ResNet uses *additive* skip connections (residual); UNet uses *concatenation* skip connections (preserves separate feature channels).
- **"The decoder just upsamples."** — It also has conv layers to combine the upsampled features with skip connection features.
- **"UNet only works for segmentation."** — UNet is used in diffusion models as the noise predictor $\epsilon_\theta$.

---

## Intuition

### The U-Shaped Architecture

```
Input (1, 572, 572)
  │
  ▼
[Conv Block] → 64 channels ─────────────────────────────────→ [Concat + Conv Block] → 64
  │                                                             ▲
  ▼ MaxPool                                                     │ UpConv
[Conv Block] → 128 channels ──────────────────→ [Concat + Conv Block] → 128
  │                                                ▲
  ▼ MaxPool                                        │ UpConv
[Conv Block] → 256 channels ──────→ [Concat + Conv Block] → 256
  │                                    ▲
  ▼ MaxPool                            │ UpConv
[Conv Block] → 512 channels → [Concat + Conv Block] → 512
  │                              ▲
  ▼ MaxPool                      │ UpConv
[Conv Block] → 1024 channels ───┘
            (Bottleneck)
```

### Why Skip Connections?

```
Without skips:                    With skips:
High-res input                    High-res input
    ↓ downsample                      ↓ downsample
Low-res features                  Low-res features ──→ (saved)
    ↓ upsample                        ↓ upsample
Blurry output (lost detail)       Concat(upsampled, saved) → Sharp output!
```

The encoder captures **what** (semantic content) but loses **where** (spatial detail). Skip connections provide the decoder with the **where** information directly.

### Concatenation vs Addition

```
Skip via concatenation:           Skip via addition:
Encoder features: [C channels]   Encoder features: [C channels]
Decoder features: [C channels]   Decoder features: [C channels]
Result: [2C channels]            Result: [C channels]
→ More parameters, more expressive → Fewer parameters, forces alignment
```

UNet uses concatenation for richer feature combination.

---

## Math

### Encoder Block

Each encoder level applies two 3x3 convolutions with ReLU, then 2x2 max pooling:

$$h^{(l)} = \text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(\text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(h^{(l-1)}))))))$$
$$h^{(l)}_{\text{down}} = \text{MaxPool}_{2\times2}(h^{(l)})$$

Spatial dimensions halve: $(H, W) \to (H/2, W/2)$. Channels double: $C \to 2C$.

### Decoder Block

Each decoder level applies upsampling, concatenates the skip connection, then two 3x3 convolutions:

$$h^{(l)}_{\text{up}} = \text{ConvTranspose}_{2\times2}(h^{(l+1)}_{\text{dec}})$$
$$h^{(l)}_{\text{cat}} = \text{Concat}(h^{(l)}_{\text{up}}, h^{(l)}_{\text{enc}})$$
$$h^{(l)}_{\text{dec}} = \text{ConvBlock}(h^{(l)}_{\text{cat}})$$

### Shape Tracking

For input $(B, 1, 256, 256)$ with 4 encoder levels:

| Level | After Conv Block | After Pool/Up | Channels |
|---|---|---|---|
| Enc 1 | $(B, 64, 256, 256)$ | $(B, 64, 128, 128)$ | 64 |
| Enc 2 | $(B, 128, 128, 128)$ | $(B, 128, 64, 64)$ | 128 |
| Enc 3 | $(B, 256, 64, 64)$ | $(B, 256, 32, 32)$ | 256 |
| Enc 4 | $(B, 512, 32, 32)$ | $(B, 512, 16, 16)$ | 512 |
| Bottleneck | $(B, 1024, 16, 16)$ | — | 1024 |
| Dec 4 | $(B, 512, 32, 32)$ | — | 512 |
| Dec 3 | $(B, 256, 64, 64)$ | — | 256 |
| Dec 2 | $(B, 128, 128, 128)$ | — | 128 |
| Dec 1 | $(B, 64, 256, 256)$ | — | 64 |
| Output | $(B, C_{out}, 256, 256)$ | — | $C_{out}$ |

### Segmentation Loss

**Cross-entropy per pixel**:
$$\mathcal{L} = -\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\sum_{c=1}^{C} y_{ijc} \log \hat{y}_{ijc}$$

**Dice loss** (handles class imbalance):
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}$$

---

## Code

### UNet in PyTorch

```python
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder path
        for f in features:
            self.encoders.append(ConvBlock(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder path
        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))  # f*2 because of concat

        # Final 1x1 conv
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse

        # Decoder
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)      # upsample
            skip = skip_connections[i // 2]
            x = torch.cat([skip, x], dim=1)  # concat along channels
            x = self.decoders[i + 1](x)  # conv block

        return self.final(x)
```

### Shape Verification

```python
model = UNet(in_channels=1, out_channels=2)
x = torch.randn(1, 1, 256, 256)
print(model(x).shape)  # torch.Size([1, 2, 256, 256])
```

# Architectures

**Prerequisites**: CNNs (Study Guide 06), batch normalization (Study Guide 05)
**USAAIO Relevance**: Round 1 tests understanding of skip connections, parameter counting for known architectures, and identifying design principles. Round 2 requires implementing ResNet blocks from scratch. Understanding architectural trade-offs is essential for model selection.

---

## Discovery

### The Depth Revolution

A key question in deep learning history: does depth matter? Can a wider but shallower network match a deeper one?

In theory, a single wide hidden layer can approximate any function (universal approximation). In practice, deeper networks are exponentially more efficient — they can represent hierarchical features with far fewer parameters.

But there was a paradox: very deep networks (20+ layers) trained **worse** than shallower ones, even on the training set. This was NOT overfitting — it was an optimization problem. The vanishing gradient made early layers nearly impossible to train.

> **Socratic question**: If a 20-layer network can represent everything a 10-layer network can (by setting the extra 10 layers to identity), why does the 20-layer network sometimes achieve HIGHER training error?
>
> *Because gradient descent cannot easily find the identity mapping. The optimization landscape is harder to navigate in deeper networks. ResNet's skip connections solve this by making the identity mapping the DEFAULT — the network only needs to learn the residual.*

### Timeline of Key Architectures

```
1998: LeNet-5 ────→ 2012: AlexNet ────→ 2014: VGG ────→ 2014: GoogLeNet
 (5 layers)          (8 layers)          (16-19 layers)    (22 layers)
                                                                 ↓
                                              2015: ResNet (152 layers!)
```

---

## Intuition

### VGG: The Power of Small Filters

VGG's key insight: replace large filters with stacks of $3 \times 3$ filters.

Two $3\times3$ convolutions have the same receptive field as one $5\times5$, but with:
- Fewer parameters: $2 \times (9C^2) = 18C^2$ vs $25C^2$
- More nonlinearity: 2 ReLU activations instead of 1
- Better gradient flow: shorter paths through the graph

```
VGG-16 Architecture:
┌────────────────────────────────────────┐
│ Input: (3, 224, 224)                   │
├────────────────────────────────────────┤
│ 2× Conv(3, 64, 3) + MaxPool(2)        │  → (64, 112, 112)
│ 2× Conv(64, 128, 3) + MaxPool(2)      │  → (128, 56, 56)
│ 3× Conv(128, 256, 3) + MaxPool(2)     │  → (256, 28, 28)
│ 3× Conv(256, 512, 3) + MaxPool(2)     │  → (512, 14, 14)
│ 3× Conv(512, 512, 3) + MaxPool(2)     │  → (512, 7, 7)
├────────────────────────────────────────┤
│ Flatten → FC(25088, 4096) → FC(4096, 4096) → FC(4096, 1000) │
└────────────────────────────────────────┘
Total: ~138M parameters (most in FC layers!)
```

### GoogLeNet / Inception: Parallel Pathways

Instead of choosing a single filter size, use ALL sizes in parallel and let the network decide:

```
Inception Module:
                    ┌─────────────┐
                    │   Input     │
                    └──┬──┬──┬──┬┘
                       │  │  │  │
                    ┌──┘  │  │  └──┐
                    │     │  │     │
                 ┌──▼──┐┌─▼─┐┌─▼─┐┌──▼──────┐
                 │1×1  ││1×1││1×1││MaxPool3×3│
                 │conv ││   ││   ││          │
                 └──┬──┘│   ││   │└────┬─────┘
                    │  ┌▼──┐┌▼──┐     │
                    │  │3×3││5×5│  ┌──▼──┐
                    │  │   ││   │  │1×1  │
                    │  └─┬─┘└─┬─┘  └──┬──┘
                    │    │    │       │
                    └──┬─┴──┬─┴───┬──┘
                       │    │     │
                    ┌──▼────▼─────▼──┐
                    │  Concatenate   │
                    │  along channels│
                    └────────────────┘
```

The $1\times1$ convolutions before $3\times3$ and $5\times5$ reduce channel dimensions, dramatically cutting computation.

### ResNet: Skip Connections

The key innovation — add the input directly to the output:

```
Residual Block:
         x
         │
    ┌────┴────┐
    │         │
    ▼         │
 ┌─────┐     │
 │Conv │     │
 │ BN  │     │
 │ReLU │     │
 ├─────┤     │
 │Conv │     │
 │ BN  │     │
 └──┬──┘     │
    │         │
    ▼         │
   [+] ◄─────┘   ← Skip connection (identity)
    │
    ▼
  ReLU
    │
    ▼
  output = ReLU(F(x) + x)
```

**Why it works**:

1. **Gradient highway**: $\frac{\partial}{\partial x}[F(x) + x] = F'(x) + I$. Even if $F'(x) \approx 0$, the gradient is still $I$ — it flows directly through the skip connection.

2. **Easy to learn identity**: If the optimal transformation is close to identity, the network only needs to learn $F(x) \approx 0$ (the residual), which is easier than learning the full mapping.

3. **Enables extreme depth**: ResNets with 152 layers train successfully, while plain networks degrade after ~20 layers.

### Bottleneck Block (ResNet-50+)

For deeper ResNets, a bottleneck design reduces computation:

```
Bottleneck Block:
    x (256 channels)
    │
    ▼
  1×1 Conv (256 → 64)    ← reduce channels
    │
    ▼
  3×3 Conv (64 → 64)     ← main computation (cheap!)
    │
    ▼
  1×1 Conv (64 → 256)    ← restore channels
    │
   [+] ◄── x              ← skip connection
    ▼
```

---

## Math

### VGG Parameter Count

For VGG-16:

| Block | Layers | Params per layer | Total |
|---|---|---|---|
| Block 1 | Conv(3,64,3) × 2 | $3 \times 64 \times 9 + 64 = 1{,}792$; $64 \times 64 \times 9 + 64 = 36{,}928$ | $38{,}720$ |
| Block 2 | Conv(64,128,3) × 2 | $64 \times 128 \times 9 + 128 \times 2 = 147{,}712$ | $147{,}712$ |
| Block 3 | Conv(128,256,3) × 3 | $295{,}168 + 590{,}080 \times 2$ | $1{,}475{,}328$ |
| Block 4 | Conv(256,512,3) × 3 | $1{,}180{,}160 + 2{,}359{,}808 \times 2$ | $5{,}899{,}776$ |
| Block 5 | Conv(512,512,3) × 3 | $2{,}359{,}808 \times 3$ | $7{,}079{,}424$ |
| FC1 | Linear(25088, 4096) | $102{,}764{,}544$ | |
| FC2 | Linear(4096, 4096) | $16{,}781{,}312$ | |
| FC3 | Linear(4096, 1000) | $4{,}097{,}000$ | |

**Key observation**: ~90% of parameters are in the FC layers, but ~90% of computation is in the conv layers.

### ResNet Identity vs. Projection Shortcut

**Identity shortcut** (when dimensions match): $y = F(x) + x$

**Projection shortcut** (when dimensions change): $y = F(x) + W_s x$

where $W_s$ is a $1\times1$ convolution that matches dimensions. Used when stride > 1 or when channels change.

Parameter overhead of projection: $C_{out} \times C_{in} \times 1 \times 1$ (minimal).

### Inception $1\times1$ Convolution as Dimensionality Reduction

A $1\times1$ convolution with $C_{out} < C_{in}$ acts as a per-pixel fully connected layer that reduces the channel dimension:

$$\text{params} = C_{in} \times C_{out} + C_{out}$$

Before a $5\times5$ convolution: without $1\times1$ reduction, params $= C_{in} \times C_{out} \times 25$. With $1\times1$ reduction to $C_r$ channels first: params $= C_{in} \times C_r + C_r \times C_{out} \times 25$.

If $C_r \ll C_{in}$, this is a massive reduction.

---

## Code

### ResNet Block Implementation

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet basic block (used in ResNet-18, ResNet-34)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride=stride, bias=False),           # 1×1 conv
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)                             # (B, C_out, H', W')

        out = torch.relu(self.bn1(self.conv1(x)))              # (B, C_out, H', W')
        out = self.bn2(self.conv2(out))                         # (B, C_out, H', W')

        out = out + identity                                     # Skip connection
        out = torch.relu(out)                                    # ReLU after addition
        return out                                               # (B, C_out, H', W')
```

### Bottleneck Block (ResNet-50+)

```python
class BottleneckBlock(nn.Module):
    """ResNet bottleneck block (used in ResNet-50, 101, 152)."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = torch.relu(self.bn1(self.conv1(x)))    # 1×1: reduce
        out = torch.relu(self.bn2(self.conv2(out)))   # 3×3: convolve
        out = self.bn3(self.conv3(out))                # 1×1: expand

        out = out + identity
        out = torch.relu(out)
        return out
```

### Simple VGG-like Network

```python
def make_vgg_block(in_c, out_c, num_convs):
    """Create a VGG block: num_convs Conv-BN-ReLU layers + MaxPool."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),     # (B,3,32,32) → (B,64,16,16)
            make_vgg_block(64, 128, 2),   # → (B,128,8,8)
            make_vgg_block(128, 256, 3),  # → (B,256,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

### Comparing Training With/Without Skip Connections

```python
import torch
import torch.nn as nn

class PlainNet(nn.Module):
    """Plain deep network (no skip connections)."""
    def __init__(self, depth=20, width=64):
        super().__init__()
        layers = [nn.Linear(width, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.net(x))

class ResidualNet(nn.Module):
    """Deep network with skip connections."""
    def __init__(self, depth=20, width=64):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x)) + x      # Skip connection!
        return self.head(x)

# The ResidualNet will train much more easily for depth >= 20
```

# Convolutional Neural Networks (CNNs)

**Prerequisites**: Forward/backward propagation, activation functions, batch normalization
**USAAIO Relevance**: CNNs are fundamental for vision tasks. Round 1 tests output shape computation and parameter counting. Round 2 requires implementing conv2d from scratch (without `nn.Conv2d`). Receptive field analysis appears in architecture design questions.

---

## Discovery

### Why Not Use MLPs for Images?

A 256x256 RGB image has $256 \times 256 \times 3 = 196{,}608$ pixels. A single fully connected hidden layer with 1024 neurons would need $196{,}608 \times 1024 \approx 200$ million parameters — just for one layer.

Three fundamental problems:
1. **Too many parameters**: Overfitting and memory/compute requirements
2. **No spatial awareness**: A fully connected layer treats pixel (0,0) and pixel (255,255) the same way
3. **No translation invariance**: An MLP trained on cats in the center cannot recognize cats in the corner

> **Socratic question**: If you see a vertical edge at position (10, 50) in an image, should the detector for "vertical edge" be different from one at position (100, 200)?
>
> *No — the same pattern should be detected everywhere. This is the key insight of convolution: the same filter (kernel) is applied at every spatial position.*

Convolution solves all three problems:
1. **Weight sharing**: One small kernel is reused across the entire image
2. **Local connectivity**: Each neuron only looks at a small local patch
3. **Translation equivariance**: The same pattern is detected regardless of position

### Historical Note

Yann LeCun's LeNet-5 (1998) demonstrated CNNs for handwritten digit recognition, but it took until 2012 (AlexNet winning ImageNet) for CNNs to dominate computer vision. The key enablers were GPU computing and large datasets.

---

## Intuition

### The Convolution Operation

A convolution slides a small kernel (filter) across the input, computing a dot product at each position:

```
Input (5×5):                 Kernel (3×3):        Output (3×3):
┌───┬───┬───┬───┬───┐       ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │       │ 1 │ 0 │ 1 │       Position (0,0):
├───┼───┼───┼───┼───┤       ├───┼───┼───┤       1·1 + 0·0 + 1·1
│ 0 │ 1 │ 0 │ 1 │ 0 │       │ 0 │ 1 │ 0 │     + 0·0 + 1·1 + 0·0
├───┼───┼───┼───┼───┤       ├───┼───┼───┤     + 1·1 + 0·0 + 1·1
│ 1 │ 0 │ 1 │ 0 │ 1 │       │ 1 │ 0 │ 1 │     = 4
├───┼───┼───┼───┼───┤       └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

The kernel slides with stride $S$ and optionally pads the input with $P$ zeros on each side.

### Stride and Padding

```
Stride = 1 (default):           Stride = 2:
┌─┬─┬─┐─ ─ ─                  ┌─┬─┬─┐─ ─ ─
│K│K│K│                        │K│K│K│
├─┼─┼─┤                        ├─┼─┼─┤
│K│K│K│→  next: shift by 1     │K│K│K│→  next: shift by 2
├─┼─┼─┤                        ├─┼─┼─┤
│K│K│K│                        │K│K│K│
└─┴─┴─┘                        └─┴─┴─┘

Padding = 0 (valid):           Padding = 1 (same for K=3, S=1):
Input 5×5 → Output 3×3         Input 5×5 → Output 5×5
(shrinks by K-1)               (0 border added)
```

**"Same" padding**: $P = \lfloor K/2 \rfloor$ with $S=1$ keeps spatial dimensions unchanged.

### Multi-Channel Convolution

Real convolutions operate on **volumes** — the kernel has the same depth as the input:

```
Input: (C_in, H, W)     Kernel: (C_in, K, K)     Output: one (H', W') feature map
  ┌──────┐                 ┌────┐
  │ C_in │                 │C_in│
  │ × H  │     *           │× K │    = one 2D map
  │ × W  │                 │× K │
  └──────┘                 └────┘

C_out kernels → C_out output feature maps → Output: (C_out, H', W')
```

Each output channel is produced by one kernel. The kernel sums across all input channels.

### Pooling

Pooling reduces spatial dimensions:

```
Max Pool (2×2, stride 2):
┌───┬───┬───┬───┐         ┌───┬───┐
│ 1 │ 3 │ 2 │ 1 │         │ 4 │ 6 │
├───┼───┼───┼───┤    →    ├───┼───┤
│ 4 │ 2 │ 6 │ 5 │         │ 8 │ 7 │
├───┼───┼───┼───┤         └───┴───┘
│ 7 │ 1 │ 3 │ 2 │
├───┼───┼───┼───┤         (4×4 → 2×2)
│ 8 │ 6 │ 5 │ 7 │
└───┴───┴───┴───┘
```

Pooling has **zero learnable parameters**. It provides:
- Spatial downsampling (reduces computation)
- Slight translation invariance
- Increased receptive field

---

## Math

### Output Spatial Dimensions

For input spatial size $H_{in} \times W_{in}$, kernel size $K$, padding $P$, stride $S$:

$$H_{out} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} - K + 2P}{S} \right\rfloor + 1$$

**Common configurations**:
- $K=3, P=1, S=1$: output = input (same convolution)
- $K=3, P=1, S=2$: output = input/2 (halves spatial dims)
- $K=1, P=0, S=1$: output = input (pointwise / 1x1 convolution)

### Parameter Count

For `Conv2d(C_in, C_out, K, bias=True)`:

$$\text{params} = C_{out} \times (C_{in} \times K \times K + 1)$$

The "$+1$" accounts for one bias per output channel.

**Example**: `Conv2d(3, 64, 3)` → $64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1{,}792$.

### Convolution as Matrix Multiplication (im2col)

Convolution can be reformulated as a matrix multiplication. For each sliding window position, extract a column of the input (the patch), stack all columns into a matrix, then multiply by the flattened kernel:

$$Y = W \cdot X_{\text{col}}$$

where $W \in \mathbb{R}^{C_{out} \times (C_{in} \cdot K \cdot K)}$ and $X_{\text{col}} \in \mathbb{R}^{(C_{in} \cdot K \cdot K) \times (H_{out} \cdot W_{out})}$.

This is how convolution is actually implemented on GPUs — it leverages highly optimized GEMM (general matrix multiply) routines.

### Receptive Field

The receptive field is the region of the input that affects a single output neuron:

For $L$ convolutional layers, each with kernel size $K_l$ and stride $S_l$:

$$r_L = 1 + \sum_{l=1}^{L} (K_l - 1) \prod_{i=1}^{l-1} S_i$$

**Example**: Two $3\times3$ conv layers with stride 1, then $2\times2$ max pool with stride 2, then $3\times3$ conv with stride 1:
- After layer 1: $r = 1 + (3-1) \times 1 = 3$
- After layer 2: $r = 3 + (3-1) \times 1 = 5$
- After pool: $r = 5 + (2-1) \times 1 = 6$... (but pool stride changes subsequent calculations)

For the common case of all stride-1 convolutions with kernel $K$:
$$r_L = 1 + L(K-1)$$

So 5 layers of $3\times3$ convolutions have receptive field $1 + 5 \times 2 = 11$.

### FLOPs per Conv Layer

For output spatial size $H_{out} \times W_{out}$:

$$\text{FLOPs} \approx 2 \times C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}$$

---

## Code

### Conv2d from Scratch (im2col)

```python
import torch
import torch.nn.functional as F

def conv2d_naive(x, weight, bias=None, stride=1, padding=0):
    """
    Manual 2D convolution using loops (slow but clear).
    x: (B, C_in, H, W)
    weight: (C_out, C_in, K_h, K_w)
    Returns: (B, C_out, H_out, W_out)
    """
    if padding > 0:
        x = F.pad(x, [padding]*4)                      # Pad all sides

    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1

    output = torch.zeros(B, C_out, H_out, W_out)       # (B, C_out, H_out, W_out)

    for b in range(B):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = x[b, :, h_start:h_start+K_h, w_start:w_start+K_w]
                    output[b, co, i, j] = (patch * weight[co]).sum()
                    if bias is not None:
                        output[b, co, i, j] += bias[co]
    return output

# Verify against PyTorch
x = torch.randn(2, 3, 8, 8)                             # (B=2, C=3, H=8, W=8)
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
my_out = conv2d_naive(x, conv.weight, conv.bias, stride=1, padding=1)
pt_out = conv(x)
print(f"Match: {torch.allclose(my_out, pt_out, atol=1e-5)}")  # True
```

### Conv2d using Unfold (Vectorized)

```python
def conv2d_unfold(x, weight, bias=None, stride=1, padding=0):
    """
    Conv2d using unfold (im2col approach — fast, no loops).
    x: (B, C_in, H, W)
    weight: (C_out, C_in, K_h, K_w)
    Returns: (B, C_out, H_out, W_out)
    """
    if padding > 0:
        x = F.pad(x, [padding]*4)

    B, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1

    # Unfold: extract all patches as columns
    # (B, C_in, H, W) → (B, C_in*K_h*K_w, H_out*W_out)
    x_unfold = x.unfold(2, K_h, stride).unfold(3, K_w, stride)   # (B, C_in, H_out, W_out, K_h, K_w)
    x_unfold = x_unfold.contiguous().reshape(B, C_in*K_h*K_w, H_out*W_out)

    # Flatten weight: (C_out, C_in*K_h*K_w)
    w_flat = weight.reshape(C_out, -1)

    # Matrix multiply: (C_out, C_in*K*K) @ (B, C_in*K*K, H_out*W_out) → (B, C_out, H_out*W_out)
    out = torch.bmm(w_flat.unsqueeze(0).expand(B, -1, -1), x_unfold)

    if bias is not None:
        out = out + bias.reshape(1, -1, 1)

    return out.reshape(B, C_out, H_out, W_out)                    # (B, C_out, H_out, W_out)
```

### Parameter Counting for CNN Architectures

```python
import torch.nn as nn

def count_conv_params(C_in, C_out, K, bias=True):
    """Count parameters for a Conv2d layer."""
    params = C_out * C_in * K * K
    if bias:
        params += C_out
    return params

# Example: simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # (B,3,32,32) → (B,32,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (B,32,32,32) → (B,32,16,16)
            nn.Conv2d(32, 64, 3, padding=1),   # (B,32,16,16) → (B,64,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (B,64,16,16) → (B,64,8,8)
        )
        self.classifier = nn.Linear(64*8*8, num_classes)

    def forward(self, x):
        x = self.features(x)                   # (B,3,32,32) → (B,64,8,8)
        x = x.flatten(1)                        # (B,64,8,8) → (B,4096)
        x = self.classifier(x)                  # (B,4096) → (B,10)
        return x

model = SimpleCNN()
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total:,}")
# Conv1: 3*32*3*3 + 32 = 896
# Conv2: 32*64*3*3 + 64 = 18,496
# Linear: 64*8*8*10 + 10 = 40,970
# Total: 60,362
```

### Output Shape Calculator

```python
def conv_output_shape(H_in, W_in, K, P=0, S=1):
    """Compute output spatial dimensions after convolution."""
    H_out = (H_in - K + 2*P) // S + 1
    W_out = (W_in - K + 2*P) // S + 1
    return H_out, W_out

# Trace shapes through the SimpleCNN
print(conv_output_shape(32, 32, 3, P=1, S=1))  # Conv1: (32, 32)
print(conv_output_shape(32, 32, 2, P=0, S=2))  # Pool1: (16, 16)
print(conv_output_shape(16, 16, 3, P=1, S=1))  # Conv2: (16, 16)
print(conv_output_shape(16, 16, 2, P=0, S=2))  # Pool2: (8, 8)
```

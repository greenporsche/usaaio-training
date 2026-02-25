# Architecture Exercises

**Topic**: VGG, ResNet, GoogLeNet, skip connections, design principles
**Difficulty**: Intermediate → Advanced

---

## Exercise 1: ResNet Skip Connection Analysis

Consider a residual block:

$$y = F(x) + x, \quad F(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

with $x, y \in \mathbb{R}^{64}$ (64-dimensional feature vectors).

1. Compute the parameter count of this block.
2. Compute $\frac{\partial y}{\partial x}$.
3. Why does the identity term in the gradient help with training?
4. If $F(x) = 0$ (the network learns zero residual), what does the block compute?

<details>
<summary>Solution</summary>

**1.** Parameters:
- $W_1$: $64 \times 64 = 4{,}096$, $b_1$: $64$
- $W_2$: $64 \times 64 = 4{,}096$, $b_2$: $64$
- Total: $8{,}320$

**2.** Gradient:
$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + I$$

where $I$ is the $64 \times 64$ identity matrix.

The Jacobian $\frac{\partial F(x)}{\partial x}$ depends on the specific $x$ (due to ReLU), but the key point is the additive $I$ term.

**3.** The identity matrix $I$ in the gradient ensures:
- Even if $\frac{\partial F}{\partial x} \approx 0$ (vanishing gradients through $F$), the gradient is still $\approx I$
- Gradients flow directly through the skip connection without any multiplicative decay
- In a chain of $L$ residual blocks, the gradient includes a term $I^L = I$ — perfect gradient flow regardless of depth

**4.** If $F(x) = 0$: $y = 0 + x = x$. The block is the identity function. This means the "default behavior" of a residual block is to pass the input through unchanged. The network only needs to learn the RESIDUAL (deviation from identity), which is typically small. Learning a small residual is easier than learning the full mapping from scratch.

**Key insight**: Skip connections change the optimization problem from "learn $H(x)$" to "learn $H(x) - x$." Since the optimal $H$ is often close to identity, the residual $F(x) = H(x) - x$ is close to zero, which is an easier starting point for optimization.
</details>

---

## Exercise 2: VGG vs. Modern Design

Compare these two network designs for input $(3, 32, 32)$:

**Design A (VGG-style)**:
```
Conv(3, 64, 3, pad=1) → ReLU → MaxPool(2) →
Conv(64, 128, 3, pad=1) → ReLU → MaxPool(2) →
Flatten → Linear(128*8*8, 4096) → ReLU → Linear(4096, 10)
```

**Design B (Modern)**:
```
Conv(3, 64, 3, pad=1) → BN → ReLU → MaxPool(2) →
Conv(64, 128, 3, pad=1) → BN → ReLU → GlobalAvgPool →
Linear(128, 10)
```

1. Compute parameter count for each.
2. Which is likely to train faster? Why?
3. Which is more prone to overfitting?

<details>
<summary>Solution</summary>

**Design A parameters**:
- Conv1: $64 \times 3 \times 9 + 64 = 1{,}792$
- Conv2: $128 \times 64 \times 9 + 128 = 73{,}856$
- Linear1: $128 \times 8 \times 8 \times 4096 + 4096 = 33{,}558{,}528$
- Linear2: $4096 \times 10 + 10 = 40{,}970$
- **Total A: 33,675,146**

**Design B parameters**:
- Conv1: $1{,}792$
- BN1: $64 \times 2 = 128$
- Conv2: $73{,}856$
- BN2: $128 \times 2 = 256$
- Linear: $128 \times 10 + 10 = 1{,}290$
- **Total B: 77,322**

**Design B has 435x fewer parameters!**

**2.** Design B trains faster because:
- Batch normalization stabilizes training and allows higher learning rates
- Far fewer parameters mean each epoch is faster
- Global average pooling eliminates the massive FC layer bottleneck

**3.** Design A is far more prone to overfitting because:
- 33.5M parameters vs 77K — massive model capacity with no regularization
- Large FC layers memorize spatial patterns instead of being translation-invariant
- No batch normalization means less implicit regularization

**Key insight**: The VGG-to-modern evolution has three key changes: (1) add BN for stable training, (2) replace large FC layers with Global Average Pooling, (3) use skip connections for depth. These three innovations enable networks that are simultaneously deeper, faster to train, and less prone to overfitting.
</details>

---

## Exercise 3: Projection Shortcut Computation

A ResNet block uses a projection shortcut when input and output dimensions differ. Given:

- Input: $x$ of shape $(B, 64, 32, 32)$
- Block with stride 2 that doubles channels:
  - Conv2d(64, 128, 3, stride=2, padding=1) → BN → ReLU
  - Conv2d(128, 128, 3, stride=1, padding=1) → BN
- Projection shortcut: Conv2d(64, 128, 1, stride=2) → BN

1. What is the output shape of the main path?
2. What is the output shape of the shortcut path?
3. Do they match for addition? Why is the projection needed?
4. How many parameters does the projection shortcut add?

<details>
<summary>Solution</summary>

**1. Main path**:
- Conv1: $(B, 64, 32, 32) \to (B, 128, 16, 16)$ — stride 2 halves spatial, 128 channels
- Conv2: $(B, 128, 16, 16) \to (B, 128, 16, 16)$ — same spatial, same channels

**2. Shortcut path**:
- Conv1x1: $(B, 64, 32, 32) \to (B, 128, 16, 16)$ — stride 2 halves spatial, 1x1 changes channels

**3. Yes, both are $(B, 128, 16, 16)$.** The projection is needed because the input has shape $(B, 64, 32, 32)$ — different both in channels (64 vs 128) and spatial dims (32 vs 16). Without the projection, $F(x) + x$ would fail due to shape mismatch.

**4. Projection parameters**:
- Conv2d(64, 128, 1, stride=2, bias=False): $128 \times 64 \times 1 \times 1 = 8{,}192$
- BN(128): $128 \times 2 = 256$
- Total projection: $8{,}448$

Compared to main path:
- Conv1 (bias=False): $128 \times 64 \times 9 = 73{,}728$
- BN1: $256$
- Conv2 (bias=False): $128 \times 128 \times 9 = 147{,}456$
- BN2: $256$
- Main path total: $221{,}696$

Projection is only 3.8% of the main path parameters.

**Key insight**: The $1 \times 1$ convolution with stride is an elegant way to match dimensions cheaply. It changes channels (via the $1 \times 1$ filter count) and halves spatial size (via stride) in one operation with minimal parameters.
</details>

---

## Exercise 4: Inception Module Efficiency

In GoogLeNet, an inception module processes input of shape $(B, 192, 28, 28)$.

Compare two approaches for producing 128 output channels with $5 \times 5$ convolution:

**Approach A**: Direct $5\times5$ convolution
```
Conv2d(192, 128, 5, padding=2)
```

**Approach B**: $1\times1$ reduction followed by $5\times5$
```
Conv2d(192, 32, 1) → ReLU → Conv2d(32, 128, 5, padding=2)
```

1. Compute parameter count for each.
2. Compute the ratio (A/B).
3. What is the trade-off?

<details>
<summary>Solution</summary>

**Approach A**: $128 \times (192 \times 25) + 128 = 614{,}528$

**Approach B**:
- $1\times1$: $32 \times (192 \times 1) + 32 = 6{,}176$
- $5\times5$: $128 \times (32 \times 25) + 128 = 102{,}528$
- Total: $108{,}704$

**Ratio**: $614{,}528 / 108{,}704 = 5.65\times$

Approach A uses 5.65 times more parameters!

**3. Trade-off**:
- **Pro**: The $1\times1$ bottleneck dramatically reduces computation and parameters while keeping the same receptive field ($5\times5$)
- **Con**: Information is compressed from 192 to 32 channels before the $5\times5$ convolution. If the 192 input channels carry complementary information that cannot be summarized in 32 channels, the bottleneck loses information
- **In practice**: The bottleneck works very well because most channels carry redundant information. The $1\times1$ convolution learns an optimal 32-dimensional projection of the 192-channel input.

**Key insight**: $1\times1$ convolutions are the secret weapon of efficient architecture design. They act as per-pixel fully connected layers that can change channel dimensions cheaply. This idea appears in Inception, ResNet bottlenecks, MobileNets, and many modern architectures.
</details>

---

## Exercise 5: Training Depth Comparison

Consider training three networks on CIFAR-10 (32x32 RGB images, 10 classes):

| Network | Layers | Skip Connections? | Batch Norm? |
|---|---|---|---|
| Plain-10 | 10 conv layers | No | No |
| Plain-30 | 30 conv layers | No | No |
| ResNet-30 | 30 conv layers | Yes (every 2 layers) | Yes |

Predict the training error curve for each (sketch or describe):

<details>
<summary>Solution</summary>

**Plain-10**: Moderate training. Converges to reasonable training loss after many epochs. 10 layers is shallow enough that vanishing gradients are not severe. Final training accuracy: ~85-90%.

**Plain-30**: **Worse** than Plain-10 despite being deeper! The vanishing gradient problem prevents early layers from learning useful features. Training loss decreases slowly and plateaus at a HIGHER value than Plain-10. This is the **degradation problem** that He et al. (2015) identified — it is not overfitting (training error is also high), it is an optimization failure.

**ResNet-30**: Best of all three. Skip connections solve the vanishing gradient problem. Batch norm stabilizes training and allows larger learning rates. Converges faster than Plain-10 and achieves lower training error. Final training accuracy: ~95%+.

```
Training Loss:
high │╲
     │ ╲ Plain-30 (plateaus high)
     │  ╲─────────────────────
     │  ╲
     │   ╲ Plain-10 (converges OK)
     │    ╲──────────────────
     │    ╲
     │     ╲ ResNet-30 (best)
     │      ╲────────────────
low  │
     └────────────────────────
     0    epoch →           100
```

**Key insight**: This is the central experiment of the ResNet paper. Deeper networks should perform at least as well as shallower ones (because they can learn identity mappings for extra layers). But without skip connections, gradient-based optimization fails to find these solutions. Skip connections make the identity mapping the default, so the optimizer only needs to learn residuals.
</details>

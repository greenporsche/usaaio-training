# CNN Exercises

**Topic**: Convolution, output shapes, parameter counting, pooling, receptive field
**Difficulty**: Foundational → Advanced

---

## Exercise 1: Output Shape Computation

For each configuration, compute the output spatial dimensions $(H_{out}, W_{out})$ using:

$$H_{out} = \left\lfloor\frac{H_{in} - K + 2P}{S}\right\rfloor + 1$$

| Input $(H, W)$ | Kernel $K$ | Padding $P$ | Stride $S$ | Output $(H_{out}, W_{out})$ |
|---|---|---|---|---|
| $(32, 32)$ | 3 | 0 | 1 | ? |
| $(32, 32)$ | 3 | 1 | 1 | ? |
| $(32, 32)$ | 5 | 2 | 1 | ? |
| $(32, 32)$ | 3 | 1 | 2 | ? |
| $(224, 224)$ | 7 | 3 | 2 | ? |
| $(28, 28)$ | 5 | 0 | 1 | ? |
| $(14, 14)$ | 2 | 0 | 2 | ? (max pool) |

<details>
<summary>Solution</summary>

| Input | $K$ | $P$ | $S$ | $H_{out} = \lfloor(H-K+2P)/S\rfloor + 1$ | Output |
|---|---|---|---|---|---|
| $(32,32)$ | 3 | 0 | 1 | $\lfloor(32-3+0)/1\rfloor+1 = 30$ | $(30, 30)$ |
| $(32,32)$ | 3 | 1 | 1 | $\lfloor(32-3+2)/1\rfloor+1 = 32$ | $(32, 32)$ ← "same" |
| $(32,32)$ | 5 | 2 | 1 | $\lfloor(32-5+4)/1\rfloor+1 = 32$ | $(32, 32)$ ← "same" |
| $(32,32)$ | 3 | 1 | 2 | $\lfloor(32-3+2)/2\rfloor+1 = 16$ | $(16, 16)$ ← halved |
| $(224,224)$ | 7 | 3 | 2 | $\lfloor(224-7+6)/2\rfloor+1 = 112$ | $(112, 112)$ ← ResNet conv1 |
| $(28,28)$ | 5 | 0 | 1 | $\lfloor(28-5+0)/1\rfloor+1 = 24$ | $(24, 24)$ |
| $(14,14)$ | 2 | 0 | 2 | $\lfloor(14-2+0)/2\rfloor+1 = 7$ | $(7, 7)$ |

**Key insight**: For "same" padding with stride 1: $P = \lfloor K/2 \rfloor$. For halving spatial dims with stride 2 and $K=3$: $P=1$. These are the two most common configurations in practice.
</details>

---

## Exercise 2: Parameter Counting

Compute the total number of learnable parameters for each layer. Assume `bias=True` unless stated otherwise.

```
1. Conv2d(3, 64, 3)
2. Conv2d(64, 128, 3)
3. Conv2d(128, 256, 3, bias=False)
4. Conv2d(512, 512, 1)        # 1×1 convolution
5. MaxPool2d(2, 2)
6. BatchNorm2d(64)
7. Linear(512 * 7 * 7, 4096)
```

<details>
<summary>Solution</summary>

Formula: $\text{params} = C_{out} \times (C_{in} \times K \times K + \text{bias})$

```
1. Conv2d(3, 64, 3):      64 × (3 × 3 × 3) + 64  = 64 × 27 + 64  = 1,728 + 64   = 1,792
2. Conv2d(64, 128, 3):    128 × (64 × 3 × 3) + 128 = 128 × 576 + 128 = 73,728 + 128 = 73,856
3. Conv2d(128, 256, 3, bias=False): 256 × (128 × 3 × 3) = 256 × 1152 = 294,912
4. Conv2d(512, 512, 1):   512 × (512 × 1 × 1) + 512 = 262,144 + 512 = 262,656
5. MaxPool2d(2, 2):       0 (pooling has NO learnable parameters)
6. BatchNorm2d(64):       64 × 2 = 128 (γ and β; running mean/var are NOT parameters)
7. Linear(25088, 4096):   25088 × 4096 + 4096 = 102,760,448 + 4096 = 102,764,544
```

**Total**: $1,792 + 73,856 + 294,912 + 262,656 + 0 + 128 + 102,764,544 = 103,397,888$

**Key insight**: The Linear layer has 99.4% of all parameters! This is why modern architectures replace large FC layers with Global Average Pooling (GAP), which maps $(C, H, W) \to (C,)$ with zero parameters.
</details>

---

## Exercise 3: Manual 2D Convolution

Compute the convolution output by hand:

```
Input (4×4, 1 channel):          Kernel (3×3):
┌───┬───┬───┬───┐               ┌───┬───┬───┐
│ 1 │ 0 │ 2 │ 1 │               │ 1 │ 0 │-1 │
├───┼───┼───┼───┤               ├───┼───┼───┤
│ 0 │ 1 │ 1 │ 0 │               │ 0 │ 1 │ 0 │
├───┼───┼───┼───┤               ├───┼───┼───┤
│ 2 │ 0 │ 1 │ 2 │               │-1 │ 0 │ 1 │
├───┼───┼───┼───┤               └───┴───┴───┘
│ 1 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┘               Bias = 0
                                 Stride = 1, Padding = 0
```

What is the output shape and values?

<details>
<summary>Solution</summary>

Output shape: $\lfloor(4-3+0)/1\rfloor + 1 = 2$, so $(2, 2)$.

**Position (0,0)**: patch = $\begin{bmatrix}1&0&2\\0&1&1\\2&0&1\end{bmatrix}$

$\sum = 1(1) + 0(0) + 2(-1) + 0(0) + 1(1) + 1(0) + 2(-1) + 0(0) + 1(1) = 1+0-2+0+1+0-2+0+1 = -1$

**Position (0,1)**: patch = $\begin{bmatrix}0&2&1\\1&1&0\\0&1&2\end{bmatrix}$

$\sum = 0(1) + 2(0) + 1(-1) + 1(0) + 1(1) + 0(0) + 0(-1) + 1(0) + 2(1) = 0+0-1+0+1+0+0+0+2 = 2$

**Position (1,0)**: patch = $\begin{bmatrix}0&1&1\\2&0&1\\1&1&0\end{bmatrix}$

$\sum = 0(1) + 1(0) + 1(-1) + 2(0) + 0(1) + 1(0) + 1(-1) + 1(0) + 0(1) = 0+0-1+0+0+0-1+0+0 = -2$

**Position (1,1)**: patch = $\begin{bmatrix}1&1&0\\0&1&2\\1&0&1\end{bmatrix}$

$\sum = 1(1) + 1(0) + 0(-1) + 0(0) + 1(1) + 2(0) + 1(-1) + 0(0) + 1(1) = 1+0+0+0+1+0-1+0+1 = 2$

**Output**:
$$\begin{bmatrix}-1 & 2 \\ -2 & 2\end{bmatrix}$$

**Key insight**: This kernel computes a horizontal edge detector — it subtracts the left column from the right column. Positive values indicate right-is-brighter, negative values indicate left-is-brighter.
</details>

---

## Exercise 4: Receptive Field Computation

Compute the receptive field (in pixels of the original input) for the final feature map in this network:

```
Conv2d(3, 32, 3, stride=1, padding=1)    # Layer 1
Conv2d(32, 32, 3, stride=1, padding=1)   # Layer 2
MaxPool2d(2, stride=2)                    # Layer 3
Conv2d(32, 64, 3, stride=1, padding=1)   # Layer 4
Conv2d(64, 64, 3, stride=1, padding=1)   # Layer 5
MaxPool2d(2, stride=2)                    # Layer 6
Conv2d(64, 128, 3, stride=1, padding=1)  # Layer 7
```

Use the formula: $r_l = r_{l-1} + (K_l - 1) \times j_{l-1}$ where $j_l$ is the "jump" (product of all strides up to layer $l$).

<details>
<summary>Solution</summary>

Track both receptive field $r$ and jump $j$ at each layer:

| Layer | Type | $K$ | $S$ | $j_{in}$ | $r_{in}$ | $j_{out} = j_{in} \times S$ | $r_{out} = r_{in} + (K-1) \times j_{in}$ |
|---|---|---|---|---|---|---|---|
| Input | — | — | — | 1 | 1 | 1 | 1 |
| 1 | Conv 3×3 | 3 | 1 | 1 | 1 | 1 | $1 + 2(1) = 3$ |
| 2 | Conv 3×3 | 3 | 1 | 1 | 3 | 1 | $3 + 2(1) = 5$ |
| 3 | Pool 2×2 | 2 | 2 | 1 | 5 | 2 | $5 + 1(1) = 6$ |
| 4 | Conv 3×3 | 3 | 1 | 2 | 6 | 2 | $6 + 2(2) = 10$ |
| 5 | Conv 3×3 | 3 | 1 | 2 | 10 | 2 | $10 + 2(2) = 14$ |
| 6 | Pool 2×2 | 2 | 2 | 2 | 14 | 4 | $14 + 1(2) = 16$ |
| 7 | Conv 3×3 | 3 | 1 | 4 | 16 | 4 | $16 + 2(4) = 24$ |

**Receptive field = 24 pixels.**

**Key insight**: Pooling layers with stride 2 double the "jump," which amplifies the receptive field growth of subsequent conv layers. Without pooling (all stride 1), the receptive field after 7 layers of 3×3 convs would be only $1 + 7 \times 2 = 15$. With 2 pooling layers, it grows to 24 — a 60% increase.
</details>

---

## Exercise 5: Full Architecture Shape Trace

Trace the tensor shape through this complete CNN. Input: RGB image batch of shape $(16, 3, 32, 32)$.

```python
nn.Conv2d(3, 32, 3, padding=1)      # 1
nn.BatchNorm2d(32)                   # 2
nn.ReLU()                            # 3
nn.MaxPool2d(2, 2)                   # 4
nn.Conv2d(32, 64, 3, padding=1)      # 5
nn.BatchNorm2d(64)                   # 6
nn.ReLU()                            # 7
nn.MaxPool2d(2, 2)                   # 8
nn.Flatten()                          # 9
nn.Linear(64*8*8, 256)               # 10
nn.ReLU()                            # 11
nn.Linear(256, 10)                    # 12
```

Also compute the total parameter count.

<details>
<summary>Solution</summary>

**Shape trace**:

| # | Layer | Output Shape |
|---|---|---|
| — | Input | $(16, 3, 32, 32)$ |
| 1 | Conv2d(3, 32, 3, padding=1) | $(16, 32, 32, 32)$ |
| 2 | BatchNorm2d(32) | $(16, 32, 32, 32)$ |
| 3 | ReLU | $(16, 32, 32, 32)$ |
| 4 | MaxPool2d(2, 2) | $(16, 32, 16, 16)$ |
| 5 | Conv2d(32, 64, 3, padding=1) | $(16, 64, 16, 16)$ |
| 6 | BatchNorm2d(64) | $(16, 64, 16, 16)$ |
| 7 | ReLU | $(16, 64, 16, 16)$ |
| 8 | MaxPool2d(2, 2) | $(16, 64, 8, 8)$ |
| 9 | Flatten | $(16, 4096)$ |
| 10 | Linear(4096, 256) | $(16, 256)$ |
| 11 | ReLU | $(16, 256)$ |
| 12 | Linear(256, 10) | $(16, 10)$ |

**Parameter count**:

| Layer | Parameters |
|---|---|
| Conv2d(3, 32, 3) | $32 \times 3 \times 9 + 32 = 896$ |
| BatchNorm2d(32) | $32 \times 2 = 64$ |
| Conv2d(32, 64, 3) | $64 \times 32 \times 9 + 64 = 18{,}496$ |
| BatchNorm2d(64) | $64 \times 2 = 128$ |
| Linear(4096, 256) | $4096 \times 256 + 256 = 1{,}048{,}832$ |
| Linear(256, 10) | $256 \times 10 + 10 = 2{,}570$ |
| **Total** | **$1{,}070{,}986$** |

**Key insight**: The Linear(4096, 256) layer has 98% of all parameters. This is the "bottleneck" of CNN design — the transition from spatial features to flat features. Global Average Pooling (replacing flatten with `mean(dim=[2,3])`) would give shape $(16, 64)$ and reduce the first linear layer to $64 \times 256 + 256 = 16{,}640$ — a 63x reduction.
</details>

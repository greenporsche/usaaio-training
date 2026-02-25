# Deep Learning — Cheat Sheet

> Quick reference for USAAIO 2026 | AI 410

---

## Multi-Layer Perceptron (MLP)

$$y = \sigma(W_2 \cdot \sigma(W_1 x + b_1) + b_2)$$

| Component | Formula | Shape |
|---|---|---|
| Input | $x$ | $(B, d_{in})$ |
| Hidden layer | $h = \sigma(W_1 x + b_1)$ | $(B, d_h)$ where $W_1 \in \mathbb{R}^{d_h \times d_{in}}, b_1 \in \mathbb{R}^{d_h}$ |
| Output layer | $y = W_2 h + b_2$ | $(B, d_{out})$ where $W_2 \in \mathbb{R}^{d_{out} \times d_h}, b_2 \in \mathbb{R}^{d_{out}}$ |

**Universal Approximation Theorem**: A single hidden layer MLP with sufficient neurons can approximate any continuous function on a compact set to arbitrary accuracy.

**Parameter count**: $\text{params}(W_1) = d_{in} \times d_h + d_h$, $\text{params}(W_2) = d_h \times d_{out} + d_{out}$.

Total for one hidden layer: $(d_{in} + 1) \cdot d_h + (d_h + 1) \cdot d_{out}$.

---

## Forward Propagation

Layer-by-layer computation:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = \sigma(z^{[l]})$$

where $a^{[0]} = x$ (input), $z^{[l]}$ is pre-activation, $a^{[l]}$ is post-activation.

**Shape tracking**: If $a^{[l-1]} \in \mathbb{R}^{n_{l-1}}$ and layer $l$ has $n_l$ neurons:
- $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$
- $b^{[l]} \in \mathbb{R}^{n_l}$
- $z^{[l]}, a^{[l]} \in \mathbb{R}^{n_l}$

---

## Backpropagation

**Chain rule for gradients**:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \cdot (a^{[l-1]})^T$$

$$\frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}}$$

$$\frac{\partial L}{\partial a^{[l-1]}} = (W^{[l]})^T \cdot \frac{\partial L}{\partial z^{[l]}}$$

**Backward pass** (from output to input):

$$\delta^{[l]} = \frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \odot \sigma'(z^{[l]})$$

For the last layer with MSE loss: $\delta^{[L]} = (a^{[L]} - y) \odot \sigma'(z^{[L]})$.

For hidden layers: $\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(z^{[l]})$.

---

## Activation Functions

| Function | Formula | Derivative | Range | Issue |
|---|---|---|---|---|
| ReLU | $\max(0, x)$ | $\begin{cases}1 & x>0\\0 & x\leq 0\end{cases}$ | $[0, \infty)$ | Dying ReLU |
| Leaky ReLU | $\max(\alpha x, x)$ | $\begin{cases}1 & x>0\\\alpha & x\leq 0\end{cases}$ | $(-\infty, \infty)$ | — |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | $(0, 1)$ | Vanishing gradient |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | $(-1, 1)$ | Vanishing gradient |
| GELU | $x \cdot \Phi(x)$ | $\Phi(x) + x \cdot \phi(x)$ | $\approx(-0.17, \infty)$ | — |
| Swish | $x \cdot \sigma(x)$ | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ | $\approx(-0.28, \infty)$ | — |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $p_i(\delta_{ij} - p_j)$ | $(0, 1)$, sums to 1 | Output layer only |

---

## Batch Normalization

**Training**:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

**Inference**: Use running mean/variance instead of batch statistics.

**Parameters**: $\gamma$ (scale) and $\beta$ (shift) are learnable. $\mu_{running}$ and $\sigma^2_{running}$ are tracked (not learnable).

---

## Dropout

**Training**: Randomly zero elements with probability $p$, scale remaining by $\frac{1}{1-p}$:

$$\tilde{h} = \frac{1}{1-p} \cdot h \odot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

**Inference**: No dropout. Output unchanged (scaling during training handles this).

---

## Convolutional Neural Networks (CNNs)

### Conv2D Output Size

$$H_{out} = \left\lfloor\frac{H_{in} - K + 2P}{S}\right\rfloor + 1$$

$$W_{out} = \left\lfloor\frac{W_{in} - K + 2P}{S}\right\rfloor + 1$$

where $K$ = kernel size, $P$ = padding, $S$ = stride.

### Parameter Count

| Layer | Parameters | Bias |
|---|---|---|
| `Conv2d(C_in, C_out, K)` | $C_{out} \times C_{in} \times K \times K$ | $+ C_{out}$ |
| `Conv2d(C_in, C_out, K, groups=g)` | $C_{out} \times \frac{C_{in}}{g} \times K \times K$ | $+ C_{out}$ |
| `Linear(in, out)` | $in \times out$ | $+ out$ |
| `BatchNorm2d(C)` | $2C$ ($\gamma$ and $\beta$) | — |

### Pooling

- **Max Pool**: $\text{out}(i,j) = \max_{(m,n) \in \text{window}} \text{in}(i \cdot S + m, j \cdot S + n)$
- **Average Pool**: $\text{out}(i,j) = \text{mean}_{(m,n) \in \text{window}} \text{in}(i \cdot S + m, j \cdot S + n)$
- Same output size formula as conv, but **zero parameters**.

### Receptive Field

For $L$ layers with kernel size $K$ and stride $S$:

$$r_L = 1 + \sum_{l=1}^{L} (K_l - 1) \prod_{i=1}^{l-1} S_i$$

For uniform $K$ and $S=1$: $r_L = 1 + L(K-1)$.

---

## Architectures

| Architecture | Key Idea | Depth |
|---|---|---|
| VGG | Stack $3\times3$ convolutions, double channels at each pool | 16–19 layers |
| ResNet | Skip connections: $F(x) + x$ | 18–152+ layers |
| GoogLeNet | Inception modules (parallel $1\times1$, $3\times3$, $5\times5$, pool) | 22 layers |

### ResNet Skip Connection

$$\text{output} = F(x) + x$$

If dimensions mismatch, use projection: $\text{output} = F(x) + W_s x$.

**Why it works**: Gradients flow directly through the skip connection ($\frac{\partial}{\partial x}[F(x)+x] = F'(x) + I$), preventing vanishing gradients.

---

## Transfer Learning

| Strategy | What to Do | When |
|---|---|---|
| Feature extraction | Freeze all base layers, train new head only | Small dataset, similar domain |
| Fine-tuning (partial) | Freeze early layers, train later layers + head | Medium dataset |
| Full fine-tuning | Unfreeze all, train with small LR | Large dataset |

```python
# Feature extraction pattern
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, num_classes)  # Replace head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

---

## Quick Formulas

| What | Formula |
|---|---|
| Total params in MLP | $\sum_{l=1}^{L} (n_{l-1} + 1) \cdot n_l$ |
| Conv2D output spatial | $\lfloor\frac{W - K + 2P}{S}\rfloor + 1$ |
| Conv2D params | $C_{out}(C_{in} K^2 + 1)$ |
| Batch norm params | $2C$ (learnable), $2C$ (tracked) |
| Dropout expected output | $E[\tilde{h}] = h$ (unbiased due to $\frac{1}{1-p}$ scaling) |
| ResNet identity block | $y = \text{ReLU}(BN(W_2 \cdot \text{ReLU}(BN(W_1 x))) + x)$ |
| Softmax + CE loss gradient | $\frac{\partial L}{\partial z_i} = p_i - y_i$ (elegant simplification) |

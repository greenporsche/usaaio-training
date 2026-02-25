# Backpropagation

**Prerequisites**: Chain rule (multivariable calculus), forward propagation (Study Guide 02)
**USAAIO Relevance**: **Heavily tested.** Round 1 asks you to derive gradients by hand for small networks. Round 2 requires implementing custom backward passes. Understanding gradient flow is essential for diagnosing training problems.

---

## Discovery

### The Credit Assignment Problem

After a network makes a prediction and we compute the loss, we face a fundamental question: **which weights are responsible for the error, and how should we adjust them?**

This is the credit assignment problem. In a network with millions of parameters, each one contributes a tiny amount to the final output. Backpropagation solves this by computing the gradient of the loss with respect to every parameter — telling us exactly how each weight affects the loss.

> **Socratic question**: Why can't we just perturb each weight one at a time and measure how the loss changes?
>
> *Answer: That would require $N+1$ forward passes (one per parameter plus the original). For a network with 1 million parameters, that's 1 million forward passes per gradient step. Backpropagation computes ALL gradients in a single backward pass — roughly the cost of 2 forward passes.*

### Historical Context

Backpropagation was popularized by Rumelhart, Hinton, and Williams in their 1986 Nature paper "Learning representations by back-propagating errors." Although the chain rule had been known for centuries, and similar algorithms existed in control theory (Werbos, 1974), the 1986 paper demonstrated that backprop could learn useful internal representations in multi-layer networks, reviving neural network research after the Minsky & Papert critique.

---

## Intuition

### Computational Graphs

Every forward pass builds a **computational graph** — a DAG (directed acyclic graph) where nodes are operations and edges carry values:

```
Forward pass for y = σ(w₁x₁ + w₂x₂ + b):

x₁ ──→ [×w₁] ──→ [+] ──→ [+b] ──→ [σ] ──→ ŷ ──→ [L] ──→ loss
                    ↑
x₂ ──→ [×w₂] ──┘
```

Backpropagation walks this graph **in reverse**, applying the chain rule at each node:

```
Backward pass:

∂L/∂x₁ ←── [×w₁] ←── [+] ←── [+b] ←── [σ'] ←── [1] ←── ∂L/∂ŷ ←── ∂L/∂loss = 1
                         ↓
∂L/∂x₂ ←── [×w₂] ←──┘
```

At each node, the incoming gradient is multiplied by the local gradient (the derivative of that node's operation).

### The Chain Rule in Action

For a composed function $L(f(g(x)))$:

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}$$

In a neural network, this chain has as many links as there are layers.

**Key insight**: Each layer receives the gradient from the layer above ($\frac{\partial L}{\partial a^{[l]}}$), multiplies it by its local derivative ($\sigma'(z^{[l]})$ and $W^{[l]}$), and passes the result down.

### A Concrete Example by Hand

Network: 1 input, 1 hidden neuron, 1 output. Sigmoid activation. MSE loss.

```
x=0.5  →  [×w₁=0.3, +b₁=0.1]  →  [σ]  →  h  →  [×w₂=0.7, +b₂=0.2]  →  [σ]  →  ŷ
                                                                                     ↓
                                                         y=1.0  →  [MSE]  →  loss
```

**Forward pass**:
1. $z_1 = w_1 x + b_1 = 0.3 \times 0.5 + 0.1 = 0.25$
2. $h = \sigma(z_1) = \sigma(0.25) = \frac{1}{1+e^{-0.25}} \approx 0.5622$
3. $z_2 = w_2 h + b_2 = 0.7 \times 0.5622 + 0.2 = 0.5935$
4. $\hat{y} = \sigma(z_2) = \sigma(0.5935) \approx 0.6441$
5. $L = (\hat{y} - y)^2 = (0.6441 - 1)^2 = 0.1267$

**Backward pass**:
1. $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2(0.6441 - 1) = -0.7118$
2. $\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z_2) = -0.7118 \times 0.6441(1-0.6441) = -0.7118 \times 0.2292 = -0.1632$
3. $\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial z_2} \cdot h = -0.1632 \times 0.5622 = -0.0917$
4. $\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} = -0.1632$
5. $\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} \cdot w_2 = -0.1632 \times 0.7 = -0.1142$
6. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \sigma'(z_1) = -0.1142 \times 0.5622(1-0.5622) = -0.1142 \times 0.2461 = -0.0281$
7. $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \cdot x = -0.0281 \times 0.5 = -0.0141$
8. $\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} = -0.0281$

All gradients are negative — we should increase all weights/biases to reduce the loss (push $\hat{y}$ toward 1).

---

## Math

### General Backpropagation Equations

Given an $L$-layer network with forward pass as defined in Study Guide 02, define:

$$\delta^{[l]} \equiv \frac{\partial L}{\partial z^{[l]}} \in \mathbb{R}^{n_l}$$

**Output layer** (with MSE loss and no output activation):

$$\delta^{[L]} = \frac{\partial L}{\partial a^{[L]}} = \frac{2}{n_L}(a^{[L]} - y)$$

**Output layer** (with cross-entropy loss + softmax):

$$\delta^{[L]} = a^{[L]} - y \quad \text{(elegant simplification — softmax and CE gradients combine)}$$

**Hidden layers** (for $l = L-1, \ldots, 1$):

$$\delta^{[l]} = \left((W^{[l+1]})^T \delta^{[l+1]}\right) \odot \sigma'(z^{[l]})$$

where $\odot$ is the Hadamard (element-wise) product.

**Parameter gradients**:

$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T \in \mathbb{R}^{n_l \times n_{l-1}}$$

$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]} \in \mathbb{R}^{n_l}$$

### Batched Backpropagation

For a batch of $B$ samples, let $\Delta^{[l]} \in \mathbb{R}^{B \times n_l}$ where each row is $\delta^{[l]}$ for one sample:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{B} (\Delta^{[l]})^T A^{[l-1]} \in \mathbb{R}^{n_l \times n_{l-1}}$$

$$\frac{\partial L}{\partial b^{[l]}} = \frac{1}{B} \sum_{i=1}^{B} \delta_i^{[l]} \in \mathbb{R}^{n_l}$$

### Gradient Flow and the Vanishing Gradient Problem

For a deep network, the gradient at layer $l$ involves a product of many terms:

$$\delta^{[l]} = \left(\prod_{k=l+1}^{L} (W^{[k]})^T \text{diag}(\sigma'(z^{[k]}))\right) \delta^{[L]}$$

If $\sigma'(z^{[k]}) < 1$ at many layers (as happens with sigmoid, whose maximum derivative is 0.25), this product shrinks exponentially. This is the **vanishing gradient problem** — early layers receive negligible gradients and barely learn.

Solutions:
- **ReLU activation**: $\sigma'(z) = 1$ for $z > 0$, so gradients pass through unchanged
- **Batch normalization**: Keeps pre-activations in a well-behaved range
- **Skip connections** (ResNet): Provide gradient shortcuts that bypass layers

---

## Code

### Backpropagation by Hand in PyTorch

```python
import torch

# Network: 2 → 2 → 1, ReLU hidden, MSE loss
# Initialize weights
W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)   # (2, 2)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)                  # (2,)
W2 = torch.tensor([[0.5, 0.6]], requires_grad=True)                 # (1, 2)
b2 = torch.tensor([0.0], requires_grad=True)                        # (1,)

x = torch.tensor([1.0, 2.0])   # input
y = torch.tensor([1.0])         # target

# Forward pass
z1 = W1 @ x + b1               # (2,): [0.5, 1.1]
a1 = torch.relu(z1)             # (2,): [0.5, 1.1]
z2 = W2 @ a1 + b2               # (1,): [0.91]
y_hat = z2                       # (1,): [0.91]

# Loss
loss = (y_hat - y).pow(2).mean() # scalar: (0.91 - 1)^2 = 0.0081

# Backward pass (automatic)
loss.backward()

# Print gradients
print(f"dL/dW2 = {W2.grad}")    # gradient w.r.t. output weights
print(f"dL/db2 = {b2.grad}")
print(f"dL/dW1 = {W1.grad}")    # gradient w.r.t. hidden weights
print(f"dL/db1 = {b1.grad}")
```

### Manual Backprop (Verifying Against Autograd)

```python
import torch

x = torch.tensor([1.0, 2.0])
y = torch.tensor([1.0])

W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
b1 = torch.tensor([0.0, 0.0])
W2 = torch.tensor([[0.5, 0.6]])
b2 = torch.tensor([0.0])

# Forward
z1 = W1 @ x + b1         # [0.5, 1.1]
a1 = torch.relu(z1)       # [0.5, 1.1]
z2 = W2 @ a1 + b2         # [0.91]
y_hat = z2                 # [0.91]
loss = (y_hat - y) ** 2   # [0.0081]

# Manual backward
dL_dy_hat = 2 * (y_hat - y)                # [−0.18]
dL_dz2 = dL_dy_hat                          # [−0.18] (no activation on output)

dL_dW2 = dL_dz2.unsqueeze(1) * a1.unsqueeze(0)  # (1,1) * (1,2) → (1, 2)
dL_db2 = dL_dz2                                   # (1,)

dL_da1 = W2.T @ dL_dz2                     # (2, 1) @ (1,) → (2,)
relu_grad = (z1 > 0).float()                # [1, 1] (both positive)
dL_dz1 = dL_da1 * relu_grad                 # (2,) element-wise

dL_dW1 = dL_dz1.unsqueeze(1) * x.unsqueeze(0)  # (2,1) * (1,2) → (2, 2)
dL_db1 = dL_dz1                                   # (2,)

# Verify against autograd
W1_ag = W1.clone().requires_grad_(True)
b1_ag = b1.clone().requires_grad_(True)
W2_ag = W2.clone().requires_grad_(True)
b2_ag = b2.clone().requires_grad_(True)

z1_ag = W1_ag @ x + b1_ag
a1_ag = torch.relu(z1_ag)
z2_ag = W2_ag @ a1_ag + b2_ag
loss_ag = (z2_ag - y) ** 2
loss_ag.backward()

assert torch.allclose(dL_dW1, W1_ag.grad, atol=1e-6)
assert torch.allclose(dL_dW2, W2_ag.grad, atol=1e-6)
print("Manual backprop matches autograd!")
```

### Implementing a Full MLP with Custom Backward

```python
import torch

class ManualMLP:
    """MLP with manual forward and backward pass (no autograd)."""

    def __init__(self, dims):
        """dims: list of layer sizes, e.g., [784, 256, 10]"""
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            # Xavier initialization
            W = torch.randn(dims[i+1], dims[i]) * (2.0 / (dims[i] + dims[i+1])) ** 0.5
            b = torch.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

        # Cache for backward pass
        self.z_cache = []
        self.a_cache = []

    def forward(self, x):
        """Forward pass. x: (B, d_in). Returns: (B, d_out)."""
        self.a_cache = [x]
        self.z_cache = []
        a = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b                        # (B, n_in) @ (n_in, n_out)^T → (B, n_out)
            self.z_cache.append(z)
            if i < len(self.weights) - 1:           # ReLU on all but last
                a = torch.relu(z)
            else:
                a = z                                # No activation on output
            self.a_cache.append(a)
        return a

    def backward(self, y_hat, y_true):
        """Compute gradients. Returns: (dW_list, db_list)."""
        B = y_hat.shape[0]
        dW_list = []
        db_list = []

        # Output gradient (MSE loss)
        delta = (2.0 / B) * (y_hat - y_true)       # (B, d_out)

        for i in reversed(range(len(self.weights))):
            # Parameter gradients
            dW = delta.T @ self.a_cache[i] / B      # (n_out, B) @ (B, n_in) → (n_out, n_in)
            db = delta.mean(dim=0)                    # (n_out,)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                # Propagate gradient to previous layer
                da = delta @ self.weights[i]          # (B, n_out) @ (n_out, n_in) → (B, n_in)
                relu_mask = (self.z_cache[i-1] > 0).float()
                delta = da * relu_mask                # (B, n_in)

        return dW_list, db_list
```

# Forward Propagation

**Prerequisites**: Matrix multiplication, MLP structure (Study Guide 01)
**USAAIO Relevance**: Round 1 frequently asks you to trace a forward pass by hand with concrete numbers. Round 2 requires implementing forward passes for custom layers. Shape tracking is essential for debugging.

---

## Discovery

### How Does a Neural Network Compute?

Given a trained neural network (with fixed weights and biases), how does it actually produce an output from an input? The answer is **forward propagation**: data flows forward through the network, layer by layer, from input to output.

> **Socratic question**: If you have a network with 3 layers and the input is a vector $x = [1, 2]$, how many matrix multiplications happen? How many additions? How many activation function evaluations?
>
> *Answer: 3 matrix multiplications (one per layer), 3 bias additions, and 2 or 3 activation function evaluations (depending on whether the output layer has an activation).*

Forward propagation is deterministic: the same input always produces the same output (ignoring dropout). It is the "inference" step — what happens when you deploy a model.

### Why "Forward"?

The name distinguishes it from **backpropagation** (Study Guide 03), which flows gradients backward from the output to the input. Forward = data flows input → output. Backward = gradients flow output → input.

---

## Intuition

### Step-by-Step with Concrete Numbers

Let us trace a forward pass through a tiny network: 2 inputs, 3 hidden neurons, 1 output. ReLU activation on the hidden layer.

```
Architecture: 2 → 3 → 1

W₁ = [[0.1, 0.2],     b₁ = [0.1, -0.1, 0.0]
      [0.3, 0.4],
      [0.5, 0.6]]

W₂ = [[0.7, 0.8, 0.9]]  b₂ = [0.1]

Input: x = [1.0, 2.0]
```

**Step 1: First linear layer**

$$z^{[1]} = W_1 x + b_1$$

$$z^{[1]} = \begin{bmatrix}0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6\end{bmatrix} \begin{bmatrix}1.0 \\ 2.0\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.1 \\ 0.0\end{bmatrix} = \begin{bmatrix}0.5 \\ 1.1 \\ 1.7\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.1 \\ 0.0\end{bmatrix} = \begin{bmatrix}0.6 \\ 1.0 \\ 1.7\end{bmatrix}$$

**Step 2: Activation**

$$a^{[1]} = \text{ReLU}(z^{[1]}) = \begin{bmatrix}\max(0, 0.6) \\ \max(0, 1.0) \\ \max(0, 1.7)\end{bmatrix} = \begin{bmatrix}0.6 \\ 1.0 \\ 1.7\end{bmatrix}$$

(All positive, so ReLU is identity here.)

**Step 3: Second linear layer**

$$z^{[2]} = W_2 a^{[1]} + b_2$$

$$z^{[2]} = \begin{bmatrix}0.7 & 0.8 & 0.9\end{bmatrix} \begin{bmatrix}0.6 \\ 1.0 \\ 1.7\end{bmatrix} + 0.1 = (0.42 + 0.80 + 1.53) + 0.1 = 2.85$$

**Output**: $y = 2.85$ (no activation on output for regression).

### The Matrix View

For a batch of $B$ inputs, forward propagation is a sequence of batched matrix operations:

```
X:     (B, d_in)
       ↓
Z1 = X @ W1.T + b1    →  (B, d_in) @ (d_in, d_h).T ... but PyTorch Linear stores W as (d_h, d_in)
                          Actually: Z1 = X @ W1.T + b1  →  (B, d_h)
       ↓
A1 = ReLU(Z1)          →  (B, d_h)   [element-wise]
       ↓
Z2 = A1 @ W2.T + b2   →  (B, d_out)
       ↓
Y = Z2                 →  (B, d_out)
```

**Critical PyTorch detail**: `nn.Linear(d_in, d_out)` stores weight as shape `(d_out, d_in)`. The forward pass computes `x @ W.T + b`, which is equivalent to `F.linear(x, W, b)`.

### Shape Tracking Template

For any network, you can track shapes through each layer:

```
Input:          (B, 784)
Linear(784,256): (B, 784) @ (784, 256) → (B, 256)
ReLU:           (B, 256)  [unchanged]
Linear(256,128): (B, 256) @ (256, 128) → (B, 128)
ReLU:           (B, 128)  [unchanged]
Linear(128,10): (B, 128) @ (128, 10)  → (B, 10)
```

---

## Math

### General Forward Pass

For an $L$-layer network, the forward pass computes:

$$a^{[0]} = x \in \mathbb{R}^{n_0}$$

For each layer $l = 1, \ldots, L$:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} \in \mathbb{R}^{n_l}$$
$$a^{[l]} = \sigma^{[l]}(z^{[l]}) \in \mathbb{R}^{n_l}$$

The final output is $\hat{y} = a^{[L]}$.

### Batched Forward Pass

For a batch of $B$ inputs $X \in \mathbb{R}^{B \times n_0}$:

$$Z^{[l]} = A^{[l-1]} (W^{[l]})^T + \mathbf{1}_B b^{[l]T} \in \mathbb{R}^{B \times n_l}$$

where $\mathbf{1}_B$ is a column vector of ones (broadcasting handles this in PyTorch).

Or equivalently in PyTorch notation (where `Linear` weight is $(n_l, n_{l-1})$):

$$Z^{[l]} = A^{[l-1]} W^{[l]T} + b^{[l]}$$

### Computational Cost

For a single forward pass through layer $l$ with $n_{l-1}$ inputs and $n_l$ outputs:

- **Multiplications**: $n_{l-1} \times n_l$ (matrix multiply)
- **Additions**: $n_{l-1} \times n_l$ (accumulation) $+ n_l$ (bias)
- **Total FLOPs**: $\approx 2 \times n_{l-1} \times n_l$ per sample

For a batch of $B$ samples: $\approx 2B \times n_{l-1} \times n_l$ FLOPs per layer.

### Caching for Backpropagation

During training, the forward pass must **cache intermediate values** ($z^{[l]}$ and $a^{[l]}$) because backpropagation needs them:

- $a^{[l-1]}$ is needed to compute $\frac{\partial L}{\partial W^{[l]}}$
- $z^{[l]}$ is needed to compute $\sigma'(z^{[l]})$

This is why training uses much more memory than inference.

---

## Code

### Manual Forward Pass

```python
import torch

# Network: 2 → 3 → 1
W1 = torch.tensor([[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]])     # (3, 2)
b1 = torch.tensor([0.1, -0.1, 0.0])  # (3,)
W2 = torch.tensor([[0.7, 0.8, 0.9]]) # (1, 3)
b2 = torch.tensor([0.1])              # (1,)

x = torch.tensor([1.0, 2.0])          # (2,)

# Forward pass — step by step
z1 = W1 @ x + b1           # (3,): pre-activation layer 1
a1 = torch.relu(z1)         # (3,): post-activation layer 1
z2 = W2 @ a1 + b2           # (1,): pre-activation layer 2 (output)
y = z2                       # (1,): output (no activation for regression)

print(f"z1 = {z1}")   # tensor([0.6000, 1.0000, 1.7000])
print(f"a1 = {a1}")   # tensor([0.6000, 1.0000, 1.7000])
print(f"y  = {y}")     # tensor([2.8500])
```

### Batched Forward Pass

```python
# Batch of 4 inputs
X = torch.tensor([[1.0, 2.0],
                   [0.5, -1.0],
                   [3.0, 0.0],
                   [-1.0, 1.0]])   # (4, 2)

# Forward pass with batching
Z1 = X @ W1.T + b1                 # (4, 2) @ (2, 3) + (3,) → (4, 3)
A1 = torch.relu(Z1)                 # (4, 3)
Z2 = A1 @ W2.T + b2                 # (4, 3) @ (3, 1) + (1,) → (4, 1)
Y = Z2                               # (4, 1)

print(f"Z1 shape: {Z1.shape}")  # torch.Size([4, 3])
print(f"A1 shape: {A1.shape}")  # torch.Size([4, 3])
print(f"Y shape:  {Y.shape}")   # torch.Size([4, 1])
```

### Forward Pass with Shape Annotations

```python
import torch.nn as nn

class AnnotatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # W: (256, 784), b: (256,)
        self.fc2 = nn.Linear(256, 128)   # W: (128, 256), b: (128,)
        self.fc3 = nn.Linear(128, 10)    # W: (10, 128),  b: (10,)

    def forward(self, x):
        # x: (B, 784)
        x = torch.relu(self.fc1(x))      # (B, 784) → (B, 256) → ReLU → (B, 256)
        x = torch.relu(self.fc2(x))      # (B, 256) → (B, 128) → ReLU → (B, 128)
        x = self.fc3(x)                   # (B, 128) → (B, 10) — no activation (logits)
        return x                           # (B, 10)

model = AnnotatedMLP()
x = torch.randn(32, 784)                  # (32, 784)
logits = model(x)                          # (32, 10)
```

### Verifying with PyTorch Internals

```python
# Access weights from nn.Linear
model = AnnotatedMLP()
W1 = model.fc1.weight    # shape: (256, 784)
b1 = model.fc1.bias      # shape: (256,)

# Manual computation should match
x = torch.randn(1, 784)
manual_z1 = x @ W1.T + b1                 # (1, 784) @ (784, 256) + (256,) → (1, 256)
pytorch_z1 = model.fc1(x)                  # should be identical
assert torch.allclose(manual_z1, pytorch_z1, atol=1e-6)
```

# MLPs and Universal Approximation

**Prerequisites**: Linear algebra (matrix multiplication), basic calculus (function composition)
**USAAIO Relevance**: MLPs are the foundation of all deep learning. Round 1 tests your understanding of how neurons compose functions. Round 2 requires building MLPs from scratch.

---

## Discovery

### From Perceptrons to Multi-Layer Networks

In 1958, Frank Rosenblatt built the Perceptron — a single neuron that computes a weighted sum of inputs and applies a threshold:

$$y = \text{sign}(w^T x + b)$$

This can classify linearly separable data: points that can be divided by a hyperplane. But what about data that is NOT linearly separable?

> **Socratic question**: Can a single perceptron learn the XOR function?
>
> | $x_1$ | $x_2$ | XOR |
> |---|---|---|
> | 0 | 0 | 0 |
> | 0 | 1 | 1 |
> | 1 | 0 | 1 |
> | 1 | 1 | 0 |
>
> Try drawing a single straight line that separates the 1s from the 0s. You cannot — XOR is not linearly separable.

This was the insight of Minsky & Papert (1969): single-layer perceptrons have fundamental limitations. Their book caused the first "AI winter" — funding for neural network research dried up.

The solution? **Add hidden layers.** A multi-layer perceptron (MLP) stacks multiple layers of neurons, with nonlinear activation functions between them. This breaks the linearity barrier.

### The Universal Approximation Theorem

In 1989, George Cybenko proved something remarkable: a feedforward network with a single hidden layer containing enough neurons can approximate **any continuous function** on a compact set to any desired accuracy.

More precisely: for any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\epsilon > 0$, there exists an MLP with one hidden layer such that $|g(x) - f(x)| < \epsilon$ for all $x$ in the domain, where $g$ is the MLP's output.

> **Important caveat**: The theorem says such a network *exists*, not that gradient descent can *find* it. And "enough neurons" might mean an astronomically large hidden layer. In practice, deeper (more layers) networks are far more efficient than wider (more neurons per layer) ones.

---

## Intuition

### How an MLP Computes

An MLP is a function composition. Each layer transforms its input, and the layers stack:

```
Input x ──→ [Linear₁ → Activation₁] ──→ [Linear₂ → Activation₂] ──→ ... ──→ Output y
              Layer 1 (hidden)             Layer 2 (hidden)              Output layer
```

Each linear layer computes an affine transformation: $z = Wx + b$. The activation function $\sigma$ introduces nonlinearity.

Without activation functions, stacking layers is useless:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$$

No matter how many linear layers you stack, the result is still a single linear transformation. **Nonlinearity is essential.**

### XOR with One Hidden Layer

Here is how an MLP solves XOR with 2 hidden neurons:

```
x₁ ──┬──(w=1, w=1, b=-0.5)──→ h₁ = σ(x₁ + x₂ - 0.5)     (AND-like)
     │╲                         ↘
     │ ╲                         → output = σ(h₁ - 2h₂ + 0.5)
     │  ╲                       ↗
x₂ ──┴──(w=1, w=1, b=-1.5)──→ h₂ = σ(x₁ + x₂ - 1.5)     (OR-like complement)
```

With step function activation:
- $h_1 = 1$ when $x_1 + x_2 \geq 0.5$ (at least one input is 1) — this is OR
- $h_2 = 1$ when $x_1 + x_2 \geq 1.5$ (both inputs are 1) — this is AND

Then: $\text{output} = h_1 - 2h_2 + 0.5 > 0$ only when $h_1=1, h_2=0$ — exactly XOR.

### Geometric View: Half-Plane Intersections

Each neuron in the first hidden layer defines a **half-plane** in input space (a line in 2D, a plane in 3D). The activation function decides which side is "on" or "off."

The second layer combines these half-planes using Boolean-like logic (AND, OR, NOT) to carve out arbitrary decision regions:

```
Single neuron:              Two neurons + output:
     ___________                    ___________
    |     |     |              _   |  /       |
    | + + | - - |             |+| |  / - - - |
    | + + | - - |    →→→      |+| | / + + + |
    | + + | - - |             |_| |/ _ _ _ _ |
    |_____|_____|                  |__________|
    (one line)                 (intersection of two half-planes)
```

With enough neurons, you can approximate any connected region. This is the geometric intuition behind universal approximation.

---

## Math

### MLP Definition

An $L$-layer MLP maps input $x \in \mathbb{R}^{n_0}$ to output $y \in \mathbb{R}^{n_L}$ through:

$$a^{[0]} = x$$
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \quad l = 1, \ldots, L$$
$$a^{[l]} = \sigma(z^{[l]}), \quad l = 1, \ldots, L-1$$
$$y = a^{[L]} = z^{[L]} \quad \text{(or } \sigma_{out}(z^{[L]}) \text{ for classification)}$$

where:
- $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ are weight matrices
- $b^{[l]} \in \mathbb{R}^{n_l}$ are bias vectors
- $\sigma$ is the activation function (applied element-wise)
- $n_l$ is the number of neurons in layer $l$

### Parameter Count

Total learnable parameters:

$$\text{params} = \sum_{l=1}^{L} (n_{l-1} \cdot n_l + n_l) = \sum_{l=1}^{L} (n_{l-1} + 1) \cdot n_l$$

**Example**: MLP with architecture $784 \to 256 \to 128 \to 10$:
- Layer 1: $(784 + 1) \times 256 = 200{,}960$
- Layer 2: $(256 + 1) \times 128 = 32{,}896$
- Layer 3: $(128 + 1) \times 10 = 1{,}290$
- **Total**: $235{,}146$

### Universal Approximation Theorem (Informal)

**Theorem** (Cybenko 1989, Hornik 1991): Let $\sigma$ be any non-constant, bounded, continuous activation function (e.g., sigmoid). For any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\epsilon > 0$, there exists an integer $N$ and parameters $\{w_i, b_i, v_i\}$ such that:

$$\left| f(x) - \sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i) \right| < \epsilon \quad \forall x \in [0,1]^n$$

**Extensions**: The theorem also holds for unbounded activations like ReLU (Leshno et al. 1993). Width is not the only path — depth can also provide universal approximation with bounded width (Lu et al. 2017).

---

## Code

### MLP in PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: int, dimension of input features
            hidden_dims: list of int, neurons per hidden layer
            output_dim: int, number of output classes/values
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))   # Affine: (B, prev) → (B, h)
            layers.append(nn.ReLU())                     # Nonlinearity
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))   # Output: (B, last_h) → (B, out)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)                            # (B, input_dim) → (B, output_dim)

# Example: 784 → 256 → 128 → 10
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
x = torch.randn(32, 784)   # batch of 32 images (flattened 28x28)
y = model(x)                # shape: (32, 10) — logits for 10 classes
```

### Solving XOR from Scratch

```python
import torch

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # (4, 2)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)                # (4, 1)

# Manual MLP: 2 → 2 → 1
W1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # (2, 2)
b1 = torch.tensor([-0.5, -1.5])               # (2,)
W2 = torch.tensor([[1.0, -2.0]])               # (1, 2)
b2 = torch.tensor([0.5])                       # (1,)

def sigmoid(x):
    return 1 / (1 + torch.exp(-10 * x))  # steep sigmoid approximates step

# Forward pass
h = sigmoid(X @ W1.T + b1)       # (4, 2)
out = sigmoid(h @ W2.T + b2)     # (4, 1)
print(out.round())                # tensor([[0], [1], [1], [0]])
```

### Counting Parameters

```python
def count_parameters(model):
    """Count total and per-layer learnable parameters."""
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        print(f"  {name}: {param.shape} → {n:,} params")
        total += n
    print(f"  Total: {total:,}")
    return total

model = MLP(784, [256, 128], 10)
count_parameters(model)
# network.0.weight: torch.Size([256, 784]) → 200,704 params
# network.0.bias: torch.Size([256]) → 256 params
# network.2.weight: torch.Size([128, 256]) → 32,768 params
# network.2.bias: torch.Size([128]) → 128 params
# network.4.weight: torch.Size([10, 128]) → 1,280 params
# network.4.bias: torch.Size([10]) → 10 params
# Total: 235,146
```

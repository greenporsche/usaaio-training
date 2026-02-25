# nn.Module Exercises

**Topic**: Custom modules, forward(), parameters, Sequential, ModuleList
**Difficulty**: Intermediate → Advanced

---

## Exercise 1: Debug the Module

This custom module has THREE bugs. Find and fix all of them.

```python
import torch
import torch.nn as nn

class BrokenNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.extra_weights = torch.randn(hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x * self.extra_weights
        x = self.fc2(x)
        return x

model = BrokenNet(784, 256, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

<details>
<summary>Solution</summary>

Three bugs:

1. **Missing `super().__init__()`**: Without this, `nn.Module` internals are not initialized, and parameter registration fails.

2. **`self.extra_weights` is a plain tensor, not `nn.Parameter`**: It will not be found by `model.parameters()`, not updated by the optimizer, and not moved by `model.to(device)`.

3. **Shape mismatch**: `self.extra_weights` has shape `(hidden_dim,)` but after `fc1`, `x` has shape `(B, hidden_dim)`. This works via broadcasting, but the intent is likely per-feature scaling — should be `nn.Parameter` with proper shape.

Fixed code:

```python
class FixedNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()                                      # Bug 1: call super
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.extra_weights = nn.Parameter(torch.randn(hidden_dim))  # Bug 2: wrap in Parameter

    def forward(self, x):
        x = torch.relu(self.fc1(x))               # (B, hidden_dim)
        x = x * self.extra_weights                 # (B, hidden_dim) * (hidden_dim,) broadcasts OK
        x = self.fc2(x)                            # (B, out_dim)
        return x

model = FixedNet(784, 256, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
# Should include: 784*256 + 256 + 256 + 256*10 + 10 = 203,786
```

**Key insight**: Forgetting `super().__init__()` is the most common `nn.Module` bug. PyTorch will raise an obscure error about `_modules` not being defined. Always include it.
</details>

---

## Exercise 2: Count Parameters

Without running the code, calculate the exact number of learnable parameters in this model.

```python
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

<details>
<summary>Solution</summary>

| Layer | Weight | Bias | Parameters |
|---|---|---|---|
| `nn.Linear(100, 64)` | 100 x 64 = 6,400 | 64 | 6,464 |
| `nn.ReLU()` | 0 | 0 | 0 |
| `nn.Linear(64, 64)` | 64 x 64 = 4,096 | 64 | 4,160 |
| `nn.BatchNorm1d(64)` | 64 (gamma) | 64 (beta) | 128 |
| `nn.ReLU()` | 0 | 0 | 0 |
| `nn.Linear(64, 10)` | 64 x 10 = 640 | 10 | 650 |
| **Total** | | | **11,402** |

Note: `BatchNorm1d` also maintains `running_mean` and `running_var` (64 each), but these are **buffers**, not parameters — they are not updated by the optimizer (they are updated by the forward pass during training).

**Key insight**: `nn.Linear(in, out)` has `out * in + out` parameters (weight matrix is stored as `(out, in)`, plus bias of size `(out,)`). BatchNorm has 2 learnable parameters per feature (scale and shift), plus 2 buffers per feature.
</details>

---

## Exercise 3: Build a Residual Block

Implement a residual block that computes $\text{output} = \text{ReLU}(\text{BN}(\text{Linear}(x))) + x$.

Requirements:
- Input and output shapes must match (both `(B, dim)`)
- Use batch normalization after the linear layer, before ReLU
- The skip connection adds the input directly to the output

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, x):
        # YOUR CODE HERE
        pass

# Test
block = ResidualBlock(128)
x = torch.randn(32, 128)
out = block(x)
assert out.shape == (32, 128)
print(f"Parameters: {sum(p.numel() for p in block.parameters())}")
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (B, dim)
        residual = x                           # Save for skip connection
        out = self.fc(x)                       # (B, dim)
        out = self.bn(out)                     # (B, dim)
        out = torch.relu(out + residual)       # (B, dim) — add then activate
        return out
```

Parameters: `dim * dim + dim` (Linear) + `2 * dim` (BatchNorm) = `128*128 + 128 + 256 = 16,768`

**Key insight**: The residual connection ($x + F(x)$) lets gradients flow directly through the skip, preventing vanishing gradients in deep networks. This is the core idea behind ResNets.
</details>

---

## Exercise 4: ModuleList vs Python List

What is wrong with this code? The model trains but the "hidden layers" are never updated.

```python
class BadModel(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

model = BadModel([784, 256, 128, 64, 10])
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
# Prints: 0 ← BUG!
```

<details>
<summary>Solution</summary>

The bug is using a plain Python list instead of `nn.ModuleList`. Plain lists are invisible to `nn.Module` — the contained layers are not registered, so `parameters()` returns nothing and the optimizer never updates them.

```python
class FixedModel(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([                      # Use ModuleList!
            nn.Linear(dims[i], dims[i+1])
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

model = FixedModel([784, 256, 128, 64, 10])
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
# 784*256+256 + 256*128+128 + 128*64+64 + 64*10+10 = 242,762
```

**The same bug applies to dictionaries**: use `nn.ModuleDict` instead of plain `dict`.

**Key insight**: `nn.Module` only discovers sub-modules that are assigned as attributes (direct assignment or via `nn.ModuleList` / `nn.ModuleDict`). Python containers (list, dict, tuple) are opaque to the module system.
</details>

---

## Exercise 5: Implement Scaled Dot-Product Attention as nn.Module

Implement the scaled dot-product attention operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This module should NOT contain any learnable parameters (it is a pure computation). The projection layers ($W_Q$, $W_K$, $W_V$) belong to the parent MHA module.

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (B, H, L_q, d_k)
            K: (B, H, L_k, d_k)
            V: (B, H, L_k, d_v)
            mask: optional (B, 1, L_q, L_k) or (1, 1, L_q, L_k), True = ignore
        Returns:
            output: (B, H, L_q, d_v)
            attn_weights: (B, H, L_q, L_k)
        """
        # YOUR CODE HERE
        pass

# Test
B, H, L, d_k = 2, 4, 8, 16
Q = torch.randn(B, H, L, d_k)
K = torch.randn(B, H, L, d_k)
V = torch.randn(B, H, L, d_k)
attn = ScaledDotProductAttention()
out, weights = attn(Q, K, V)
assert out.shape == (B, H, L, d_k)
assert weights.shape == (B, H, L, L)
assert torch.allclose(weights.sum(dim=-1), torch.ones(B, H, L))
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (B, H, L_q, L_k)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over keys dimension
        attn_weights = torch.softmax(scores, dim=-1)       # (B, H, L_q, L_k)

        # Weighted sum of values
        output = attn_weights @ V                           # (B, H, L_q, d_v)

        return output, attn_weights
```

**Key insight**: This is the core computation of all transformer models. Note that `K.transpose(-2, -1)` swaps the last two dimensions (L_k and d_k), which is exactly what we need for the dot product. The `mask` with `-inf` becomes 0 after softmax, effectively ignoring those positions.
</details>

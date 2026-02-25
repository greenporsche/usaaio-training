# 03 — Multi-Head Attention (MHA)

> **THE CORE FILE — Discovery — Intuition — Mastery — Competition**
>
> This is the most important section. The 2025 Round 2 Problem 2, Parts 1-5 (35 points) directly tested MHA shapes and implementation.

---

## Discovery

### The Problem: One Attention Head Isn't Enough

A single attention head computes one set of attention weights — one "view" of which tokens are relevant to which. But language has many simultaneous relationships:

- Syntactic: "The **cat** that **sat** on the mat **was** fluffy" — subject-verb agreement
- Semantic: "The cat sat on the **mat**" — location information
- Coreference: "**It** was fluffy" — "it" refers to "cat"

A single head must compromise between these different notions of relevance.

### The Solution: Multiple Heads in Parallel

Vaswani et al. (2017): Run **multiple** attention operations in parallel, each with its own learned projections, then combine results. Different heads can learn to focus on different types of relationships.

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions." — Vaswani et al.

---

## Intuition

### The Subspace Interpretation

Think of the model dimension $D$ as containing many types of information mixed together. Each head projects the input into a different **subspace** of dimension $D_{qk}$, computes attention in that subspace, and the results are concatenated.

```
Head 1: focuses on syntactic relationships  → D_qk dimensions
Head 2: focuses on semantic similarity      → D_qk dimensions
Head 3: focuses on positional proximity     → D_qk dimensions
...
Head H: focuses on some other pattern       → D_qk dimensions

Concatenate all heads → H * D_v dimensions
Project back → D dimensions
```

### Parameter Efficiency

A naive approach would use $H$ completely separate attention modules, each with full $D$-dimensional projections. MHA is smarter: it uses **one large projection matrix** that's equivalent to $H$ smaller per-head projections, computed as a single matrix multiplication.

---

## Mastery

### Per-Head Projection Matrices

For head $h \in \{1, \dots, H\}$:

$$W^Q_h \in \mathbb{R}^{D_1 \times D_{qk}}, \quad W^K_h \in \mathbb{R}^{D_2 \times D_{qk}}, \quad W^V_h \in \mathbb{R}^{D_2 \times D_v}$$

Each head $h$ computes:

$$Q_h = X_1 W^Q_h, \quad K_h = X_2 W^K_h, \quad V_h = X_2 W^V_h$$

$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h) = \text{softmax}\!\left(\frac{Q_h K_h^T}{\sqrt{D_{qk}}}\right) V_h$$

**Note**: $X_1$ is the attending sequence, $X_2$ is the sequence being attended to.
- Self-attention: $X_1 = X_2 = X$, and $D_1 = D_2$
- Cross-attention: $X_1 \neq X_2$, dimensions may differ

### Concatenated Projection Matrices

Instead of $H$ separate matrix multiplications, we stack the per-head matrices:

$$W^Q = [W^Q_1 \;|\; W^Q_2 \;|\; \cdots \;|\; W^Q_H] \in \mathbb{R}^{D_1 \times (H \cdot D_{qk})}$$

$$W^K = [W^K_1 \;|\; W^K_2 \;|\; \cdots \;|\; W^K_H] \in \mathbb{R}^{D_2 \times (H \cdot D_{qk})}$$

$$W^V = [W^V_1 \;|\; W^V_2 \;|\; \cdots \;|\; W^V_H] \in \mathbb{R}^{D_2 \times (H \cdot D_v)}$$

Now one matrix multiply computes all heads at once:

$$Q_{\text{all}} = X_1 W^Q \in \mathbb{R}^{L_1 \times (H \cdot D_{qk})}$$

### The Tensor Reshaping Choreography

This is the KEY technique. After projecting, we reshape to separate heads and compute attention in parallel.

**Step-by-step with shapes** (self-attention, $D_1 = D_2 = D$):

```python
# Input
x = ...                                                       # (B, L, D)

# Step 1: Project — one matmul for all heads
Q = self.W_Q(x)                                              # (B, L, H*D_qk)
K = self.W_K(x)                                              # (B, L, H*D_qk)
V = self.W_V(x)                                              # (B, L, H*D_v)

# Step 2: Reshape — split last dim into (H, D_qk)
Q = Q.reshape(B, L, self.H, self.D_qk)                      # (B, L, H, D_qk)
K = K.reshape(B, L, self.H, self.D_qk)                      # (B, L, H, D_qk)
V = V.reshape(B, L, self.H, self.D_v)                        # (B, L, H, D_v)

# Step 3: Permute — bring heads before sequence length
Q = Q.permute(0, 2, 1, 3)                                    # (B, H, L, D_qk)
K = K.permute(0, 2, 1, 3)                                    # (B, H, L, D_qk)
V = V.permute(0, 2, 1, 3)                                    # (B, H, L, D_v)

# Step 4: Compute attention (batched across B and H simultaneously)
logits = Q @ K.mT / (self.D_qk ** 0.5)                      # (B, H, L, L)
alpha = F.softmax(logits, dim=-1)                             # (B, H, L, L)
O = alpha @ V                                                 # (B, H, L, D_v)

# Step 5: Merge heads — reverse the reshape
O = O.permute(0, 2, 1, 3)                                    # (B, L, H, D_v)
O = O.reshape(B, L, self.H * self.D_v)                       # (B, L, H*D_v)

# Step 6: Output projection
output = self.W_O(O)                                          # (B, L, D)
```

**Why this works**: After `permute(0, 2, 1, 3)`, the tensor has shape `(B, H, L, D)`. PyTorch's `@` operator works on the last two dimensions, treating `B` and `H` as batch dimensions. So `Q @ K.mT` computes $H$ separate attention matrices simultaneously — **no loops needed**.

### Output Projection

$$W^O \in \mathbb{R}^{(H \cdot D_v) \times D_1}$$

The output projection mixes information across heads. This is critical — without $W^O$, the heads would be independent and couldn't share information.

### Complete Shape Summary Table

| Tensor | Shape | Description |
|--------|-------|-------------|
| $X_1$ (input) | $(B, L_1, D_1)$ | Attending sequence |
| $X_2$ (input) | $(B, L_2, D_2)$ | Being attended sequence |
| $Q$ (projected) | $(B, L_1, H \cdot D_{qk})$ | All queries concatenated |
| $Q$ (reshaped) | $(B, H, L_1, D_{qk})$ | Queries split by head |
| $K$ (reshaped) | $(B, H, L_2, D_{qk})$ | Keys split by head |
| $V$ (reshaped) | $(B, H, L_2, D_v)$ | Values split by head |
| logits | $(B, H, L_1, L_2)$ | Attention scores |
| $\alpha$ | $(B, H, L_1, L_2)$ | Attention weights |
| $O$ (per-head) | $(B, H, L_1, D_v)$ | Per-head output |
| $O$ (merged) | $(B, L_1, H \cdot D_v)$ | Concatenated heads |
| output | $(B, L_1, D_1)$ | Final output |

### Complete PyTorch Implementation (NO LOOPS)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMHA(nn.Module):
    """Multi-Head Attention — no loops, USAAIO competition style."""

    def __init__(self, D1: int, D2: int, H: int, D_qk: int, D_v: int):
        """
        D1:   hidden dim of attending sequence
        D2:   hidden dim of being attended sequence
        H:    number of heads
        D_qk: per-head query/key dimension
        D_v:  per-head value dimension
        """
        super().__init__()
        self.H = H
        self.D_qk = D_qk
        self.D_v = D_v

        self.W_Q = nn.Linear(D1, H * D_qk, bias=False)  # (D1, H*D_qk)
        self.W_K = nn.Linear(D2, H * D_qk, bias=False)  # (D2, H*D_qk)
        self.W_V = nn.Linear(D2, H * D_v, bias=False)    # (D2, H*D_v)
        self.W_O = nn.Linear(H * D_v, D1, bias=False)    # (H*D_v, D1)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        X1: (B, L1, D1) — attending sequence
        X2: (B, L2, D2) — being attended sequence

        For self-attention: X1 = X2
        For cross-attention: X1 != X2

        Returns: (B, L1, D1)
        """
        B, L1, _ = X1.shape
        _, L2, _ = X2.shape

        # Project (one matmul per projection, all heads at once)
        Q = self.W_Q(X1)                                          # (B, L1, H*D_qk)
        K = self.W_K(X2)                                          # (B, L2, H*D_qk)
        V = self.W_V(X2)                                          # (B, L2, H*D_v)

        # Reshape: split heads
        Q = Q.reshape(B, L1, self.H, self.D_qk).permute(0, 2, 1, 3)  # (B, H, L1, D_qk)
        K = K.reshape(B, L2, self.H, self.D_qk).permute(0, 2, 1, 3)  # (B, H, L2, D_qk)
        V = V.reshape(B, L2, self.H, self.D_v).permute(0, 2, 1, 3)   # (B, H, L2, D_v)

        # Scaled dot-product attention
        logits = Q @ K.mT / (self.D_qk ** 0.5)                   # (B, H, L1, L2)
        alpha = F.softmax(logits, dim=-1)                          # (B, H, L1, L2)
        O = alpha @ V                                              # (B, H, L1, D_v)

        # Merge heads
        O = O.permute(0, 2, 1, 3).reshape(B, L1, self.H * self.D_v)  # (B, L1, H*D_v)

        # Output projection
        return self.W_O(O)                                         # (B, L1, D1)
```

### Verification

```python
# Test
B, L, D, H, D_qk, D_v = 2, 5, 16, 4, 8, 8
mha = MyMHA(D, D, H, D_qk, D_v)
x = torch.randn(B, L, D)
out = mha(x, x)  # self-attention
print(out.shape)  # torch.Size([2, 5, 16]) ✓
```

### Parameter Count

| Parameter | Shape | Count |
|-----------|-------|-------|
| $W^Q$ | $(D_1, H \cdot D_{qk})$ | $D_1 \cdot H \cdot D_{qk}$ |
| $W^K$ | $(D_2, H \cdot D_{qk})$ | $D_2 \cdot H \cdot D_{qk}$ |
| $W^V$ | $(D_2, H \cdot D_v)$ | $D_2 \cdot H \cdot D_v$ |
| $W^O$ | $(H \cdot D_v, D_1)$ | $H \cdot D_v \cdot D_1$ |

For self-attention with $D_1 = D_2 = D$ and $D_{qk} = D_v = D/H$:

$$\text{Total} = 4 \cdot D^2$$

---

## Competition Connections

### 2025 Round 2 Problem 2, Parts 1-5

This problem gave specific dimensions and asked:

- **Part 1**: Shape of per-head $W^Q_h, W^K_h, W^V_h$
- **Part 2**: Shape of concatenated $W^Q, W^K, W^V$
- **Part 3**: The scaled dot-product attention formula
- **Part 4**: Shape of output projection $W^O$
- **Part 5**: Implement `MyMHA` as `nn.Module` with NO LOOPS

The implementation in Part 5 was worth 10 points and required:
- Using `nn.Linear` for all projections
- `reshape` and `permute` (not loops over heads)
- Correct shapes at every step
- `K.mT` for batched transpose

### Common Mistakes on Exams

1. **Wrong reshape order**: `reshape(B, H, L, D_qk)` instead of `reshape(B, L, H, D_qk)` — you must reshape BEFORE permuting.

2. **Forgetting permute**: After reshape you have `(B, L, H, D)` but attention needs heads as a batch dim, so `(B, H, L, D)`.

3. **Using `.T` instead of `.mT`**: `.T` reverses ALL dimensions, `.mT` only transposes the last two.

4. **Scaling by wrong factor**: Scale by $\sqrt{D_{qk}}$ (per-head dim), not $\sqrt{D}$.

5. **Using loops**: `for h in range(H): ...` is explicitly forbidden and loses points.

### Practice: Shape Derivation

Given: $B=4, L_1=10, L_2=20, D_1=512, D_2=256, H=8, D_{qk}=64, D_v=64$

Derive the shape after each operation in MHA.

<details>
<summary>Solution</summary>

```
X1:             (4, 10, 512)
X2:             (4, 20, 256)

W_Q(X1):        (4, 10, 512)   — 512 = 8*64 = H*D_qk
W_K(X2):        (4, 20, 512)
W_V(X2):        (4, 20, 512)   — 512 = 8*64 = H*D_v

Q reshaped:     (4, 10, 8, 64)
Q permuted:     (4, 8, 10, 64)

K reshaped:     (4, 20, 8, 64)
K permuted:     (4, 8, 20, 64)

V reshaped:     (4, 20, 8, 64)
V permuted:     (4, 8, 20, 64)

Q @ K.mT:       (4, 8, 10, 64) @ (4, 8, 64, 20) = (4, 8, 10, 20)
alpha:          (4, 8, 10, 20)
alpha @ V:      (4, 8, 10, 20) @ (4, 8, 20, 64) = (4, 8, 10, 64)

O permuted:     (4, 10, 8, 64)
O reshaped:     (4, 10, 512)

W_O(O):         (4, 10, 512)
```

</details>

---

### Key Takeaways

1. **MHA = project once (all heads), reshape, attend in parallel, merge, project out.** No loops.
2. **The reshaping pattern** `(B, L, H*D) → reshape(B, L, H, D) → permute(0,2,1,3) → (B, H, L, D)` is the single most important pattern in this unit.
3. **The reverse**: `permute(0,2,1,3) → reshape(B, L, H*D)` to merge heads back.
4. **$W^O$ is essential** — it lets heads communicate by mixing their outputs.
5. **Shape tracking is non-negotiable** — annotate every line. On the exam, wrong shapes = wrong answer.

---

*Previous: [02 — Scaled Dot-Product](02-scaled-dot-product.md) | Next: [04 — Cross-Attention & Masked Attention](04-cross-attention-masked.md)*

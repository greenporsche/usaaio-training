# 02 — Scaled Dot-Product Attention

> **Discovery — Intuition — Mastery — Competition**

---

## Discovery

### The Problem: Softmax Saturation

Vaswani et al. (2017) noticed that as the dimension $d_k$ of queries and keys grows, the dot products $q \cdot k$ grow in magnitude. Large dot products push softmax into regions where its gradients are extremely small (near 0 or 1), causing **vanishing gradients** and slow training.

Their elegant fix: divide by $\sqrt{d_k}$.

This simple scaling restored stable training and became a fundamental component of all transformer architectures.

---

## Intuition

### Why Dot Products Grow

Imagine you flip more and more coins and sum the results. With 10 coins, you might get sums between 0-10. With 1000 coins, sums range from ~450 to ~550. The **magnitude** of the sum grows with the number of terms.

Dot products work the same way. If $q$ and $k$ are $d_k$-dimensional vectors with random entries, their dot product is a sum of $d_k$ random terms. More dimensions = larger magnitude = more extreme softmax outputs.

### The Softmax Problem Visually

```
d_k = 4:   scores = [1.2, 0.8, -0.3, 0.5]  → softmax ≈ [0.35, 0.24, 0.08, 0.18] ✓ spread out
d_k = 512: scores = [15.2, -8.3, 22.1, 3.7] → softmax ≈ [0.00, 0.00, 1.00, 0.00] ✗ peaked!
```

When softmax is peaked (nearly one-hot), gradients are tiny, and the model can't learn to adjust attention.

---

## Mastery

### Formal Derivation: Variance of Dot Products

**Theorem**: If $q_i, k_i$ are i.i.d. random variables with mean 0 and variance 1, then:

$$\text{Var}(q \cdot k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

**Proof**:

Since $q_i$ and $k_i$ are independent with $\mathbb{E}[q_i] = 0$, $\text{Var}(q_i) = 1$:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1$$

Since the terms $q_i k_i$ are independent:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Therefore $\text{Std}(q \cdot k) = \sqrt{d_k}$.

**Consequence**: After scaling by $1/\sqrt{d_k}$:

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

The scaled dot products have unit variance regardless of dimension. $\square$

### The Complete Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Important**: The scaling factor is $\sqrt{d_k}$ where $d_k$ is the **per-head** key dimension $D_{qk}$, NOT the model dimension $D$ and NOT $H \cdot D_{qk}$.

### Softmax Review

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

**Properties**:
- Outputs are in $(0, 1)$ and sum to 1 (probability distribution)
- Monotone: larger inputs get larger outputs
- Translation invariant: $\text{softmax}(z + c) = \text{softmax}(z)$ for any constant $c$

**Numerical stability trick**: Compute $\text{softmax}(z - \max(z))$ to avoid overflow. PyTorch's `F.softmax` does this automatically.

### Gradient Analysis

The Jacobian of softmax:

$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j)$$

When softmax is nearly one-hot (say $\text{softmax}(z)_k \approx 1$ for some $k$):
- $\frac{\partial}{\partial z_j} \approx 0$ for all $j$ — **vanishing gradients**

When softmax is uniform ($\text{softmax}(z)_i \approx 1/L$ for all $i$):
- Gradients are well-behaved, learning can proceed

Scaling keeps softmax in the moderate regime where gradients flow.

### Alternative Attention Scores

While scaled dot-product is standard, other scoring functions exist:

| Method | Formula | Complexity |
|--------|---------|-----------|
| Dot-product | $q \cdot k$ | $O(d)$ |
| **Scaled dot-product** | $q \cdot k / \sqrt{d_k}$ | $O(d)$ |
| Additive (Bahdanau) | $v^T \tanh(W_1 q + W_2 k)$ | $O(d)$ + params |
| Multiplicative | $q^T W k$ | $O(d^2)$ |

Scaled dot-product won because it is simple, parameter-free, and parallelizable with matrix multiplication.

### Implementation

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,  # (..., L1, D_qk)
    K: torch.Tensor,  # (..., L2, D_qk)
    V: torch.Tensor,  # (..., L2, D_v)
    mask: torch.Tensor = None  # (..., L1, L2) or broadcastable
) -> torch.Tensor:
    """
    Returns: (..., L1, D_v)
    """
    D_qk = Q.shape[-1]

    # Compute scaled scores
    scores = Q @ K.mT / (D_qk ** 0.5)    # (..., L1, L2)

    # Apply mask (for causal attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Normalize
    alpha = F.softmax(scores, dim=-1)      # (..., L1, L2)

    # Weighted sum
    output = alpha @ V                     # (..., L1, D_v)

    return output
```

**Key points**:
- `...` in shapes means "any number of batch/head dimensions"
- `K.mT` transposes last two dimensions: $(..., L_2, D_{qk}) \to (..., D_{qk}, L_2)$
- Mask is applied BEFORE softmax (setting masked positions to $-\infty$ so they get probability 0)

### Computational Complexity

For sequences of length $L$ with dimension $D_{qk}$:

| Operation | Complexity |
|-----------|-----------|
| $QK^T$ | $O(L^2 \cdot D_{qk})$ |
| softmax | $O(L^2)$ |
| $\alpha V$ | $O(L^2 \cdot D_v)$ |
| **Total** | $O(L^2 \cdot \max(D_{qk}, D_v))$ |

The $O(L^2)$ memory and computation is the well-known bottleneck of standard attention. This motivates efficient attention variants (not covered in USAAIO but good to know: FlashAttention, linear attention, etc.).

---

## Competition Connections

### USAAIO-Style Questions

1. **Derivation**: "Prove that the variance of the unscaled dot product of two random vectors with i.i.d. $\mathcal{N}(0,1)$ entries is $d_k$."

2. **Analysis**: "For $d_k = 64$ and two random unit-variance vectors, what is the expected standard deviation of their dot product? What is the standard deviation after scaling?"

3. **Computation**: Given specific small matrices, compute $QK^T/\sqrt{d_k}$, apply softmax, multiply by $V$.

### Practice Problem

Given $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$:

Compute the output of scaled dot-product attention (with $d_k = 2$).

<details>
<summary>Solution</summary>

$QK^T = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$

$QK^T / \sqrt{2} = \begin{pmatrix} 1/\sqrt{2} & 0 \\ 0 & 1/\sqrt{2} \end{pmatrix} \approx \begin{pmatrix} 0.707 & 0 \\ 0 & 0.707 \end{pmatrix}$

Row 1 softmax: $\frac{e^{0.707}}{e^{0.707} + e^0} = \frac{2.028}{2.028 + 1} = 0.670$, so $[0.670, 0.330]$

Row 2 softmax: $[0.330, 0.670]$ (by symmetry)

$\alpha = \begin{pmatrix} 0.670 & 0.330 \\ 0.330 & 0.670 \end{pmatrix}$

Output = $\alpha V = \begin{pmatrix} 0.670 \times 1 + 0.330 \times 3 & 0.670 \times 2 + 0.330 \times 4 \\ 0.330 \times 1 + 0.670 \times 3 & 0.330 \times 2 + 0.670 \times 4 \end{pmatrix} = \begin{pmatrix} 1.660 & 2.660 \\ 2.340 & 3.340 \end{pmatrix}$

</details>

---

### Key Takeaways

1. **Scaling by $\sqrt{d_k}$ prevents softmax saturation** — this is a critical numerical stability technique.
2. **The variance derivation is a standard exam question** — memorize the proof.
3. **Scale by the per-head $D_{qk}$**, not the model dimension $D$.
4. **Masking is applied before softmax** using $-\infty$ to zero out attention to certain positions.
5. **Attention has $O(L^2)$ complexity** in sequence length — the fundamental bottleneck.

---

*Previous: [01 — Self-Attention](01-self-attention.md) | Next: [03 — Multi-Head Attention](03-multi-head-attention.md)*

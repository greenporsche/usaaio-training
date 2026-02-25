# 08 — Multi-Head Latent Attention (MLA) & KV-Cache

> **Discovery — Intuition — Mastery — Competition**
>
> **Competition-critical**: 2025 Round 2 Problem 2, Parts 9-14 (35 points) tested MLA decomposition, proofs, and KV-cache analysis. This is the most advanced topic and the highest-scoring section.

---

## Discovery

### The Problem: Even GQA Isn't Enough

GQA reduces KV-cache by a factor of $H/G$, but for very large models (DeepSeek-V2 with 236B parameters), the cache is still enormous. The fundamental issue: KV-cache stores projections of the input at each position, and these projections live in a high-dimensional space.

### The Insight: Low-Rank Structure

DeepSeek-AI (2024) observed that the K/V projections don't need to be full-rank. Instead of storing $H$ separate key/value vectors per position, compress the input to a low-rank **latent** representation and reconstruct K/V from it.

### Multi-Head Latent Attention (MLA)

The key idea: decompose each head's K/V projection matrix into a **shared low-rank down-projection** followed by **per-head up-projections**.

---

## Intuition

### The Bottleneck Architecture

```
Input (D dims)
    │
    ▼
Shared Down-Projection W^{DKV}: D → r     ← compress to r dimensions
    │
    ├─── Per-head Up-Projection W^{UK}_1: r → D_qk  (head 1 keys)
    ├─── Per-head Up-Projection W^{UK}_2: r → D_qk  (head 2 keys)
    ├─── ...
    ├─── Per-head Up-Projection W^{UV}_1: r → D_v   (head 1 values)
    ├─── Per-head Up-Projection W^{UV}_2: r → D_v   (head 2 values)
    └─── ...
```

The bottleneck dimension $r$ controls the trade-off: smaller $r$ = less cache, but potentially less expressiveness.

**For KV-cache**: We only store the compressed representation $c = xW^{DKV} \in \mathbb{R}^r$, not the full K/V vectors.

---

## Mastery

### Formal Decomposition

For head $h$:

$$W^K_h = W^{UK}_h \cdot W^{DKV} \in \mathbb{R}^{D \times D_{qk}}$$

$$W^V_h = W^{UV}_h \cdot W^{DKV} \in \mathbb{R}^{D \times D_v}$$

where:
- $W^{DKV} \in \mathbb{R}^{D \times r}$ — shared down-projection (D to r)
- $W^{UK}_h \in \mathbb{R}^{r \times D_{qk}}$ — per-head key up-projection (r to D_qk)
- $W^{UV}_h \in \mathbb{R}^{r \times D_v}$ — per-head value up-projection (r to D_v)

**Note on matrix multiplication order**: $K_h = X \cdot W^K_h = X \cdot W^{DKV} \cdot (W^{UK}_h)^T$ when using the convention where $W^{UK}_h$ maps FROM latent TO key space. Check the exact convention used in the problem statement.

In the USAAIO formulation: $K_h = X W^{UK}_h W^{DKV}$ where:
- $W^{DKV} \in \mathbb{R}^{D \times r}$ maps input to latent
- Wait — let me state it precisely as in the 2025 exam:

$$K_h = X \cdot W^K_h = X \cdot W^{UK}_h \cdot W^{DKV}$$

where $W^{UK}_h \in \mathbb{R}^{D \times r}$ and $W^{DKV} \in \mathbb{R}^{r \times D_{qk}}$.

**IMPORTANT**: The exact convention varies. In some formulations:
- $W^{DKV}$ is the shared down-projection $D \to r$
- $W^{UK}_h$ is the per-head up-projection $r \to D_{qk}$

The key structural point is: $W^K_h = A_h \cdot B$ or $W^K_h = B \cdot A_h$ where $B$ is shared across heads and the bottleneck rank is $r$.

For this study guide, we use:

$$c = X W^{DKV}, \quad W^{DKV} \in \mathbb{R}^{D \times r} \quad \text{(shared, compresses input)}$$

$$K_h = c \cdot W^{UK}_h, \quad W^{UK}_h \in \mathbb{R}^{r \times D_{qk}} \quad \text{(per-head, for keys)}$$

$$V_h = c \cdot W^{UV}_h, \quad W^{UV}_h \in \mathbb{R}^{r \times D_v} \quad \text{(per-head, for values)}$$

So: $K_h = X W^{DKV} W^{UK}_h$ and equivalently $W^K_h = W^{DKV} W^{UK}_h$.

### Proof: GQA $\subseteq$ MLA (via SVD)

**Claim**: Any GQA configuration can be represented as MLA.

**Proof**:

Consider a GQA model with $G$ groups. For group $g$, the key projection is $W^K_g \in \mathbb{R}^{D \times D_{qk}}$.

Stack all group key matrices: $W^K_{\text{all}} = [W^K_1 \;|\; W^K_2 \;|\; \cdots \;|\; W^K_G] \in \mathbb{R}^{D \times (G \cdot D_{qk})}$.

Take the SVD: $W^K_{\text{all}} = U \Sigma V^T$ where:
- $U \in \mathbb{R}^{D \times \rho}$, $\Sigma \in \mathbb{R}^{\rho \times \rho}$, $V^T \in \mathbb{R}^{\rho \times (G \cdot D_{qk})}$
- $\rho = \text{rank}(W^K_{\text{all}}) \leq \min(D, G \cdot D_{qk})$

Set:
- $W^{DKV} = U \Sigma \in \mathbb{R}^{D \times \rho}$ (or take the rank-$r$ truncation for $r \geq \rho$)
- For head $h$ in group $g$: $W^{UK}_h = V^T_{g\text{-block}} \in \mathbb{R}^{\rho \times D_{qk}}$ (the corresponding block of $V^T$)

Then: $W^{DKV} \cdot W^{UK}_h = U\Sigma V^T_{g\text{-block}} = W^K_g$ for heads in group $g$.

Similarly for V projections (using SVD of stacked V matrices, or choosing $r$ large enough to accommodate both).

Therefore, any GQA can be expressed as MLA with rank $r \geq \text{rank}([W^K_1 | \cdots | W^K_G | W^V_1 | \cdots | W^V_G])$. $\square$

### GQA-to-MLA Conversion (NumPy Implementation)

```python
import numpy as np

def gqa_to_mla(W_K_groups, W_V_groups, r=None):
    """
    Convert GQA weights to MLA form.

    W_K_groups: list of G arrays, each (D, D_qk)
    W_V_groups: list of G arrays, each (D, D_v)
    r: target rank (default: full rank)

    Returns: W_DKV, list of W_UK, list of W_UV
    """
    # Stack all K and V matrices
    W_all = np.concatenate(W_K_groups + W_V_groups, axis=1)  # (D, G*D_qk + G*D_v)

    # SVD
    U, S, Vt = np.linalg.svd(W_all, full_matrices=False)

    # Determine rank
    rho = np.sum(S > 1e-10)
    if r is None:
        r = rho
    assert r >= rho, f"r={r} must be >= rank={rho} for exact reconstruction"

    # Shared down-projection
    W_DKV = U[:, :r] * S[:r]  # (D, r) — absorb singular values

    # Per-group up-projections
    Vt_r = Vt[:r, :]  # (r, G*D_qk + G*D_v)

    D_qk = W_K_groups[0].shape[1]
    D_v = W_V_groups[0].shape[1]
    G = len(W_K_groups)

    W_UK_list = []
    W_UV_list = []

    for g in range(G):
        W_UK_list.append(Vt_r[:, g * D_qk : (g + 1) * D_qk])        # (r, D_qk)
        W_UV_list.append(Vt_r[:, G * D_qk + g * D_v : G * D_qk + (g + 1) * D_v])  # (r, D_v)

    return W_DKV, W_UK_list, W_UV_list
```

### Proof: GQA $\subsetneq$ MLA (Strict Containment — Counterexample)

**Claim**: There exist MLA configurations that cannot be represented by any GQA.

**Proof by counterexample**:

Consider MLA with $H = 2$ heads, $r = 2$, $D = 2$, $D_{qk} = 1$:

$$W^{DKV} = I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad W^{UK}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad W^{UK}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

Then:
- $W^K_1 = W^{DKV} W^{UK}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$
- $W^K_2 = W^{DKV} W^{UK}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

These are two DISTINCT key matrices — each of rank 1.

In GQA, for this to work with $G = 1$ (the only non-MHA option), ALL heads must share the same $W^K_g$. But $W^K_1 \neq W^K_2$, so $G = 1$ GQA cannot represent this.

For $G = 2 = H$ (i.e., MHA), the configuration IS representable. But MHA with 2 heads stores $2 \times D_{qk} = 2$ values per position, while this MLA stores $r = 2$ values. In general, we can construct examples where MLA with rank $r < 2 \times G \times D_{qk}$ achieves the same expressiveness as GQA with $G$ groups, which is impossible for any GQA with fewer than $G$ groups.

More concretely: Choose $H = 4$, $D_{qk} = 1$, $r = 2$, and make all 4 heads have distinct $W^K_h$. Any GQA with $G < 4$ groups must share $W^K$ within groups, so at most $G$ distinct key matrices. But MLA has 4 distinct ones. And $G = 4 = H$ is just MHA, which requires cache $2H D_{qk} = 8 > r = 2$. $\square$

**The key insight**: MLA can represent configurations where all heads have DISTINCT key matrices (like MHA) but only cache $r$ values (like MQA). GQA must trade distinctness for cache savings.

### Reduced Matrices for Efficient MLA Inference

The full MLA attention for head $h$ is:

$$\text{logits}_h = \frac{(X_1 W^Q_h)(X_2 W^{DKV} W^{UK}_h)^T}{\sqrt{D_{qk}}}$$

We can rearrange to avoid materializing full K:

$$\text{logits}_h = \frac{(X_1 W^Q_h)(W^{UK}_h)^T(X_2 W^{DKV})^T}{\sqrt{D_{qk}}}$$

Define the **reduced query projection**:

$$\hat{W}^Q_h = W^Q_h (W^{UK}_h)^T \in \mathbb{R}^{D_1 \times r}$$

And the **compressed cache**: $C = X_2 W^{DKV} \in \mathbb{R}^{L_2 \times r}$

Then:

$$\text{logits}_h = \frac{(X_1 \hat{W}^Q_h) C^T}{\sqrt{D_{qk}}}$$

Similarly for values, define:

$$\hat{W}^O_h = W^{UV}_h (W^O_h)^T$$

where $W^O_h$ is head $h$'s slice of the output projection. Then the output can be computed directly from $C$ without materializing $V$.

### Complete Efficient MLA Implementation

```python
class MyMLA(nn.Module):
    """Multi-Head Latent Attention."""

    def __init__(self, D: int, H: int, D_qk: int, D_v: int, r: int):
        super().__init__()
        self.H = H
        self.D_qk = D_qk
        self.D_v = D_v
        self.r = r

        # Query projection (standard, per-head)
        self.W_Q = nn.Linear(D, H * D_qk, bias=False)

        # Shared down-projection to latent space
        self.W_DKV = nn.Linear(D, r, bias=False)

        # Per-head up-projections from latent space
        self.W_UK = nn.Linear(r, H * D_qk, bias=False)  # keys
        self.W_UV = nn.Linear(r, H * D_v, bias=False)    # values

        # Output projection
        self.W_O = nn.Linear(H * D_v, D, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, _ = X.shape

        # Standard query projection
        Q = self.W_Q(X)                                        # (B, L, H*D_qk)
        Q = Q.reshape(B, L, self.H, self.D_qk).permute(0, 2, 1, 3)
        #                                                       (B, H, L, D_qk)

        # Compress input to latent representation
        C = self.W_DKV(X)                                      # (B, L, r)

        # Up-project to get keys and values
        K = self.W_UK(C)                                       # (B, L, H*D_qk)
        V = self.W_UV(C)                                       # (B, L, H*D_v)

        K = K.reshape(B, L, self.H, self.D_qk).permute(0, 2, 1, 3)
        #                                                       (B, H, L, D_qk)
        V = V.reshape(B, L, self.H, self.D_v).permute(0, 2, 1, 3)
        #                                                       (B, H, L, D_v)

        # Standard attention
        logits = Q @ K.mT / (self.D_qk ** 0.5)                # (B, H, L, L)
        alpha = F.softmax(logits, dim=-1)                       # (B, H, L, L)
        O = alpha @ V                                           # (B, H, L, D_v)

        O = O.permute(0, 2, 1, 3).reshape(B, L, self.H * self.D_v)
        return self.W_O(O)                                     # (B, L, D)
```

### Efficient MLA with Reduced Matrices (Caches Only C)

```python
class MyMLAEfficient(nn.Module):
    """MLA with reduced matrices — caches only the compressed C."""

    def __init__(self, D: int, H: int, D_qk: int, D_v: int, r: int):
        super().__init__()
        self.H = H
        self.D_qk = D_qk
        self.D_v = D_v
        self.r = r

        # Reduced query: absorbs W_UK into W_Q
        # Shape: (D, H*r) instead of (D, H*D_qk)
        self.W_Q_hat = nn.Linear(D, H * r, bias=False)

        # Shared compression
        self.W_DKV = nn.Linear(D, r, bias=False)

        # For values: need W_UV per head
        self.W_UV = nn.Linear(r, H * D_v, bias=False)

        self.W_O = nn.Linear(H * D_v, D, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, _ = X.shape

        # Compressed cache
        C = self.W_DKV(X)                                     # (B, L, r)

        # Reduced query (operates in latent space)
        Q_hat = self.W_Q_hat(X)                                # (B, L, H*r)
        Q_hat = Q_hat.reshape(B, L, self.H, self.r).permute(0, 2, 1, 3)
        #                                                       (B, H, L, r)

        # Attention in latent space (no need to materialize full K!)
        logits = Q_hat @ C.unsqueeze(1).mT / (self.D_qk ** 0.5)
        #         (B, H, L, r) @ (B, 1, r, L) → (B, H, L, L)
        alpha = F.softmax(logits, dim=-1)                       # (B, H, L, L)

        # Values still need up-projection
        V = self.W_UV(C)                                       # (B, L, H*D_v)
        V = V.reshape(B, L, self.H, self.D_v).permute(0, 2, 1, 3)
        #                                                       (B, H, L, D_v)

        O = alpha @ V                                           # (B, H, L, D_v)
        O = O.permute(0, 2, 1, 3).reshape(B, L, self.H * self.D_v)
        return self.W_O(O)                                     # (B, L, D)
```

---

## KV-Cache Analysis

### What Is the KV-Cache?

During autoregressive generation, at each step $t$, we need K and V from ALL previous positions $1, \dots, t-1$. Recomputing them is wasteful, so we cache them.

### Cache Requirements by Architecture

| Architecture | What's cached per position | Size per position per layer |
|-------------|---------------------------|---------------------------|
| **MHA** | $K_1, \dots, K_H, V_1, \dots, V_H$ | $2 \times H \times D_{qk}$ |
| **GQA** ($G$ groups) | $K_1, \dots, K_G, V_1, \dots, V_G$ | $2 \times G \times D_{qk}$ |
| **MQA** ($G=1$) | $K, V$ (single head) | $2 \times D_{qk}$ |
| **MLA** (rank $r$) | $C = xW^{DKV}$ (compressed) | $r$ |

### Concrete Example

Model: $D = 4096$, $H = 32$, $D_{qk} = 128$, 80 layers, 8192 context length, FP16 (2 bytes).

| Method | Cache per layer | Total cache | vs. MHA |
|--------|----------------|-------------|---------|
| MHA | $2 \times 32 \times 128 \times 8192 \times 2$ = 128 MB | 10.0 GB | 1.0x |
| GQA ($G=8$) | $2 \times 8 \times 128 \times 8192 \times 2$ = 32 MB | 2.5 GB | 0.25x |
| MQA ($G=1$) | $2 \times 128 \times 8192 \times 2$ = 4 MB | 0.31 GB | 0.031x |
| MLA ($r=512$) | $512 \times 8192 \times 2$ = 8 MB | 0.625 GB | 0.0625x |

MLA with $r = 512$ achieves **16x** compression vs MHA while maintaining high expressiveness.

### When Is MLA Better Than GQA?

MLA wins when $r < 2 \times G \times D_{qk}$, which is the regime where:
- The KV projections have structure that can be captured by a low-rank approximation
- The model is large enough that KV-cache is a bottleneck

---

## Competition Connections

### 2025 Round 2 Problem 2, Parts 9-14

- **Part 9** (10 pts): Prove GQA $\subseteq$ MLA via SVD
- **Part 10** (5 pts): Implement GQA-to-MLA conversion in NumPy
- **Part 11** (10 pts): Counterexample showing GQA $\subsetneq$ MLA
- **Part 12** (10 pts): Derive reduced Q/K/V/O matrices for efficient MLA
- **Part 13** (5 pts): Implement reduced MLA and verify equivalence with standard MLA
- **Part 14** (5 pts): KV-cache analysis comparing MHA vs MLA

### Key Skills

1. **SVD manipulation**: Must be comfortable factoring matrices and reassembling them
2. **Rank arguments**: Understanding when column repetition doesn't change rank
3. **Counterexample construction**: Finding specific matrices that demonstrate strict containment
4. **Matrix absorption**: Rearranging products to eliminate intermediate materializations
5. **Memory analysis**: Computing exact cache sizes for different architectures

### Practice Problem

Given $D=64, H=4, D_{qk}=16, D_v=16, r=8$:

1. What are the shapes of $W^{DKV}, W^{UK}_h, W^{UV}_h$?
2. How much KV-cache does MHA use per position? MLA?
3. If GQA uses $G=2$ groups, how much cache per position?
4. Is $r=8$ sufficient to represent GQA with $G=2$ groups exactly?

<details>
<summary>Answers</summary>

1. $W^{DKV}: (64, 8)$, $W^{UK}_h: (8, 16)$, $W^{UV}_h: (8, 16)$
2. MHA: $2 \times 4 \times 16 = 128$. MLA: $r = 8$.
3. GQA: $2 \times 2 \times 16 = 64$.
4. We need $r \geq \text{rank}([W^K_1 | W^K_2 | W^V_1 | W^V_2])$. The stacked matrix is $(64, 2 \times 16 + 2 \times 16) = (64, 64)$ and could have rank up to 64. So $r = 8$ may NOT be sufficient for exact representation — it depends on the actual rank. For randomly initialized GQA with $D = 64$, the rank is likely $\min(64, 64) = 64 > 8$. So no, $r = 8$ is not sufficient in general.

</details>

---

### Key Takeaways

1. **MLA decomposes K/V projections** into shared down-projection + per-head up-projections.
2. **KV-cache stores only $r$ values** per position instead of $2HD_{qk}$.
3. **GQA $\subseteq$ MLA** via SVD decomposition of the stacked weight matrices.
4. **GQA $\subsetneq$ MLA**: MLA can have all heads distinct while caching minimally.
5. **Reduced matrices** absorb up-projections into query/output, computing attention directly in latent space.
6. **The SVD proof and counterexample are the highest-value exam topics** — practice them until they're automatic.

---

*Previous: [07 — Grouped Query Attention](07-gqa.md)*

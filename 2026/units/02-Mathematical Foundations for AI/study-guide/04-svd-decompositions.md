# 04 — Singular Value Decomposition (SVD)

**Prerequisites**: `03-eigenvalues-eigenvectors.md` (eigendecomposition, spectral theorem)
**USAAIO Relevance**: **High priority.** SVD generalizes eigendecomposition to non-square matrices and is the theoretical foundation for PCA, latent semantic analysis, and low-rank approximations. USAAIO tests SVD derivation, truncated SVD, and the Eckart-Young theorem.

---

## Discovery

It's 1936, and you're Carl Eckart and Gale Young at the University of Chicago. You've been handed a large table of experimental data — say, 1000 measurements across 50 variables. Most of the information seems to live in a much smaller number of "patterns." You want to find the best way to compress this matrix while losing as little information as possible.

**Motivating challenge**: You have a matrix $A \in \mathbb{R}^{100 \times 50}$ representing 100 data points in 50 dimensions. You can't eigendecompose it — it's not square! Yet you still want to find the "important directions." How?

Hint: Even though $A$ is not square, the matrices $A^\top A$ (size $50 \times 50$) and $AA^\top$ (size $100 \times 100$) ARE square and symmetric. What are their eigenvalues telling you?

**Socratic questions**:
1. $A^\top A$ is symmetric and PSD. What does the spectral theorem guarantee? (Real, non-negative eigenvalues and orthogonal eigenvectors)
2. If $A^\top A \mathbf{v} = \sigma^2 \mathbf{v}$, what does $\sigma$ represent geometrically? ($\sigma$ is the "stretch factor" — how much $A$ stretches in direction $\mathbf{v}$)
3. The eigenvectors of $A^\top A$ give input directions. How do we find the corresponding *output* directions? (Apply $A$ to those eigenvectors: $\mathbf{u} = A\mathbf{v}/\sigma$)

**Misconception trap**: Students often think SVD and eigendecomposition are the same thing. Eigendecomposition requires a square matrix and uses the same basis for input and output. SVD works for ANY matrix and uses *two different* orthonormal bases — one for the input space and one for the output space.

---

## Intuition

What you just discovered is the Singular Value Decomposition — arguably the most useful matrix factorization in all of applied mathematics.

### Geometric Picture

SVD says: any linear transformation can be decomposed into three steps:

```
  Input space          Intermediate         Output space

   /|                    |                    /
  / |    V^T rotates     | Sigma scales      / |    U rotates
 /  |   ────────────>    |  ──────────────> /  |
|   |                    |                 |   |
 \  |                    |                  \  |
  \ |                    |                   \ |
   \|                    |                    \|
```

1. **$V^\top$**: Rotate input to align with the "natural axes" of the transformation
2. **$\Sigma$**: Scale along each axis by the singular values $\sigma_1 \geq \sigma_2 \geq \cdots$
3. **$U$**: Rotate the scaled result to the output space

### Why SVD is So Powerful

The singular values $\sigma_i$ tell you the "importance" of each direction:
- Large $\sigma_i$: this direction carries a lot of information
- Small $\sigma_i$: this direction is mostly noise — you can throw it away
- This leads directly to optimal compression (truncated SVD)

### What Goes Wrong Without SVD?

- No principled way to compress high-dimensional data
- Can't do PCA on non-square data matrices
- Can't compute pseudoinverses for least-squares regression
- No optimal low-rank matrix approximation

---

## Math

### Theorem: Singular Value Decomposition

For any matrix $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$, there exist:
- Orthogonal matrix $U \in \mathbb{R}^{m \times m}$ (left singular vectors)
- Diagonal matrix $\Sigma \in \mathbb{R}^{m \times n}$ with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ on the diagonal
- Orthogonal matrix $V \in \mathbb{R}^{n \times n}$ (right singular vectors)

such that:

$$A = U \Sigma V^\top$$

### Derivation from Eigendecomposition

**Step 1**: Consider $A^\top A \in \mathbb{R}^{n \times n}$. It is symmetric and PSD:
- Symmetric: $(A^\top A)^\top = A^\top A$ $\checkmark$
- PSD: $\mathbf{x}^\top (A^\top A) \mathbf{x} = \|A\mathbf{x}\|^2 \geq 0$ $\checkmark$

By the spectral theorem: $A^\top A = V \Lambda_V V^\top$ where $\Lambda_V = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$ with $\sigma_i^2 \geq 0$.

**Step 2**: Define singular values $\sigma_i = \sqrt{\lambda_i(A^\top A)}$ and arrange $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots = \sigma_n$.

**Step 3**: For $i = 1, \ldots, r$, define the left singular vectors:

$$\mathbf{u}_i = \frac{A\mathbf{v}_i}{\sigma_i}$$

**Verification** that $\{\mathbf{u}_i\}$ are orthonormal:

$$\mathbf{u}_i^\top \mathbf{u}_j = \frac{(A\mathbf{v}_i)^\top (A\mathbf{v}_j)}{\sigma_i \sigma_j} = \frac{\mathbf{v}_i^\top A^\top A \mathbf{v}_j}{\sigma_i \sigma_j} = \frac{\mathbf{v}_i^\top \sigma_j^2 \mathbf{v}_j}{\sigma_i \sigma_j} = \frac{\sigma_j}{\sigma_i} \delta_{ij} = \delta_{ij}$$

$\checkmark$ (using orthonormality of $V$: $\mathbf{v}_i^\top \mathbf{v}_j = \delta_{ij}$)

**Step 4**: Extend $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ to a full orthonormal basis of $\mathbb{R}^m$ to get $U$.

**Step 5**: Verify $A = U\Sigma V^\top$ by checking $A\mathbf{v}_i = \sigma_i \mathbf{u}_i$ for $i = 1, \ldots, r$ and $A\mathbf{v}_i = \mathbf{0}$ for $i > r$. $\blacksquare$

### Compact/Reduced SVD

The full SVD has $U \in \mathbb{R}^{m \times m}$ which can be wasteful. The **compact SVD** uses only the first $r$ columns:

$$A = U_r \Sigma_r V_r^\top$$

where $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$.

### Truncated SVD (Rank-$k$ Approximation)

For $k < r$, keep only the top $k$ singular values:

$$A_k = U_k \Sigma_k V_k^\top = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

where $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$, $V_k \in \mathbb{R}^{n \times k}$.

### Eckart-Young-Mirsky Theorem

**Theorem**: The truncated SVD $A_k$ is the best rank-$k$ approximation of $A$ in both Frobenius and spectral norms:

$$A_k = \arg\min_{\text{rank}(B) = k} \|A - B\|_F$$

The error is:

$$\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

$$\|A - A_k\|_2 = \sigma_{k+1}$$

*Reasoning required*: Understanding why the truncated SVD is optimal (you should be able to sketch the proof idea, which relies on the fact that any rank-$k$ matrix can capture at most $k$ singular directions).

### Connections

| SVD quantity | Eigendecomposition of $A^\top A$ | Eigendecomposition of $AA^\top$ |
|---|---|---|
| $V$ (right singular vectors) | Eigenvectors of $A^\top A$ | — |
| $U$ (left singular vectors) | — | Eigenvectors of $AA^\top$ |
| $\sigma_i^2$ | Eigenvalues of $A^\top A$ | Eigenvalues of $AA^\top$ (same!) |

### Pseudoinverse via SVD

For $A = U\Sigma V^\top$, the **Moore-Penrose pseudoinverse** is:

$$A^+ = V \Sigma^+ U^\top$$

where $\Sigma^+$ inverts the nonzero singular values: $\Sigma^+_{ii} = 1/\sigma_i$ for $\sigma_i > 0$.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def svd_from_scratch(A: np.ndarray) -> tuple:
    """Compute SVD via eigendecomposition of A^T A and AA^T.

    Args:
        A: shape (m, n)
    Returns:
        U: shape (m, m)
        S: shape (min(m,n),) — singular values
        Vt: shape (n, n) — V transposed
    """
    m, n = A.shape  # m, n

    # Step 1: Eigendecompose A^T A to get V and singular values
    AtA = A.T @ A  # (n, n) = (n, m) @ (m, n)
    eigenvalues_v, V = np.linalg.eigh(AtA)  # (n,), (n, n)

    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues_v)[::-1]  # (n,)
    eigenvalues_v = eigenvalues_v[idx]  # (n,)
    V = V[:, idx]  # (n, n)

    # Singular values = sqrt of eigenvalues (clip tiny negatives from numerics)
    S = np.sqrt(np.maximum(eigenvalues_v, 0))  # (n,)

    # Step 2: Compute U from U = A V Sigma^{-1}
    r = np.sum(S > 1e-10)  # rank
    U_partial = np.zeros((m, r))  # (m, r)
    for i in range(r):
        U_partial[:, i] = A @ V[:, i] / S[i]  # (m,) = (m, n) @ (n,) / scalar

    # Extend to full orthonormal basis of R^m
    # Use QR on U_partial extended with random vectors
    if r < m:
        random_vecs = np.random.randn(m, m - r)  # (m, m-r)
        U_extended = np.column_stack([U_partial, random_vecs])  # (m, m)
        U, _ = np.linalg.qr(U_extended)  # (m, m)
    else:
        U = U_partial  # (m, m)

    return U, S[:min(m, n)], V.T  # (m, m), (min(m,n),), (n, n)

def truncated_svd(A: np.ndarray, k: int) -> tuple:
    """Compute rank-k approximation via truncated SVD.

    Args:
        A: shape (m, n)
        k: rank of approximation
    Returns:
        A_k: shape (m, n) — rank-k approximation
        explained_ratio: fraction of squared Frobenius norm captured
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)  # (m, r), (r,), (r, n)
    r = len(S)

    # Truncate to rank k
    U_k = U[:, :k]  # (m, k)
    S_k = S[:k]  # (k,)
    Vt_k = Vt[:k, :]  # (k, n)

    # Reconstruct
    A_k = U_k * S_k[np.newaxis, :]  # (m, k) — broadcasting
    A_k = A_k @ Vt_k  # (m, n) = (m, k) @ (k, n)

    # Alternatively: A_k = U_k @ np.diag(S_k) @ Vt_k

    # Explained ratio
    total_energy = np.sum(S**2)  # scalar
    captured_energy = np.sum(S_k**2)  # scalar
    explained_ratio = captured_energy / total_energy  # scalar

    return A_k, explained_ratio

# --- Demo ---
np.random.seed(42)
A = np.random.randn(6, 4)  # (6, 4)

# NumPy SVD (ground truth)
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # (6, 4), (4,), (4, 4)
print(f"Singular values: {S}")

# Verify: A = U diag(S) V^T
A_reconstructed = U * S[np.newaxis, :] @ Vt  # (6, 4)
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Truncated SVD
for k in [1, 2, 3, 4]:
    A_k, ratio = truncated_svd(A, k)
    error = np.linalg.norm(A - A_k, 'fro')
    print(f"Rank-{k}: error={error:.4f}, explained={ratio:.4f}")

# Verify Eckart-Young: error^2 = sum of discarded sigma^2
A_2, _ = truncated_svd(A, 2)
error_sq = np.linalg.norm(A - A_2, 'fro')**2  # scalar
expected_sq = np.sum(S[2:]**2)  # scalar
print(f"\nEckart-Young verification: {error_sq:.6f} == {expected_sq:.6f}")

# SVD from scratch
U_scratch, S_scratch, Vt_scratch = svd_from_scratch(A)
print(f"\nFrom-scratch singular values: {S_scratch[:4]}")
print(f"Difference from numpy: {np.linalg.norm(np.sort(S)[::-1] - np.sort(S_scratch[:4])[::-1]):.2e}")
```

### PyTorch Equivalent

```python
import torch

A = torch.randn(6, 4)  # (6, 4)

# Full SVD
U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # (6, 4), (4,), (4, 4)

# Truncated SVD (manual)
k = 2
A_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]  # (6, 4)

# For large-scale truncated SVD, use torch.svd_lowrank
U_k, S_k, V_k = torch.svd_lowrank(A, q=k)  # (6, k), (k,), (4, k)
A_k_fast = U_k @ torch.diag(S_k) @ V_k.T  # (6, 4)

# Batch SVD
batch_A = torch.randn(32, 10, 5)  # (B, m, n)
batch_U, batch_S, batch_Vh = torch.linalg.svd(batch_A)  # (B, m, m), (B, min), (B, n, n)
```

---

## Resources

- [Steve Brunton: SVD Overview](https://www.youtube.com/watch?v=nbBvuuNVfco) — excellent visual walkthrough
- MML Book, Chapter 4.5: Singular Value Decomposition
- [Stanford CS168: The Matrix Perspective](https://web.stanford.edu/class/cs168/) — computational perspective on SVD

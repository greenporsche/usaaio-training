# 05 — Projections and PCA

**Prerequisites**: `03-eigenvalues-eigenvectors.md` (eigendecomposition), `04-svd-decompositions.md` (SVD, truncated SVD)
**USAAIO Relevance**: **Very high priority — Round 1 favorite topic.** PCA is one of the most frequently tested concepts. You need both the maximum variance formulation and the minimum reconstruction error formulation, plus the connection to SVD.

---

## Discovery

It's 1901, and you're Karl Pearson at University College London. You have a cloud of data points in high-dimensional space, and you want to find the line (or plane, or hyperplane) that best represents the data. "Best" means: if I project all points onto this line, the spread of the projected points should be as large as possible — I want to preserve as much variation as possible.

**Motivating challenge**: You have 2D data points that form an elongated elliptical cloud:

```
        y
        |     . .
        |   . . . .
        | . . . . . .
        |. . . . . . .
        | . . . . . .
        |   . . . .
        |     . .
  ------+-----------> x
```

If you could keep only ONE direction (compressing 2D to 1D), which direction would you choose? The horizontal axis? The vertical axis? Or some diagonal?

**Socratic questions**:
1. What does "spread" mean mathematically? (Variance!)
2. If you project data onto direction $\mathbf{w}$, what is the variance of the projection? ($\mathbf{w}^\top C \mathbf{w}$ where $C$ is the covariance matrix)
3. What constraint should $\mathbf{w}$ satisfy? ($\|\mathbf{w}\| = 1$ — otherwise you could make variance infinite by scaling $\mathbf{w}$)
4. You want to maximize $\mathbf{w}^\top C \mathbf{w}$ subject to $\|\mathbf{w}\| = 1$. Does this look like an eigenvalue problem?

**Misconception trap**: PCA finds directions of maximum *variance*, not maximum *distance*. If your data is centered at a point far from the origin but has very low variance, PCA will still find low-spread directions. That's why centering the data is the critical first step.

---

## Intuition

What you just discovered is **Principal Component Analysis** — the idea that the most informative directions in data are the eigenvectors of the covariance matrix, and their importance is ranked by the corresponding eigenvalues.

### Geometric Picture: Projection

Projection of vector $\mathbf{b}$ onto vector $\mathbf{a}$:

```
        b
       /|
      / |
     /  | (error: b - proj)
    /   |
   /----|---> a
   proj_a(b)
```

The projection minimizes $\|\mathbf{b} - \text{proj}\|$ — it's the closest point on the line through $\mathbf{a}$.

### PCA as Finding the Best View

Imagine a 3D object (say, a sculpture). You want to photograph it from the angle that reveals the most detail. That "best viewing angle" is the first principal component — the direction along which the data is most spread out.

```
  Raw data (3D):        PC1 direction:      Projected (1D):
   . .                       /               |
  . . . .              . . ./. .             .|.|..|...|.|.
 . . . . .            . . ./. . .            (most spread)
  . . . .              . ../. .
   . .                    /
```

### Two Equivalent Formulations

**Maximum variance**: Find $\mathbf{w}$ that maximizes $\text{Var}(\mathbf{w}^\top \bar{X})$ — projected data should be as spread out as possible.

**Minimum reconstruction error**: Find $\mathbf{w}$ that minimizes $\sum_i \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$ where $\hat{\mathbf{x}}_i$ is the reconstruction from the projection.

These are mathematically equivalent! This is a deep and beautiful result.

### What Goes Wrong Without PCA?

- Curse of dimensionality: too many features, too little data
- Visualization: can't plot 1000-dimensional data without dimensionality reduction
- Noise: many dimensions are just noise; PCA separates signal from noise
- Computation: downstream algorithms are faster on reduced data

---

## Math

### Projection: Scalar Case

The **projection** of $\mathbf{b}$ onto $\mathbf{a}$:

$$\text{proj}_\mathbf{a}(\mathbf{b}) = \frac{\langle \mathbf{a}, \mathbf{b} \rangle}{\langle \mathbf{a}, \mathbf{a} \rangle} \mathbf{a} = \frac{\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top \mathbf{a}} \mathbf{a}$$

If $\mathbf{a}$ is a unit vector ($\|\mathbf{a}\| = 1$): $\text{proj}_\mathbf{a}(\mathbf{b}) = (\mathbf{a}^\top \mathbf{b}) \mathbf{a}$

**Derivation**: We want $\hat{\mathbf{b}} = c\mathbf{a}$ that minimizes $\|\mathbf{b} - c\mathbf{a}\|^2$.

$$\frac{d}{dc}\|\mathbf{b} - c\mathbf{a}\|^2 = \frac{d}{dc}(\mathbf{b}^\top\mathbf{b} - 2c\mathbf{a}^\top\mathbf{b} + c^2\mathbf{a}^\top\mathbf{a}) = -2\mathbf{a}^\top\mathbf{b} + 2c\mathbf{a}^\top\mathbf{a} = 0$$

$$c = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}} \qquad \blacksquare$$

### Projection Matrix

The projection matrix onto the column space of $A \in \mathbb{R}^{m \times n}$ (full column rank):

$$P = A(A^\top A)^{-1}A^\top$$

**Properties**:
- $P^2 = P$ (idempotent: projecting twice = projecting once)
- $P^\top = P$ (symmetric)
- $\text{rank}(P) = n$ (dimension of the subspace)
- $I - P$ projects onto the orthogonal complement

**Proof of idempotence**:
$$P^2 = A(A^\top A)^{-1}A^\top A(A^\top A)^{-1}A^\top = A(A^\top A)^{-1}A^\top = P \qquad \blacksquare$$

### PCA: Maximum Variance Formulation

Given data matrix $\bar{X} \in \mathbb{R}^{N \times d}$ (centered: columns have zero mean), we want to find direction $\mathbf{w} \in \mathbb{R}^d$ with $\|\mathbf{w}\| = 1$ that maximizes the variance of the projected data:

$$\max_{\|\mathbf{w}\|=1} \text{Var}(\bar{X}\mathbf{w}) = \max_{\|\mathbf{w}\|=1} \frac{1}{N-1} \|\bar{X}\mathbf{w}\|^2 = \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \underbrace{\left(\frac{\bar{X}^\top \bar{X}}{N-1}\right)}_{C} \mathbf{w}$$

where $C \in \mathbb{R}^{d \times d}$ is the **sample covariance matrix**.

**Using Lagrange multipliers** (constrained optimization with $\|\mathbf{w}\|^2 = 1$):

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top C \mathbf{w} - \lambda(\mathbf{w}^\top \mathbf{w} - 1)$$

$$\nabla_\mathbf{w} \mathcal{L} = 2C\mathbf{w} - 2\lambda\mathbf{w} = 0 \implies C\mathbf{w} = \lambda\mathbf{w}$$

This is an **eigenvalue equation**! The optimal $\mathbf{w}$ is an eigenvector of $C$, and the variance in that direction is:

$$\mathbf{w}^\top C \mathbf{w} = \mathbf{w}^\top \lambda \mathbf{w} = \lambda$$

So the maximum variance direction is the eigenvector with the **largest eigenvalue** $\lambda_1$. The second principal component is the eigenvector with the second largest eigenvalue, and so on.

*Reasoning required*: This derivation is a USAAIO favorite.

### PCA: Minimum Reconstruction Error Formulation

Alternatively, find the $k$-dimensional subspace that minimizes reconstruction error:

$$\min_{W \in \mathbb{R}^{d \times k}} \sum_{i=1}^{N} \|\mathbf{x}_i - WW^\top \mathbf{x}_i\|^2 \quad \text{s.t. } W^\top W = I_k$$

The projection of $\mathbf{x}_i$ onto the subspace spanned by columns of $W$ is $WW^\top \mathbf{x}_i$, and the reconstruction error is the squared distance between $\mathbf{x}_i$ and its projection.

**Expanding the objective**:

$$\sum_i \|\mathbf{x}_i - WW^\top\mathbf{x}_i\|^2 = \sum_i \mathbf{x}_i^\top \mathbf{x}_i - \sum_i \mathbf{x}_i^\top WW^\top \mathbf{x}_i$$

The first term is constant, so minimizing the error is equivalent to maximizing:

$$\sum_i \mathbf{x}_i^\top WW^\top \mathbf{x}_i = \text{tr}(W^\top \bar{X}^\top \bar{X} W) = (N-1)\text{tr}(W^\top C W)$$

By the spectral theorem, this is maximized when $W$ contains the top-$k$ eigenvectors of $C$. $\blacksquare$

### PCA via SVD

Since $\bar{X} = U\Sigma V^\top$ (SVD):

$$C = \frac{\bar{X}^\top \bar{X}}{N-1} = \frac{V\Sigma^2 V^\top}{N-1}$$

The eigenvectors of $C$ are the columns of $V$ (right singular vectors of $\bar{X}$), and the eigenvalues are $\sigma_i^2/(N-1)$.

**PCA summary**:
- Principal directions = columns of $V$ (or eigenvectors of $C$)
- Principal component scores = $\bar{X}V_k$ (or $U_k \Sigma_k$)
- Variance explained by component $i$ = $\sigma_i^2 / (N-1) = \lambda_i$
- Total variance explained ratio = $\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}$

### Scree Plot

Plot eigenvalues in decreasing order. The "elbow" suggests the number of components to keep:

```
eigenvalue
  |
  |*
  | *
  |  *
  |    *
  |       *  *  *  *  *     <-- elbow here: keep ~3 components
  +--+--+--+--+--+--+---> component
     1  2  3  4  5  6  7
```

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def pca_from_scratch(X: np.ndarray, k: int) -> dict:
    """PCA from scratch using eigendecomposition of covariance matrix.

    Args:
        X: shape (N, d) — data matrix (raw, not centered)
        k: number of principal components
    Returns:
        dict with keys: components, scores, explained_variance,
                       explained_ratio, mean, reconstructed
    """
    N, d = X.shape  # N samples, d features

    # Step 1: Center the data
    mean = np.mean(X, axis=0)  # (d,)
    X_centered = X - mean  # (N, d) — broadcasting

    # Step 2: Covariance matrix
    C = (X_centered.T @ X_centered) / (N - 1)  # (d, d) = (d, N) @ (N, d)

    # Step 3: Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # (d,), (d, d)

    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]  # (d,)
    eigenvalues = eigenvalues[idx]  # (d,)
    eigenvectors = eigenvectors[:, idx]  # (d, d)

    # Step 4: Select top-k components
    W = eigenvectors[:, :k]  # (d, k) — principal directions

    # Step 5: Project
    scores = X_centered @ W  # (N, k) = (N, d) @ (d, k)

    # Step 6: Reconstruct
    X_reconstructed = scores @ W.T + mean  # (N, d) = (N, k) @ (k, d) + (d,)

    # Variance explained
    total_var = np.sum(eigenvalues)  # scalar
    explained_var = eigenvalues[:k]  # (k,)
    explained_ratio = np.sum(explained_var) / total_var  # scalar

    return {
        'components': W,  # (d, k) — principal directions
        'scores': scores,  # (N, k) — projected data
        'explained_variance': explained_var,  # (k,)
        'explained_ratio': explained_ratio,  # scalar
        'mean': mean,  # (d,)
        'reconstructed': X_reconstructed,  # (N, d)
    }

def pca_via_svd(X: np.ndarray, k: int) -> dict:
    """PCA via SVD — numerically more stable for large matrices.

    Args:
        X: shape (N, d)
        k: number of components
    Returns:
        dict with same keys as pca_from_scratch
    """
    N, d = X.shape

    # Center
    mean = np.mean(X, axis=0)  # (d,)
    X_centered = X - mean  # (N, d)

    # SVD of centered data (economy/reduced form)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # (N, r), (r,), (r, d)
    # r = min(N, d)

    # Top-k
    W = Vt[:k, :].T  # (d, k) — principal directions (transpose of first k rows of Vt)
    scores = U[:, :k] * S[:k]  # (N, k) — U_k @ Sigma_k

    # Reconstruct
    X_reconstructed = scores @ W.T + mean  # (N, d)

    # Variance explained
    total_var = np.sum(S**2) / (N - 1)  # scalar
    explained_var = S[:k]**2 / (N - 1)  # (k,)
    explained_ratio = np.sum(explained_var) / total_var  # scalar

    return {
        'components': W,
        'scores': scores,
        'explained_variance': explained_var,
        'explained_ratio': explained_ratio,
        'mean': mean,
        'reconstructed': X_reconstructed,
    }

# --- Demo ---
np.random.seed(42)

# Generate correlated 2D data
N = 200
theta = np.pi / 4  # 45 degree rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])  # (2, 2) rotation
raw = np.column_stack([
    np.random.randn(N) * 3,  # (N,) — large variance direction
    np.random.randn(N) * 0.5  # (N,) — small variance direction
])  # (N, 2)
X = raw @ R.T + np.array([5, 3])  # (N, 2) — rotated + shifted

# PCA
result = pca_from_scratch(X, k=1)
print(f"Principal direction: {result['components'].flatten()}")
print(f"Variance explained: {result['explained_ratio']:.4f}")
print(f"Reconstruction error: {np.mean((X - result['reconstructed'])**2):.4f}")

# Compare with SVD version
result_svd = pca_via_svd(X, k=1)
print(f"\nSVD-based direction: {result_svd['components'].flatten()}")
print(f"SVD variance explained: {result_svd['explained_ratio']:.4f}")

# Full PCA — scree plot data
result_full = pca_from_scratch(X, k=2)
print(f"\nAll eigenvalues: {result_full['explained_variance']}")
print(f"Variance ratios: {result_full['explained_variance'] / np.sum(result_full['explained_variance'])}")
```

### PyTorch Equivalent

```python
import torch

X = torch.randn(200, 10)  # (N, d)

# Center
mean = X.mean(dim=0)  # (d,)
X_centered = X - mean  # (N, d)

# PCA via SVD (preferred in PyTorch)
U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)  # (N, d), (d,), (d, d)

k = 3
components = Vh[:k, :].T  # (d, k)
scores = U[:, :k] * S[:k]  # (N, k)
reconstructed = scores @ components.T + mean  # (N, d)

# For large-scale PCA, use torch.pca_lowrank
U_k, S_k, V_k = torch.pca_lowrank(X_centered, q=k)  # (N, k), (k,), (d, k)

# Scikit-learn equivalent (for reference):
# from sklearn.decomposition import PCA
# pca = PCA(n_components=k)
# scores = pca.fit_transform(X.numpy())
```

---

## Resources

- [StatQuest: PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — clear explanation
- [3Blue1Brown: Change of basis](https://www.3blue1brown.com/lessons/change-of-basis) — geometric understanding of projection
- MML Book, Chapter 10: Dimensionality Reduction with PCA
- Bishop, *Pattern Recognition and Machine Learning*, Section 12.1

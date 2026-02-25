# Principal Component Analysis (PCA)

**Prerequisites**: Linear algebra — eigenvalues/eigenvectors, SVD (Unit 02), covariance matrices
**USAAIO Relevance**: **HIGH** — One of the most frequently tested topics. Expect both conceptual questions (Round 1) and from-scratch implementation (Round 2). Know the derivation cold.

---

## Discovery

It's 1901, and you're Karl Pearson, a statistician at University College London. You're studying biological measurements — skull dimensions, bone lengths, body weights — and you have a problem: **too many measurements, too hard to visualize.**

You have data on 15 body measurements for 500 people. Plotting all pairs would require $\binom{15}{2} = 105$ scatter plots. But you notice something: many measurements are correlated. Tall people tend to have long arms. Heavy people tend to have wide skulls. The 15 dimensions aren't really "15 independent things" — there's a smaller set of underlying factors.

**Your question**: Can you find a new coordinate system where the first few axes capture most of the variation in the data?

Imagine looking at a 3D cloud of points. You want to find the single direction along which the data is most spread out. That's your first principal component. Then, perpendicular to that direction, find the direction with the most remaining spread. That's your second. Together, they define a 2D plane that captures the maximum possible variance.

```
Original 3D data:              Projected onto best 2D plane:
    *                               *
   * *  *                         *  *  *
  *  *  * *      ─────>        *   *  *  *
   *  * *                        *  *  *
    *  *                           * *
```

**Socratic questions**:
- Why maximize variance? What's wrong with minimizing it? (Hint: zero-variance directions carry no information.)
- If you center the data, what matrix captures all the variance information?
- How do you find the direction that maximizes $w^T \Sigma w$ subject to $\|w\| = 1$?

**Misconception trap**: PCA finds directions of maximum *variance*, not maximum *separation between classes*. It's unsupervised — it doesn't know about labels. For class separation, use LDA (Linear Discriminant Analysis).

---

## Intuition

What Pearson discovered (and Hotelling formalized in 1933) is now the most widely used dimensionality reduction technique in all of science.

### The Core Idea

PCA finds an orthogonal rotation of the coordinate axes so that:
1. The first axis points in the direction of maximum variance.
2. The second axis is perpendicular and captures the most remaining variance.
3. And so on.

```
Original coordinates:          PCA coordinates:
  y                              PC2
  |    . . .                      |    .
  |  . . . . .                    |  . . .
  | . . . . . . .    ─────>      | . . . . .
  |  . . . . .                    |  . . .
  |    . . .                      |    .
  +──────── x                     +──────── PC1
  (correlated)                    (uncorrelated!)
```

After PCA, the components are **uncorrelated** — the covariance matrix in the new coordinates is diagonal.

### Step-by-Step Process

1. **Center** the data: subtract the mean of each feature.
2. **Compute covariance matrix**: $\Sigma = \frac{1}{n-1}X_c^T X_c$ where $X_c$ is centered.
3. **Eigendecompose**: Find eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots$ and eigenvectors $v_1, v_2, \ldots$
4. **Project**: $Z = X_c V_k$ where $V_k = [v_1, v_2, \ldots, v_k]$ (top-$k$ eigenvectors as columns).

### Variance Explained

Each eigenvalue $\lambda_k$ equals the variance captured by the $k$-th principal component.

```
Scree Plot:

Variance  |
explained |  ██
(%)       |  ██ ██
          |  ██ ██ ██
          |  ██ ██ ██ ██
          |  ██ ██ ██ ██ ██ ██
          +──────────────────── Component
             1  2  3  4  5  6

Cumulative:  45% 70% 85% 93% 97% 100%
                         ^
                    Choose k=4 for 93%
```

**Rule of thumb**: Choose $k$ such that cumulative variance $\geq$ 95% (or use elbow method).

### Connection to SVD

The SVD of centered data $X_c = U \Sigma_{\text{svd}} V^T$ gives:
- Columns of $V$ = principal component directions (same as eigenvectors of $X_c^T X_c$)
- Singular values: $\sigma_k = \sqrt{(n-1)\lambda_k}$
- Projected data: $Z = U \Sigma_{\text{svd}}$ (or equivalently $X_c V$)

**In practice, use SVD** — it's numerically more stable than computing $X_c^T X_c$ explicitly.

### Reconstruction

From $k$ components, reconstruct the original data:

$$\hat{X}_c = Z V_k^T = X_c V_k V_k^T$$

Reconstruction error: $\|X_c - \hat{X}_c\|_F^2 = \sum_{j=k+1}^{D} \lambda_j$

### Failure Cases

- **Non-linear structure**: PCA only finds linear relationships. For a spiral or curved manifold, PCA fails (use kernel PCA or t-SNE/UMAP).
- **Scale dependence**: If features have different units/scales, PCA is dominated by the highest-variance feature. **Always standardize** unless features are naturally comparable.
- **Outliers**: PCA maximizes variance, so outliers have outsized influence. Consider robust PCA for noisy data.

---

## Math

### Maximum Variance Formulation

*Reasoning required for USAAIO — be able to derive this.*

We want to find a unit vector $w$ that maximizes the variance of the projected data.

The projection of centered data $X_c$ onto $w$ gives scores $z = X_c w$ with variance:

$$\text{Var}(z) = \frac{1}{n-1}(X_c w)^T(X_c w) = w^T \left(\frac{X_c^T X_c}{n-1}\right) w = w^T \Sigma w$$

**Optimization problem**:
$$\max_{w} \; w^T \Sigma w \quad \text{subject to} \quad w^T w = 1$$

**Lagrangian**:
$$\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda(w^T w - 1)$$

**Take gradient and set to zero**:
$$\frac{\partial \mathcal{L}}{\partial w} = 2\Sigma w - 2\lambda w = 0$$

$$\Sigma w = \lambda w$$

This is the **eigenvalue equation**! The optimal $w$ is an eigenvector of $\Sigma$, and the variance along that direction is:

$$w^T \Sigma w = w^T \lambda w = \lambda$$

To maximize variance, choose the eigenvector with the **largest eigenvalue** $\lambda_1$.

### Subsequent Components

The second principal component maximizes variance subject to being orthogonal to the first:

$$\max_{w_2} w_2^T \Sigma w_2 \quad \text{s.t.} \quad w_2^T w_2 = 1, \; w_2^T w_1 = 0$$

The solution is the eigenvector with the second-largest eigenvalue. By induction, the $k$-th component corresponds to the $k$-th largest eigenvalue.

### Total Variance Decomposition

$$\text{Total variance} = \text{tr}(\Sigma) = \sum_{k=1}^{D} \lambda_k$$

**Proportion of variance explained by component $k$**:
$$\frac{\lambda_k}{\sum_{j=1}^{D} \lambda_j}$$

**Cumulative proportion**:
$$\frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}$$

### SVD Connection

*Reasoning not required for USAAIO, but essential for implementation.*

Let $X_c = U \Sigma_{\text{svd}} V^T$ be the SVD. Then:

$$X_c^T X_c = V \Sigma_{\text{svd}}^2 U^T U \Sigma_{\text{svd}} V^T = ... $$

Actually more carefully: $X_c^T X_c = (U\Sigma V^T)^T (U\Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T$

So $\frac{1}{n-1}X_c^T X_c = V \frac{\Sigma_{\text{svd}}^2}{n-1} V^T$, meaning:
- Eigenvectors of covariance = right singular vectors $V$
- Eigenvalues of covariance = $\sigma_k^2 / (n-1)$

### Reconstruction Error (Eckart-Young Theorem)

The best rank-$k$ approximation to $X_c$ (in Frobenius norm) is given by the top-$k$ SVD components:

$$\min_{\text{rank-}k \; A} \|X_c - A\|_F^2 = \sum_{j=k+1}^{D} \sigma_j^2 = (n-1)\sum_{j=k+1}^{D} \lambda_j$$

This means PCA gives the optimal linear compression.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None      # (k, D) principal directions
        self.mean_ = None            # (D,) feature means
        self.eigenvalues_ = None     # (D,) all eigenvalues
        self.explained_variance_ratio_ = None  # (k,)

    def fit(self, X):
        """Fit PCA by eigendecomposing the covariance matrix."""
        # X: (N, D)
        N, D = X.shape

        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)  # (D,)
        X_c = X - self.mean_  # (N, D)

        # Step 2: Covariance matrix
        cov = (X_c.T @ X_c) / (N - 1)  # (D, D)

        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric
        # eigh returns ascending order; reverse to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]    # (D,)
        eigenvectors = eigenvectors[:, idx]  # (D, D) columns are eigenvectors

        self.eigenvalues_ = eigenvalues

        # Step 4: Select top k components
        k = self.n_components if self.n_components else D
        self.components_ = eigenvectors[:, :k].T  # (k, D)
        self.explained_variance_ratio_ = eigenvalues[:k] / np.sum(eigenvalues)  # (k,)

        return self

    def transform(self, X):
        """Project data onto principal components."""
        # X: (N, D) -> (N, k)
        X_c = X - self.mean_  # (N, D)
        return X_c @ self.components_.T  # (N, k)

    def inverse_transform(self, Z):
        """Reconstruct data from principal components."""
        # Z: (N, k) -> (N, D)
        return Z @ self.components_ + self.mean_  # (N, D)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

# Example usage
np.random.seed(42)
# Create correlated 2D data
X = np.random.randn(200, 2) @ np.array([[2, 1], [1, 3]]) + np.array([5, 10])

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print(f"Eigenvalues: {pca.eigenvalues_}")
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Component directions:\n{pca.components_}")
```

### PCA via SVD (More Numerically Stable)

```python
def pca_svd(X, n_components):
    """PCA using SVD — preferred in practice."""
    # X: (N, D) -> Z: (N, k), components: (k, D)
    N, D = X.shape
    mean = np.mean(X, axis=0)  # (D,)
    X_c = X - mean  # (N, D)

    # Economy SVD
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    # U: (N, min(N,D)), S: (min(N,D),), Vt: (min(N,D), D)

    # Principal components
    components = Vt[:n_components]  # (k, D)

    # Projected data
    Z = U[:, :n_components] * S[:n_components]  # (N, k)

    # Variance explained
    eigenvalues = S ** 2 / (N - 1)
    explained_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

    return Z, components, explained_ratio
```

### scikit-learn Equivalent

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

# Standardize (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

# Scree plot
pca_full = PCA().fit(X_scaled)
plt.bar(range(1, 5), pca_full.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.show()

# 2D scatter
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris PCA Projection')
plt.show()
```

---

## Resources

- Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space." *Philosophical Magazine*.
- Hotelling, H. (1933). "Analysis of a Complex of Statistical Variables Into Principal Components." *Journal of Educational Psychology*.
- ISLR Chapter 12.2 — PCA
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- 3Blue1Brown: "Eigenvectors and Eigenvalues" — excellent visual intuition

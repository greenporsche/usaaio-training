# Dimensionality Reduction: t-SNE & UMAP

**Prerequisites**: PCA (04-pca.md), probability distributions (Unit 02), KL divergence
**USAAIO Relevance**: Low–Medium for computation, but conceptual understanding tested in Round 1. Know when to use t-SNE vs PCA vs UMAP.

---

## Discovery

It's 2008, and you're Laurens van der Maaten at Tilburg University. PCA works beautifully for linear structure, but you're dealing with a nightmare: the MNIST handwritten digit dataset. 784 dimensions (28x28 pixels), 10 classes, and the data lives on a complex, curved manifold in that high-dimensional space.

You try PCA and project to 2D:

```
PCA projection of MNIST:       What you WANT to see:

    3 3                           0 0 0
  7   5 5                        0 0 0
 7 3  5                               1 1 1
  8 8 5 3                             1 1
 8   9 7                         2 2 2
  9 9                            2 2
                                       3 3 3
(all classes mixed together!)    (clean clusters!)
```

PCA fails because the structure is **non-linear**. Digits that look similar (8 and 3) might be far apart in PCA space because PCA only captures linear variance, not the curved structure of the data manifold.

**Your question**: Can you find a 2D representation that preserves the *neighborhood structure* — keeping similar points close and dissimilar points far apart — even if the true structure is non-linear?

**Socratic questions**:
- If you could only preserve pairwise distances between points, what distance matrix would you work with?
- Why might preserving *all* pairwise distances be too strict? What if you only preserve *local* neighborhoods?
- How would you set up an optimization problem that says "the 2D arrangement should have similar neighborhood structure to the high-D arrangement"?

---

## Intuition

### The Non-Linear Problem

PCA is a linear projection: $z = Wx$. It can rotate and project but cannot "unfold" a manifold.

```
Swiss Roll (3D):                PCA (2D):               t-SNE (2D):

  ╭──╮                         ██████                   ╭───╮
  │  ╰──╮                      ██████                   │   │
  │     │                       ██████                   ╰───╯
  ╰──╮  │                      (squished!)              (unfolded!)
     ╰──╯
```

PCA collapses the roll onto a plane, mixing layers. t-SNE "unrolls" it by preserving local distances.

### t-SNE: Matching Neighborhood Distributions

t-SNE works in three steps:

**Step 1**: In **high-dimensional** space, define a probability distribution over pairs of points. Close points get high probability, far points get low.

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

This is a Gaussian kernel — close points have high probability. The bandwidth $\sigma_i$ is chosen so each point has a specified number of effective neighbors (set by **perplexity**).

**Step 2**: In **low-dimensional** space (2D), define a similar distribution, but using a **Student-t** distribution (heavier tails):

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

**Step 3**: Minimize the **KL divergence** between $P$ and $Q$ using gradient descent:

$$C = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### Why Student-t in Low-D?

This is the key insight. In high dimensions, moderate distances are common (the "curse of dimensionality"). If you used Gaussians in both spaces, points at moderate distance in high-D would need to be *very close* in 2D to maintain the same probability — causing a "crowding problem."

The heavy-tailed Student-t allows moderate-distance points in high-D to be farther apart in 2D without paying a big penalty. This creates better-separated clusters.

```
Gaussian (thin tails):    Student-t (heavy tails):
   ╭─╮                      ╭──╮
  ╱   ╲                    ╱    ╲
 ╱     ╲                  ╱      ╲
╱       ╲                ╱    ──── ╲────
─────────────           ──────────────────
  (rapid decay)           (slow decay = room for separation)
```

### Perplexity

Perplexity is roughly "how many neighbors each point should have." It controls $\sigma_i$:

$$\text{Perplexity} = 2^{H(P_i)} \quad \text{where} \quad H(P_i) = -\sum_j p_{j|i}\log_2 p_{j|i}$$

- Low perplexity (5): Focus on very local structure. Many small clusters.
- High perplexity (50): Consider more global structure. Fewer, larger clusters.
- **Typical range**: 5–50. Always try multiple values!

### UMAP: A Topological Approach

UMAP (McInnes et al., 2018) takes a different philosophical approach:

1. Build a **fuzzy simplicial set** (weighted neighborhood graph) in high-D.
2. Build a similar graph in low-D.
3. Minimize the **cross-entropy** between the two graphs.

In practice, UMAP is:
- **Faster** than t-SNE (especially on large datasets)
- **Preserves more global structure** (relative distances between clusters are more meaningful)
- **Deterministic** with a fixed seed (t-SNE is highly stochastic)
- **Parametric** variant available (can transform new points)

### When to Use What

| Method | Linear? | Preserves | Speed | New points? | Use when... |
|--------|---------|-----------|-------|-------------|-------------|
| PCA | Yes | Global variance | Fast | Yes | Preprocessing, linear relationships |
| t-SNE | No | Local neighborhoods | Slow | No* | Publication-quality 2D visualization |
| UMAP | No | Local + some global | Medium | Yes (approx) | Large datasets, exploration |

*t-SNE has no inverse transform; new points require re-running on all data.

### Critical Warnings About t-SNE

1. **Cluster sizes are meaningless** — t-SNE normalizes local densities.
2. **Distances between clusters are meaningless** — only within-cluster structure is reliable.
3. **Different runs give different layouts** — use fixed random seeds for reproducibility.
4. **Perplexity matters a lot** — always try multiple values.
5. **Not suitable for downstream ML** — use PCA for feature extraction, t-SNE only for visualization.

---

## Math

### t-SNE Objective

*Reasoning not required for full derivation, but understand KL divergence.*

**Symmetric SNE affinities** (high-D):

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

where $p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$

**Low-D affinities** (Student-t with 1 degree of freedom = Cauchy):

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

**Objective**: $C = KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

**Gradient** (for updating $y_i$):

$$\frac{\partial C}{\partial y_i} = 4\sum_{j} (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$$

This gradient has an attractive-repulsive interpretation:
- When $p_{ij} > q_{ij}$: points should be *closer* in low-D (attractive force)
- When $p_{ij} < q_{ij}$: points should be *farther* in low-D (repulsive force)

### KL Divergence Review

$$KL(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} \geq 0$$

- $KL = 0$ iff $P = Q$.
- **Asymmetric**: $KL(P \| Q) \neq KL(Q \| P)$.
- $KL(P \| Q)$ penalizes heavily when $P$ is large but $Q$ is small (i.e., neighbors in high-D that are far apart in low-D).

### UMAP Objective (Simplified)

*Reasoning not required for USAAIO.*

UMAP minimizes a fuzzy set cross-entropy:

$$C = \sum_{i,j} \left[v_{ij} \log\frac{v_{ij}}{w_{ij}} + (1 - v_{ij})\log\frac{1-v_{ij}}{1-w_{ij}}\right]$$

where $v_{ij}$ are high-D fuzzy membership strengths and $w_{ij} = (1 + a\|y_i - y_j\|^{2b})^{-1}$ are low-D similarities.

---

## Code

### t-SNE with scikit-learn

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load MNIST-like data (8x8 digits)
X, y = load_digits(return_X_y=True)
print(f"Data shape: {X.shape}")  # (1797, 64)

# t-SNE with different perplexities
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, perp in zip(axes, [5, 30, 50]):
    Z = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=42,
        n_iter=1000
    ).fit_transform(X)  # (1797, 2)

    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f'Perplexity = {perp}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.colorbar(scatter, ax=axes, label='Digit')
plt.suptitle('t-SNE: Effect of Perplexity')
plt.tight_layout()
plt.show()
```

### UMAP

```python
import umap

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,    # like perplexity — local vs global
    min_dist=0.1,      # minimum distance in embedding
    random_state=42
)
Z_umap = reducer.fit_transform(X)  # (1797, 2)

plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(label='Digit')
plt.title('UMAP projection')
plt.show()

# Transform new points (UMAP supports this, t-SNE does not)
Z_new = reducer.transform(X[:10])
```

### Comparing PCA, t-SNE, UMAP

```python
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
Z_pca = PCA(n_components=2).fit_transform(X)
axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[0].set_title('PCA')

# t-SNE
Z_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[1].set_title('t-SNE (perplexity=30)')

# UMAP
Z_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
axes[2].scatter(Z_umap[:, 0], Z_umap[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
axes[2].set_title('UMAP')

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('Dimensionality Reduction Comparison on Digits')
plt.tight_layout()
plt.show()
```

---

## Resources

- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579–2605.
- McInnes, L. et al. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*.
- [Distill: "How to Use t-SNE Effectively"](https://distill.pub/2016/misread-tsne/) — Essential reading on t-SNE pitfalls.
- [UMAP documentation](https://umap-learn.readthedocs.io/)

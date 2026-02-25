# K-Means Clustering

**Prerequisites**: Linear algebra (distances, norms), calculus (taking derivatives), Python/NumPy
**USAAIO Relevance**: **HIGH** — K-means is a staple of Round 1 (trace iterations, compute centroids) and Round 2 (implement from scratch with k-means++). Know Lloyd's algorithm cold.

---

## Discovery

It's 1957, and you're Stuart Lloyd at Bell Labs. You're working on a problem in signal processing called **pulse-code modulation**: how to optimally quantize a continuous signal into a discrete set of levels. If you have $k$ levels, where should you place them to minimize the total distortion?

Consider a simpler version: you have 1000 customers scattered across a city, and you need to place 3 warehouses. Where do you put them to minimize the total distance customers travel?

```
Before optimization:              After optimization:

    ·  ·                              ·  ·
  ·  · W · ·                        ·  ·★· ·
 · ·  · · ·                        · ·  · · ·
    ·  ·                               ·  ·
  · ·  · ·     · ·                  · ·  · ·     · ·
 · · ·  W  ·  ·  ·  ·             · · · ★  ·  ·  ·  ·
  · ·  · ·   ·  · ·                · ·  · ·   · ★·
    ·  ·     W · · ·                  ·  ·     · · ·
                                  (warehouses moved to cluster centers)
```

**Socratic questions**:
- If the warehouses are fixed, how should you assign each customer? (Nearest warehouse, obviously.)
- If the assignments are fixed, where should each warehouse go? (Center of its assigned customers — the mean!)
- What happens if you alternate these two steps?

**Misconception trap**: K-means does NOT find the globally optimal clustering — it finds a **local minimum** that depends on initialization. Run it multiple times with different starting points!

---

## Intuition

What you just discovered is **Lloyd's algorithm** (published 1982, but developed in 1957). It alternates between two simple steps:

### The Two Steps

```
ASSIGN step:                    UPDATE step:
Assign each point to            Move each centroid to the
the nearest centroid            mean of its assigned points

  ·  ·  ★₁                       ·  ·  ★₁←(moved)
 ·¹ ·¹  ·¹                      ·  ·   ·
  ·¹  ·²  ·²                     ·  ·  ·
   ·²  ★₂  ·²                     ·  ★₂←(moved)  ·
    ·²  ·²                          ·  ·

Superscript = cluster assignment    ★ = new centroid position
```

**Assign**: For each point, find the closest centroid.
**Update**: Recompute each centroid as the mean of its cluster.
**Repeat** until assignments don't change.

### Why It Converges

The objective function (inertia / WCSS):

$$J = \sum_{j=1}^{k}\sum_{i \in C_j} \|x_i - \mu_j\|^2$$

- The **assign** step minimizes $J$ over assignments (each point picks its nearest centroid).
- The **update** step minimizes $J$ over centroid positions (the mean minimizes sum of squared distances).
- Each step decreases (or doesn't change) $J$.
- There are finitely many possible assignments.
- Therefore, the algorithm must terminate in a finite number of steps.

**But**: $J$ has many local minima. The algorithm finds one of them, not necessarily the best.

### K-Means++ Initialization

Random initialization often leads to bad local minima. **K-means++** (Arthur & Vassilvitskii, 2007) is a smarter initialization:

```
1. Pick first centroid randomly from data points.
2. For each remaining point, compute D(x) = distance to nearest existing centroid.
3. Pick next centroid with probability proportional to D(x)².
4. Repeat steps 2-3 until k centroids chosen.
5. Run Lloyd's algorithm from these starting centroids.
```

**Intuition**: Points far from existing centroids are more likely to be chosen, spreading centroids across the data.

**Guarantee**: K-means++ gives a solution within $O(\log k)$ of optimal in expectation.

### Choosing $k$

**Elbow Method**: Plot $J$ vs $k$. Look for an "elbow" where the rate of decrease sharply changes.

```
J (inertia)
  |╲
  | ╲
  |  ╲
  |   ╲____
  |        ╲_____
  |              ╲________
  +──────────────────────── k
  1  2  3  4  5  6  7  8
            ^
         elbow at k=4
```

**Silhouette Score**: For each point $i$:
- $a_i$ = mean distance to points in same cluster
- $b_i$ = mean distance to points in nearest other cluster
- $s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$

$s_i \approx 1$: well-clustered. $s_i \approx 0$: on boundary. $s_i < 0$: possibly misassigned.

Average over all points: higher is better. Choose $k$ with highest average silhouette.

### Failure Cases

```
K-means fails:                  K-means succeeds:

Concentric circles:             Spherical/convex clusters:
    ○ ○ ○ ○                        ● ● ●     ○ ○ ○
  ○  ● ● ●  ○                     ● ● ●     ○ ○ ○
  ○  ● ● ●  ○                     ● ● ●     ○ ○ ○
    ○ ○ ○ ○
(can't separate with centroids)  (centroids work perfectly)
```

K-means assumes **spherical, similarly-sized clusters**. It fails on:
- Non-convex shapes (crescents, rings)
- Clusters of very different sizes or densities
- High-dimensional data (distances become less meaningful — "curse of dimensionality")

---

## Math

### Objective Function

*Reasoning required for USAAIO.*

$$J = \sum_{j=1}^{k}\sum_{i \in C_j} \|x_i - \mu_j\|^2$$

where $C_j$ is the set of points in cluster $j$ and $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$ is the centroid.

### Assign Step Optimality

For fixed centroids $\mu_1, \ldots, \mu_k$, the optimal assignment is:

$$c_i = \arg\min_{j \in \{1,\ldots,k\}} \|x_i - \mu_j\|^2$$

This minimizes $J$ over assignments because each point independently contributes $\|x_i - \mu_{c_i}\|^2$.

### Update Step Optimality

For fixed assignments, the optimal centroid for cluster $j$ minimizes:

$$\min_{\mu_j} \sum_{i \in C_j} \|x_i - \mu_j\|^2$$

Taking the gradient and setting to zero:

$$\frac{\partial}{\partial \mu_j}\sum_{i \in C_j}\|x_i - \mu_j\|^2 = \sum_{i \in C_j} -2(x_i - \mu_j) = 0$$

$$\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$$

The optimal centroid is the **mean** of the cluster — hence the name "k-means."

### Convergence Proof

*Reasoning not required for USAAIO, but understand the argument.*

1. $J$ is bounded below by 0.
2. Each assign step: $J$ decreases or stays the same (each point moves to a closer centroid).
3. Each update step: $J$ decreases or stays the same (mean minimizes sum of squared distances).
4. There are at most $k^n$ possible assignments (finite).
5. $J$ strictly decreases with each assignment change.
6. Therefore, the algorithm terminates in finite steps. $\square$

### Silhouette Score

*Reasoning not required, but know the formula.*

For point $i$ in cluster $C_I$:

$$a_i = \frac{1}{|C_I| - 1}\sum_{j \in C_I, j \neq i}\|x_i - x_j\|$$

$$b_i = \min_{J \neq I}\frac{1}{|C_J|}\sum_{j \in C_J}\|x_i - x_j\|$$

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

### Connection to Gaussian Mixture Models

K-means is a special case of EM for Gaussian mixtures where:
- All covariances are $\sigma^2 I$ (spherical, equal)
- As $\sigma \to 0$, soft assignments become hard assignments
- Centroids correspond to mixture means

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def kmeans_plus_plus_init(X, k, rng):
    """K-means++ initialization."""
    # X: (N, D) -> centroids: (k, D)
    N, D = X.shape
    centroids = np.empty((k, D))

    # First centroid: random data point
    idx = rng.integers(N)
    centroids[0] = X[idx]

    for c in range(1, k):
        # Compute squared distances to nearest existing centroid
        dists = np.min(
            np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :c, :]) ** 2, axis=2),
            axis=1
        )  # (N,)

        # Sample proportional to D(x)^2
        probs = dists / dists.sum()
        idx = rng.choice(N, p=probs)
        centroids[c] = X[idx]

    return centroids  # (k, D)

def kmeans(X, k, max_iters=100, seed=42):
    """K-means clustering with k-means++ initialization."""
    # X: (N, D) -> labels: (N,), centroids: (k, D), inertia: float
    N, D = X.shape
    rng = np.random.default_rng(seed)

    # Initialize centroids with k-means++
    centroids = kmeans_plus_plus_init(X, k, rng)  # (k, D)

    for iteration in range(max_iters):
        # ASSIGN: each point to nearest centroid
        # Compute all pairwise distances: (N, k)
        dists = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )  # (N, k)
        labels = np.argmin(dists, axis=1)  # (N,)

        # UPDATE: move centroids to cluster means
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(axis=0)  # (D,)
            else:
                # Empty cluster: reinitialize randomly
                new_centroids[j] = X[rng.integers(N)]

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Compute inertia (WCSS)
    inertia = sum(
        np.sum((X[labels == j] - centroids[j]) ** 2)
        for j in range(k)
    )

    return labels, centroids, inertia

def silhouette_score(X, labels):
    """Compute mean silhouette score."""
    # X: (N, D), labels: (N,) -> float
    N = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k <= 1 or k >= N:
        return 0.0

    scores = np.zeros(N)
    for i in range(N):
        # a_i: mean intra-cluster distance
        same_mask = (labels == labels[i])
        same_mask[i] = False
        if same_mask.sum() == 0:
            scores[i] = 0
            continue
        a_i = np.mean(np.sqrt(np.sum((X[same_mask] - X[i]) ** 2, axis=1)))

        # b_i: mean distance to nearest other cluster
        b_i = float('inf')
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_mask = labels == label
            mean_dist = np.mean(np.sqrt(np.sum((X[other_mask] - X[i]) ** 2, axis=1)))
            b_i = min(b_i, mean_dist)

        scores[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(scores)

# Example: elbow method
def elbow_method(X, k_range=range(1, 11), seed=42):
    """Compute inertia for different k values."""
    inertias = []
    for k in k_range:
        _, _, inertia = kmeans(X, k, seed=seed)
        inertias.append(inertia)
    return list(k_range), inertias
```

### scikit-learn Equivalent

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-means with k-means++ (default)
km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X)

print(f"Inertia: {km.inertia_:.2f}")
print(f"Centroids:\n{km.cluster_centers_}")
print(f"Silhouette score: {silhouette_score(X, labels):.3f}")

# Elbow plot
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()

# Silhouette analysis
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

plt.plot(range(2, 11), sil_scores, 'ro-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()
```

---

## Resources

- Lloyd, S. (1982). "Least Squares Quantization in PCM." *IEEE Trans. Information Theory*, 28(2), 129–137. (Originally a 1957 Bell Labs technical report.)
- Arthur, D. & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." *SODA*.
- ISLR Chapter 12.4 — K-Means Clustering
- [scikit-learn: K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)

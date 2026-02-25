# Dimensionality Reduction (t-SNE & UMAP) — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: t-SNE Affinities

Given 4 points with pairwise Euclidean distances:

| | $x_1$ | $x_2$ | $x_3$ | $x_4$ |
|---|-------|-------|-------|-------|
| $x_1$ | 0 | 1 | 5 | 6 |
| $x_2$ | 1 | 0 | 4 | 5 |
| $x_3$ | 5 | 4 | 0 | 1 |
| $x_4$ | 6 | 5 | 1 | 0 |

Assume $\sigma_i = 1$ for all $i$ (for simplicity).

1. Compute the conditional probability $p_{2|1}$ (how much $x_1$ considers $x_2$ a neighbor).
2. Compute $p_{3|1}$ and $p_{4|1}$.
3. Which point does $x_1$ consider its strongest neighbor?
4. Compute the symmetric affinity $p_{12} = \frac{p_{2|1} + p_{1|2}}{2n}$ where $n = 4$.

<details>
<summary>Solution</summary>

**Part 1**: $p_{2|1} = \frac{\exp(-d_{12}^2 / 2)}{\sum_{k \neq 1}\exp(-d_{1k}^2 / 2)}$

Numerator: $\exp(-1^2/2) = \exp(-0.5) = 0.6065$

Denominator: $\exp(-0.5) + \exp(-25/2) + \exp(-36/2) = 0.6065 + 3.73 \times 10^{-6} + 1.52 \times 10^{-8}$

$\approx 0.6065$ (other terms are negligible!)

$p_{2|1} \approx 0.6065 / 0.6065 \approx 1.0$ (almost all the probability mass)

**Part 2**:

$p_{3|1} = \frac{\exp(-25/2)}{0.6065} = \frac{3.73 \times 10^{-6}}{0.6065} \approx 6.15 \times 10^{-6}$

$p_{4|1} = \frac{\exp(-18)}{0.6065} = \frac{1.52 \times 10^{-8}}{0.6065} \approx 2.51 \times 10^{-8}$

**Part 3**: $x_1$ overwhelmingly considers $x_2$ as its neighbor ($p_{2|1} \approx 1.0$), with $x_3$ and $x_4$ being essentially zero. The Gaussian kernel creates very sharp neighbor distinctions with $\sigma = 1$.

**Part 4**: By symmetry of distances, $p_{1|2} \approx 1.0$ as well (since $x_2$'s nearest point is also $x_1$).

$p_{12} = \frac{p_{2|1} + p_{1|2}}{2 \times 4} = \frac{1.0 + 1.0}{8} = 0.25$

This is a very high symmetric affinity — these two points are strong mutual neighbors.

</details>

---

## Exercise 2: Perplexity Interpretation

The perplexity of the conditional distribution $P_i$ for point $x_i$ is defined as:

$$\text{Perp}(P_i) = 2^{H(P_i)}$$

where $H(P_i) = -\sum_j p_{j|i}\log_2 p_{j|i}$ is the entropy.

1. If $x_i$ has exactly 3 equally likely neighbors and all other points have zero probability, what is $H(P_i)$? What is the perplexity?
2. If $x_i$ has one neighbor with probability 0.9 and nine other neighbors each with probability 0.011, approximate the entropy and perplexity.
3. A user sets perplexity = 30. Roughly how many effective neighbors does each point have?
4. Why should perplexity be smaller than the number of data points?

<details>
<summary>Solution</summary>

**Part 1**: With 3 equally likely neighbors: $p_{j|i} = 1/3$ for three points.

$H = -3 \times \frac{1}{3}\log_2\frac{1}{3} = -\log_2\frac{1}{3} = \log_2 3 \approx 1.585$ bits

Perplexity $= 2^{1.585} = 3.0$

So perplexity = number of effective neighbors for a uniform distribution.

**Part 2**: $H = -0.9\log_2(0.9) - 9 \times 0.011\log_2(0.011)$

$= -0.9(-0.152) - 0.099(-6.507) = 0.137 + 0.644 = 0.781$ bits

Perplexity $= 2^{0.781} \approx 1.72$

Despite having 10 neighbors, the effective number is only ~1.7 because almost all mass is on one point.

**Part 3**: Perplexity 30 means each point has roughly **30 effective neighbors**. The algorithm adjusts $\sigma_i$ for each point until its conditional distribution has entropy $\log_2(30) \approx 4.91$ bits.

**Part 4**: Perplexity must be smaller than $n$ because it represents the number of effective neighbors per point. If perplexity $\geq n$, each point would consider ALL other points as neighbors, which means the neighborhood structure carries no information — every point is equally close. This defeats the purpose of t-SNE, which relies on local neighborhood differences.

In practice, perplexity should be much less than $n$ (typically 5–50), even for datasets with thousands of points, to preserve local structure.

</details>

---

## Exercise 3: KL Divergence Asymmetry

Consider two distributions:

$P$: $p_1 = 0.8, p_2 = 0.2$

$Q$: $q_1 = 0.5, q_2 = 0.5$

1. Compute $KL(P \| Q)$.
2. Compute $KL(Q \| P)$.
3. Are they equal? What does the asymmetry mean for t-SNE?
4. In t-SNE, we minimize $KL(P \| Q)$ where $P$ is the high-D distribution and $Q$ is the low-D distribution. What happens when $p_{ij}$ is large but $q_{ij}$ is small? What about $p_{ij}$ small but $q_{ij}$ large?

<details>
<summary>Solution</summary>

**Part 1**: $KL(P \| Q) = 0.8\ln\frac{0.8}{0.5} + 0.2\ln\frac{0.2}{0.5}$

$= 0.8\ln(1.6) + 0.2\ln(0.4) = 0.8(0.470) + 0.2(-0.916) = 0.376 - 0.183 = 0.193$

**Part 2**: $KL(Q \| P) = 0.5\ln\frac{0.5}{0.8} + 0.5\ln\frac{0.5}{0.2}$

$= 0.5\ln(0.625) + 0.5\ln(2.5) = 0.5(-0.470) + 0.5(0.916) = -0.235 + 0.458 = 0.223$

**Part 3**: No, $KL(P \| Q) = 0.193 \neq 0.223 = KL(Q \| P)$. KL divergence is asymmetric.

**Part 4**: In $KL(P \| Q) = \sum p_{ij} \log(p_{ij}/q_{ij})$:

- **$p_{ij}$ large, $q_{ij}$ small**: The term $p_{ij}\log(p_{ij}/q_{ij})$ is large and positive. This is **heavily penalized**. It means neighbors in high-D that are far apart in low-D — t-SNE strongly pulls them together. This preserves local structure.

- **$p_{ij}$ small, $q_{ij}$ large**: The term is near zero (because $p_{ij} \approx 0$ multiplies everything). This is **barely penalized**. It means non-neighbors in high-D that end up close in low-D — t-SNE doesn't care much. This is why distances between clusters in t-SNE can be misleading.

</details>

---

## Exercise 4: t-SNE Interpretation Pitfalls

Examine this hypothetical t-SNE plot:

```
           Cluster A       Cluster B
            (large)          (small)
          ○ ○ ○ ○ ○        ● ●
         ○ ○ ○ ○ ○ ○       ● ●
          ○ ○ ○ ○ ○
                                    Cluster C
                                   (medium, far away)
                                    ▲ ▲ ▲
                                    ▲ ▲ ▲
```

A colleague makes three claims. Evaluate each:

1. "Cluster A has more data points than Cluster B because it's bigger."
2. "Cluster C is more different from A and B because it's farther away."
3. "The data has exactly 3 natural clusters."
4. What would you need to verify these claims?

<details>
<summary>Solution</summary>

**Claim 1**: "Cluster A has more data points than Cluster B because it's bigger."

**FALSE.** t-SNE normalizes local densities — it adjusts $\sigma_i$ per point to match the specified perplexity. A dense cluster in high-D will be "puffed up" in t-SNE, and a sparse cluster may be compressed. Cluster sizes in t-SNE reflect neither count nor density reliably.

**Claim 2**: "Cluster C is more different from A and B because it's farther away."

**FALSE.** Inter-cluster distances in t-SNE are NOT meaningful. t-SNE optimizes local neighborhoods, not global distances. Cluster C might be equidistant from A and B in the original space, yet appear far away due to the optimization landscape. Different random seeds could place C between A and B instead.

**Claim 3**: "The data has exactly 3 natural clusters."

**UNCERTAIN.** t-SNE can create artificial clusters from continuous data by tearing apart gradients. It can also merge clusters or create subclusters depending on perplexity. You should try multiple perplexity values. If the 3-cluster structure is consistent across perplexities, it's more reliable.

**Part 4**: To verify:

1. **Cluster sizes**: Check the actual number of points per cluster in the original data (or color by group size).
2. **Inter-cluster distances**: Compute pairwise distances between cluster centroids in the original high-dimensional space, or use UMAP which better preserves global structure.
3. **Number of clusters**: Run clustering algorithms (k-means with silhouette, DBSCAN) on the original high-D data. Try different perplexities in t-SNE (5, 15, 30, 50) and see if the structure is consistent.

</details>

---

## Exercise 5: When to Use Which Method

For each scenario, recommend the best dimensionality reduction method (PCA, t-SNE, or UMAP) and explain why:

1. **Preprocessing for a downstream ML model**: You have 500 features and want to reduce to 50 before training a classifier.
2. **Publication figure**: You need a beautiful 2D visualization of 5000 single-cell RNA-seq samples for a biology paper.
3. **Exploratory analysis**: You're exploring a new dataset of 1,000,000 customer behavior vectors and want a quick overview.
4. **Feature extraction from images**: You want to compress 784-dimensional MNIST images into a compact representation that can be used for nearest-neighbor search.
5. **Detecting outliers**: You want to identify anomalous samples in a 100-dimensional sensor dataset.

<details>
<summary>Solution</summary>

**Scenario 1**: **PCA**. For preprocessing, PCA is ideal because: (a) it has an inverse transform for reconstruction, (b) it's fast, (c) it preserves global variance which is what downstream classifiers need, (d) it handles new data (just project using the learned components), and (e) 50 components is too many for t-SNE/UMAP which are designed for 2-3D.

**Scenario 2**: **t-SNE** (or UMAP). For publication-quality 2D visualization of biological data, t-SNE is the standard in genomics. It creates visually appealing, well-separated clusters. For 5000 points, runtime is acceptable. Use multiple perplexity values and random seeds to ensure robustness. UMAP is also acceptable and increasingly popular in this field.

**Scenario 3**: **UMAP**. With 1,000,000 points, t-SNE is too slow ($O(n^2)$ or $O(n\log n)$ with Barnes-Hut). UMAP scales much better and provides good global structure for exploration. PCA could be a first pass (reduce to 50D), followed by UMAP to 2D.

**Scenario 4**: **PCA**. For feature extraction that supports nearest-neighbor search, PCA is best because: (a) it preserves Euclidean distances (approximately) after projection, (b) it can transform new query images, (c) distance-based retrieval needs global distance preservation, which t-SNE explicitly discards.

**Scenario 5**: **PCA**. For outlier detection, PCA is preferred because: (a) outliers have high reconstruction error (distance from PCA subspace), which is a direct outlier score, (b) the Mahalanobis distance in PC space is interpretable, (c) PCA preserves global structure, so true outliers (far from the data manifold) stand out. t-SNE might distort outlier positions.

</details>

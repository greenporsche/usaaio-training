# K-Means Clustering — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: Trace K-Means Iterations

Given 6 points in 2D and initial centroids $\mu_1 = (0, 0)$ and $\mu_2 = (5, 5)$:

| Point | $x$ | $y$ |
|-------|-----|-----|
| A | 1 | 1 |
| B | 1.5 | 2 |
| C | 3 | 4 |
| D | 5 | 7 |
| E | 3.5 | 5 |
| F | 4.5 | 5 |

1. **Iteration 1, Assign**: Assign each point to its nearest centroid (use squared Euclidean distance).
2. **Iteration 1, Update**: Compute the new centroids.
3. **Iteration 2, Assign**: Reassign points to new centroids. Did any points change clusters?
4. Compute the inertia (WCSS) after iteration 2.

<details>
<summary>Solution</summary>

**Part 1**: Distances to $\mu_1 = (0,0)$ and $\mu_2 = (5,5)$:

| Point | $d^2(\mu_1)$ | $d^2(\mu_2)$ | Cluster |
|-------|-------------|-------------|---------|
| A(1,1) | $1+1=2$ | $16+16=32$ | 1 |
| B(1.5,2) | $2.25+4=6.25$ | $12.25+9=21.25$ | 1 |
| C(3,4) | $9+16=25$ | $4+1=5$ | 2 |
| D(5,7) | $25+49=74$ | $0+4=4$ | 2 |
| E(3.5,5) | $12.25+25=37.25$ | $2.25+0=2.25$ | 2 |
| F(4.5,5) | $20.25+25=45.25$ | $0.25+0=0.25$ | 2 |

Cluster 1: {A, B}, Cluster 2: {C, D, E, F}

**Part 2**: New centroids:

$\mu_1 = \frac{1}{2}((1,1) + (1.5,2)) = (1.25, 1.5)$

$\mu_2 = \frac{1}{4}((3,4) + (5,7) + (3.5,5) + (4.5,5)) = (\frac{16}{4}, \frac{21}{4}) = (4.0, 5.25)$

**Part 3**: Distances to new centroids:

| Point | $d^2(\mu_1 = (1.25, 1.5))$ | $d^2(\mu_2 = (4, 5.25))$ | Cluster |
|-------|---------------------------|--------------------------|---------|
| A(1,1) | $0.0625+0.25=0.3125$ | $9+18.0625=27.0625$ | 1 |
| B(1.5,2) | $0.0625+0.25=0.3125$ | $6.25+10.5625=16.8125$ | 1 |
| C(3,4) | $3.0625+6.25=9.3125$ | $1+1.5625=2.5625$ | 2 |
| D(5,7) | $14.0625+30.25=44.3125$ | $1+3.0625=4.0625$ | 2 |
| E(3.5,5) | $5.0625+12.25=17.3125$ | $0.25+0.0625=0.3125$ | 2 |
| F(4.5,5) | $10.5625+12.25=22.8125$ | $0.25+0.0625=0.3125$ | 2 |

No points changed clusters. The algorithm has **converged** after 2 iterations.

**Part 4**: Inertia:

Cluster 1: $0.3125 + 0.3125 = 0.625$

Cluster 2: $2.5625 + 4.0625 + 0.3125 + 0.3125 = 7.25$

Total inertia = $0.625 + 7.25 = 7.875$

</details>

---

## Exercise 2: K-Means++ Initialization

Given 5 points: $A(0,0)$, $B(1,0)$, $C(10,0)$, $D(10,1)$, $E(11,0)$.

We want $k = 2$ centroids using k-means++.

1. Suppose the first centroid is randomly chosen as point $A(0,0)$. Compute $D(x)^2$ for each remaining point.
2. Compute the probability of selecting each point as the second centroid.
3. Which point is most likely to be chosen? Why does this make intuitive sense?
4. Compare this to naive random initialization where both centroids might be chosen from the left cluster $\{A, B\}$.

<details>
<summary>Solution</summary>

**Part 1**: $D(x)^2$ = squared distance to nearest existing centroid (which is $A(0,0)$):

| Point | $D(x)^2$ |
|-------|----------|
| B(1,0) | $1^2 + 0^2 = 1$ |
| C(10,0) | $10^2 + 0^2 = 100$ |
| D(10,1) | $10^2 + 1^2 = 101$ |
| E(11,0) | $11^2 + 0^2 = 121$ |

**Part 2**: Total = $1 + 100 + 101 + 121 = 323$

| Point | Probability |
|-------|-----------|
| B | 1/323 = 0.3% |
| C | 100/323 = 31.0% |
| D | 101/323 = 31.3% |
| E | 121/323 = 37.5% |

**Part 3**: **Point E** is most likely (37.5%), followed closely by D and C. This makes intuitive sense because k-means++ preferentially selects points that are **far from existing centroids**. The right cluster {C, D, E} is far from A, so one of them should be the second centroid — and k-means++ achieves this with high probability (99.7% chance of picking from the right cluster).

**Part 4**: With naive random initialization, there's a $\frac{2}{5} \times \frac{1}{4} = \frac{2}{20} = 10\%$ chance both centroids come from {A, B}, which would result in a terrible clustering (one centroid for 2 points, another for 3 points that are far away). K-means++ essentially eliminates this failure mode.

</details>

---

## Exercise 3: Elbow Method and Silhouette Score

You run k-means for $k = 1, \ldots, 6$ and obtain:

| $k$ | Inertia | Avg Silhouette |
|-----|---------|---------------|
| 1 | 500 | N/A |
| 2 | 200 | 0.72 |
| 3 | 100 | 0.68 |
| 4 | 60 | 0.55 |
| 5 | 50 | 0.42 |
| 6 | 45 | 0.35 |

1. Using the elbow method, what $k$ would you choose? Explain your reasoning.
2. Using the silhouette method, what $k$ would you choose?
3. The two methods disagree. Which would you trust more and why?
4. Why does the silhouette score decrease for larger $k$ in this example?

<details>
<summary>Solution</summary>

**Part 1**: The "elbow" is at **$k = 3$**. Inertia drops sharply from $k=1$ to $k=3$ (500 → 200 → 100), then the rate of decrease slows dramatically (100 → 60 → 50 → 45). The biggest "bend" in the curve is at $k=3$.

Decrease rates: $k=2$ drops by 300, $k=3$ drops by 100, $k=4$ drops by 40, $k=5$ drops by 10, $k=6$ drops by 5.

**Part 2**: The maximum silhouette score is at **$k = 2$** (0.72). The silhouette method recommends $k = 2$.

**Part 3**: It depends on the data and goal, but the **elbow method suggests $k=3$** might be more appropriate. Here's why:

- The silhouette score favors $k=2$ because two large clusters have well-separated points. But this might be too coarse.
- The elbow clearly shows $k=3$ captures significant additional structure (100-point inertia reduction).
- In practice, domain knowledge should guide the decision. If you know there are 3 natural groups, go with $k=3$ despite lower silhouette.

Both methods are heuristics. When they disagree, examine the actual clusters visually if possible.

**Part 4**: The silhouette score decreases because:
- With more clusters, points near cluster boundaries have smaller $b_i$ (nearest other cluster is closer) while $a_i$ remains similar.
- When true data has 2-3 natural clusters, forcing more clusters splits natural groups, putting some points close to the boundary of the split, reducing their silhouette score.
- Silhouette inherently penalizes over-segmentation.

</details>

---

## Exercise 4: K-Means Convergence and Local Minima

Consider 4 points: $(0,0)$, $(1,0)$, $(4,0)$, $(5,0)$.

We run k-means with $k = 2$.

1. **Init A**: $\mu_1 = (0,0)$, $\mu_2 = (5,0)$. Trace to convergence. What is the final inertia?
2. **Init B**: $\mu_1 = (0,0)$, $\mu_2 = (1,0)$. Trace to convergence. What is the final inertia?
3. Are the two results the same? What does this tell you about k-means?
4. What is the globally optimal clustering? Which initialization found it?

<details>
<summary>Solution</summary>

**Part 1**: Init A: $\mu_1 = (0,0)$, $\mu_2 = (5,0)$

Assign: $(0,0) \to \mu_1$, $(1,0) \to \mu_1$, $(4,0) \to \mu_2$, $(5,0) \to \mu_2$

Update: $\mu_1 = (0.5, 0)$, $\mu_2 = (4.5, 0)$

Reassign: $(0,0) \to \mu_1$ (dist $0.25$ vs $20.25$), $(1,0) \to \mu_1$ (dist $0.25$ vs $12.25$), $(4,0) \to \mu_2$ (dist $12.25$ vs $0.25$), $(5,0) \to \mu_2$ (dist $20.25$ vs $0.25$). No changes.

Converged. Clusters: $\{(0,0), (1,0)\}$ and $\{(4,0), (5,0)\}$.

Inertia: $(0.25 + 0.25) + (0.25 + 0.25) = 1.0$

**Part 2**: Init B: $\mu_1 = (0,0)$, $\mu_2 = (1,0)$

Assign: $(0,0) \to \mu_1$, $(1,0) \to \mu_2$, $(4,0) \to \mu_2$ (dist $16$ vs $9$), $(5,0) \to \mu_2$ (dist $25$ vs $16$)

Update: $\mu_1 = (0, 0)$, $\mu_2 = \frac{1}{3}((1,0) + (4,0) + (5,0)) = (10/3, 0) \approx (3.33, 0)$

Reassign: $(0,0) \to \mu_1$ (dist $0$ vs $11.11$), $(1,0) \to \mu_1$ (dist $1$ vs $5.44$), $(4,0) \to \mu_2$ (dist $16$ vs $0.44$), $(5,0) \to \mu_2$ (dist $25$ vs $2.78$). $(1,0)$ changed from $\mu_2$ to $\mu_1$!

Update: $\mu_1 = (0.5, 0)$, $\mu_2 = (4.5, 0)$

This is the same as Init A's result. Converged. Inertia = 1.0.

**Part 3**: In this case, both initializations converged to the same result. But this is fortunate — with more complex data, different initializations often lead to different local minima with different inertia values.

**Part 4**: The globally optimal $k=2$ clustering is $\{(0,0), (1,0)\}$ and $\{(4,0), (5,0)\}$ with inertia = 1.0. Both initializations found the global optimum here because the data has a clear gap between $x=1$ and $x=4$.

If the data were $\{0, 2, 3, 5\}$ instead, different initializations could lead to $\{0, 2\}$ / $\{3, 5\}$ (inertia = 4) or $\{0\}$ / $\{2, 3, 5\}$ (inertia = 4.67), illustrating how initialization affects the result.

</details>

---

## Exercise 5: K-Means Limitations

For each dataset configuration, explain whether k-means would succeed or fail, and suggest an alternative if it fails:

```
1. Two concentric circles:    2. Two elongated ellipses:
      ○ ○ ○ ○                    ● ● ● ● ● ● ●
    ○  ● ● ●  ○                      ○ ○ ○ ○ ○ ○ ○
    ○  ● ● ●  ○
      ○ ○ ○ ○

3. Three spherical clusters     4. Two clusters of very
   of different sizes:             different densities:
   ● (100 pts)   ●● (500 pts)     ···●··· (sparse)
                  ●●                ○○○○○○ (dense)
        ○○ (200 pts)
```

<details>
<summary>Solution</summary>

**1. Concentric circles**: **FAIL**. K-means uses Euclidean distance to centroids, which creates convex (spherical) boundaries. It cannot separate an inner circle from an outer ring — the centroids would both end up near the center. **Alternative**: Spectral clustering (uses graph connectivity), DBSCAN (density-based), or kernel k-means.

**2. Elongated ellipses**: **PARTIALLY SUCCEED, depending on orientation**. If the ellipses are well-separated, k-means can assign points correctly even though the cluster shapes aren't spherical. However, if they are close together or overlapping, k-means will create a boundary perpendicular to the line between centroids, which may cut through the ellipses incorrectly. **Alternative**: Gaussian Mixture Models (GMM) with full covariance matrices can model elliptical clusters naturally.

**3. Different-sized clusters**: **SUCCEED (mostly)**. K-means can handle clusters of different sizes (number of points) as long as they are well-separated and roughly spherical. The centroids will be at the means, and the Voronoi boundaries will correctly partition the space. However, k-means may "steal" points from the small cluster if it's between two large clusters. **No alternative needed** unless clusters overlap.

**4. Different densities**: **FAIL (or POOR)**. K-means tends to split the sparse cluster and merge it partially with the dense cluster to equalize the within-cluster distances. The centroid of the sparse cluster will be pulled toward the center, and the Voronoi boundary won't respect the density difference. **Alternative**: DBSCAN (which explicitly uses density to define clusters), or GMM (which can model different variances).

</details>

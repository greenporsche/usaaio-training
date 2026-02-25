# Unit 05 — ML2 Unsupervised Learning: Cheat Sheet

> Dense reference for USAAIO Round 1 & 2. Keep this open during practice.

---

## Decision Trees

| Impurity | Formula | Range |
|----------|---------|-------|
| Gini | $G = 1 - \sum_{k=1}^{K} p_k^2$ | $[0, 1-1/K]$ |
| Entropy | $H = -\sum_{k=1}^{K} p_k \log_2 p_k$ | $[0, \log_2 K]$ |
| Misclassification | $E = 1 - \max_k p_k$ | $[0, 1-1/K]$ |

**Information Gain**: $IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$

**Splitting**: Greedily pick feature + threshold maximizing IG (or Gini reduction). Recurse on children.

**Pruning**: Pre-pruning (max depth, min samples) or post-pruning (cost-complexity: $R_\alpha(T) = R(T) + \alpha |T|$).

---

## Random Forests (Bagging)

- **Bootstrap**: Sample $n$ points with replacement $\Rightarrow$ ~63.2% unique per tree.
- **Feature subsampling**: At each split, consider $m$ random features.
  - Classification: $m \approx \sqrt{p}$ | Regression: $m \approx p/3$
- **Prediction**: Majority vote (classification) or mean (regression).
- **OOB Error**: Use ~36.8% out-of-bag samples as free validation.
- **Feature importance**: Mean decrease in impurity or permutation importance.
- **Variance reduction**: $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ where $\rho$ = pairwise correlation, $B$ = number of trees.

---

## Boosting

### AdaBoost
1. Init weights $w_i = 1/n$.
2. Fit weak learner $h_t$, compute error $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$.
3. Learner weight: $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$.
4. Update: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$, then normalize.
5. Final: $H(x) = \text{sign}\left(\sum_t \alpha_t h_t(x)\right)$.

### Gradient Boosting
1. Init $F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$.
2. Compute pseudo-residuals: $r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$.
3. Fit tree $h_m$ to residuals, update $F_m = F_{m-1} + \eta \cdot h_m$.

**Key hyperparameters**: Learning rate $\eta$ (smaller = more trees needed), max depth, number of estimators.

---

## Principal Component Analysis (PCA)

**Objective**: $\max_{w} w^T \Sigma w \quad \text{s.t.} \quad \|w\| = 1$

**Solution**: Eigenvectors of covariance matrix $\Sigma = \frac{1}{n-1}X_c^T X_c$ (centered data).

**Steps**:
1. Center data: $X_c = X - \bar{X}$
2. Compute covariance: $\Sigma = \frac{1}{n-1}X_c^T X_c$
3. Eigendecompose: $\Sigma v_k = \lambda_k v_k$
4. Sort by $\lambda_1 \geq \lambda_2 \geq \cdots$
5. Project: $Z = X_c V_k$ where $V_k$ = top-$k$ eigenvectors

**Variance explained**: $\frac{\lambda_k}{\sum_j \lambda_j}$ | **Cumulative**: choose $k$ s.t. $\frac{\sum_{j=1}^{k}\lambda_j}{\sum_j \lambda_j} \geq 0.95$

**SVD connection**: $X_c = U\Sigma V^T \Rightarrow$ columns of $V$ are principal components.

---

## Dimensionality Reduction (t-SNE / UMAP)

### t-SNE
- **High-dim**: $p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$, symmetrize $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$
- **Low-dim**: $q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$ (Student-t with 1 df)
- **Objective**: Minimize $KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$
- **Perplexity** $\approx$ effective number of neighbors (typically 5–50).
- **Non-parametric**: No inverse mapping. Distances between clusters are meaningless.

### UMAP
- Topological approach: builds fuzzy simplicial sets. Faster, preserves more global structure.
- Key params: `n_neighbors` (local vs global), `min_dist` (cluster tightness).

---

## K-Means Clustering

**Lloyd's Algorithm**:
1. Initialize $k$ centroids $\mu_1, \ldots, \mu_k$.
2. **Assign**: $c_i = \arg\min_j \|x_i - \mu_j\|^2$
3. **Update**: $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$
4. Repeat until convergence.

**Objective**: $J = \sum_{j=1}^{k}\sum_{i \in C_j} \|x_i - \mu_j\|^2$ (within-cluster sum of squares / inertia).

**Convergence**: $J$ decreases monotonically; finite partitions $\Rightarrow$ terminates. Converges to local minimum.

**K-means++**: Init first centroid randomly. Pick next centroid with probability $\propto D(x)^2$ (distance to nearest existing centroid). Gives $O(\log k)$-competitive guarantee.

**Choosing $k$**: Elbow method (plot $J$ vs $k$) | Silhouette score: $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ where $a_i$ = mean intra-cluster dist, $b_i$ = mean nearest-cluster dist.

---

## Support Vector Machines (SVM)

**Hard margin** (linearly separable): $\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1$

**Margin width**: $\frac{2}{\|w\|}$

**Soft margin** (with slack $\xi_i$): $\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$

**Hinge loss**: $L = \max(0, 1 - y_i f(x_i))$. SVM = hinge loss + L2 regularization.

**Dual form**: $\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j x_i^T x_j$ s.t. $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$.

**Support vectors**: Points with $\alpha_i > 0$ (on or inside margin).

**Kernel trick**: Replace $x_i^T x_j$ with $K(x_i, x_j)$:
| Kernel | $K(x, z)$ |
|--------|-----------|
| Linear | $x^T z$ |
| Polynomial | $(x^T z + c)^d$ |
| RBF/Gaussian | $\exp(-\gamma\|x-z\|^2)$ |

**Key insight**: Decision boundary depends only on support vectors, not all data.

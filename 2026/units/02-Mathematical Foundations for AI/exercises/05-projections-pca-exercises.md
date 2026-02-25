# Exercises: Projections and PCA

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 5.1 — Vector Projection

**(a)** Compute the projection of $\mathbf{b} = \begin{bmatrix}3\\4\end{bmatrix}$ onto $\mathbf{a} = \begin{bmatrix}1\\0\end{bmatrix}$.

**(b)** Compute the projection of $\mathbf{b} = \begin{bmatrix}3\\4\end{bmatrix}$ onto $\mathbf{a} = \begin{bmatrix}1\\1\end{bmatrix}$.

**(c)** Verify that the error vector $\mathbf{b} - \text{proj}_\mathbf{a}(\mathbf{b})$ is orthogonal to $\mathbf{a}$ in both cases.

<details><summary>Solution</summary>

**(a)** $\text{proj}_\mathbf{a}(\mathbf{b}) = \frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\mathbf{a} = \frac{3}{1}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix}$

**(b)** $\text{proj}_\mathbf{a}(\mathbf{b}) = \frac{3+4}{1+1}\begin{bmatrix}1\\1\end{bmatrix} = \frac{7}{2}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}3.5\\3.5\end{bmatrix}$

**(c)** Case (a): $\mathbf{e} = \begin{bmatrix}3\\4\end{bmatrix} - \begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}0\\4\end{bmatrix}$, $\mathbf{e}^\top\mathbf{a} = 0$ $\checkmark$

Case (b): $\mathbf{e} = \begin{bmatrix}3\\4\end{bmatrix} - \begin{bmatrix}3.5\\3.5\end{bmatrix} = \begin{bmatrix}-0.5\\0.5\end{bmatrix}$, $\mathbf{e}^\top\mathbf{a} = -0.5 + 0.5 = 0$ $\checkmark$

</details>

---

## Exercise 5.2 — Projection Matrix Properties

Let $P = \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix}$.

**(a)** Verify $P^2 = P$ (idempotent).

**(b)** Verify $P^\top = P$ (symmetric).

**(c)** What subspace does $P$ project onto?

**(d)** Write down the projection matrix $P^\perp$ onto the orthogonal complement.

<details><summary>Solution</summary>

**(a)** $P^2 = \frac{1}{4}\begin{bmatrix}1&1\\1&1\end{bmatrix}\begin{bmatrix}1&1\\1&1\end{bmatrix} = \frac{1}{4}\begin{bmatrix}2&2\\2&2\end{bmatrix} = \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix} = P$ $\checkmark$

**(b)** $P^\top = \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix}^\top = \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix} = P$ $\checkmark$

**(c)** $P$ projects onto $\text{span}\left(\begin{bmatrix}1\\1\end{bmatrix}\right)$ — the line $y = x$.

To verify: $P\begin{bmatrix}1\\1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}2\\2\end{bmatrix} = \begin{bmatrix}1\\1\end{bmatrix}$ (already on the line, unchanged).

**(d)** $P^\perp = I - P = \begin{bmatrix}1&0\\0&1\end{bmatrix} - \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$

This projects onto $\text{span}\left(\begin{bmatrix}1\\-1\end{bmatrix}\right)$ — the line $y = -x$.

</details>

---

## Exercise 5.3 — PCA by Hand

Given 4 data points in 2D:

$$\mathbf{x}_1 = \begin{bmatrix}2\\1\end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix}4\\3\end{bmatrix}, \quad \mathbf{x}_3 = \begin{bmatrix}6\\5\end{bmatrix}, \quad \mathbf{x}_4 = \begin{bmatrix}8\\7\end{bmatrix}$$

**(a)** Compute the mean and center the data.

**(b)** Compute the $2 \times 2$ covariance matrix.

**(c)** Find the first principal component direction and the variance explained.

<details><summary>Solution</summary>

**(a)** Mean: $\boldsymbol{\mu} = \frac{1}{4}\begin{bmatrix}20\\16\end{bmatrix} = \begin{bmatrix}5\\4\end{bmatrix}$

Centered data: $\bar{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$:

$\bar{\mathbf{x}}_1 = \begin{bmatrix}-3\\-3\end{bmatrix}, \bar{\mathbf{x}}_2 = \begin{bmatrix}-1\\-1\end{bmatrix}, \bar{\mathbf{x}}_3 = \begin{bmatrix}1\\1\end{bmatrix}, \bar{\mathbf{x}}_4 = \begin{bmatrix}3\\3\end{bmatrix}$

**(b)** $\bar{X} = \begin{bmatrix}-3&-3\\-1&-1\\1&1\\3&3\end{bmatrix}$

$C = \frac{1}{3}\bar{X}^\top\bar{X} = \frac{1}{3}\begin{bmatrix}20&20\\20&20\end{bmatrix} = \begin{bmatrix}20/3&20/3\\20/3&20/3\end{bmatrix}$

**(c)** Eigenvalues: $(20/3 - \lambda)^2 - (20/3)^2 = 0 \implies \lambda(lambda - 40/3) = 0$

$\lambda_1 = 40/3 \approx 13.33$, $\lambda_2 = 0$.

PC1 direction (for $\lambda_1$): $(C - \frac{40}{3}I)\mathbf{w} = 0 \implies \begin{bmatrix}-20/3&20/3\\20/3&-20/3\end{bmatrix}\mathbf{w} = 0 \implies \mathbf{w}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$

Variance explained: $\frac{40/3}{40/3 + 0} = \mathbf{100\%}$

The data lies perfectly on the line $y = x - 1$, so one component captures everything.

</details>

---

## Exercise 5.4 — Maximum Variance vs Minimum Error

Prove that the maximum variance formulation and minimum reconstruction error formulation of PCA are equivalent for finding the first principal component.

<details><summary>Solution</summary>

**Maximum variance**: maximize $\mathbf{w}^\top C \mathbf{w}$ s.t. $\|\mathbf{w}\| = 1$.

**Minimum error**: minimize $\sum_i \|\bar{\mathbf{x}}_i - (\mathbf{w}^\top\bar{\mathbf{x}}_i)\mathbf{w}\|^2$ s.t. $\|\mathbf{w}\| = 1$.

Expand the reconstruction error:

$$\sum_i \|\bar{\mathbf{x}}_i - (\mathbf{w}^\top\bar{\mathbf{x}}_i)\mathbf{w}\|^2 = \sum_i \left(\|\bar{\mathbf{x}}_i\|^2 - (\mathbf{w}^\top\bar{\mathbf{x}}_i)^2\right)$$

(using the Pythagorean theorem: $\|\mathbf{x}\|^2 = \|\text{proj}\|^2 + \|\text{error}\|^2$)

$$= \sum_i \|\bar{\mathbf{x}}_i\|^2 - \sum_i (\mathbf{w}^\top\bar{\mathbf{x}}_i)^2$$

$$= \text{const} - \mathbf{w}^\top\left(\sum_i \bar{\mathbf{x}}_i\bar{\mathbf{x}}_i^\top\right)\mathbf{w} = \text{const} - (N-1)\mathbf{w}^\top C \mathbf{w}$$

Minimizing this is equivalent to maximizing $\mathbf{w}^\top C \mathbf{w}$.

Therefore both formulations yield the same solution: $\mathbf{w} = $ eigenvector of $C$ with largest eigenvalue. $\blacksquare$

</details>

---

## Exercise 5.5 — PCA via SVD

Show that PCA of data matrix $\bar{X} \in \mathbb{R}^{N \times d}$ (centered) can be computed via SVD $\bar{X} = U\Sigma V^\top$ without ever forming the covariance matrix $C$.

Specifically, identify the principal directions and scores in terms of $U$, $\Sigma$, $V$.

<details><summary>Solution</summary>

The covariance matrix is:

$$C = \frac{\bar{X}^\top\bar{X}}{N-1} = \frac{(U\Sigma V^\top)^\top(U\Sigma V^\top)}{N-1} = \frac{V\Sigma^\top U^\top U\Sigma V^\top}{N-1} = \frac{V\Sigma^2 V^\top}{N-1}$$

(using $U^\top U = I$)

This is the eigendecomposition of $C$:
- **Eigenvectors** of $C$ = columns of $V$ = **right singular vectors** of $\bar{X}$
- **Eigenvalues** of $C$ = $\sigma_i^2 / (N-1)$

**Principal directions**: top-$k$ columns of $V$ (or first $k$ rows of $V^\top$)

**Principal component scores**: $\bar{X}V_k = U\Sigma V^\top V_k = U_k \Sigma_k$

So $Z = U_k \Sigma_k$ gives the scores directly from SVD without forming $C$.

**Why this matters**: For $N \ll d$ (more features than samples), the $d \times d$ covariance matrix is huge. SVD of $\bar{X}$ ($N \times d$) is much more efficient.

</details>

---

## Exercise 5.6 — USAAIO Competition Style

A dataset has covariance matrix eigenvalues $\lambda_1 = 10, \lambda_2 = 5, \lambda_3 = 3, \lambda_4 = 1, \lambda_5 = 0.5$.

**(a)** How many principal components are needed to capture at least 90% of the variance?

**(b)** What is the reconstruction error (in terms of variance lost) when using $k = 2$ components?

**(c)** If the original data is 5D and you project to 2D, what is the shape of the projection matrix $W$? What is the shape of the projected data (for $N$ samples)?

<details><summary>Solution</summary>

Total variance = $10 + 5 + 3 + 1 + 0.5 = 19.5$

**(a)** Cumulative variance:
- $k=1$: $10/19.5 = 51.3\%$
- $k=2$: $15/19.5 = 76.9\%$
- $k=3$: $18/19.5 = 92.3\% > 90\%$

**$k = 3$ components** needed for $\geq 90\%$.

**(b)** Variance captured with $k=2$: $\lambda_1 + \lambda_2 = 15$

Variance lost: $\lambda_3 + \lambda_4 + \lambda_5 = 3 + 1 + 0.5 = \mathbf{4.5}$

This equals $\sum_{i=k+1}^{d} \lambda_i$, which by the equivalence with SVD, also equals $\|X - X_k\|_F^2 / (N-1)$.

**(c)** Projection matrix $W \in \mathbb{R}^{5 \times 2}$ (columns are the top 2 eigenvectors of $C$).

Projected data: $Z = \bar{X}W \in \mathbb{R}^{N \times 2}$.

Shape: $(N, 5) \times (5, 2) = (N, 2)$.

</details>

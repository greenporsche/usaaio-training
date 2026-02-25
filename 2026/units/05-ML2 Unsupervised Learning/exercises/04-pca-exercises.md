# PCA — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: Compute PCA by Hand (2D)

Given 4 centered data points in 2D:

$$X_c = \begin{bmatrix} -2 & -1 \\ -1 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix}$$

1. Compute the covariance matrix $\Sigma = \frac{1}{n-1}X_c^T X_c$.
2. Find the eigenvalues and eigenvectors of $\Sigma$.
3. What proportion of variance does the first principal component explain?
4. Project the data onto the first principal component.

<details>
<summary>Solution</summary>

**Part 1**: $n = 4$.

$X_c^T X_c = \begin{bmatrix} -2 & -1 & 1 & 2 \\ -1 & 0 & 0 & 1 \end{bmatrix}\begin{bmatrix} -2 & -1 \\ -1 & 0 \\ 1 & 0 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 10 & 4 \\ 4 & 2 \end{bmatrix}$

$\Sigma = \frac{1}{3}\begin{bmatrix} 10 & 4 \\ 4 & 2 \end{bmatrix} = \begin{bmatrix} 10/3 & 4/3 \\ 4/3 & 2/3 \end{bmatrix}$

**Part 2**: Eigenvalue equation: $\det(\Sigma - \lambda I) = 0$

$(10/3 - \lambda)(2/3 - \lambda) - (4/3)^2 = 0$

$\lambda^2 - 4\lambda + 20/9 - 16/9 = 0$

$\lambda^2 - 4\lambda + 4/9 = 0$

$\lambda = \frac{4 \pm \sqrt{16 - 16/9}}{2} = \frac{4 \pm \sqrt{128/9}}{2} = \frac{4 \pm \frac{8\sqrt{2}}{3}}{2}$

$\lambda_1 = 2 + \frac{4\sqrt{2}}{3} \approx 2 + 1.886 = 3.886$

$\lambda_2 = 2 - \frac{4\sqrt{2}}{3} \approx 2 - 1.886 = 0.114$

For $\lambda_1 \approx 3.886$: $(\Sigma - \lambda_1 I)v = 0$

$(10/3 - 3.886)v_1 + (4/3)v_2 = 0 \Rightarrow -0.553v_1 + 1.333v_2 = 0 \Rightarrow v_2 = 0.414 v_1$

Normalized: $v_1 \approx [0.924, 0.383]^T$

**Part 3**: Total variance = $\lambda_1 + \lambda_2 = 4$.

PC1 explains $\lambda_1 / 4 = 3.886/4 \approx 97.1\%$ of variance.

**Part 4**: Project: $z = X_c v_1$

$z = \begin{bmatrix} -2(0.924) + (-1)(0.383) \\ -1(0.924) + 0(0.383) \\ 1(0.924) + 0(0.383) \\ 2(0.924) + 1(0.383) \end{bmatrix} = \begin{bmatrix} -2.231 \\ -0.924 \\ 0.924 \\ 2.231 \end{bmatrix}$

</details>

---

## Exercise 2: Variance Explained and Choosing Components

A PCA on a 10-dimensional dataset produces the following eigenvalues:

| Component | Eigenvalue |
|-----------|-----------|
| 1 | 15.2 |
| 2 | 8.7 |
| 3 | 4.1 |
| 4 | 2.3 |
| 5 | 1.5 |
| 6 | 0.8 |
| 7 | 0.5 |
| 8 | 0.3 |
| 9 | 0.1 |
| 10 | 0.05 |

1. Compute the total variance.
2. Compute the proportion of variance explained by each of the first 5 components.
3. Compute the cumulative variance explained.
4. How many components are needed to retain 95% of the variance?
5. How many for 99%?

<details>
<summary>Solution</summary>

**Part 1**: Total = $15.2 + 8.7 + 4.1 + 2.3 + 1.5 + 0.8 + 0.5 + 0.3 + 0.1 + 0.05 = 33.55$

**Part 2**: Proportions:

| Component | Proportion |
|-----------|-----------|
| 1 | 15.2/33.55 = 45.3% |
| 2 | 8.7/33.55 = 25.9% |
| 3 | 4.1/33.55 = 12.2% |
| 4 | 2.3/33.55 = 6.9% |
| 5 | 1.5/33.55 = 4.5% |

**Part 3**: Cumulative:

| Components | Cumulative |
|------------|-----------|
| 1 | 45.3% |
| 1-2 | 71.2% |
| 1-3 | 83.5% |
| 1-4 | 90.3% |
| 1-5 | 94.8% |
| 1-6 | 97.2% |
| 1-7 | 98.7% |

**Part 4**: For 95%, we need **6 components** (cumulative = 97.2% > 95%). With 5 components we only reach 94.8%.

**Part 5**: For 99%, we need **8 components** (cumulative at 8 = 98.7 + 0.3/33.55 = 98.7 + 0.9 = 99.6% > 99%). Let me recalculate:

Cumulative at 7: $(15.2+8.7+4.1+2.3+1.5+0.8+0.5)/33.55 = 33.1/33.55 = 98.7\%$

Cumulative at 8: $33.1 + 0.3 = 33.4 \Rightarrow 33.4/33.55 = 99.6\%$

So **8 components** for 99%.

</details>

---

## Exercise 3: PCA and Standardization

Consider a dataset with two features: height (in cm, range 150-200) and weight (in kg, range 50-100).

The covariance matrix (unstandardized) is:

$$\Sigma = \begin{bmatrix} 225 & 30 \\ 30 & 100 \end{bmatrix}$$

1. Without standardizing, which feature will PC1 be most aligned with? Why?
2. Compute the total variance. What fraction comes from height?
3. If you standardize both features (divide by their standard deviations), the correlation matrix becomes:

$$R = \begin{bmatrix} 1.0 & 0.2 \\ 0.2 & 1.0 \end{bmatrix}$$

   What are the eigenvalues and eigenvectors of $R$?

4. After standardization, what fraction of variance does PC1 explain?
5. When should you use the covariance matrix vs. the correlation matrix for PCA?

<details>
<summary>Solution</summary>

**Part 1**: PC1 will be mostly aligned with **height** because its variance (225) is much larger than weight's (100). PCA maximizes variance, so it will point roughly along the height direction. This is misleading — height only has more variance because centimeters give bigger numbers than kilograms!

**Part 2**: Total variance = 225 + 100 = 325. Height fraction = 225/325 = 69.2%.

**Part 3**: For the correlation matrix $R$:

$\det(R - \lambda I) = (1-\lambda)^2 - 0.04 = 0$

$(1-\lambda)^2 = 0.04$

$1 - \lambda = \pm 0.2$

$\lambda_1 = 1.2, \quad \lambda_2 = 0.8$

For $\lambda_1 = 1.2$: $(1-1.2)v_1 + 0.2v_2 = 0 \Rightarrow v_2 = v_1$

$v_1 = \frac{1}{\sqrt{2}}[1, 1]^T$ (equal contribution from both features!)

For $\lambda_2 = 0.8$: $v_2 = \frac{1}{\sqrt{2}}[1, -1]^T$

**Part 4**: PC1 explains $1.2/2.0 = 60\%$ of variance after standardization. This is a much more balanced picture than the 69.2% dominated by height units.

**Part 5**:
- **Correlation matrix** (standardized PCA): When features have different units or very different scales. This is the default recommendation.
- **Covariance matrix**: When features are in comparable units and you want variance differences to be meaningful (e.g., all features are pixel intensities in an image).

</details>

---

## Exercise 4: PCA Reconstruction Error

You perform PCA on a dataset with $N = 100$ samples and $D = 5$ features. The eigenvalues are $\lambda = [10, 5, 2, 0.5, 0.1]$.

1. If you keep $k = 2$ components, what is the reconstruction error (sum of discarded eigenvalues)?
2. What fraction of total variance is lost?
3. Write the reconstruction formula for the centered data.
4. If $x_i = [3, 1, -2, 0, 4]$ and the mean is $\bar{x} = [1, 0, -1, 1, 2]$, and the first two principal components are $v_1 = [0.5, 0.5, -0.5, 0, 0.5]$ and $v_2 = [0.5, -0.5, 0.5, 0.5, 0]$ (approximately), compute the projection and reconstruction.

<details>
<summary>Solution</summary>

**Part 1**: Discarded eigenvalues: $\lambda_3 + \lambda_4 + \lambda_5 = 2 + 0.5 + 0.1 = 2.6$

Reconstruction error = 2.6 (this is $\sum_{j=k+1}^{D} \lambda_j$, equal to $\|X_c - \hat{X}_c\|_F^2 / (n-1)$... actually, total reconstruction error summed over all samples is $(n-1) \times 2.6 = 99 \times 2.6 = 257.4$).

**Part 2**: Total variance = $10 + 5 + 2 + 0.5 + 0.1 = 17.6$

Retained variance = $(10 + 5)/17.6 = 85.2\%$

Lost variance = $2.6/17.6 = 14.8\%$

**Part 3**: For centered data $x_c = x - \bar{x}$:

$\hat{x}_c = (x_c^T v_1)v_1 + (x_c^T v_2)v_2 = V_k V_k^T x_c$

$\hat{x} = \hat{x}_c + \bar{x}$

**Part 4**:

Centered: $x_c = [3-1, 1-0, -2-(-1), 0-1, 4-2] = [2, 1, -1, -1, 2]$

Projection:
$z_1 = x_c^T v_1 = 2(0.5) + 1(0.5) + (-1)(-0.5) + (-1)(0) + 2(0.5) = 1 + 0.5 + 0.5 + 0 + 1 = 3.0$

$z_2 = x_c^T v_2 = 2(0.5) + 1(-0.5) + (-1)(0.5) + (-1)(0.5) + 2(0) = 1 - 0.5 - 0.5 - 0.5 + 0 = -0.5$

Reconstruction (centered):
$\hat{x}_c = 3.0 \times [0.5, 0.5, -0.5, 0, 0.5] + (-0.5) \times [0.5, -0.5, 0.5, 0.5, 0]$
$= [1.5, 1.5, -1.5, 0, 1.5] + [-0.25, 0.25, -0.25, -0.25, 0]$
$= [1.25, 1.75, -1.75, -0.25, 1.5]$

Reconstruction (original): $\hat{x} = \hat{x}_c + \bar{x} = [2.25, 1.75, -2.75, 0.75, 3.5]$

Compare to original $x = [3, 1, -2, 0, 4]$. Reconstruction error for this point: $\|x - \hat{x}\|^2 = 0.75^2 + 0.75^2 + 0.75^2 + 0.75^2 + 0.5^2 = 0.5625 \times 4 + 0.25 = 2.5$

</details>

---

## Exercise 5: PCA vs. Feature Selection

A dataset has 4 features. After PCA, the first two principal components are:

$$PC_1 = 0.5 x_1 + 0.5 x_2 + 0.5 x_3 + 0.5 x_4$$
$$PC_2 = 0.5 x_1 - 0.5 x_2 + 0.5 x_3 - 0.5 x_4$$

These two components capture 90% of the variance.

1. Is it correct to say "features $x_1$ and $x_3$ are most important because they contribute positively to both PCs"? Why or why not?
2. What is the key difference between PCA (dimensionality reduction) and feature selection (e.g., choosing the top-2 features by variance)?
3. Can PCA components be interpreted as physical/meaningful quantities? Give an example where they can and where they cannot.
4. If you wanted to reduce to 2 features while keeping interpretability, would you use PCA or feature selection?

<details>
<summary>Solution</summary>

**Part 1**: **No, this is incorrect.** All four features contribute equally in magnitude ($|0.5|$) to both PCs. The sign only indicates direction, not importance. All features have equal weight $0.5^2 = 0.25$ in each PC. The loadings tell you the *direction* of the component, not feature importance.

**Part 2**:
- **PCA**: Creates new features that are *linear combinations* of all original features. Maximizes variance in the new coordinate system. Every PC uses all original features.
- **Feature selection**: Picks a subset of original features. The selected features retain their original meaning, but you might miss important combinations.

Key difference: PCA uses information from ALL features in every component, while feature selection discards some features entirely.

**Part 3**:
- **Can be interpreted**: In genetics, PC1 often corresponds to geographic ancestry (north-south gradient in European population genetics). In climate science, PCs often correspond to known climate patterns (e.g., El Nino).
- **Cannot be interpreted**: In a dataset of random survey questions, PCs might be arbitrary combinations with no meaningful interpretation. "$0.3 \times \text{height} + 0.7 \times \text{income} - 0.2 \times \text{age}$" may capture variance but doesn't correspond to any real concept.

**Part 4**: **Feature selection** if interpretability is paramount. PCA components are linear combinations that may not have clear meaning. If you need to explain your model to stakeholders or if domain knowledge suggests specific features matter, feature selection is preferable. However, if maximum information compression matters more than interpretability, PCA is better.

</details>

# Kernel Methods — Exercises

> 5 exercises covering kernel computation, kernel properties, kernel ridge regression, and feature maps

---

## Exercise 1: Compute the Kernel Matrix (Compute This)

Given three data points in $\mathbb{R}^2$:

$$x_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad x_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad x_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Tasks**:
1. Compute the kernel matrix $K$ using the **polynomial kernel** $K(x, x') = (x^Tx' + 1)^2$.
2. Verify that $K$ is symmetric.
3. Compute the eigenvalues of $K$ and verify that $K$ is PSD.

<details>
<summary>Solution</summary>

**1. Compute inner products first**:

| | $x_1$ | $x_2$ | $x_3$ |
|---|---|---|---|
| $x_1$ | 1 | 0 | 1 |
| $x_2$ | 0 | 1 | 1 |
| $x_3$ | 1 | 1 | 2 |

Now apply $(x^Tx' + 1)^2$:

$$K = \begin{bmatrix} (1+1)^2 & (0+1)^2 & (1+1)^2 \\ (0+1)^2 & (1+1)^2 & (1+1)^2 \\ (1+1)^2 & (1+1)^2 & (2+1)^2 \end{bmatrix} = \begin{bmatrix} 4 & 1 & 4 \\ 1 & 4 & 4 \\ 4 & 4 & 9 \end{bmatrix}$$

**2.** $K_{ij} = K_{ji}$ for all $i, j$ — verified by inspection. ✓

**3.** Eigenvalues (computed numerically or via characteristic polynomial):

$\det(K - \lambda I) = 0$

The eigenvalues are approximately $\lambda_1 \approx 14.77$, $\lambda_2 \approx 2.23$, $\lambda_3 \approx 0.0$.

All eigenvalues $\geq 0$, so $K$ is PSD. ✓

(The near-zero eigenvalue indicates that the kernel matrix is nearly rank-2, reflecting the fact that the data lies in $\mathbb{R}^2$.)

</details>

---

## Exercise 2: Verify the Kernel Trick (Derive That)

For the polynomial kernel $K(x, x') = (x^Tx')^2$ with $x \in \mathbb{R}^2$:

1. Show that the feature map $\phi(x) = (x_1^2, \sqrt{2}\,x_1 x_2, x_2^2)^T$ satisfies $\phi(x)^T\phi(x') = K(x, x')$.
2. What is the dimension of the feature space for the kernel $(x^Tx')^d$ with $x \in \mathbb{R}^p$?
3. For the kernel $(x^Tx' + c)^d$, the feature space is even larger. Why?

<details>
<summary>Solution</summary>

**1.** Compute $\phi(x)^T\phi(x')$:

$$\phi(x)^T\phi(x') = x_1^2 {x_1'}^2 + 2x_1 x_2 x_1' x_2' + x_2^2 {x_2'}^2$$

Now compute $(x^Tx')^2$:

$$\left(x_1 x_1' + x_2 x_2'\right)^2 = x_1^2 {x_1'}^2 + 2x_1 x_2 x_1' x_2' + x_2^2 {x_2'}^2$$

These are equal. ✓

**Key**: The $\sqrt{2}$ factor on $x_1 x_2$ is essential. Without it, the cross term would be $x_1 x_2 x_1' x_2'$ instead of $2 x_1 x_2 x_1' x_2'$.

**2.** The feature space dimension for $(x^Tx')^d$ with $x \in \mathbb{R}^p$ is $\binom{p + d - 1}{d}$. This counts the number of monomials of degree exactly $d$ in $p$ variables (with repetition).

For $p = 2, d = 2$: $\binom{3}{2} = 3$ (which matches: $x_1^2, x_1 x_2, x_2^2$).

For $p = 100, d = 5$: $\binom{104}{5} = 96,560,646$ — enormous!

**3.** The kernel $(x^Tx' + c)^d$ with $c > 0$ includes monomials of all degrees from $0$ to $d$ (by the binomial theorem, $(x^Tx' + c)^d = \sum_{k=0}^{d} \binom{d}{k} c^{d-k} (x^Tx')^k$). Each $(x^Tx')^k$ corresponds to degree-$k$ monomials. So the feature space includes all monomials up to degree $d$, which is $\sum_{k=0}^{d} \binom{p+k-1}{k} = \binom{p+d}{d}$.

</details>

---

## Exercise 3: Identify the Error (Debug)

```python
def rbf_kernel(X1, X2, sigma=1.0):
    """Compute RBF kernel matrix."""
    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x^T x'
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2 - 2 * X1 @ X2.T          # BUG?
    return np.exp(-dist_sq / (2 * sigma))           # BUG?
```

Find **two bugs** in this implementation.

<details>
<summary>Solution</summary>

**Bug 1**: `sq2` has shape `(n2, 1)` but needs to be transposed to `(1, n2)` for proper broadcasting.

The line `dist_sq = sq1 + sq2 - 2 * X1 @ X2.T` should be:

```python
dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
```

`sq1` is `(n1, 1)` and `sq2.T` is `(1, n2)`, so broadcasting gives `(n1, n2)`. Without the transpose, `sq1 + sq2` would try to broadcast `(n1, 1) + (n2, 1)` which either fails (if `n1 != n2`) or gives the wrong result.

**Bug 2**: The denominator in the exponential should be `2 * sigma ** 2`, not `2 * sigma`.

The RBF kernel is $\exp(-\|x - x'\|^2 / (2\sigma^2))$. The current code computes $\exp(-\|x - x'\|^2 / (2\sigma))$, which has the wrong scaling.

```python
return np.exp(-dist_sq / (2 * sigma ** 2))
```

</details>

---

## Exercise 4: Kernel Ridge Regression by Hand (Compute This)

Given 3 training points with RBF kernel ($\sigma = 1$) and $\lambda = 1$:

$$x_1 = 0, \quad x_2 = 1, \quad x_3 = 2, \quad y = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

**Tasks**:
1. Compute the kernel matrix $K_{ij} = \exp(-\frac{(x_i - x_j)^2}{2})$.
2. Compute $\alpha = (K + \lambda I)^{-1}y$.
3. Predict $\hat{y}(x_* = 0.5)$ by computing $k_*^T\alpha$ where $k_{*i} = K(x_*, x_i)$.

<details>
<summary>Solution</summary>

**1. Kernel matrix**:

| $K_{ij}$ | $x_1=0$ | $x_2=1$ | $x_3=2$ |
|---|---|---|---|
| $x_1=0$ | $e^0 = 1$ | $e^{-0.5} \approx 0.607$ | $e^{-2} \approx 0.135$ |
| $x_2=1$ | $0.607$ | $1$ | $0.607$ |
| $x_3=2$ | $0.135$ | $0.607$ | $1$ |

$$K \approx \begin{bmatrix} 1 & 0.607 & 0.135 \\ 0.607 & 1 & 0.607 \\ 0.135 & 0.607 & 1 \end{bmatrix}$$

**2.** $K + I = \begin{bmatrix} 2 & 0.607 & 0.135 \\ 0.607 & 2 & 0.607 \\ 0.135 & 0.607 & 2 \end{bmatrix}$

Solving $(K + I)\alpha = y$ numerically:

$\alpha \approx \begin{bmatrix} -0.192 \\ 0.688 \\ -0.192 \end{bmatrix}$

(Note the symmetry: $\alpha_1 = \alpha_3$ because $y_1 = y_3 = 0$ and the kernel structure is symmetric about $x_2$.)

**3.** Compute $k_*$ for $x_* = 0.5$:

$$k_{*1} = e^{-0.25/2} = e^{-0.125} \approx 0.882$$
$$k_{*2} = e^{-0.25/2} = e^{-0.125} \approx 0.882$$
$$k_{*3} = e^{-2.25/2} = e^{-1.125} \approx 0.325$$

$$\hat{y}(0.5) = k_*^T\alpha = 0.882(-0.192) + 0.882(0.688) + 0.325(-0.192)$$

$$\approx -0.169 + 0.607 - 0.062 = 0.376$$

The prediction at $x = 0.5$ is approximately $0.376$, which is between the neighboring targets $y_1 = 0$ and $y_2 = 1$, as expected.

</details>

---

## Exercise 5: True/False with Justification

1. **Any symmetric matrix is a valid kernel matrix.**
2. **The RBF kernel with $\sigma \to 0$ makes every point its own cluster (overfitting).**
3. **If $K_1$ and $K_2$ are valid kernel matrices, then $K_1 + K_2$ is also valid.**
4. **Kernel ridge regression has $O(d^3)$ time complexity, where $d$ is the feature dimension.**
5. **The kernel trick allows us to work in infinite-dimensional spaces in finite time.**

<details>
<summary>Solution</summary>

1. **FALSE**. A valid kernel matrix must be symmetric **and** positive semi-definite (PSD). The matrix $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ is symmetric but has eigenvalues $3$ and $-1$, so it's not PSD and therefore not a valid kernel matrix.

2. **TRUE**. As $\sigma \to 0$, $K(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / (2\sigma^2)) \to 0$ for $i \neq j$ and $K(x_i, x_i) = 1$. The kernel matrix approaches the identity $I$, meaning each point is only similar to itself. The resulting model essentially memorizes each training point independently — extreme overfitting.

3. **TRUE**. If $K_1$ and $K_2$ are PSD, then for any vector $v$: $v^T(K_1 + K_2)v = v^TK_1v + v^TK_2v \geq 0 + 0 = 0$. So $K_1 + K_2$ is PSD and therefore a valid kernel matrix. This corresponds to using the feature map $\phi(x) = [\phi_1(x)^T, \phi_2(x)^T]^T$ (concatenation).

4. **FALSE**. Kernel ridge regression has $O(n^3)$ time complexity, where $n$ is the number of **data points**, not features. The main operation is inverting the $n \times n$ kernel matrix $(K + \lambda I)^{-1}$. The kernel trick eliminates dependence on feature dimension $d$ (which could be infinite), but introduces cubic dependence on $n$.

5. **TRUE**. The RBF kernel $K(x, x') = \exp(-\|x-x'\|^2 / (2\sigma^2))$ corresponds to an infinite-dimensional feature map (via Taylor expansion of the exponential). Yet we only ever compute $K(x_i, x_j)$ — a finite scalar for each pair — and work with the finite $n \times n$ kernel matrix. We never explicitly construct the infinite-dimensional features.

</details>

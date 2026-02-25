# Linear Regression — Exercises

> 5 exercises covering normal equation, gradient descent, geometry, and implementation

---

## Exercise 1: Compute the Prediction (Compute This)

Given the design matrix and weight vector:

$$X = \begin{bmatrix} 1 & 2 & 3 \\ 1 & 0 & -1 \\ 1 & 1 & 1 \end{bmatrix}, \quad w = \begin{bmatrix} 1 \\ -2 \\ 3 \end{bmatrix}$$

**Tasks**:
1. Compute $\hat{y} = Xw$.
2. If $y = [4, 5, 2]^T$, compute the MSE loss.
3. Compute the gradient $\nabla_w \mathcal{L} = \frac{2}{n}X^T(Xw - y)$.

<details>
<summary>Solution</summary>

**1. Prediction**:

$$Xw = \begin{bmatrix} 1(1) + 2(-2) + 3(3) \\ 1(1) + 0(-2) + (-1)(3) \\ 1(1) + 1(-2) + 1(3) \end{bmatrix} = \begin{bmatrix} 6 \\ -2 \\ 2 \end{bmatrix}$$

**2. MSE**:

$$\text{MSE} = \frac{1}{3}\left[(6-4)^2 + (-2-5)^2 + (2-2)^2\right] = \frac{1}{3}(4 + 49 + 0) = \frac{53}{3} \approx 17.67$$

**3. Gradient**:

Residual: $r = Xw - y = [2, -7, 0]^T$

$$\nabla_w \mathcal{L} = \frac{2}{3}X^Tr = \frac{2}{3}\begin{bmatrix} 1 & 1 & 1 \\ 2 & 0 & 1 \\ 3 & -1 & 1 \end{bmatrix}\begin{bmatrix} 2 \\ -7 \\ 0 \end{bmatrix} = \frac{2}{3}\begin{bmatrix} -5 \\ 4 \\ 13 \end{bmatrix} = \begin{bmatrix} -10/3 \\ 8/3 \\ 26/3 \end{bmatrix}$$

</details>

---

## Exercise 2: Derive the Normal Equation (Derive That)

Starting from the MSE loss $\mathcal{L}(w) = \frac{1}{n}(Xw - y)^T(Xw - y)$:

1. Expand the quadratic form.
2. Compute $\nabla_w \mathcal{L}$ using matrix calculus.
3. Set the gradient to zero and solve for $w$.
4. State the condition under which the solution is unique.

<details>
<summary>Solution</summary>

**1. Expand**:

$$\mathcal{L} = \frac{1}{n}\left(w^TX^TXw - 2y^TXw + y^Ty\right)$$

**2. Gradient** (using $\nabla_w(w^TAw) = 2Aw$ for symmetric $A$ and $\nabla_w(b^Tw) = b$):

$$\nabla_w \mathcal{L} = \frac{1}{n}\left(2X^TXw - 2X^Ty\right) = \frac{2}{n}X^T(Xw - y)$$

**3. Set to zero**:

$$X^T(Xw - y) = 0 \implies X^TXw = X^Ty \implies \hat{w} = (X^TX)^{-1}X^Ty$$

**4. Uniqueness**: The solution is unique if and only if $X^TX$ is invertible, which requires $X$ to have full column rank ($\text{rank}(X) = d$). This fails when $n < d$ (underdetermined) or when columns of $X$ are linearly dependent.

</details>

---

## Exercise 3: Identify the Error (Debug)

A student implements linear regression gradient descent:

```python
def linear_regression_gd(X, y, lr=0.01, n_steps=1000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(n_steps):
        grad = X.T @ (X @ w - y)         # BUG HERE?
        w = w + lr * grad                 # BUG HERE?
    return w
```

Find **two bugs** and explain why each is wrong.

<details>
<summary>Solution</summary>

**Bug 1**: The gradient is missing the scaling factor $\frac{2}{n}$.

Correct: `grad = (2 / n) * X.T @ (X @ w - y)`

Without the $\frac{2}{n}$ factor, the gradient is scaled by $n$, which means the effective learning rate depends on the dataset size. This makes the hyperparameter $lr$ not portable across different dataset sizes.

**Bug 2**: The update should subtract the gradient, not add it. Gradient descent moves in the direction of *negative* gradient.

Correct: `w = w - lr * grad`

Adding the gradient performs gradient *ascent*, which would maximize the loss instead of minimizing it.

</details>

---

## Exercise 4: Fill in the Shape (Shape Reasoning)

Given $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $w \in \mathbb{R}^d$, fill in the shapes:

| Expression | Shape |
|---|---|
| $X^TX$ | ? |
| $(X^TX)^{-1}$ | ? |
| $X^Ty$ | ? |
| $(X^TX)^{-1}X^Ty$ | ? |
| $H = X(X^TX)^{-1}X^T$ (hat matrix) | ? |
| $Hy$ | ? |
| $y - Hy$ (residual) | ? |
| $\nabla_w \mathcal{L}$ | ? |

<details>
<summary>Solution</summary>

| Expression | Shape | Reasoning |
|---|---|---|
| $X^TX$ | $(d, d)$ | $(d, n) \times (n, d)$ |
| $(X^TX)^{-1}$ | $(d, d)$ | Inverse of a $d \times d$ matrix |
| $X^Ty$ | $(d,)$ | $(d, n) \times (n,)$ |
| $(X^TX)^{-1}X^Ty$ | $(d,)$ | $(d, d) \times (d,)$ — this is $\hat{w}$ |
| $H$ | $(n, n)$ | $(n, d) \times (d, d) \times (d, n)$ |
| $Hy$ | $(n,)$ | $(n, n) \times (n,)$ — these are predictions $\hat{y}$ |
| $y - Hy$ | $(n,)$ | Residuals live in $\mathbb{R}^n$ |
| $\nabla_w \mathcal{L}$ | $(d,)$ | Same shape as $w$ (gradient has same shape as variable) |

**Key insight**: The hat matrix $H$ is $n \times n$ — it projects any vector in $\mathbb{R}^n$ onto the column space of $X$. It's called the "hat" matrix because it puts a hat on $y$: $\hat{y} = Hy$.

</details>

---

## Exercise 5: True/False with Justification

For each statement, determine if it's TRUE or FALSE and provide a brief justification.

1. **The MSE loss for linear regression is always convex.**
2. **If we add a new feature to $X$, the training MSE can only decrease or stay the same.**
3. **Gradient descent with any positive learning rate will converge to the optimal solution.**
4. **The normal equation and gradient descent always give the same solution.**
5. **The residual vector $y - Xw$ is orthogonal to every column of $X$ at the optimal $w$.**

<details>
<summary>Solution</summary>

1. **TRUE**. The Hessian of MSE is $\frac{2}{n}X^TX$, which is positive semi-definite (PSD) for any $X$. A function with PSD Hessian everywhere is convex. (It's strictly convex when $X$ has full column rank.)

2. **TRUE**. Adding a feature expands the column space of $X$. The projection of $y$ onto a larger subspace is at least as close (and potentially closer) to $y$. Formally, the optimal loss with $d+1$ features is $\leq$ the optimal loss with $d$ features because we can always set the new weight to zero and recover the old solution.

3. **FALSE**. If the learning rate is too large, gradient descent can diverge. Specifically, we need $\eta < \frac{2}{\lambda_{\max}(X^TX/n)}$ where $\lambda_{\max}$ is the largest eigenvalue. With too large a step, the iterates oscillate with increasing magnitude.

4. **TRUE** (with caveats). If $X^TX$ is invertible and gradient descent converges, both methods find the same unique minimizer. However, if $X^TX$ is singular, the normal equation gives one solution (e.g., the minimum-norm solution with pseudoinverse), while gradient descent with $w^{(0)} = 0$ converges to the minimum-norm solution specifically.

5. **TRUE**. This is exactly the normal equation: $X^T(y - Xw) = 0$, which states that the residual is orthogonal to the column space of $X$. This is the geometric interpretation of least squares as a projection.

</details>

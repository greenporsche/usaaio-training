# Bias-Variance & Regularization — Exercises

> 5 exercises covering bias-variance decomposition, Ridge, Lasso, and regularization effects

---

## Exercise 1: Compute the Ridge Solution (Compute This)

Given:

$$X = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad y = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \lambda = 2$$

**Tasks**:
1. Compute $X^TX$.
2. Compute $(X^TX + \lambda I)$.
3. Compute $(X^TX + \lambda I)^{-1}$.
4. Compute $\hat{w}_{\text{Ridge}} = (X^TX + \lambda I)^{-1}X^Ty$.
5. Compare with the OLS solution $\hat{w}_{\text{OLS}} = (X^TX)^{-1}X^Ty$.

<details>
<summary>Solution</summary>

**1.** $X^TX = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}^T\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$

**2.** $X^TX + 2I = \begin{bmatrix} 4 & 0 \\ 0 & 4 \end{bmatrix}$

**3.** $(X^TX + 2I)^{-1} = \begin{bmatrix} 1/4 & 0 \\ 0 & 1/4 \end{bmatrix}$

**4.** $X^Ty = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$

$\hat{w}_{\text{Ridge}} = \begin{bmatrix} 1/4 & 0 \\ 0 & 1/4 \end{bmatrix}\begin{bmatrix} 4 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}$

**5.** $\hat{w}_{\text{OLS}} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/2 \end{bmatrix}\begin{bmatrix} 4 \\ 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$

**Observation**: Ridge shrinks both weights from $(2, 1)$ to $(1, 0.5)$ — exactly halved. In general, Ridge applies a shrinkage factor of $\frac{\sigma_j^2}{\sigma_j^2 + \lambda}$ to each component. Here, all singular values of $X$ are equal ($\sigma = \sqrt{2}$), so the shrinkage is uniform: $\frac{2}{2 + 2} = 0.5$.

</details>

---

## Exercise 2: Derive Bias-Variance Decomposition (Derive That)

Let $y = f(x) + \epsilon$ where $E[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$, and $\epsilon$ is independent of the training set.

**Prove** that:

$$E_D\left[(y - \hat{f}(x))^2\right] = \left(f(x) - E_D[\hat{f}(x)]\right)^2 + E_D\left[(\hat{f}(x) - E_D[\hat{f}(x)])^2\right] + \sigma^2$$

Show every step. Clearly state where you use independence and where you use $E[\epsilon] = 0$.

<details>
<summary>Solution</summary>

Let $\bar{f} = E_D[\hat{f}(x)]$ for brevity.

**Step 1**: Expand $(y - \hat{f})^2$.

$$E[(y - \hat{f})^2] = E[(f + \epsilon - \hat{f})^2]$$

$$= E[((f - \hat{f}) + \epsilon)^2]$$

$$= E[(f - \hat{f})^2] + 2E[\epsilon(f - \hat{f})] + E[\epsilon^2]$$

**Step 2**: Handle the cross term. Since $\epsilon$ is independent of the training set $D$ (and hence independent of $\hat{f}$) and $f(x)$ is a fixed function:

$$E[\epsilon(f - \hat{f})] = E[\epsilon] \cdot E[f - \hat{f}] = 0 \cdot E[f - \hat{f}] = 0$$

(Used: independence of $\epsilon$ and $\hat{f}$, and $E[\epsilon] = 0$.)

Also, $E[\epsilon^2] = \text{Var}(\epsilon) + (E[\epsilon])^2 = \sigma^2 + 0 = \sigma^2$.

So: $E[(y - \hat{f})^2] = E[(f - \hat{f})^2] + \sigma^2$

**Step 3**: Decompose $E[(f - \hat{f})^2]$ using the add-and-subtract trick with $\bar{f}$.

$$E[(f - \hat{f})^2] = E[(f - \bar{f} + \bar{f} - \hat{f})^2]$$

$$= (f - \bar{f})^2 + 2(f - \bar{f})E[\bar{f} - \hat{f}] + E[(\bar{f} - \hat{f})^2]$$

Note: $(f - \bar{f})$ is a constant (not random), so it comes out of the expectation.

**Step 4**: The cross term vanishes because $E[\bar{f} - \hat{f}] = \bar{f} - E[\hat{f}] = \bar{f} - \bar{f} = 0$.

**Step 5**: Identify the terms.

$$E[(y - \hat{f})^2] = \underbrace{(f - \bar{f})^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f} - \bar{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

$\square$

</details>

---

## Exercise 3: Identify the Error (Debug)

A student writes Ridge regression with L1 regularization by mistake:

```python
def ridge_regression(X, y, lam):
    n, d = X.shape
    # Ridge should use L2 regularization
    w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    return w

def ridge_loss(X, y, w, lam):
    n = X.shape[0]
    mse = np.mean((X @ w - y) ** 2)
    reg = lam * np.sum(np.abs(w))      # ← Is this correct for Ridge?
    return mse + reg
```

1. Find the bug and explain why it's wrong.
2. What would the correct regularization term be?
3. The closed-form solution in `ridge_regression` is correct for Ridge. Would it still be correct if the loss function used L1 regularization?

<details>
<summary>Solution</summary>

**1. Bug**: The regularization term in `ridge_loss` uses `np.sum(np.abs(w))`, which is the **L1 norm** ($\|w\|_1$). This is the **Lasso** penalty, not Ridge.

**2. Correct Ridge regularization**: `reg = lam * np.sum(w ** 2)` or equivalently `lam * np.dot(w, w)`. This is the **L2 squared norm** ($\|w\|_2^2$).

**3. No**. The closed-form solution $(X^TX + \lambda I)^{-1}X^Ty$ is derived by setting the gradient of the L2-regularized loss to zero. The L1 norm $|w_j|$ is not differentiable at $w_j = 0$, so we cannot simply set the gradient to zero and get a closed-form solution. Lasso requires iterative methods like coordinate descent.

</details>

---

## Exercise 4: Soft Thresholding (Compute This)

The Lasso solution for orthonormal features is given by the soft-thresholding operator:

$$\hat{w}_j = \text{sign}(\hat{w}_j^{\text{OLS}}) \cdot \max(|\hat{w}_j^{\text{OLS}}| - \lambda, 0)$$

Given OLS weights $\hat{w}^{\text{OLS}} = [3.0, -1.5, 0.8, -0.3, 0.1]$ and $\lambda = 0.5$:

1. Compute the Lasso weights $\hat{w}^{\text{Lasso}}$.
2. Which features are "selected" (nonzero)?
3. What value of $\lambda$ would set exactly 3 of the 5 weights to zero?

<details>
<summary>Solution</summary>

**1. Apply soft thresholding to each component**:

| $j$ | $w_j^{\text{OLS}}$ | $|w_j^{\text{OLS}}|$ | $\max(|w_j^{\text{OLS}}| - 0.5, 0)$ | $\text{sign}$ | $w_j^{\text{Lasso}}$ |
|---|---|---|---|---|---|
| 1 | 3.0 | 3.0 | 2.5 | + | **2.5** |
| 2 | -1.5 | 1.5 | 1.0 | - | **-1.0** |
| 3 | 0.8 | 0.8 | 0.3 | + | **0.3** |
| 4 | -0.3 | 0.3 | 0 | - | **0** |
| 5 | 0.1 | 0.1 | 0 | + | **0** |

$\hat{w}^{\text{Lasso}} = [2.5, -1.0, 0.3, 0, 0]$

**2.** Features 1, 2, and 3 are selected (nonzero). Features 4 and 5 are zeroed out.

**3.** To set exactly 3 weights to zero, we need $\lambda$ such that the third-smallest absolute value is zeroed out but the second-largest is not. The sorted absolute values are: $[0.1, 0.3, 0.8, 1.5, 3.0]$.

We need $0.8 \leq \lambda < 1.5$, so any $\lambda \in [0.8, 1.5)$ sets features 3, 4, and 5 to zero while keeping features 1 and 2 nonzero.

</details>

---

## Exercise 5: True/False with Justification

1. **Ridge regression always produces a unique solution, even when $X^TX$ is singular.**
2. **Lasso with $\lambda \to \infty$ sets all weights to zero.**
3. **Increasing $\lambda$ in Ridge regression always increases training error.**
4. **The bias-variance decomposition shows that the irreducible noise $\sigma^2$ depends on the model.**
5. **Elastic Net always selects more features than Lasso for the same L1 penalty strength.**

<details>
<summary>Solution</summary>

1. **TRUE**. The Ridge solution is $(X^TX + \lambda I)^{-1}X^Ty$. For $\lambda > 0$, the matrix $X^TX + \lambda I$ has eigenvalues $\sigma_i^2 + \lambda > 0$, so it's always invertible. This is a key advantage of Ridge over OLS.

2. **TRUE**. As $\lambda \to \infty$, the regularization penalty dominates. The optimal solution that minimizes $\|Xw - y\|^2 + \lambda\|w\|_1$ for large $\lambda$ is $w = 0$ (since any nonzero $w$ incurs an unbounded penalty). This applies to both Lasso and Ridge.

3. **TRUE**. At $\lambda = 0$, Ridge gives the OLS solution, which minimizes training MSE. For $\lambda > 0$, the solution is constrained away from the OLS optimum (shrunk toward zero), so training MSE must increase. Formally, the OLS objective value is $\leq$ the Ridge objective value evaluated at the Ridge solution, since the Ridge solution was chosen to minimize a different (penalized) objective.

4. **FALSE**. The irreducible noise $\sigma^2 = \text{Var}(\epsilon)$ is a property of the data-generating process, not the model. No model can reduce it — it represents inherent randomness in the labels. That's why it's called "irreducible."

5. **Not necessarily TRUE** — this is **FALSE** in general. While elastic net's L2 component encourages grouped selection (if features are correlated, it tends to select all of them together rather than arbitrarily picking one), it doesn't *always* select more features. The number of selected features depends on the specific data, correlations, and the relative strengths of L1 and L2 penalties.

</details>

# Loss Functions — Exercises

> 5 exercises covering loss computation, convexity proofs, gradient comparison, and loss selection

---

## Exercise 1: Compute Multiple Losses (Compute This)

Given true values $y = [3, -1, 2, 10]$ and predictions $\hat{y} = [2.5, -0.5, 3, 4]$:

**Tasks**:
1. Compute MSE, MAE, and Huber loss ($\delta = 1.0$) for each residual and the average.
2. Which loss is most affected by the outlier ($y=10, \hat{y}=4$)?
3. What is the gradient of each loss w.r.t. $\hat{y}_4$ (the prediction for the outlier)?

<details>
<summary>Solution</summary>

**1.** Residuals: $r = y - \hat{y} = [0.5, -0.5, -1, 6]$

| $i$ | $r_i$ | $r_i^2$ (MSE) | $|r_i|$ (MAE) | Huber ($\delta=1$) |
|---|---|---|---|---|
| 1 | 0.5 | 0.25 | 0.5 | 0.125 (quadratic: $\frac{1}{2}(0.5)^2$) |
| 2 | -0.5 | 0.25 | 0.5 | 0.125 |
| 3 | -1.0 | 1.0 | 1.0 | 0.5 (at boundary: $\frac{1}{2}(1)^2$) |
| 4 | 6.0 | 36.0 | 6.0 | 5.5 (linear: $1 \cdot 6 - 0.5$) |
| **Average** | | **9.375** | **2.0** | **1.5625** |

**2.** MSE is most affected. The outlier contributes 36/37.5 = 96% of the total MSE, but only 6/8 = 75% of MAE and 5.5/6.25 = 88% of Huber. MSE amplifies large errors quadratically.

**3.** Gradient w.r.t. $\hat{y}_4$ (where $r_4 = 6$):

- **MSE**: $\frac{\partial}{\partial \hat{y}_4}(\hat{y}_4 - y_4)^2 = 2(\hat{y}_4 - y_4) = 2(-6) = -12$. Divided by $n$: $-12/4 = -3.0$
- **MAE**: $\frac{\partial}{\partial \hat{y}_4}|\hat{y}_4 - y_4| = \text{sign}(\hat{y}_4 - y_4) = -1$. Divided by $n$: $-1/4 = -0.25$
- **Huber** (in linear regime since $|r_4| = 6 > \delta = 1$): gradient = $-\delta \cdot \text{sign}(r_4) = -1$. Divided by $n$: $-0.25$

The MSE gradient for the outlier is 12x larger than the MAE gradient — this is why MSE is sensitive to outliers and MAE is robust.

</details>

---

## Exercise 2: Prove Hinge Loss is Convex (Derive That)

The hinge loss for a single sample is:

$$\ell(w) = \max(0, 1 - y \cdot w^Tx)$$

where $y \in \{-1, +1\}$ and $x \in \mathbb{R}^d$.

1. Show that $g(w) = 1 - y \cdot w^Tx$ is a convex function of $w$.
2. Show that $h(w) = 0$ is a convex function of $w$.
3. Use the fact that $\max$ of convex functions is convex to conclude that the hinge loss is convex.
4. Is the hinge loss differentiable everywhere? If not, where does it fail?

<details>
<summary>Solution</summary>

**1.** $g(w) = 1 - yw^Tx$ is an **affine** function of $w$ (linear + constant). Affine functions are both convex and concave. To verify: for any $\lambda \in [0,1]$:

$$g(\lambda w_1 + (1-\lambda)w_2) = 1 - y(\lambda w_1 + (1-\lambda)w_2)^Tx$$
$$= \lambda(1 - yw_1^Tx) + (1-\lambda)(1 - yw_2^Tx) = \lambda g(w_1) + (1-\lambda)g(w_2)$$

Equality holds, so $g$ is affine (hence convex). ✓

**2.** $h(w) = 0$ is a constant function. Constants are convex (trivially: $\lambda \cdot 0 + (1-\lambda) \cdot 0 = 0$). ✓

**3.** The hinge loss is $\ell(w) = \max(g(w), h(w)) = \max(0, 1 - yw^Tx)$.

**Theorem**: If $f_1, f_2$ are convex, then $\max(f_1, f_2)$ is convex.

**Proof**: For any $\lambda \in [0, 1]$:

$$\max(f_1, f_2)(\lambda w_1 + (1-\lambda)w_2)$$

$$= \max(f_1(\lambda w_1 + (1-\lambda)w_2), f_2(\lambda w_1 + (1-\lambda)w_2))$$

$$\leq \max(\lambda f_1(w_1) + (1-\lambda)f_1(w_2), \lambda f_2(w_1) + (1-\lambda)f_2(w_2))$$

$$\leq \lambda \max(f_1(w_1), f_2(w_1)) + (1-\lambda)\max(f_1(w_2), f_2(w_2))$$

The first inequality uses convexity of $f_1, f_2$. The second uses $\max(\alpha, \beta) \leq \max(a, b) + \max(c, d)$ when $\alpha \leq a + c$ and $\beta \leq b + d$ (not exactly; more precisely, $\max(\lambda a + (1-\lambda)b, \lambda c + (1-\lambda)d) \leq \lambda \max(a,c) + (1-\lambda)\max(b,d)$).

So the hinge loss is convex. $\square$

**4.** The hinge loss is **not differentiable** at $yw^Tx = 1$ (where $\max$ switches between its two arguments). The left derivative is $-yx$ and the right derivative is $0$. At this point, the **subdifferential** is the convex hull: $\partial \ell = \{-\alpha yx : \alpha \in [0, 1]\}$.

</details>

---

## Exercise 3: Identify the Error (Debug)

```python
def binary_cross_entropy(y_true, y_pred):
    """Compute BCE loss."""
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(y_pred))  # BUG

def hinge_loss(y_true, scores):
    """y_true in {0, 1}, scores = w^T x"""
    return np.mean(np.maximum(0, 1 - y_true * scores))                         # BUG
```

Find **two bugs** and explain the fix.

<details>
<summary>Solution</summary>

**Bug 1 — `binary_cross_entropy`**: The second term should be `np.log(1 - y_pred)`, not `np.log(y_pred)`.

Current: `y_true * np.log(y_pred) + (1 - y_true) * np.log(y_pred)`
Correct: `y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)`

Without the `1 -`, when $y_i = 0$, the loss becomes $-\log(\hat{y}_i)$ instead of $-\log(1 - \hat{y}_i)$. This would *reward* the model for predicting high probability for the negative class (the exact opposite of what we want).

Additionally, best practice is to clip predictions: `y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)` to avoid `log(0)`.

**Bug 2 — `hinge_loss`**: The hinge loss expects labels in $\{-1, +1\}$, but the docstring says `y_true in {0, 1}`. If labels are actually $\{0, 1\}$, then when $y_i = 0$, the term becomes $\max(0, 1 - 0 \cdot \text{score}) = 1$ regardless of the score — the loss is constant and provides no gradient.

Fix: Either convert labels `y_true = 2 * y_true - 1` (maps $\{0,1\} \to \{-1,+1\}$), or change the docstring to require $\{-1, +1\}$ labels.

```python
def hinge_loss(y_true, scores):
    """y_true in {-1, +1}, scores = w^T x"""
    return np.mean(np.maximum(0, 1 - y_true * scores))
```

</details>

---

## Exercise 4: Loss Function Selection (Fill in the Analysis)

For each scenario, choose the best loss function and justify:

1. **Regression with clean data, no outliers.** Options: MSE, MAE, Huber.
2. **Regression with heavy-tailed noise (many outliers).** Options: MSE, MAE, Huber.
3. **Binary classification where you want probability estimates.** Options: BCE, Hinge.
4. **Binary classification where you want maximum margin.** Options: BCE, Hinge.
5. **You're building an SVM and want support vectors.** Options: MSE, Hinge, Squared Hinge.

<details>
<summary>Solution</summary>

**1. MSE** for clean data. When there are no outliers, MSE's quadratic penalty provides the strongest gradient signal and the smoothest optimization landscape. It gives the conditional mean, which is the optimal predictor under Gaussian noise. MSE also has a unique, easily computable global minimum for linear models.

**2. Huber** (or MAE). With outliers, MSE's quadratic penalty causes the model to "chase" outliers, distorting the fit. MAE is robust but not smooth (gradient is $\pm 1$ everywhere, leading to oscillation near the optimum). Huber gives the best of both: smooth near zero (like MSE) and robust far from zero (like MAE). Choose $\delta$ based on the expected noise level.

**3. BCE** (binary cross-entropy). BCE is derived from maximum likelihood estimation for Bernoulli outcomes, so the model naturally outputs calibrated probabilities. Hinge loss has a flat region (zero gradient) for well-classified points, so sigmoid applied to hinge-loss scores does not produce well-calibrated probabilities.

**4. Hinge loss**. The hinge loss explicitly optimizes for margin: it's zero when the classification is correct with margin $\geq 1$, and linearly penalizes violations. BCE, by contrast, always wants higher confidence (never gives zero loss for a correct prediction), which means no margin-based interpretation.

**5. Hinge loss**. The hinge loss creates **support vectors** — only points with $y_i f(x_i) \leq 1$ contribute to the gradient. The squared hinge ($\max(0, 1-yf)^2$) also works but makes all margin-violating points contribute with varying weight (no sharp support vector boundary). MSE is not appropriate for classification.

</details>

---

## Exercise 5: True/False with Justification

1. **Cross-entropy loss can be negative.**
2. **The gradient of MSE is always larger in magnitude than the gradient of MAE.**
3. **Hinge loss is zero for all correctly classified points.**
4. **A convex loss function guarantees that gradient descent finds the global minimum.**
5. **The Huber loss is differentiable at $|r| = \delta$.**

<details>
<summary>Solution</summary>

1. **FALSE**. Binary cross-entropy: $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$. Since $0 < \hat{y} < 1$, we have $\log\hat{y} < 0$ and $\log(1-\hat{y}) < 0$. The whole expression (with the negative sign out front) is always $\geq 0$. The minimum is 0, achieved when $\hat{y} = y$ (perfect prediction).

2. **FALSE**. For small residuals $|r| < 1$: MSE gradient $= 2r$, MAE gradient $= \text{sign}(r)$. When $|r| = 0.1$: $|2r| = 0.2 < 1 = |\text{sign}(r)|$. So MAE has a *larger* gradient for small errors. For large residuals $|r| > 0.5$: MSE gradient is larger. The crossover is at $|r| = 0.5$ (where $2|r| = 1$).

3. **FALSE**. Hinge loss is zero only when the point is correctly classified **with margin $\geq 1$**: $y_i f(x_i) \geq 1$. A correctly classified point with $0 < y_i f(x_i) < 1$ (correct but inside the margin) still incurs nonzero hinge loss. This is the key feature that creates the "margin band" in SVMs.

4. **TRUE** (with mild caveats). For a convex loss, every local minimum is a global minimum. Gradient descent with a sufficiently small learning rate converges to a point with zero gradient, which is the global minimum for strictly convex losses. For merely convex losses (not strictly), the minimum may not be unique, but gradient descent still converges to *a* global minimizer. The caveat: the learning rate must be chosen appropriately (e.g., $\eta < 2/L$ where $L$ is the Lipschitz constant of the gradient).

5. **TRUE**. By design, the Huber loss transitions smoothly between quadratic and linear regimes at $|r| = \delta$. Specifically:

From the quadratic side ($|r| \leq \delta$): derivative at $r = \delta$ is $r = \delta$.

From the linear side ($|r| > \delta$): derivative at $r = \delta^+$ is $\delta \cdot \text{sign}(r) = \delta$.

Both sides agree, so the Huber loss is differentiable at $|r| = \delta$. This was a deliberate design choice — Huber specifically constructed this loss to be smooth everywhere while remaining robust.

</details>

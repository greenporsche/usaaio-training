# Exercises: Convex Optimization

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 8.1 — Convexity Check

Determine whether each function is convex, strictly convex, or neither. Justify using the Hessian.

**(a)** $f(x) = e^x$

**(b)** $f(x, y) = x^2 - y^2$

**(c)** $f(\mathbf{x}) = \|\mathbf{x}\|^2 + 3$ for $\mathbf{x} \in \mathbb{R}^n$

<details><summary>Solution</summary>

**(a)** $f''(x) = e^x > 0$ for all $x$. **Strictly convex.** $\checkmark$

**(b)** $H = \begin{bmatrix}2 & 0 \\ 0 & -2\end{bmatrix}$. Eigenvalues: $2$ and $-2$. Since there's a negative eigenvalue, $H$ is indefinite. **Neither convex nor concave.** The function is a saddle.

**(c)** $f(\mathbf{x}) = \mathbf{x}^\top\mathbf{x} + 3$, so $H = 2I_n \succ 0$. **Strictly convex.** $\checkmark$

</details>

---

## Exercise 8.2 — Gradient Descent by Hand

Run 3 iterations of gradient descent on $f(x) = x^2 + 2x + 1 = (x+1)^2$ starting at $x_0 = 3$ with learning rate $\eta = 0.3$.

<details><summary>Solution</summary>

$f'(x) = 2x + 2$

**Iteration 1**: $f'(3) = 8$, $x_1 = 3 - 0.3(8) = 3 - 2.4 = \mathbf{0.6}$

**Iteration 2**: $f'(0.6) = 3.2$, $x_2 = 0.6 - 0.3(3.2) = 0.6 - 0.96 = \mathbf{-0.36}$

**Iteration 3**: $f'(-0.36) = 1.28$, $x_3 = -0.36 - 0.3(1.28) = -0.36 - 0.384 = \mathbf{-0.744}$

The true minimum is at $x^* = -1$. After 3 iterations: $x_3 = -0.744$, error = $0.256$.

The error decreases by factor $|1 - 2\eta| = |1 - 0.6| = 0.4$ per iteration (for quadratics with $f'' = 2$). So convergence: $|x_t - x^*| = 4 \cdot (0.4)^t$.

</details>

---

## Exercise 8.3 — Learning Rate Analysis

For $f(x) = \frac{1}{2}x^2$ (so $f'(x) = x$):

**(a)** Write the gradient descent update.

**(b)** After $t$ iterations starting from $x_0$, what is $x_t$?

**(c)** For what values of $\eta$ does gradient descent converge?

**(d)** What is the optimal $\eta$?

<details><summary>Solution</summary>

**(a)** $x_{t+1} = x_t - \eta x_t = (1 - \eta)x_t$

**(b)** $x_t = (1 - \eta)^t x_0$

**(c)** Converges iff $|1 - \eta| < 1$, i.e., $\mathbf{0 < \eta < 2}$.

- $\eta = 0$: no progress
- $0 < \eta < 1$: monotone convergence (each step gets closer)
- $\eta = 1$: converges in ONE step ($x_1 = 0$)
- $1 < \eta < 2$: oscillating convergence (overshoots but converges)
- $\eta = 2$: oscillates forever between $\pm x_0$
- $\eta > 2$: diverges

**(d)** Optimal $\eta = 1$ (or $\eta = 1/L$ where $L = f'' = 1$). Converges in a single step.

</details>

---

## Exercise 8.4 — Lagrangian Method

Minimize $f(x, y) = x^2 + y^2$ subject to $2x + y = 5$.

<details><summary>Solution</summary>

**Lagrangian**: $\mathcal{L}(x, y, \nu) = x^2 + y^2 + \nu(2x + y - 5)$

**Stationarity conditions**:

$\frac{\partial\mathcal{L}}{\partial x} = 2x + 2\nu = 0 \implies x = -\nu$

$\frac{\partial\mathcal{L}}{\partial y} = 2y + \nu = 0 \implies y = -\nu/2$

**Primal feasibility**: $2x + y = 5$

$2(-\nu) + (-\nu/2) = 5 \implies -5\nu/2 = 5 \implies \nu = -2$

**Solution**: $x^* = 2, y^* = 1$

$f^* = 4 + 1 = \mathbf{5}$

**Geometric interpretation**: The point $(2, 1)$ is the closest point on the line $2x + y = 5$ to the origin.

</details>

---

## Exercise 8.5 — KKT Conditions

Minimize $f(x) = x^2$ subject to $x \geq 3$ (equivalently: $g(x) = 3 - x \leq 0$).

**(a)** Write the Lagrangian.

**(b)** Apply KKT conditions to find the solution.

**(c)** Verify using geometric reasoning.

<details><summary>Solution</summary>

**(a)** $\mathcal{L}(x, \lambda) = x^2 + \lambda(3 - x)$

**(b)** KKT conditions:

1. **Stationarity**: $\frac{\partial\mathcal{L}}{\partial x} = 2x - \lambda = 0 \implies \lambda = 2x$

2. **Primal feasibility**: $3 - x \leq 0 \implies x \geq 3$

3. **Dual feasibility**: $\lambda \geq 0$

4. **Complementary slackness**: $\lambda(3 - x) = 0$

From complementary slackness: either $\lambda = 0$ or $x = 3$.

**Case 1**: $\lambda = 0 \implies$ from stationarity: $2x = 0 \implies x = 0$. But $x = 0$ violates $x \geq 3$. Infeasible.

**Case 2**: $x = 3 \implies \lambda = 6 > 0$ $\checkmark$ (dual feasibility satisfied)

**Solution**: $x^* = 3$, $\lambda^* = 6$, $f^* = 9$.

**(c)** The unconstrained minimum of $x^2$ is at $x = 0$, but $x \geq 3$ forces us to the boundary. The closest feasible point is $x = 3$, so $f^* = 9$. $\checkmark$

</details>

---

## Exercise 8.6 — USAAIO Competition Style

Consider gradient descent with momentum on the "narrow valley" function:

$$f(x, y) = \frac{1}{2}(100x^2 + y^2)$$

**(a)** What is the condition number of the Hessian? Why does this make gradient descent struggle?

**(b)** Compute 2 iterations of standard gradient descent from $(x_0, y_0) = (1, 1)$ with $\eta = 0.01$.

**(c)** Why would momentum help on this problem?

<details><summary>Solution</summary>

**(a)** Hessian: $H = \begin{bmatrix}100&0\\0&1\end{bmatrix}$

Condition number: $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{100}{1} = \mathbf{100}$

A high condition number means the curvature is very different in different directions. Gradient descent oscillates back and forth in the high-curvature direction ($x$) while making slow progress in the low-curvature direction ($y$). The learning rate must be small enough for the steepest direction ($\eta < 2/\lambda_{\max} = 0.02$), which makes progress painfully slow in the flat direction.

**(b)** $\nabla f = \begin{bmatrix}100x\\y\end{bmatrix}$

**Iteration 1**: $\nabla f(1,1) = \begin{bmatrix}100\\1\end{bmatrix}$

$(x_1, y_1) = (1, 1) - 0.01\begin{bmatrix}100\\1\end{bmatrix} = (0, 0.99)$

**Iteration 2**: $\nabla f(0, 0.99) = \begin{bmatrix}0\\0.99\end{bmatrix}$

$(x_2, y_2) = (0, 0.99) - 0.01\begin{bmatrix}0\\0.99\end{bmatrix} = (0, 0.9801)$

After 2 iterations: $x$ converged (to 0), but $y$ is barely moving (0.9801 vs target 0).

**(c)** Momentum accumulates velocity in consistent directions. The $y$-gradient is small but consistent, so momentum builds up speed along $y$. The $x$-gradient oscillates (changes sign), so momentum dampens those oscillations. Net effect: much faster convergence along the valley floor.

</details>

# Exercises: Multivariable Calculus

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 7.1 — Gradient Computation

Compute the gradient of $f(x, y, z) = x^2y + yz^3 + \sin(xz)$ at the point $(1, 2, 0)$.

<details><summary>Solution</summary>

$$\frac{\partial f}{\partial x} = 2xy + z\cos(xz)$$

$$\frac{\partial f}{\partial y} = x^2 + z^3$$

$$\frac{\partial f}{\partial z} = 3yz^2 + x\cos(xz)$$

At $(1, 2, 0)$:

$$\nabla f = \begin{bmatrix}2(1)(2) + 0 \cdot \cos(0) \\ 1 + 0 \\ 0 + 1 \cdot \cos(0)\end{bmatrix} = \begin{bmatrix}4\\1\\1\end{bmatrix}$$

The gradient points in the direction of steepest ascent. Its magnitude $\|\nabla f\| = \sqrt{16+1+1} = \sqrt{18} = 3\sqrt{2}$ is the maximum rate of change.

</details>

---

## Exercise 7.2 — Matrix Calculus Identity

For $\mathbf{x} \in \mathbb{R}^n$ and symmetric $A \in \mathbb{R}^{n \times n}$:

**(a)** Derive $\nabla_\mathbf{x}(\mathbf{x}^\top A \mathbf{x})$ from first principles (using the definition of partial derivatives).

**(b)** Verify your answer for $A = \begin{bmatrix}2&1\\1&3\end{bmatrix}$ at $\mathbf{x} = \begin{bmatrix}1\\2\end{bmatrix}$.

<details><summary>Solution</summary>

**(a)** $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} = \sum_{i,j} x_i A_{ij} x_j$

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj}x_j + \sum_i x_i A_{ik}$$

The first sum is $(A\mathbf{x})_k$. The second sum is $(A^\top\mathbf{x})_k$.

$$\nabla_\mathbf{x} f = A\mathbf{x} + A^\top\mathbf{x} = (A + A^\top)\mathbf{x}$$

For symmetric $A$ ($A = A^\top$): $\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = 2A\mathbf{x}$ $\blacksquare$

**(b)** $\nabla f = 2A\mathbf{x} = 2\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = 2\begin{bmatrix}4\\7\end{bmatrix} = \begin{bmatrix}8\\14\end{bmatrix}$

Verify numerically: $f(\mathbf{x}) = [1\ 2]\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = [1\ 2]\begin{bmatrix}4\\7\end{bmatrix} = 18$

$f(\mathbf{x} + \epsilon\mathbf{e}_1) \approx f(1.001, 2) = [1.001\ 2]\begin{bmatrix}4.002\\7.001\end{bmatrix} = 4.006 + 14.002 = 18.008$

Numerical derivative: $(18.008 - 18)/0.001 = 8$ $\checkmark$

</details>

---

## Exercise 7.3 — Jacobian Computation

Compute the Jacobian of $\mathbf{f}(\mathbf{x}) = \begin{bmatrix}x_1^2 + x_2 \\ x_1 x_2 \\ e^{x_1}\end{bmatrix}$ at $\mathbf{x} = \begin{bmatrix}0\\1\end{bmatrix}$.

<details><summary>Solution</summary>

$$\mathbf{J} = \begin{bmatrix}\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \\ \frac{\partial f_3}{\partial x_1} & \frac{\partial f_3}{\partial x_2}\end{bmatrix} = \begin{bmatrix}2x_1 & 1 \\ x_2 & x_1 \\ e^{x_1} & 0\end{bmatrix}$$

At $\mathbf{x} = (0, 1)$:

$$\mathbf{J} = \begin{bmatrix}0 & 1 \\ 1 & 0 \\ 1 & 0\end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

Shape check: $\mathbf{f}: \mathbb{R}^2 \to \mathbb{R}^3$, so $\mathbf{J} \in \mathbb{R}^{3 \times 2}$ $\checkmark$

</details>

---

## Exercise 7.4 — Chain Rule for Vectors

Let $L = \|\sigma(W\mathbf{x})\|^2$ where $\sigma(z) = \max(0, z)$ (ReLU applied element-wise), $W \in \mathbb{R}^{2 \times 3}$, $\mathbf{x} \in \mathbb{R}^3$.

Given $W = \begin{bmatrix}1&0&-1\\0&1&1\end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix}2\\1\\-1\end{bmatrix}$, compute $\frac{\partial L}{\partial \mathbf{x}}$.

<details><summary>Solution</summary>

**Forward pass**:

$\mathbf{z} = W\mathbf{x} = \begin{bmatrix}1&0&-1\\0&1&1\end{bmatrix}\begin{bmatrix}2\\1\\-1\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix}$

$\mathbf{a} = \sigma(\mathbf{z}) = \begin{bmatrix}3\\0\end{bmatrix}$ (ReLU: max(3,0)=3, max(0,0)=0)

$L = \|\mathbf{a}\|^2 = 9$

**Backward pass** (chain rule):

$\frac{\partial L}{\partial \mathbf{a}} = 2\mathbf{a} = \begin{bmatrix}6\\0\end{bmatrix}$

$\frac{\partial \mathbf{a}}{\partial \mathbf{z}} = \text{diag}(\mathbf{1}[\mathbf{z} > 0]) = \begin{bmatrix}1&0\\0&0\end{bmatrix}$ (ReLU derivative: 1 if $z > 0$, 0 otherwise)

$\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial \mathbf{a}}{\partial \mathbf{z}} \cdot \frac{\partial L}{\partial \mathbf{a}} = \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}6\\0\end{bmatrix} = \begin{bmatrix}6\\0\end{bmatrix}$

$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = W = \begin{bmatrix}1&0&-1\\0&1&1\end{bmatrix}$

$\frac{\partial L}{\partial \mathbf{x}} = W^\top \frac{\partial L}{\partial \mathbf{z}} = \begin{bmatrix}1&0\\0&1\\-1&1\end{bmatrix}\begin{bmatrix}6\\0\end{bmatrix} = \begin{bmatrix}6\\0\\-6\end{bmatrix}$

</details>

---

## Exercise 7.5 — Hessian and Convexity

For $f(\mathbf{x}) = x_1^2 + x_1 x_2 + x_2^2$:

**(a)** Compute the gradient and Hessian.

**(b)** Determine whether $f$ is convex, strictly convex, or neither.

**(c)** Find the minimum of $f$.

<details><summary>Solution</summary>

**(a)** Gradient: $\nabla f = \begin{bmatrix}2x_1 + x_2 \\ x_1 + 2x_2\end{bmatrix}$

Hessian: $H = \begin{bmatrix}2 & 1 \\ 1 & 2\end{bmatrix}$ (constant — $f$ is quadratic)

**(b)** Eigenvalues of $H$: $(2-\lambda)^2 - 1 = 0 \implies \lambda = 1, 3$

Both eigenvalues are strictly positive, so $H \succ 0$ everywhere.

$f$ is **strictly convex**. $\checkmark$

**(c)** Set $\nabla f = \mathbf{0}$:
$2x_1 + x_2 = 0$ and $x_1 + 2x_2 = 0$

From the first: $x_2 = -2x_1$. Substituting: $x_1 + 2(-2x_1) = -3x_1 = 0 \implies x_1 = 0, x_2 = 0$.

Minimum at $\mathbf{x}^* = \begin{bmatrix}0\\0\end{bmatrix}$, $f^* = 0$.

Since $f$ is strictly convex, this is the unique global minimum.

</details>

---

## Exercise 7.6 — USAAIO Competition Style: MSE Gradient

The mean squared error loss for linear regression is:

$$L(\mathbf{w}) = \frac{1}{2N}\|X\mathbf{w} - \mathbf{y}\|^2$$

where $X \in \mathbb{R}^{N \times d}$, $\mathbf{w} \in \mathbb{R}^d$, $\mathbf{y} \in \mathbb{R}^N$.

**(a)** Derive $\nabla_\mathbf{w} L$.

**(b)** Set the gradient to zero and solve for the optimal $\mathbf{w}^*$ (assuming $X^\top X$ is invertible).

**(c)** What is the Hessian? Is $L$ convex?

<details><summary>Solution</summary>

**(a)** Expand: $L = \frac{1}{2N}(X\mathbf{w} - \mathbf{y})^\top(X\mathbf{w} - \mathbf{y}) = \frac{1}{2N}(\mathbf{w}^\top X^\top X \mathbf{w} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{y}^\top\mathbf{y})$

Using matrix calculus identities:

$$\nabla_\mathbf{w} L = \frac{1}{2N}(2X^\top X\mathbf{w} - 2X^\top\mathbf{y}) = \frac{1}{N}X^\top(X\mathbf{w} - \mathbf{y})$$

$\blacksquare$

**(b)** $\nabla_\mathbf{w} L = 0 \implies X^\top X\mathbf{w}^* = X^\top\mathbf{y}$ (normal equations)

$$\boxed{\mathbf{w}^* = (X^\top X)^{-1}X^\top\mathbf{y}}$$

**(c)** Hessian: $H = \frac{1}{N}X^\top X$

$X^\top X$ is always PSD (since $\mathbf{z}^\top X^\top X\mathbf{z} = \|X\mathbf{z}\|^2 \geq 0$).

Therefore $H \succeq 0$ and **$L$ is convex**. If $X$ has full column rank, $H \succ 0$ and $L$ is strictly convex (unique minimum).

</details>

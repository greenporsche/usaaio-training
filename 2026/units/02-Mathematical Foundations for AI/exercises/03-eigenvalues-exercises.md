# Exercises: Eigenvalues and Eigenvectors

**Target time**: 2-5 minutes per exercise | **Total**: 7 exercises

---

## Exercise 3.1 — Compute Eigenvalues and Eigenvectors

Find all eigenvalues and eigenvectors of:

$$A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$$

<details><summary>Solution</summary>

**Characteristic equation**: $\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0$

**Eigenvalues**: $\lambda_1 = 5$, $\lambda_2 = 2$.

**Eigenvector for $\lambda_1 = 5$**:

$(A - 5I)\mathbf{x} = \begin{bmatrix}-1&2\\1&-2\end{bmatrix}\mathbf{x} = \mathbf{0} \implies x_1 = 2x_2$

$\mathbf{v}_1 = \begin{bmatrix}2\\1\end{bmatrix}$ (or any scalar multiple)

**Eigenvector for $\lambda_2 = 2$**:

$(A - 2I)\mathbf{x} = \begin{bmatrix}2&2\\1&1\end{bmatrix}\mathbf{x} = \mathbf{0} \implies x_1 = -x_2$

$\mathbf{v}_2 = \begin{bmatrix}-1\\1\end{bmatrix}$

**Verification**: $A\mathbf{v}_1 = \begin{bmatrix}4&2\\1&3\end{bmatrix}\begin{bmatrix}2\\1\end{bmatrix} = \begin{bmatrix}10\\5\end{bmatrix} = 5\begin{bmatrix}2\\1\end{bmatrix}$ $\checkmark$

</details>

---

## Exercise 3.2 — Eigenvalues of Triangular Matrix

Without computing the characteristic polynomial, find the eigenvalues of:

$$T = \begin{bmatrix} 3 & 1 & 4 \\ 0 & -2 & 5 \\ 0 & 0 & 7 \end{bmatrix}$$

Also compute $\text{tr}(T)$ and $\det(T)$ from the eigenvalues.

<details><summary>Solution</summary>

For triangular matrices (upper or lower), **the eigenvalues are the diagonal entries**.

$\det(T - \lambda I) = (3-\lambda)(-2-\lambda)(7-\lambda) = 0$

**Eigenvalues**: $\lambda_1 = 3, \lambda_2 = -2, \lambda_3 = 7$.

$\text{tr}(T) = \lambda_1 + \lambda_2 + \lambda_3 = 3 + (-2) + 7 = \mathbf{8}$

$\det(T) = \lambda_1 \cdot \lambda_2 \cdot \lambda_3 = 3 \cdot (-2) \cdot 7 = \mathbf{-42}$

Verify: trace from diagonal: $3 + (-2) + 7 = 8$ $\checkmark$. Determinant of upper triangular: product of diagonal $= 3 \cdot (-2) \cdot 7 = -42$ $\checkmark$.

</details>

---

## Exercise 3.3 — Eigendecomposition

Given $A = \begin{bmatrix} 5 & 4 \\ 1 & 2 \end{bmatrix}$:

**(a)** Find the eigendecomposition $A = Q\Lambda Q^{-1}$.

**(b)** Use it to compute $A^3$.

<details><summary>Solution</summary>

**(a)** Characteristic equation: $(5-\lambda)(2-\lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda-6)(\lambda-1) = 0$

Eigenvalues: $\lambda_1 = 6, \lambda_2 = 1$.

For $\lambda_1 = 6$: $(A-6I)\mathbf{x} = \begin{bmatrix}-1&4\\1&-4\end{bmatrix}\mathbf{x} = \mathbf{0} \implies \mathbf{v}_1 = \begin{bmatrix}4\\1\end{bmatrix}$

For $\lambda_2 = 1$: $(A-I)\mathbf{x} = \begin{bmatrix}4&4\\1&1\end{bmatrix}\mathbf{x} = \mathbf{0} \implies \mathbf{v}_2 = \begin{bmatrix}-1\\1\end{bmatrix}$

$$Q = \begin{bmatrix}4&-1\\1&1\end{bmatrix}, \quad \Lambda = \begin{bmatrix}6&0\\0&1\end{bmatrix}$$

$Q^{-1} = \frac{1}{4+1}\begin{bmatrix}1&1\\-1&4\end{bmatrix} = \frac{1}{5}\begin{bmatrix}1&1\\-1&4\end{bmatrix}$

**(b)** $A^3 = Q\Lambda^3 Q^{-1} = Q\begin{bmatrix}216&0\\0&1\end{bmatrix}Q^{-1}$

$$= \begin{bmatrix}4&-1\\1&1\end{bmatrix}\begin{bmatrix}216&0\\0&1\end{bmatrix}\frac{1}{5}\begin{bmatrix}1&1\\-1&4\end{bmatrix}$$

$$= \frac{1}{5}\begin{bmatrix}864&-1\\216&1\end{bmatrix}\begin{bmatrix}1&1\\-1&4\end{bmatrix} = \frac{1}{5}\begin{bmatrix}865&860\\215&220\end{bmatrix} = \begin{bmatrix}173&172\\43&44\end{bmatrix}$$

</details>

---

## Exercise 3.4 — Spectral Theorem

Let $S = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ (symmetric).

**(a)** Find eigenvalues and eigenvectors.

**(b)** Verify the eigenvectors are orthogonal.

**(c)** Write $S = Q\Lambda Q^\top$ and verify $Q$ is orthogonal.

<details><summary>Solution</summary>

**(a)** $(3-\lambda)^2 - 1 = 0 \implies \lambda^2 - 6\lambda + 8 = 0 \implies \lambda_1 = 4, \lambda_2 = 2$

For $\lambda_1 = 4$: $\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{x} = 0 \implies \mathbf{v}_1 = \begin{bmatrix}1\\1\end{bmatrix}$

For $\lambda_2 = 2$: $\begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf{x} = 0 \implies \mathbf{v}_2 = \begin{bmatrix}-1\\1\end{bmatrix}$

**(b)** $\mathbf{v}_1 \cdot \mathbf{v}_2 = (1)(-1) + (1)(1) = 0$ $\checkmark$ Orthogonal!

**(c)** Normalize: $\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$, $\mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}-1\\1\end{bmatrix}$

$$Q = \frac{1}{\sqrt{2}}\begin{bmatrix}1&-1\\1&1\end{bmatrix}, \quad \Lambda = \begin{bmatrix}4&0\\0&2\end{bmatrix}$$

Verify $Q^\top Q = I$:
$$Q^\top Q = \frac{1}{2}\begin{bmatrix}1&1\\-1&1\end{bmatrix}\begin{bmatrix}1&-1\\1&1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}2&0\\0&2\end{bmatrix} = I \checkmark$$

Verify $S = Q\Lambda Q^\top$:
$$Q\Lambda Q^\top = \frac{1}{2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}\begin{bmatrix}4&0\\0&2\end{bmatrix}\begin{bmatrix}1&1\\-1&1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}4&-2\\4&2\end{bmatrix}\begin{bmatrix}1&1\\-1&1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}6&2\\2&6\end{bmatrix} = \begin{bmatrix}3&1\\1&3\end{bmatrix} \checkmark$$

</details>

---

## Exercise 3.5 — True/False with Justification

**(a)** If $\lambda$ is an eigenvalue of $A$, then $\lambda^2$ is an eigenvalue of $A^2$.

**(b)** If $A$ has eigenvalue 0, then $A$ is invertible.

**(c)** A $3 \times 3$ real matrix always has at least one real eigenvalue.

<details><summary>Solution</summary>

**(a) True.** If $A\mathbf{x} = \lambda\mathbf{x}$, then $A^2\mathbf{x} = A(A\mathbf{x}) = A(\lambda\mathbf{x}) = \lambda A\mathbf{x} = \lambda^2\mathbf{x}$. $\blacksquare$

**(b) False** — in the opposite direction. If $A$ has eigenvalue 0, then $\det(A) = \prod \lambda_i = 0$, so $A$ is **not invertible** (singular).

**(c) True.** The characteristic polynomial of a $3 \times 3$ real matrix is a degree-3 polynomial with real coefficients. By the Intermediate Value Theorem (or the fact that complex roots of real polynomials come in conjugate pairs), a degree-3 real polynomial always has at least one real root.

</details>

---

## Exercise 3.6 — Power Method Convergence

Consider $A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$ with starting vector $\mathbf{b}_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

Compute $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ using the power method (multiply by $A$, then normalize). What eigenvalue/eigenvector does it converge to?

<details><summary>Solution</summary>

**Iteration 1**: $A\mathbf{b}_0 = \begin{bmatrix}2\\1\end{bmatrix}$, $\|\cdot\| = \sqrt{5}$, $\mathbf{b}_1 = \frac{1}{\sqrt{5}}\begin{bmatrix}2\\1\end{bmatrix} \approx \begin{bmatrix}0.894\\0.447\end{bmatrix}$

**Iteration 2**: $A\mathbf{b}_1 = \frac{1}{\sqrt{5}}\begin{bmatrix}4\\1\end{bmatrix}$, $\|\cdot\| = \frac{\sqrt{17}}{\sqrt{5}}$, $\mathbf{b}_2 = \frac{1}{\sqrt{17}}\begin{bmatrix}4\\1\end{bmatrix} \approx \begin{bmatrix}0.970\\0.243\end{bmatrix}$

**Iteration 3**: $A\mathbf{b}_2 = \frac{1}{\sqrt{17}}\begin{bmatrix}8\\1\end{bmatrix}$, $\|\cdot\| = \frac{\sqrt{65}}{\sqrt{17}}$, $\mathbf{b}_3 = \frac{1}{\sqrt{65}}\begin{bmatrix}8\\1\end{bmatrix} \approx \begin{bmatrix}0.992\\0.124\end{bmatrix}$

The method converges to $\begin{bmatrix}1\\0\end{bmatrix}$ with eigenvalue $\lambda = 2$ (the dominant eigenvalue).

**Convergence rate**: The ratio $|\lambda_2/\lambda_1| = 1/2$, so each iteration halves the error in the non-dominant component. After $k$ iterations, the non-dominant component decays as $(1/2)^k$.

</details>

---

## Exercise 3.7 — USAAIO Competition Style

Let $A \in \mathbb{R}^{n \times n}$ have eigenvalues $\lambda_1, \ldots, \lambda_n$.

**(a)** Prove that $A + cI$ has eigenvalues $\lambda_1 + c, \ldots, \lambda_n + c$.

**(b)** Prove that if $A$ is invertible, then $A^{-1}$ has eigenvalues $1/\lambda_1, \ldots, 1/\lambda_n$.

**(c)** Using (a) and (b), find the eigenvalues of $(A - 3I)^{-1}$ in terms of $\lambda_i$ (assuming $\lambda_i \neq 3$ for all $i$).

<details><summary>Solution</summary>

**(a)** If $A\mathbf{x} = \lambda_i \mathbf{x}$, then:

$(A + cI)\mathbf{x} = A\mathbf{x} + c\mathbf{x} = \lambda_i\mathbf{x} + c\mathbf{x} = (\lambda_i + c)\mathbf{x}$

So $\mathbf{x}$ is an eigenvector of $A + cI$ with eigenvalue $\lambda_i + c$. $\blacksquare$

**(b)** If $A\mathbf{x} = \lambda_i \mathbf{x}$ with $\lambda_i \neq 0$, multiply both sides by $A^{-1}$:

$\mathbf{x} = \lambda_i A^{-1}\mathbf{x} \implies A^{-1}\mathbf{x} = \frac{1}{\lambda_i}\mathbf{x}$

So $\mathbf{x}$ is an eigenvector of $A^{-1}$ with eigenvalue $1/\lambda_i$. $\blacksquare$

**(c)** By (a), $A - 3I$ has eigenvalues $\lambda_i - 3$. By (b), $(A - 3I)^{-1}$ has eigenvalues:

$$\frac{1}{\lambda_i - 3}, \quad i = 1, \ldots, n$$

</details>

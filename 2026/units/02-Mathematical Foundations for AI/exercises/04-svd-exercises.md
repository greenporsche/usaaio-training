# Exercises: Singular Value Decomposition

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 4.1 — SVD of a Simple Matrix

Compute the full SVD of:

$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$$

<details><summary>Solution</summary>

For a diagonal matrix, the SVD is trivial:

$A^\top A = \begin{bmatrix}9&0\\0&4\end{bmatrix}$, eigenvalues: $\sigma_1^2 = 9, \sigma_2^2 = 4$

Singular values: $\sigma_1 = 3, \sigma_2 = 2$ (already ordered).

$V = I_2$ (eigenvectors of $A^\top A$ are standard basis), $U = I_2$ (since $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$).

$$A = \begin{bmatrix}1&0\\0&1\end{bmatrix}\begin{bmatrix}3&0\\0&2\end{bmatrix}\begin{bmatrix}1&0\\0&1\end{bmatrix}^\top = I \cdot \Sigma \cdot I$$

For a diagonal matrix with non-negative entries, $U = V = I$ and $\Sigma$ equals the matrix itself.

</details>

---

## Exercise 4.2 — SVD from $A^\top A$

Compute the SVD of:

$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}$$

<details><summary>Solution</summary>

**Step 1**: $A^\top A = \begin{bmatrix}1&0&1\\1&1&0\end{bmatrix}\begin{bmatrix}1&1\\0&1\\1&0\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}$

**Step 2**: Eigenvalues of $A^\top A$: $(2-\lambda)^2 - 1 = 0 \implies \lambda_1 = 3, \lambda_2 = 1$

Singular values: $\sigma_1 = \sqrt{3}, \sigma_2 = 1$.

**Step 3**: Eigenvectors of $A^\top A$ (right singular vectors $V$):

For $\lambda = 3$: $\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{v} = 0 \implies \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$

For $\lambda = 1$: $\begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf{v} = 0 \implies \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}-1\\1\end{bmatrix}$

**Step 4**: Left singular vectors: $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$

$\mathbf{u}_1 = \frac{1}{\sqrt{3}} A\mathbf{v}_1 = \frac{1}{\sqrt{3}} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}2\\1\\1\end{bmatrix} = \frac{1}{\sqrt{6}}\begin{bmatrix}2\\1\\1\end{bmatrix}$

$\mathbf{u}_2 = \frac{1}{1} A\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}0\\1\\-1\end{bmatrix}$

(Extend to $\mathbf{u}_3$ orthogonal to both: $\mathbf{u}_3 = \frac{1}{\sqrt{3}}\begin{bmatrix}-1\\1\\1\end{bmatrix}$)

$$U = \begin{bmatrix}\frac{2}{\sqrt{6}}&0&\frac{-1}{\sqrt{3}}\\\frac{1}{\sqrt{6}}&\frac{1}{\sqrt{2}}&\frac{1}{\sqrt{3}}\\\frac{1}{\sqrt{6}}&\frac{-1}{\sqrt{2}}&\frac{1}{\sqrt{3}}\end{bmatrix}, \quad \Sigma = \begin{bmatrix}\sqrt{3}&0\\0&1\\0&0\end{bmatrix}, \quad V = \frac{1}{\sqrt{2}}\begin{bmatrix}1&-1\\1&1\end{bmatrix}$$

</details>

---

## Exercise 4.3 — Truncated SVD and Eckart-Young

A matrix $A$ has singular values $\sigma_1 = 10, \sigma_2 = 5, \sigma_3 = 2, \sigma_4 = 0.5$.

**(a)** What is $\|A\|_F$?

**(b)** What is the Frobenius norm error of the best rank-2 approximation $A_2$?

**(c)** What fraction of the "energy" (squared Frobenius norm) is captured by $A_2$?

<details><summary>Solution</summary>

**(a)** $\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \sigma_3^2 + \sigma_4^2} = \sqrt{100 + 25 + 4 + 0.25} = \sqrt{129.25} \approx \mathbf{11.37}$

**(b)** By Eckart-Young: $\|A - A_2\|_F = \sqrt{\sigma_3^2 + \sigma_4^2} = \sqrt{4 + 0.25} = \sqrt{4.25} \approx \mathbf{2.06}$

**(c)** Energy captured = $\frac{\sigma_1^2 + \sigma_2^2}{\sigma_1^2 + \sigma_2^2 + \sigma_3^2 + \sigma_4^2} = \frac{125}{129.25} \approx \mathbf{96.7\%}$

So a rank-2 approximation captures 96.7% of the information with half the components.

</details>

---

## Exercise 4.4 — SVD and Rank

**(a)** If $A \in \mathbb{R}^{100 \times 50}$ has SVD with $\sigma_1 = 10, \sigma_2 = 5, \sigma_3 = 3$, and $\sigma_i = 0$ for $i > 3$, what is $\text{rank}(A)$?

**(b)** How many nonzero singular values does a rank-$r$ matrix have?

**(c)** If you compute SVD numerically and get singular values $[10, 5, 3, 10^{-15}, 10^{-16}]$, what is the "numerical rank"?

<details><summary>Solution</summary>

**(a)** $\text{rank}(A) = 3$ (the number of nonzero singular values).

**(b)** Exactly $r$. The number of nonzero singular values always equals the rank.

**(c)** The numerical rank is **3**. The values $10^{-15}$ and $10^{-16}$ are effectively zero (machine epsilon is about $2.2 \times 10^{-16}$). In practice, singular values below a threshold (e.g., $\epsilon \cdot \sigma_1$) are treated as zero.

</details>

---

## Exercise 4.5 — Pseudoinverse

Using the SVD, compute the pseudoinverse of:

$$A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

Verify $AA^+A = A$.

<details><summary>Solution</summary>

SVD: $U = I_2$, $\Sigma = \begin{bmatrix}1&0\\0&0\end{bmatrix}$, $V = I_2$.

Pseudoinverse: $A^+ = V\Sigma^+ U^\top$

$\Sigma^+ = \begin{bmatrix}1/1&0\\0&0\end{bmatrix} = \begin{bmatrix}1&0\\0&0\end{bmatrix}$ (invert nonzero entries, leave zeros)

$$A^+ = I \begin{bmatrix}1&0\\0&0\end{bmatrix} I = \begin{bmatrix}1&0\\0&0\end{bmatrix}$$

**Verify**: $AA^+A = \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1&0\\0&0\end{bmatrix} = \begin{bmatrix}1&0\\0&0\end{bmatrix} = A$ $\checkmark$

Note: The pseudoinverse satisfies all four Moore-Penrose conditions.

</details>

---

## Exercise 4.6 — USAAIO Competition Style

Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U\Sigma V^\top$.

**(a)** Prove that $\|A\mathbf{x}\| \leq \sigma_1 \|\mathbf{x}\|$ for all $\mathbf{x} \in \mathbb{R}^n$.

**(b)** Prove that equality holds when $\mathbf{x} = \mathbf{v}_1$ (the first right singular vector).

**(c)** Give a geometric interpretation: what does $\sigma_1$ represent?

<details><summary>Solution</summary>

**(a)** Let $\mathbf{x} = \sum_{i} c_i \mathbf{v}_i$ (expansion in right singular vector basis).

Since $V$ is orthogonal: $\|\mathbf{x}\|^2 = \sum_i c_i^2$

$A\mathbf{x} = U\Sigma V^\top \sum_i c_i \mathbf{v}_i = U\Sigma \sum_i c_i \mathbf{e}_i = U \sum_i c_i \sigma_i \mathbf{e}_i = \sum_i c_i \sigma_i \mathbf{u}_i$

Since $U$ is orthogonal: $\|A\mathbf{x}\|^2 = \sum_i c_i^2 \sigma_i^2 \leq \sigma_1^2 \sum_i c_i^2 = \sigma_1^2 \|\mathbf{x}\|^2$

Therefore $\|A\mathbf{x}\| \leq \sigma_1 \|\mathbf{x}\|$. $\blacksquare$

**(b)** For $\mathbf{x} = \mathbf{v}_1$: $c_1 = 1, c_i = 0$ for $i > 1$.

$\|A\mathbf{v}_1\|^2 = \sigma_1^2 = \sigma_1^2 \|\mathbf{v}_1\|^2$

Equality holds. $\blacksquare$

**(c)** $\sigma_1$ is the **maximum stretching factor** of $A$: the largest factor by which $A$ can increase the length of a vector. It equals the spectral norm $\|A\|_2 = \sigma_1$. Geometrically, $A$ maps the unit sphere to an ellipsoid whose semi-axes have lengths $\sigma_1 \geq \sigma_2 \geq \cdots$

</details>

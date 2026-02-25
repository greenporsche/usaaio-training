# Exercises: Matrix Operations

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 2.1 — Matrix Multiplication Views

Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$.

**(a)** Compute $AB$ using the row-column dot product view.

**(b)** Compute $AB$ using the sum of outer products view.

**(c)** Verify that both give the same result.

<details><summary>Solution</summary>

**(a)** Row-column dot products:

$(AB)_{11} = [1\ 2] \cdot [5\ 7]^\top = 5 + 14 = 19$
$(AB)_{12} = [1\ 2] \cdot [6\ 8]^\top = 6 + 16 = 22$
$(AB)_{21} = [3\ 4] \cdot [5\ 7]^\top = 15 + 28 = 43$
$(AB)_{22} = [3\ 4] \cdot [6\ 8]^\top = 18 + 32 = 50$

$$AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**(b)** Sum of outer products: $AB = \mathbf{a}_1 \mathbf{b}_1^\top + \mathbf{a}_2 \mathbf{b}_2^\top$

$$= \begin{bmatrix}1\\3\end{bmatrix}\begin{bmatrix}5&6\end{bmatrix} + \begin{bmatrix}2\\4\end{bmatrix}\begin{bmatrix}7&8\end{bmatrix} = \begin{bmatrix}5&6\\15&18\end{bmatrix} + \begin{bmatrix}14&16\\28&32\end{bmatrix} = \begin{bmatrix}19&22\\43&50\end{bmatrix}$$

**(c)** Both give $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ $\checkmark$

</details>

---

## Exercise 2.2 — Identify the Error

A student claims: "Since $AB = BA$ for real numbers, we also have $AB = BA$ for matrices." They write:

$$A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$$

$$AB = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad BA = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$

"Oops, $AB \neq BA$. So my claim was wrong."

**Question**: Is the student correct that $AB = BA$ fails? Give a general condition under which $AB = BA$ does hold.

<details><summary>Solution</summary>

**Yes**, the student's computation is correct: $AB \neq BA$ for these matrices.

**Conditions under which $AB = BA$** (any one suffices):
- $A$ and $B$ are both diagonal matrices
- $A$ and $B$ are simultaneously diagonalizable (share the same eigenvectors): $A = Q\Lambda_A Q^{-1}$, $B = Q\Lambda_B Q^{-1}$, then $AB = Q\Lambda_A\Lambda_B Q^{-1} = Q\Lambda_B\Lambda_A Q^{-1} = BA$
- One of them is a scalar multiple of the identity: $A = cI \implies AB = cB = BA$
- $A$ and $B$ are polynomials of the same matrix: if $A = p(M)$, $B = q(M)$

In general, commutativity is the exception, not the rule.

</details>

---

## Exercise 2.3 — Trace Properties

Prove that $\text{tr}(AB) = \text{tr}(BA)$ for any $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times m}$.

<details><summary>Solution</summary>

$$\text{tr}(AB) = \sum_{i=1}^{m} (AB)_{ii} = \sum_{i=1}^{m} \sum_{k=1}^{n} A_{ik} B_{ki}$$

$$\text{tr}(BA) = \sum_{k=1}^{n} (BA)_{kk} = \sum_{k=1}^{n} \sum_{i=1}^{m} B_{ki} A_{ik}$$

These are the same double sum (just with the order of summation swapped), so $\text{tr}(AB) = \text{tr}(BA)$. $\blacksquare$

Note: $AB$ is $m \times m$ and $BA$ is $n \times n$, so they may have different sizes, but their traces are equal.

</details>

---

## Exercise 2.4 — Determinant Computation

Compute the determinant of:

$$A = \begin{bmatrix} 2 & 1 & 3 \\ 0 & -1 & 2 \\ 4 & 1 & 8 \end{bmatrix}$$

<details><summary>Solution</summary>

Cofactor expansion along row 1:

$$\det(A) = 2 \det\begin{bmatrix}-1&2\\1&8\end{bmatrix} - 1 \det\begin{bmatrix}0&2\\4&8\end{bmatrix} + 3\det\begin{bmatrix}0&-1\\4&1\end{bmatrix}$$

$$= 2(-8-2) - 1(0-8) + 3(0-(-4))$$

$$= 2(-10) - 1(-8) + 3(4) = -20 + 8 + 12 = \mathbf{0}$$

Since $\det(A) = 0$, the matrix is **singular** (not invertible). This means its columns are linearly dependent.

</details>

---

## Exercise 2.5 — Inverse and Properties

Let $A = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}$.

**(a)** Compute $A^{-1}$ by hand.

**(b)** Verify $AA^{-1} = I$.

**(c)** Compute $(A^2)^{-1}$ and verify it equals $(A^{-1})^2$.

<details><summary>Solution</summary>

**(a)** Using the $2 \times 2$ formula: $A^{-1} = \frac{1}{ad - bc}\begin{bmatrix}d & -b \\ -c & a\end{bmatrix}$

$\det(A) = 2(3) - 1(5) = 1$

$$A^{-1} = \frac{1}{1}\begin{bmatrix}3&-1\\-5&2\end{bmatrix} = \begin{bmatrix}3&-1\\-5&2\end{bmatrix}$$

**(b)** $AA^{-1} = \begin{bmatrix}2&1\\5&3\end{bmatrix}\begin{bmatrix}3&-1\\-5&2\end{bmatrix} = \begin{bmatrix}6-5&-2+2\\15-15&-5+6\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix} = I$ $\checkmark$

**(c)** $A^2 = \begin{bmatrix}2&1\\5&3\end{bmatrix}\begin{bmatrix}2&1\\5&3\end{bmatrix} = \begin{bmatrix}9&5\\25&14\end{bmatrix}$

$(A^2)^{-1} = \frac{1}{9(14)-5(25)}\begin{bmatrix}14&-5\\-25&9\end{bmatrix} = \frac{1}{1}\begin{bmatrix}14&-5\\-25&9\end{bmatrix}$

$(A^{-1})^2 = \begin{bmatrix}3&-1\\-5&2\end{bmatrix}\begin{bmatrix}3&-1\\-5&2\end{bmatrix} = \begin{bmatrix}9+5&-3-2\\-15-10&5+4\end{bmatrix} = \begin{bmatrix}14&-5\\-25&9\end{bmatrix}$

$(A^2)^{-1} = (A^{-1})^2$ $\checkmark$

</details>

---

## Exercise 2.6 — Positive Definiteness

**(a)** Show that $A^\top A$ is always positive semi-definite for any real matrix $A$.

**(b)** Under what condition on $A$ is $A^\top A$ positive definite (strictly)?

**(c)** Is $B = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$ positive definite? Prove using two different methods.

<details><summary>Solution</summary>

**(a)** For any $\mathbf{x} \in \mathbb{R}^n$:

$$\mathbf{x}^\top (A^\top A) \mathbf{x} = (A\mathbf{x})^\top (A\mathbf{x}) = \|A\mathbf{x}\|^2 \geq 0$$

This holds for all $\mathbf{x}$, so $A^\top A \succeq 0$. $\blacksquare$

**(b)** $A^\top A$ is positive definite iff $\|A\mathbf{x}\|^2 > 0$ for all $\mathbf{x} \neq \mathbf{0}$, which is true iff $A\mathbf{x} \neq \mathbf{0}$ for $\mathbf{x} \neq \mathbf{0}$, i.e., $A$ has **full column rank** ($\text{null}(A) = \{\mathbf{0}\}$).

**(c)** **Method 1: Eigenvalues.** $\det(B - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3) = 0$. Eigenvalues: $\lambda_1 = 1 > 0, \lambda_2 = 3 > 0$. Since all eigenvalues are positive, $B$ is PD. $\checkmark$

**Method 2: Leading principal minors (Sylvester's criterion).**
- $B_{11} = 2 > 0$ $\checkmark$
- $\det(B) = 4 - 1 = 3 > 0$ $\checkmark$

All leading principal minors are positive, so $B$ is PD. $\checkmark$

</details>

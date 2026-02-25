# Exercises: Vector Spaces

**Target time**: 2-5 minutes per exercise | **Total**: 7 exercises

---

## Exercise 1.1 — Linear Independence Check

Determine whether the following vectors are linearly independent. Justify your answer.

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}$$

<details><summary>Solution</summary>

**Linearly dependent.** Check: $\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1$:

$$2\begin{bmatrix}4\\5\\6\end{bmatrix} - \begin{bmatrix}1\\2\\3\end{bmatrix} = \begin{bmatrix}8-1\\10-2\\12-3\end{bmatrix} = \begin{bmatrix}7\\8\\9\end{bmatrix} = \mathbf{v}_3 \checkmark$$

Alternatively, form the matrix $M = [\mathbf{v}_1\ \mathbf{v}_2\ \mathbf{v}_3]$ and compute $\det(M) = 1(5 \cdot 9 - 6 \cdot 8) - 2(4 \cdot 9 - 6 \cdot 7) + 3(4 \cdot 8 - 5 \cdot 7) = 1(-3) - 2(-6) + 3(-3) = -3 + 12 - 9 = 0$.

Since $\det(M) = 0$, the columns are linearly dependent.

</details>

---

## Exercise 1.2 — Subspace Verification

Is the set $W = \{(x, y, z) \in \mathbb{R}^3 : x + 2y - z = 0\}$ a subspace of $\mathbb{R}^3$? Prove or give a counterexample.

<details><summary>Solution</summary>

**Yes, $W$ is a subspace.** We verify the three conditions:

1. **Zero vector**: $0 + 2(0) - 0 = 0$ $\checkmark$

2. **Closed under addition**: If $\mathbf{u} = (u_1, u_2, u_3)$ and $\mathbf{v} = (v_1, v_2, v_3)$ satisfy $u_1 + 2u_2 - u_3 = 0$ and $v_1 + 2v_2 - v_3 = 0$, then:
   $(u_1+v_1) + 2(u_2+v_2) - (u_3+v_3) = (u_1+2u_2-u_3) + (v_1+2v_2-v_3) = 0 + 0 = 0$ $\checkmark$

3. **Closed under scalar multiplication**: For $c \in \mathbb{R}$:
   $cu_1 + 2(cu_2) - cu_3 = c(u_1 + 2u_2 - u_3) = c \cdot 0 = 0$ $\checkmark$

$W$ is the null space of the matrix $[1\ 2\ -1]$, which is always a subspace.

</details>

---

## Exercise 1.3 — Dimension and Basis

Find a basis for the column space of:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \\ 3 & 6 & 10 \end{bmatrix}$$

What is $\text{rank}(A)$?

<details><summary>Solution</summary>

Row reduce $A$:

$$\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \\ 3 & 6 & 10 \end{bmatrix} \xrightarrow{R_2 - 2R_1, R_3 - 3R_1} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 1 \\ 0 & 0 & 1 \end{bmatrix} \xrightarrow{R_3 - R_2} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

Pivot columns are columns 1 and 3. A basis for the column space is:

$$\left\{\begin{bmatrix}1\\2\\3\end{bmatrix}, \begin{bmatrix}3\\7\\10\end{bmatrix}\right\}$$

$\text{rank}(A) = 2$.

</details>

---

## Exercise 1.4 — Gram-Schmidt

Apply the Gram-Schmidt process to:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

<details><summary>Solution</summary>

**Step 1**: $\mathbf{u}_1 = \mathbf{v}_1 = \begin{bmatrix}1\\1\\0\end{bmatrix}$, $\mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|} = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\\0\end{bmatrix}$

**Step 2**: $\mathbf{u}_2 = \mathbf{v}_2 - \langle\mathbf{v}_2, \mathbf{e}_1\rangle \mathbf{e}_1$

$\langle\mathbf{v}_2, \mathbf{e}_1\rangle = \frac{1}{\sqrt{2}}(1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0) = \frac{1}{\sqrt{2}}$

$\mathbf{u}_2 = \begin{bmatrix}1\\0\\1\end{bmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\\0\end{bmatrix} = \begin{bmatrix}1\\0\\1\end{bmatrix} - \begin{bmatrix}1/2\\1/2\\0\end{bmatrix} = \begin{bmatrix}1/2\\-1/2\\1\end{bmatrix}$

$\|\mathbf{u}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}$

$\mathbf{e}_2 = \frac{1}{\sqrt{3/2}}\begin{bmatrix}1/2\\-1/2\\1\end{bmatrix} = \frac{1}{\sqrt{6}}\begin{bmatrix}1\\-1\\2\end{bmatrix}$

**Verification**: $\mathbf{e}_1 \cdot \mathbf{e}_2 = \frac{1}{\sqrt{2}\sqrt{6}}(1 - 1 + 0) = 0$ $\checkmark$

</details>

---

## Exercise 1.5 — True/False with Justification

For each statement, determine if it's true or false. Justify your answer.

**(a)** If $\mathbf{u} \cdot \mathbf{v} = 0$, then $\mathbf{u} = \mathbf{0}$ or $\mathbf{v} = \mathbf{0}$.

**(b)** The set of all $2 \times 2$ symmetric matrices forms a vector space (subspace of $\mathbb{R}^{2 \times 2}$).

**(c)** If $\dim(V) = n$, then any set of $n+1$ vectors in $V$ is linearly dependent.

<details><summary>Solution</summary>

**(a) False.** Counterexample: $\mathbf{u} = \begin{bmatrix}1\\0\end{bmatrix}$, $\mathbf{v} = \begin{bmatrix}0\\1\end{bmatrix}$. Both are nonzero but $\mathbf{u} \cdot \mathbf{v} = 0$.

**(b) True.** Check subspace conditions:
- The zero matrix is symmetric $\checkmark$
- Sum of symmetric matrices is symmetric: $(A+B)^\top = A^\top + B^\top = A + B$ $\checkmark$
- Scalar multiple of symmetric is symmetric: $(cA)^\top = cA^\top = cA$ $\checkmark$

This subspace has dimension 3 (basis: $\begin{bmatrix}1&0\\0&0\end{bmatrix}, \begin{bmatrix}0&1\\1&0\end{bmatrix}, \begin{bmatrix}0&0\\0&1\end{bmatrix}$).

**(c) True.** This is a fundamental theorem of linear algebra. If $\dim(V) = n$, every basis has $n$ elements, and any set of more than $n$ vectors must be linearly dependent (since no set of $n+1$ vectors can all be "in different directions" in an $n$-dimensional space).

</details>

---

## Exercise 1.6 — Fill in the Shape

A matrix $A \in \mathbb{R}^{5 \times 3}$ has $\text{rank}(A) = 3$. Fill in:

**(a)** Dimension of column space = ____

**(b)** Dimension of null space = ____

**(c)** Dimension of row space = ____

**(d)** $A^\top A$ has shape ____ and rank ____

**(e)** $AA^\top$ has shape ____ and rank ____

<details><summary>Solution</summary>

**(a)** $\dim(\text{col}(A)) = \text{rank}(A) = \mathbf{3}$

**(b)** $\dim(\text{null}(A)) = n - \text{rank}(A) = 3 - 3 = \mathbf{0}$ (only the zero vector)

**(c)** $\dim(\text{row}(A)) = \text{rank}(A) = \mathbf{3}$

**(d)** $A^\top A \in \mathbb{R}^{\mathbf{3 \times 3}}$, $\text{rank}(A^\top A) = \text{rank}(A) = \mathbf{3}$ (full rank, invertible)

**(e)** $AA^\top \in \mathbb{R}^{\mathbf{5 \times 5}}$, $\text{rank}(AA^\top) = \text{rank}(A) = \mathbf{3}$

</details>

---

## Exercise 1.7 — USAAIO Competition Style

Let $V$ be the set of all polynomials of degree at most 2, i.e., $V = \{a_0 + a_1x + a_2x^2 : a_i \in \mathbb{R}\}$.

**(a)** Show that $V$ is a vector space under standard polynomial addition and scalar multiplication.

**(b)** Give a basis for $V$ and state its dimension.

**(c)** Is $W = \{p \in V : p(1) = 0\}$ a subspace of $V$? If yes, find its dimension.

<details><summary>Solution</summary>

**(a)** $V$ satisfies all 8 vector space axioms:
- The zero polynomial $0 + 0x + 0x^2$ is in $V$ (additive identity)
- Sum of two degree-$\leq 2$ polynomials has degree $\leq 2$ (closure under addition)
- Scalar multiple of degree-$\leq 2$ polynomial has degree $\leq 2$ (closure under scalar mult.)
- All other axioms (commutativity, associativity, etc.) are inherited from properties of real-valued functions.

**(b)** Standard basis: $\{1, x, x^2\}$. Dimension = 3.

**(c)** **Yes, $W$ is a subspace.**
- Zero polynomial: $p(x) = 0$ satisfies $p(1) = 0$ $\checkmark$
- Closure: if $p(1) = 0$ and $q(1) = 0$, then $(p+q)(1) = 0$ $\checkmark$
- Scalar closure: if $p(1) = 0$, then $(cp)(1) = 0$ $\checkmark$

To find the dimension: $p(x) = a_0 + a_1 x + a_2 x^2$ with $p(1) = a_0 + a_1 + a_2 = 0$, so $a_0 = -a_1 - a_2$. Thus:

$$p(x) = (-a_1 - a_2) + a_1 x + a_2 x^2 = a_1(x - 1) + a_2(x^2 - 1)$$

Basis for $W$: $\{x - 1, x^2 - 1\}$. **Dimension = 2.**

</details>

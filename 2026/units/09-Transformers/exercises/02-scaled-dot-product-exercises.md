# Exercises 02 — Scaled Dot-Product Attention

> 5 competition-level exercises

---

## Exercise 2.1 — Variance Derivation

Let $q, k \in \mathbb{R}^{d_k}$ with entries drawn i.i.d. from $\mathcal{N}(0, \sigma^2)$.

**(a)** Compute $\mathbb{E}[q \cdot k]$.

**(b)** Compute $\text{Var}(q \cdot k)$.

**(c)** What is $\text{Var}(q \cdot k / \sqrt{d_k})$?

**(d)** Now suppose $q_i \sim \mathcal{N}(0, \sigma_q^2)$ and $k_i \sim \mathcal{N}(0, \sigma_k^2)$ independently. Compute $\text{Var}(q \cdot k)$.

**(e)** What scaling factor would you use to normalize the variance to 1 in part (d)?

---

## Exercise 2.2 — Numerical Computation

Given $Q = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V = \begin{pmatrix} 10 & 0 \\ 0 & 10 \end{pmatrix}$:

**(a)** Compute $QK^T$.

**(b)** Compute $QK^T / \sqrt{d_k}$ where $d_k = 2$.

**(c)** Apply row-wise softmax. Show your work.

**(d)** Compute the final output.

**(e)** Now compute the output WITHOUT scaling (use $QK^T$ directly in softmax). Compare with (d). Which output has more "peaked" attention?

---

## Exercise 2.3 — Softmax Properties

**(a)** Prove that softmax is translation-invariant: $\text{softmax}(z + c\mathbf{1}) = \text{softmax}(z)$ for any scalar $c$.

**(b)** Prove that the Jacobian of softmax is: $\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i(\delta_{ij} - \text{softmax}(z)_j)$.

**(c)** Show that when $\text{softmax}(z)$ is nearly one-hot (one entry close to 1), ALL partial derivatives are close to 0.

**(d)** Compute the maximum possible value of $\frac{\partial \text{softmax}(z)_i}{\partial z_i}$ for a softmax over $n$ elements. At what input does this maximum occur?

---

## Exercise 2.4 — Masking Mechanics

Consider a causal mask for $L = 4$:

$$M = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{pmatrix}$$

Given unscaled scores $S = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 2 & 1 & 2 & 3 \\ 1 & 1 & 1 & 1 \\ 3 & 0 & 1 & 2 \end{pmatrix}$:

**(a)** Apply the causal mask (set masked positions to $-\infty$).

**(b)** Compute softmax for row 2 (position 2, which can see positions 1-2 only).

**(c)** Compute softmax for row 4 (position 4, which can see all positions).

**(d)** What would happen if we masked AFTER softmax instead of before? Why is this incorrect?

---

## Exercise 2.5 — Attention Complexity Analysis

Consider standard scaled dot-product attention with $L$ tokens, $D_{qk}$ key dimension, and $D_v$ value dimension.

**(a)** What is the time complexity of computing $QK^T$?

**(b)** What is the space complexity of storing the attention matrix?

**(c)** If we double the sequence length $L$, by what factor does the computation increase?

**(d)** For $L = 8192$, $D_{qk} = 128$, how many floating-point multiplications are needed for $QK^T$? Express in billions.

**(e)** [Bonus] Linear attention replaces $\text{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^T V)$ where the parenthesization changes the computation order. What is the complexity of this formulation? Why is the order of operations important?

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

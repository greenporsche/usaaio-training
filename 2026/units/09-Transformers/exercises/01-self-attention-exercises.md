# Exercises 01 — Self-Attention

> 5 competition-level exercises

---

## Exercise 1.1 — Shape Derivation (Warm-up)

Given an input sequence $X \in \mathbb{R}^{10 \times 64}$ with projection dimensions $D_{qk} = 32$ and $D_v = 48$:

**(a)** Write the shapes of $W^Q$, $W^K$, $W^V$.

**(b)** Write the shape of the attention matrix $\alpha$.

**(c)** Write the shape of the output.

**(d)** How many learnable parameters are there in total (no bias)?

---

## Exercise 1.2 — Hand Computation

Given:

$$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad W^Q = W^K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad W^V = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

**(a)** Compute $Q$, $K$, $V$.

**(b)** Compute the raw attention scores $QK^T$ (without scaling).

**(c)** Compute the scaled scores $QK^T / \sqrt{D_{qk}}$.

**(d)** Apply softmax row-wise to get attention weights $\alpha$.

**(e)** Compute the output $\alpha V$.

**(f)** Which token does token 3 attend to most strongly? Explain why.

---

## Exercise 1.3 — Permutation Equivariance Proof

**Prove** that single-head self-attention (without positional encoding) is permutation equivariant. That is, if $P$ is a permutation matrix and $f(X) = \text{softmax}(XW^Q (XW^K)^T / \sqrt{d_k}) \cdot XW^V$, then:

$$f(PX) = P \cdot f(X)$$

*Hint*: Use the fact that $P^T P = I$ for permutation matrices.

---

## Exercise 1.4 — Attention as Kernel Smoothing

The softmax attention can be viewed as a form of kernel smoothing:

$$\text{output}_i = \sum_j \frac{K(q_i, k_j)}{\sum_{j'} K(q_i, k_{j'})} v_j$$

where $K(q, k) = \exp(q \cdot k / \sqrt{d_k})$ is the kernel function.

**(a)** What happens when the temperature (scaling factor) $\tau$ in $K(q, k) = \exp(q \cdot k / \tau)$ approaches 0? Describe the behavior of the attention weights.

**(b)** What happens when $\tau \to \infty$?

**(c)** What is the effect on gradients in each extreme case?

**(d)** Why is $\tau = \sqrt{d_k}$ a good choice? (Reference the variance derivation.)

---

## Exercise 1.5 — Self-Attention vs. Fully Connected Layer

Consider single-head self-attention with $D_{qk} = D_v = D$ and no output projection.

**(a)** Show that if the attention weights $\alpha$ are fixed (not input-dependent), self-attention reduces to a specific linear operation on $V = XW^V$. What operation?

**(b)** A fully connected layer applied across the sequence dimension would compute $Y = AXW$ for a learnable matrix $A \in \mathbb{R}^{L \times L}$. How does this differ from self-attention?

**(c)** What is the key advantage of self-attention over a fixed linear mixing of positions? Consider what happens when the sequence length changes.

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

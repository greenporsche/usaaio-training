# Exercises 07 — Grouped Query Attention

> 5 competition-level exercises

---

## Exercise 7.1 — GQA Shape Derivation

Given $D = 1024, H = 16, G = 4, D_{qk} = 64, D_v = 64$:

**(a)** How many query heads per group?

**(b)** What are the shapes of $W^Q, W^K, W^V, W^O$? Compare with MHA.

**(c)** After reshaping Q for grouped computation, what is its shape? (Use the 5D format.)

**(d)** After reshaping K with unsqueeze, what is its shape?

**(e)** What is the shape of the attention logits tensor? How does broadcasting work here?

**(f)** Trace shapes through the entire merge operation back to $(B, L, D)$.

---

## Exercise 7.2 — Rank of Repeated Matrix (Proof)

Let $A \in \mathbb{R}^{m \times n}$ and define $\tilde{A} = [A \;|\; A \;|\; \cdots \;|\; A] \in \mathbb{R}^{m \times kn}$ (A repeated $k$ times horizontally).

**(a)** Prove that $\text{rank}(\tilde{A}) = \text{rank}(A)$.

**(b)** Now let $\tilde{A}$ be formed by repeating $A$ vertically: $\tilde{A} = \begin{pmatrix} A \\ A \\ \vdots \\ A \end{pmatrix} \in \mathbb{R}^{km \times n}$. Prove that $\text{rank}(\tilde{A}) = \text{rank}(A)$.

**(c)** In GQA, if $W^K_g \in \mathbb{R}^{D \times D_{qk}}$ with $\text{rank}(W^K_g) = D_{qk}$ (full column rank), what is the rank of the "MHA-equivalent" matrix formed by repeating $W^K_g$ for all $H/G$ heads in the group?

**(d)** Compare with MHA where $W^K = [W^K_1 | \cdots | W^K_H]$ with independently random $W^K_h$. What is the expected rank of $W^K$?

**(e)** What does this rank difference imply about the expressiveness of GQA vs. MHA?

---

## Exercise 7.3 — MHA as Special Case of GQA (Proof)

**(a)** State the GQA attention formula with $G$ groups and $H/G$ heads per group.

**(b)** Substitute $G = H$ and simplify. Show that every term matches standard MHA.

**(c)** Conversely, substitute $G = 1$ and show that this yields Multi-Query Attention.

**(d)** For $H = 32$ and $D_{qk} = 128$, compute the KV-cache per position for $G = 32$ (MHA), $G = 8$, $G = 4$, $G = 1$ (MQA). Express in bytes (FP16).

**(e)** If a model has 80 layers and generates sequences of length 4096, compute the total KV-cache for each value of $G$ in part (d). Express in GB.

---

## Exercise 7.4 — Broadcasting Mechanics

Consider GQA with $B = 2, G = 2, H = 8$ (so $H/G = 4$ heads per group), $L = 3, D_{qk} = 2$.

Q has shape $(2, 2, 4, 3, 2)$ — `(B, G, H//G, L, D_qk)`.
K has shape $(2, 2, 1, 3, 2)$ — `(B, G, 1, L, D_qk)`.

**(a)** When computing `Q @ K.mT`, explain exactly how broadcasting works. Which dimension is broadcast?

**(b)** What is the shape of the result?

**(c)** Write a concrete small example: create Q and K tensors in PyTorch and verify the shape.

**(d)** Is `K.unsqueeze(2)` equivalent to `K.expand(-1, -1, 4, -1, -1)` for the purposes of the matrix multiplication? Explain.

**(e)** What is the memory difference between using `unsqueeze` (broadcasting) vs. `expand` + `clone` (explicit repetition)?

---

## Exercise 7.5 — GQA Design Trade-offs

You are deploying a model with $D = 4096, H = 32, D_{qk} = 128$ and need to generate 8192-token sequences with batch size 64.

**(a)** Compute the KV-cache for MHA ($G=32$) for one layer. Express in MB (FP16).

**(b)** Your GPU has 80 GB memory, and the model weights take 40 GB. For 40 layers, what is the maximum batch size for MHA?

**(c)** Repeat part (b) for GQA with $G = 8$.

**(d)** Repeat part (b) for GQA with $G = 1$ (MQA).

**(e)** Research shows that GQA with $G = 8$ has negligible quality degradation compared to MHA for this model size. What is the "efficiency-quality frontier" argument for choosing $G$?

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

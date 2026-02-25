# Exercises 03 — Multi-Head Attention

> 5 competition-level exercises

---

## Exercise 3.1 — Projection Matrix Shapes

A multi-head attention module has:
- Attending sequence dimension $D_1 = 512$
- Being attended sequence dimension $D_2 = 768$
- $H = 8$ heads, $D_{qk} = 64$, $D_v = 64$

**(a)** What is the shape of the per-head query projection $W^Q_h$?

**(b)** What is the shape of the concatenated key projection $W^K$?

**(c)** What is the shape of the output projection $W^O$?

**(d)** Compute the total number of parameters (no bias).

**(e)** If we change to $D_{qk} = 128, D_v = 32$, what changes? List all affected matrix shapes.

---

## Exercise 3.2 — Tensor Shape Tracing

Given $B = 4, L_1 = 12, L_2 = 20, D_1 = 256, D_2 = 256, H = 4, D_{qk} = 64, D_v = 48$.

Trace the shape of the Q tensor through every operation:

**(a)** After `self.W_Q(X1)` (linear projection)

**(b)** After `.reshape(B, L1, H, D_qk)`

**(c)** After `.permute(0, 2, 1, 3)`

**(d)** What is the shape of `Q @ K.mT`? Show the matrix multiplication dimensions.

**(e)** After `alpha @ V`, what is the output shape? Trace through the merge operation back to `(B, L1, D1)`.

---

## Exercise 3.3 — Equivalence of Loop and Vectorized MHA

Consider two implementations:

**Implementation A (loop)**:
```python
heads = []
for h in range(H):
    Q_h = X1 @ W_Q_list[h]   # (B, L1, D_qk)
    K_h = X2 @ W_K_list[h]   # (B, L2, D_qk)
    V_h = X2 @ W_V_list[h]   # (B, L2, D_v)
    attn_h = softmax(Q_h @ K_h.mT / sqrt(D_qk)) @ V_h
    heads.append(attn_h)
out = torch.cat(heads, dim=-1) @ W_O
```

**Implementation B (vectorized)**:
```python
Q = X1 @ W_Q   # W_Q: (D1, H*D_qk)
Q = Q.reshape(B, L1, H, D_qk).permute(0, 2, 1, 3)
# ... (standard MHA)
```

**(a)** Prove that if $W^Q = [W^Q_1 | W^Q_2 | \cdots | W^Q_H]$, then the first $D_{qk}$ columns of $XW^Q$ reshaped by head are identical to $XW^Q_1$.

**(b)** Why is Implementation B preferred over A in practice?

**(c)** Implementation A uses `torch.cat(heads, dim=-1)`. What operation in Implementation B corresponds to this?

---

## Exercise 3.4 — Attention Pattern Analysis

Consider a 4-token sequence with $H = 2$ heads, $D_{qk} = 2$.

Head 1 attention weights:
$$\alpha_1 = \begin{pmatrix} 0.9 & 0.03 & 0.03 & 0.04 \\ 0.02 & 0.9 & 0.04 & 0.04 \\ 0.03 & 0.02 & 0.9 & 0.05 \\ 0.04 & 0.03 & 0.03 & 0.9 \end{pmatrix}$$

Head 2 attention weights:
$$\alpha_2 = \begin{pmatrix} 0.25 & 0.25 & 0.25 & 0.25 \\ 0.7 & 0.1 & 0.1 & 0.1 \\ 0.1 & 0.7 & 0.1 & 0.1 \\ 0.1 & 0.1 & 0.7 & 0.1 \end{pmatrix}$$

**(a)** What pattern does head 1 exhibit? What relationship might it be capturing?

**(b)** What pattern does head 2 exhibit? What relationship might it be capturing?

**(c)** Why is it beneficial for different heads to learn different patterns?

**(d)** If we only had one head, could it simultaneously capture both patterns? Explain.

---

## Exercise 3.5 — Counting and Complexity

For a transformer model with:
- $D = 1024, H = 16, D_{qk} = D_v = 64$
- Sequence length $L = 2048$
- Batch size $B = 8$

**(a)** Compute the total number of parameters in one MHA layer (4 weight matrices, no bias).

**(b)** Compute the total FLOPs for one forward pass of MHA. Count: Q/K/V projections, attention score computation, weighted sum, output projection.

**(c)** What is the peak memory usage for the attention matrix during the forward pass? Express in MB (assuming FP32).

**(d)** If we double $H$ while keeping $D$ fixed (so $D_{qk} = D_v = D/(2H) = 32$), how do parameters, FLOPs, and attention matrix memory change?

**(e)** The standard transformer uses $D_{qk} = D_v = D/H$. Suggest one reason this might not be optimal, and what alternative configurations could be explored.

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

# Exercises 05 — Positional Encoding

> 5 competition-level exercises

---

## Exercise 5.1 — Sinusoidal Computation

For model dimension $d = 8$:

**(a)** Compute the frequencies $\omega_i = 1/10000^{2i/d}$ for $i = 0, 1, 2, 3$.

**(b)** Compute $PE_{(0, :)}$ (positional encoding for position 0).

**(c)** Compute $PE_{(1, :)}$ (positional encoding for position 1).

**(d)** Compute the dot product $PE_{(0, :)} \cdot PE_{(1, :)}$. Is it close to $PE_{(0, :)} \cdot PE_{(0, :)}$?

**(e)** Compute $PE_{(100, :)} \cdot PE_{(101, :)}$ and $PE_{(100, :)} \cdot PE_{(200, :)}$. What do you observe about the relationship between dot product and distance?

---

## Exercise 5.2 — Rotation Matrix Property

For a single frequency $\omega$, the sinusoidal encoding at position $p$ is:

$$e_p = \begin{pmatrix} \sin(\omega p) \\ \cos(\omega p) \end{pmatrix}$$

**(a)** Write the $2 \times 2$ rotation matrix $R_k$ such that $e_{p+k} = R_k \cdot e_p$.

**(b)** Verify that $R_k$ is an orthogonal matrix (i.e., $R_k^T R_k = I$).

**(c)** Show that $R_{k_1} R_{k_2} = R_{k_1 + k_2}$ (rotation matrices compose by adding offsets).

**(d)** Explain why this property is useful for a transformer learning to attend to relative positions.

**(e)** For the full $d$-dimensional encoding (with $d/2$ frequencies), write the block-diagonal rotation matrix that transforms $PE_{pos}$ to $PE_{pos+k}$.

---

## Exercise 5.3 — Why Not Simpler Encodings?

Consider these alternative positional encodings:

**Option A**: $PE_{pos} = pos / L$ (normalized scalar)
**Option B**: $PE_{pos} = \text{one-hot}(pos)$ (one-hot vector of length $L$)
**Option C**: Binary encoding of $pos$

**(a)** What is the problem with Option A? (Consider what happens when the model sees different sequence lengths.)

**(b)** What is the problem with Option B? (Consider parameter efficiency and generalization.)

**(c)** What is the advantage of sinusoidal encoding over binary encoding?

**(d)** Binary encoding also uses different frequencies (powers of 2). How does it differ from sinusoidal encoding in terms of smoothness?

---

## Exercise 5.4 — Positional Encoding Dimensionality

**(a)** For sinusoidal encoding with $d = 512$, how many distinct frequencies are used?

**(b)** The longest wavelength is $2\pi \cdot 10000 \approx 62{,}832$ positions. The shortest is $2\pi \approx 6.28$ positions. If your sequence has $L = 4096$ tokens, which frequency dimensions effectively vary across positions, and which are nearly constant?

**(c)** If you needed to handle sequences of length $L = 100{,}000$, would standard sinusoidal encoding still work? What might go wrong?

**(d)** Propose a modification to extend sinusoidal encoding to longer sequences. (Hint: Change the base.)

---

## Exercise 5.5 — Impact on Attention

Consider two tokens at positions $p$ and $q$ with token embeddings $t_p$ and $t_q$.

After adding positional encoding: $x_p = t_p + PE_p$ and $x_q = t_q + PE_q$.

**(a)** Expand the dot product $x_p \cdot x_q$ and identify the four terms. Which terms depend on position, which on content, and which on both?

**(b)** The term $PE_p \cdot PE_q$ contributes a position-dependent bias to the attention score. Show that for sinusoidal encodings, this bias depends only on $|p - q|$ (the distance), not on the absolute positions.

**(c)** Why is this distance-dependent bias potentially useful? Give an example from natural language.

**(d)** The cross term $t_p \cdot PE_q + PE_p \cdot t_q$ mixes content and position. Is this desirable? When might it be problematic?

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

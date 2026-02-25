# Exercises 08 — Multi-Head Latent Attention & KV-Cache

> 5 competition-level exercises

---

## Exercise 8.1 — MLA Decomposition Shapes

Given $D = 512, H = 8, D_{qk} = 64, D_v = 64, r = 32$:

**(a)** What are the shapes of $W^{DKV}, W^{UK}_h, W^{UV}_h$?

**(b)** Verify that $W^K_h = W^{DKV} W^{UK}_h$ has the correct shape.

**(c)** What is the shape of the compressed representation $C = XW^{DKV}$?

**(d)** What is the shape of the reduced query $\hat{W}^Q_h = W^Q_h (W^{UK}_h)^T$?

**(e)** Compare the total parameter count for MLA vs. MHA (just the K/V projections, ignoring Q and O).

---

## Exercise 8.2 — SVD Proof: GQA $\subseteq$ MLA

Consider GQA with $G = 2$ groups, $D = 4$, $D_{qk} = 2$, $D_v = 2$.

Group 1: $W^K_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$, $W^V_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$

Group 2: $W^K_2 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$, $W^V_2 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$

**(a)** Form the stacked matrix $W_{\text{all}} = [W^K_1 \;|\; W^K_2 \;|\; W^V_1 \;|\; W^V_2]$. What is its shape?

**(b)** What is the rank of $W_{\text{all}}$?

**(c)** Compute (or describe how to compute) the SVD $W_{\text{all}} = U\Sigma V^T$.

**(d)** Set $W^{DKV} = U\Sigma$ (taking rank $r =$ rank of $W_{\text{all}}$). What is $r$?

**(e)** Extract $W^{UK}_1$ and $W^{UK}_2$ from $V^T$ and verify $W^{DKV} W^{UK}_g = W^K_g$.

---

## Exercise 8.3 — Counterexample: GQA $\subsetneq$ MLA

**(a)** Construct an MLA configuration with $H = 4, D = 4, D_{qk} = 2, r = 4$ such that all 4 heads have DISTINCT key matrices $W^K_h$.

**(b)** Show that no GQA with $G = 1$ can represent this (since all heads would share one $W^K$).

**(c)** Show that no GQA with $G = 2$ can represent this (since heads within a group share $W^K_g$, giving at most 2 distinct matrices).

**(d)** The only GQA that works is $G = 4 = H$ (MHA). But compare the cache: MHA stores $2 \times 4 \times 2 = 16$ values per position, while this MLA stores $r = 4$ values. What is the compression ratio?

**(e)** Generalize: for $H$ heads with all distinct key matrices, what is the minimum GQA cache? What is the MLA cache? Derive when MLA is strictly more efficient.

---

## Exercise 8.4 — Reduced Matrices Derivation

Starting from the MLA attention formula:

$$\text{logits}_h = \frac{(X_1 W^Q_h)(X_2 W^{DKV} W^{UK}_h)^T}{\sqrt{D_{qk}}}$$

**(a)** Expand $(X_2 W^{DKV} W^{UK}_h)^T$ using the transpose of a product.

**(b)** Rearrange to get $\text{logits}_h = \frac{X_1 \hat{W}^Q_h (X_2 W^{DKV})^T}{\sqrt{D_{qk}}}$ where $\hat{W}^Q_h = W^Q_h (W^{UK}_h)^T$.

**(c)** What is the shape of $\hat{W}^Q_h$? Compare with $W^Q_h$.

**(d)** Define $C = X_2 W^{DKV} \in \mathbb{R}^{L_2 \times r}$. Show that $\text{logits}_h = \frac{(X_1 \hat{W}^Q_h) C^T}{\sqrt{D_{qk}}}$.

**(e)** Similarly derive the reduced output computation. If $O_h = \alpha_h (C W^{UV}_h) \in \mathbb{R}^{L_1 \times D_v}$ and the contribution of head $h$ to the final output is $O_h W^O_h$ where $W^O_h \in \mathbb{R}^{D_v \times D}$, show that this can be written as $\alpha_h C \hat{W}^O_h$ where $\hat{W}^O_h = W^{UV}_h W^O_h \in \mathbb{R}^{r \times D}$.

---

## Exercise 8.5 — KV-Cache Comprehensive Analysis

Consider three architectures, all with $D = 8192, H = 64, D_{qk} = D_v = 128$:

**Architecture A**: MHA ($G = H = 64$)
**Architecture B**: GQA ($G = 8$)
**Architecture C**: MLA ($r = 512$)

All have 80 layers. Assume FP16 (2 bytes per value).

**(a)** Compute the KV-cache per position per layer for each architecture.

**(b)** For a sequence of length $L = 128{,}000$ (128K context), compute the total cache per layer and total across all layers. Express in GB.

**(c)** If the GPU has 80 GB total, model weights take 50 GB, and activations take 10 GB, how many tokens of cache can each architecture support?

**(d)** Architecture C (MLA) uses $r = 512 < 2 \times D_{qk} = 256$ — wait, $2 \times D_{qk} = 256$ and $r = 512 > 256$. In this case, is MLA actually saving memory compared to MQA? What is the break-even point for $r$?

**(e)** DeepSeek-V2 uses $r = 512$ with $H = 128$ and $D_{qk} = 128$. For this configuration, compute the MHA, GQA ($G=1$, i.e., MQA), and MLA cache per position per layer. Which is most efficient?

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

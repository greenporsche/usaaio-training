# Unit 09 — Transformers: Competition Cheat Sheet

> **USAAIO 2026 | AI 500 | The Most Critical Unit for Round 2**
>
> The 2025 Round 2 Problem 2 (100 points, 14 parts) was entirely about MHA, GQA, and MLA.
> Master every formula and tensor shape on this sheet.

---

## Notation Table

| Symbol | Meaning | Typical values |
|--------|---------|---------------|
| $B$ | Batch size | 1–64 |
| $L$, $L_1$, $L_2$ | Sequence length (attending / being attended) | 128–2048 |
| $D$, $D_1$, $D_2$ | Model / hidden dimension | 512–4096 |
| $H$ | Number of attention heads | 8–64 |
| $D_{qk}$ | Per-head query/key dimension | $D/H$ typically |
| $D_v$ | Per-head value dimension | $D/H$ typically |
| $G$ | Number of GQA key/value groups | 1–H |
| $r$ | MLA latent rank | $r \ll D$ |
| $W^Q$, $W^K$, $W^V$ | Concatenated projection matrices | |
| $W^Q_h$, $W^K_h$, $W^V_h$ | Per-head projection matrices | |
| $W^O$ | Output projection matrix | |

---

## 1. Self-Attention (Single Head)

**Intuition**: Query = "what am I looking for?", Key = "what do I contain?", Value = "what do I offer?"

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $Q = XW^Q \in \mathbb{R}^{L \times D_{qk}}$
- $K = XW^K \in \mathbb{R}^{L \times D_{qk}}$
- $V = XW^V \in \mathbb{R}^{L \times D_v}$
- Attention weights: $\alpha = \text{softmax}(QK^T / \sqrt{D_{qk}}) \in \mathbb{R}^{L \times L}$
- Output: $\alpha V \in \mathbb{R}^{L \times D_v}$

**Why scale by $\sqrt{d_k}$?** If $q_i, k_j \sim \mathcal{N}(0,1)$ i.i.d., then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance, preventing softmax saturation.

---

## 2. Multi-Head Attention (MHA)

### Projection Matrices

| Matrix | Shape | Notes |
|--------|-------|-------|
| $W^Q$ | $(D_1, H \cdot D_{qk})$ | Concatenation of $H$ per-head $W^Q_h \in \mathbb{R}^{D_1 \times D_{qk}}$ |
| $W^K$ | $(D_2, H \cdot D_{qk})$ | Concatenation of $H$ per-head $W^K_h$ |
| $W^V$ | $(D_2, H \cdot D_v)$ | Concatenation of $H$ per-head $W^V_h$ |
| $W^O$ | $(H \cdot D_v, D_1)$ | Output projection |

### The Tensor Reshaping Choreography (MUST BE MUSCLE MEMORY)

```python
# Input: x has shape (B, L, D)

# 1. Project
Q = self.W_Q(x)                                              # (B, L, H*D_qk)

# 2. Reshape: split the last dimension into H heads
Q = Q.reshape(B, L, H, D_qk)                                # (B, L, H, D_qk)

# 3. Permute: bring heads before sequence
Q = Q.permute(0, 2, 1, 3)                                    # (B, H, L, D_qk)

# --- same for K, V ---

# 4. Scaled dot-product attention
logits = Q @ K.mT / (D_qk ** 0.5)                           # (B, H, L1, L2)
Alpha = F.softmax(logits, dim=-1)                             # (B, H, L1, L2)

# 5. Weighted sum of values
O = Alpha @ V                                                 # (B, H, L1, D_v)

# 6. Merge heads: reverse the reshape
O = O.permute(0, 2, 1, 3)                                    # (B, L1, H, D_v)
O = O.reshape(B, L1, H * D_v)                                # (B, L1, H*D_v)

# 7. Output projection
O = self.W_O(O)                                               # (B, L1, D1)
```

### Compact Form

$$\text{MultiHead}(X_1, X_2) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$$

where $\text{head}_h = \text{Attention}(X_1 W^Q_h,\; X_2 W^K_h,\; X_2 W^V_h)$

---

## 3. Cross-Attention & Masked Attention

**Cross-attention**: Queries from sequence 1 ($L_1, D_1$), keys/values from sequence 2 ($L_2, D_2$).
- $W^Q: (D_1, H \cdot D_{qk})$, $W^K: (D_2, H \cdot D_{qk})$, $W^V: (D_2, H \cdot D_v)$
- Attention matrix: $(B, H, L_1, L_2)$

**Causal (masked) attention**: Lower-triangular mask for autoregressive decoding.

```python
mask = torch.tril(torch.ones(L, L))                          # (L, L)
logits = logits.masked_fill(mask == 0, float('-inf'))         # before softmax
```

---

## 4. Positional Encoding

**Sinusoidal** (Vaswani et al.):

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**Why needed?** Self-attention is permutation equivariant — without position info, "the cat sat on the mat" and "mat the on sat cat the" produce the same attention weights.

---

## 5. Full Transformer Block

### Encoder Block (Post-Norm)
```
x = x + MultiHeadSelfAttention(x)
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)
```

### Encoder Block (Pre-Norm) — more common in modern architectures
```
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Decoder Block
```
x = x + MaskedSelfAttention(x)          # causal mask
x = LayerNorm(x)
x = x + CrossAttention(x, encoder_out)  # attend to encoder
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)
```

### FFN (Feed-Forward Network)
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Typically $W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$.

---

## 6. Grouped Query Attention (GQA)

**Motivation**: Reduce KV-cache memory. Instead of $H$ independent K/V heads, use $G$ groups. Each group serves $H/G$ query heads.

| Setting | Groups $G$ | Special case |
|---------|-----------|-------------|
| Multi-Head Attention | $G = H$ | Each query head has its own K/V |
| Grouped Query Attention | $1 < G < H$ | K/V shared within groups |
| Multi-Query Attention | $G = 1$ | All query heads share one K/V |

### Broadcasting Pattern (NO LOOPS)

```python
# K has shape (B, G, L, D_qk) — G groups
# Q has shape (B, H, L, D_qk) — H heads

# Reshape Q to expose group structure
Q = Q.reshape(B, G, H // G, L, D_qk)   # (B, G, H//G, L, D_qk)
K = K.unsqueeze(2)                       # (B, G, 1,    L, D_qk)
V = V.unsqueeze(2)                       # (B, G, 1,    L, D_v)

# Broadcasting handles the repetition — no explicit expand needed
logits = Q @ K.mT / (D_qk ** 0.5)       # (B, G, H//G, L1, L2)
```

### Key Proof: Rank of Repeated GQA Matrix

If $W^K_g \in \mathbb{R}^{D \times D_{qk}}$ is repeated $H/G$ times to form $\tilde{W}^K \in \mathbb{R}^{D \times H \cdot D_{qk}}$:

$$\text{rank}(\tilde{W}^K) = \text{rank}(W^K_g) \leq \min(D, D_{qk})$$

Repeating columns does not increase rank.

### Key Proof: MHA is Special Case of GQA

Set $G = H$. Then each group has exactly one query head, and GQA reduces to MHA. $\square$

---

## 7. Multi-Head Latent Attention (MLA)

**Core idea** (DeepSeek): Low-rank decomposition of K/V projections.

### Decomposition

$$W^K_h = W^{UK}_h \cdot W^{DKV}, \qquad W^V_h = W^{UV}_h \cdot W^{DKV}$$

where:
- $W^{DKV} \in \mathbb{R}^{D \times r}$ — shared down-projection (compresses input to rank $r$)
- $W^{UK}_h \in \mathbb{R}^{r \times D_{qk}}$ — per-head up-projection for keys
- $W^{UV}_h \in \mathbb{R}^{r \times D_v}$ — per-head up-projection for values

### GQA $\subseteq$ MLA (via SVD)

Given GQA weight $W^K_g \in \mathbb{R}^{D \times D_{qk}}$ (rank $\leq D_{qk}$), take its SVD:

$$W^K_g = U \Sigma V^T$$

Set $W^{DKV} = U\Sigma$ (or choose rank-$r$ truncation) and $W^{UK}_h = V^T$ for all heads in group $g$.

Similarly for $W^V_g$. This shows any GQA configuration can be expressed as MLA. $\square$

### GQA $\subsetneq$ MLA (Strict Containment)

**Counterexample**: Choose $W^{DKV}$ and per-head $W^{UK}_h$ such that $W^K_h = W^{UK}_h W^{DKV}$ yields $H$ distinct key matrices with rank $> D_{qk}$ collectively. In GQA with $G$ groups, heads within a group must share the same $W^K_g$, so GQA cannot represent this configuration. $\square$

### Reduced Matrices for Efficient MLA Inference

Instead of materializing full K/V, cache only the compressed representation $c = x W^{DKV} \in \mathbb{R}^{L \times r}$.

Define reduced projection matrices:

$$\hat{W}^Q_h = (W^{UK}_h)^T W^Q_h \in \mathbb{R}^{r \times D_{qk}} \quad \text{(absorbed into query)}$$

Then attention becomes:

$$\text{logits}_h = (X_1 W^Q_h)(X_2 W^{DKV} W^{UK}_h)^T / \sqrt{D_{qk}} = (X_1 W^Q_h)((X_2 W^{DKV}) \hat{W}^{UK}_h)^T / \sqrt{D_{qk}}$$

But more efficiently, absorb into query side:

$$\hat{Q}_h = X_1 \hat{W}^Q_h, \quad \hat{W}^Q_h = W^Q_h (W^{UK}_h)^T \in \mathbb{R}^{D_1 \times r}$$

Then: $\text{logits}_h = \hat{Q}_h C^T / \sqrt{D_{qk}}$ where $C = X_2 W^{DKV} \in \mathbb{R}^{L_2 \times r}$.

---

## 8. KV-Cache Comparison

| Architecture | Cache per position per layer | Total for $L$ positions |
|-------------|----------------------------|------------------------|
| **MHA** | $2 \times H \times D_{qk}$ (store K and V for all heads) | $2HLD_{qk}$ |
| **GQA** ($G$ groups) | $2 \times G \times D_{qk}$ | $2GLD_{qk}$ |
| **MQA** ($G=1$) | $2 \times D_{qk}$ | $2LD_{qk}$ |
| **MLA** (rank $r$) | $r$ (store compressed $c$ only) | $rL$ |

**Key insight**: MLA with $r \ll 2HD_{qk}$ dramatically reduces cache memory.

---

## 9. Inclusion Hierarchy

$$\text{MHA} \subset \text{GQA} \subset \text{MLA}$$

- MHA $\subset$ GQA: Set $G = H$
- GQA $\subset$ MLA: Via SVD decomposition
- GQA $\subsetneq$ MLA: Counterexample with distinct per-head key matrices

---

## 10. Quick Reference: PyTorch Shapes

```
Self-Attention MHA:
  Input:   (B, L, D)
  Q,K,V:   (B, L, H*D_qk) → reshape → (B, H, L, D_qk)
  logits:  (B, H, L, L)
  Alpha:   (B, H, L, L)
  O:       (B, H, L, D_v) → reshape → (B, L, H*D_v)
  Output:  (B, L, D)

Cross-Attention MHA:
  Input1:  (B, L1, D1),  Input2: (B, L2, D2)
  Q:       (B, H, L1, D_qk)
  K,V:     (B, H, L2, D_qk), (B, H, L2, D_v)
  logits:  (B, H, L1, L2)
  O:       (B, H, L1, D_v) → reshape → (B, L1, H*D_v)
  Output:  (B, L1, D1)

GQA:
  Q:       (B, G, H//G, L, D_qk)
  K:       (B, G, 1,    L, D_qk)
  V:       (B, G, 1,    L, D_v)
  logits:  (B, G, H//G, L1, L2)

MLA:
  C:       (B, L, r)          ← compressed cache
  Q_hat:   (B, H, L1, r)     ← reduced query
  logits:  (B, H, L1, L2)
```

---

## 11. Common Pitfalls

1. **Forgetting to scale**: Always divide by $\sqrt{D_{qk}}$, not $\sqrt{D}$ or $\sqrt{H \cdot D_{qk}}$.
2. **Wrong permute order**: It's `permute(0, 2, 1, 3)` — swap sequence and head dims.
3. **Using loops over heads**: The whole point is parallel computation via reshape/permute.
4. **Confusing `.mT` and `.T`**: Use `.mT` for batched transpose (transposes last two dims).
5. **Causal mask after softmax**: The mask must be applied BEFORE softmax (set to $-\infty$).
6. **GQA broadcasting**: Don't `expand()` then compute — use `unsqueeze` and let broadcasting handle it.
7. **MLA rank confusion**: The rank $r$ is for the shared down-projection, not per-head.

---

*Last updated: 2026-02-24 | USAAIO 2026 Training*

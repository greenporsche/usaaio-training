# 04 — Cross-Attention & Masked Attention

> **Discovery — Intuition — Mastery — Competition**

---

## Discovery

### Cross-Attention: Connecting Two Sequences

In the original transformer (Vaswani et al., 2017), the decoder must attend to the encoder's output. The decoder generates queries ("what information do I need?"), while the encoder provides keys and values ("here's what I can offer").

This is **cross-attention**: queries come from one sequence, keys and values from another.

### Masked Attention: Preventing the Future from Leaking

In autoregressive generation (like GPT), the model predicts one token at a time. When predicting token $t$, it must NOT see tokens $t+1, t+2, \dots$ — that would be cheating. A **causal mask** enforces this constraint.

---

## Intuition

### Cross-Attention as Translation

Imagine translating "Le chat est sur le tapis" to "The cat is on the mat":

When generating "cat", the decoder query asks: "What noun am I translating?" The encoder keys for "chat" match strongly, and the encoder value for "chat" provides the meaning → "cat".

```
Decoder Query("cat")  ×  Encoder Keys:
  "Le"     → low
  "chat"   → HIGH ← this is the word to translate
  "est"    → low
  "sur"    → low
  "le"     → low
  "tapis"  → low
```

### Causal Mask Visualization

For a 4-token sequence, the causal mask looks like:

```
     t1  t2  t3  t4
t1 [  1   0   0   0 ]   ← token 1 can only see itself
t2 [  1   1   0   0 ]   ← token 2 sees tokens 1-2
t3 [  1   1   1   0 ]   ← token 3 sees tokens 1-3
t4 [  1   1   1   1 ]   ← token 4 sees everything
```

Positions with 0 are set to $-\infty$ before softmax, giving them zero attention weight.

---

## Mastery

### Cross-Attention: Different Dimensions

In cross-attention, the attending and attended sequences can have different dimensions:

- Attending sequence: $X_1 \in \mathbb{R}^{B \times L_1 \times D_1}$
- Being attended sequence: $X_2 \in \mathbb{R}^{B \times L_2 \times D_2}$

**Projection matrices**:

| Matrix | Shape | Input |
|--------|-------|-------|
| $W^Q$ | $(D_1, H \cdot D_{qk})$ | Projects from $X_1$'s space |
| $W^K$ | $(D_2, H \cdot D_{qk})$ | Projects from $X_2$'s space |
| $W^V$ | $(D_2, H \cdot D_v)$ | Projects from $X_2$'s space |
| $W^O$ | $(H \cdot D_v, D_1)$ | Projects back to $X_1$'s space |

**Key constraint**: Q and K must have the same last dimension ($D_{qk}$) for the dot product. But $D_1$ and $D_2$ can differ.

### Cross-Attention Shape Flow

```python
X1: (B, L1, D1)    # attending (e.g., decoder)
X2: (B, L2, D2)    # being attended (e.g., encoder output)

Q = W_Q(X1):       (B, L1, H*D_qk) → (B, H, L1, D_qk)
K = W_K(X2):       (B, L2, H*D_qk) → (B, H, L2, D_qk)
V = W_V(X2):       (B, L2, H*D_v)  → (B, H, L2, D_v)

logits = Q @ K.mT: (B, H, L1, L2)   # L1 queries attending to L2 keys
alpha:              (B, H, L1, L2)
O = alpha @ V:      (B, H, L1, D_v)  # each of L1 positions gets D_v features
→ merge:            (B, L1, H*D_v)
→ W_O:              (B, L1, D1)       # back to attending sequence's dimension
```

**Critical difference from self-attention**: The attention matrix is $L_1 \times L_2$ (not square if $L_1 \neq L_2$).

### Masked (Causal) Self-Attention

For autoregressive models, position $i$ can only attend to positions $\leq i$.

**Implementation**:

```python
# Create causal mask (lower-triangular)
causal_mask = torch.tril(torch.ones(L, L, device=x.device))   # (L, L)

# In attention computation:
logits = Q @ K.mT / (self.D_qk ** 0.5)                       # (B, H, L, L)
logits = logits.masked_fill(causal_mask == 0, float('-inf'))   # mask future
alpha = F.softmax(logits, dim=-1)                              # (B, H, L, L)
```

**Why $-\infty$?** Because $\text{softmax}(-\infty) = 0$. The softmax function maps $-\infty$ to zero probability, effectively removing those positions from the weighted sum.

**Alternative (register as buffer)**:

```python
class MaskedMHA(nn.Module):
    def __init__(self, max_seq_len, ...):
        super().__init__()
        # Register as buffer (not a parameter, but moves with .to(device))
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask)
```

### Complete MHA with Optional Mask

```python
class MyMHA(nn.Module):
    def __init__(self, D1: int, D2: int, H: int, D_qk: int, D_v: int):
        super().__init__()
        self.H = H
        self.D_qk = D_qk
        self.D_v = D_v
        self.W_Q = nn.Linear(D1, H * D_qk, bias=False)
        self.W_K = nn.Linear(D2, H * D_qk, bias=False)
        self.W_V = nn.Linear(D2, H * D_v, bias=False)
        self.W_O = nn.Linear(H * D_v, D1, bias=False)

    def forward(
        self,
        X1: torch.Tensor,       # (B, L1, D1)
        X2: torch.Tensor,       # (B, L2, D2)
        mask: torch.Tensor = None  # (L1, L2) or (B, 1, L1, L2)
    ) -> torch.Tensor:
        B, L1, _ = X1.shape
        _, L2, _ = X2.shape

        Q = self.W_Q(X1).reshape(B, L1, self.H, self.D_qk).permute(0, 2, 1, 3)
        K = self.W_K(X2).reshape(B, L2, self.H, self.D_qk).permute(0, 2, 1, 3)
        V = self.W_V(X2).reshape(B, L2, self.H, self.D_v).permute(0, 2, 1, 3)

        logits = Q @ K.mT / (self.D_qk ** 0.5)             # (B, H, L1, L2)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        alpha = F.softmax(logits, dim=-1)                    # (B, H, L1, L2)
        O = alpha @ V                                        # (B, H, L1, D_v)

        O = O.permute(0, 2, 1, 3).reshape(B, L1, self.H * self.D_v)
        return self.W_O(O)                                   # (B, L1, D1)
```

### Padding Masks

In practice, sequences in a batch may have different lengths. Padding tokens should not receive attention.

```python
# Example: sequences of length [3, 5] in a batch, padded to L=5
#   seq 1: [tok, tok, tok, PAD, PAD]
#   seq 2: [tok, tok, tok, tok, tok]

# Padding mask: 1 where real, 0 where padding
padding_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1]
])  # (B, L2)

# Reshape for broadcasting with (B, H, L1, L2)
padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L2)
```

### Combining Causal and Padding Masks

```python
# Causal mask: (1, 1, L, L) — same for all batch items and heads
causal = torch.tril(torch.ones(1, 1, L, L))

# Padding mask: (B, 1, 1, L) — different per batch item
padding = padding_mask.unsqueeze(1).unsqueeze(2)

# Combined: element-wise AND (both must be 1)
combined_mask = causal * padding  # (B, 1, L, L) via broadcasting
```

---

## Competition Connections

### Where This Appears

- **Decoder self-attention**: Uses causal mask
- **Encoder-decoder attention**: Cross-attention from decoder to encoder
- **Vision-language models**: Cross-attention between visual and text features

### Practice Problem

Given:
- Encoder output: $(B=2, L_2=8, D_2=256)$
- Decoder input: $(B=2, L_1=5, D_1=512)$
- $H=4, D_{qk}=32, D_v=32$

1. What are the shapes of $W^Q, W^K, W^V, W^O$?
2. What is the shape of the attention matrix?
3. If we apply a causal mask, what shape is it?
4. Is a causal mask appropriate for encoder-decoder cross-attention?

<details>
<summary>Answers</summary>

1. $W^Q: (512, 128)$, $W^K: (256, 128)$, $W^V: (256, 128)$, $W^O: (128, 512)$
2. Attention matrix: $(2, 4, 5, 8)$ — 5 decoder positions attending to 8 encoder positions
3. A causal mask would be $(5, 8)$ — but this is NOT square, so standard `torch.tril` won't work directly
4. **No!** Causal masking is for self-attention in the decoder. In cross-attention, each decoder position should be able to attend to ALL encoder positions. The encoder output is already fully computed.

</details>

---

### Key Takeaways

1. **Cross-attention**: Q from one sequence, K/V from another. Projection dimensions match via $D_{qk}$, even if input dimensions differ.
2. **Causal mask**: Lower-triangular, applied BEFORE softmax, sets future positions to $-\infty$.
3. **Padding mask**: Prevents attending to padding tokens. Shaped $(B, 1, 1, L)$ for broadcasting.
4. **Causal masking applies to decoder self-attention only**, not to encoder self-attention or encoder-decoder cross-attention.
5. **The attention matrix shape is $(B, H, L_1, L_2)$** — rectangular when attending and attended sequences have different lengths.

---

*Previous: [03 — Multi-Head Attention](03-multi-head-attention.md) | Next: [05 — Positional Encoding](05-positional-encoding.md)*

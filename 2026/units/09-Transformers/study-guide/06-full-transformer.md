# 06 — Full Transformer Architecture

> **Discovery — Intuition — Mastery — Competition**

---

## Discovery

### "Attention Is All You Need" — Vaswani et al. (2017)

The original transformer replaced recurrent layers entirely with attention and feed-forward networks. The architecture has two main components:

1. **Encoder**: Processes the input sequence, producing contextualized representations
2. **Decoder**: Generates the output sequence, attending to both its own previous outputs and the encoder's representations

Key insight: By removing recurrence, all positions can be processed in parallel during training. This enabled massive speedups on GPU/TPU hardware.

---

## Intuition

### The Encoder-Decoder Mental Model

```
                    ENCODER                              DECODER
              ┌─────────────────┐                 ┌─────────────────┐
              │  Self-Attention  │                 │ Masked Self-Attn │
Input    ──→  │  + Add & Norm   │    Encoder  ──→ │  + Add & Norm    │
Sequence      │  Feed-Forward   │    Output        │ Cross-Attention  │ ──→ Output
              │  + Add & Norm   │                 │  + Add & Norm    │    Sequence
              │  × N layers     │                 │  Feed-Forward    │
              └─────────────────┘                 │  + Add & Norm    │
                                                  │  × N layers      │
                                                  └─────────────────┘
```

**Encoder**: Each token attends to all tokens in the input. Produces rich contextualized representations.

**Decoder**: Each token attends to (1) previous tokens only (causal mask) and (2) all encoder tokens (cross-attention). Generates output autoregressively.

---

## Mastery

### Layer Normalization

Before diving into the blocks, we need LayerNorm:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu, \sigma^2$ are computed across the feature dimension for each position independently, and $\gamma, \beta$ are learnable scale and shift parameters.

```python
# PyTorch provides this
norm = nn.LayerNorm(D)  # normalizes across last dimension
```

**Key difference from BatchNorm**: LayerNorm normalizes across features (last dim), not across batch. This makes it independent of batch size and suitable for variable-length sequences.

### Residual Connections

$$\text{output} = \text{sublayer}(x) + x$$

Residual connections allow gradients to flow directly through the network, enabling training of very deep models (6+ layers of transformer blocks).

### Post-Norm vs. Pre-Norm

**Post-Norm** (original Transformer):
```python
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FFN(x))
```

**Pre-Norm** (modern standard, easier to train):
```python
x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-norm is more common in practice because it provides more stable gradients and doesn't require careful learning rate warmup.

### Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- $W_1 \in \mathbb{R}^{D \times D_{ff}}$, typically $D_{ff} = 4D$
- $W_2 \in \mathbb{R}^{D_{ff} \times D}$

The FFN expands to a higher dimension, applies a nonlinearity, and projects back. It operates **independently on each position** (unlike attention, which mixes positions).

```python
class FFN(nn.Module):
    def __init__(self, D: int, D_ff: int):
        super().__init__()
        self.W1 = nn.Linear(D, D_ff)
        self.W2 = nn.Linear(D_ff, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(x)))
```

Modern variants use GELU or SwiGLU instead of ReLU, but ReLU is the USAAIO standard.

### Encoder Block (Pre-Norm)

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, D: int, H: int, D_qk: int, D_v: int, D_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.self_attn = MyMHA(D, D, H, D_qk, D_v)
        self.norm2 = nn.LayerNorm(D)
        self.ffn = FFN(D, D_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)                              # (B, L, D)
        x = x + self.self_attn(x_norm, x_norm)             # (B, L, D)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))                    # (B, L, D)

        return x
```

### Decoder Block (Pre-Norm)

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, D: int, D_enc: int, H: int, D_qk: int, D_v: int, D_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.masked_self_attn = MyMHA(D, D, H, D_qk, D_v)
        self.norm2 = nn.LayerNorm(D)
        self.cross_attn = MyMHA(D, D_enc, H, D_qk, D_v)
        self.norm3 = nn.LayerNorm(D)
        self.ffn = FFN(D, D_ff)

    def forward(
        self,
        x: torch.Tensor,          # (B, L_dec, D)
        encoder_out: torch.Tensor,  # (B, L_enc, D_enc)
        causal_mask: torch.Tensor   # (L_dec, L_dec)
    ) -> torch.Tensor:
        """Returns: (B, L_dec, D)"""

        # 1. Masked self-attention (with causal mask)
        x_norm = self.norm1(x)
        x = x + self.masked_self_attn(x_norm, x_norm, mask=causal_mask)

        # 2. Cross-attention to encoder output (no mask)
        x_norm = self.norm2(x)
        x = x + self.cross_attn(x_norm, encoder_out)

        # 3. FFN
        x = x + self.ffn(self.norm3(x))

        return x
```

### Full Transformer

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        D: int,
        H: int,
        D_qk: int,
        D_v: int,
        D_ff: int,
        N_enc: int,
        N_dec: int,
        max_len: int = 5000
    ):
        super().__init__()
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, D)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, D)
        self.pos_enc = PositionalEncoding(D, max_len)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(D, H, D_qk, D_v, D_ff)
            for _ in range(N_enc)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(D, D, H, D_qk, D_v, D_ff)
            for _ in range(N_dec)
        ])

        # Output head
        self.output_proj = nn.Linear(D, tgt_vocab_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """src: (B, L_src) token IDs → (B, L_src, D) encoder output"""
        x = self.pos_enc(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor
    ) -> torch.Tensor:
        """tgt: (B, L_tgt) token IDs → (B, L_tgt, vocab_size) logits"""
        L = tgt.size(1)
        causal_mask = torch.tril(torch.ones(L, L, device=tgt.device))

        x = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, causal_mask)

        return self.output_proj(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encode(src)
        return self.decode(tgt, encoder_out)
```

### Architecture Variants

| Model | Architecture | Key Feature |
|-------|-------------|------------|
| **Original Transformer** | Encoder-Decoder | Machine translation |
| **BERT** | Encoder only | Bidirectional, masked language modeling |
| **GPT** | Decoder only | Autoregressive, causal mask |
| **T5** | Encoder-Decoder | Text-to-text framework |
| **Vision Transformer (ViT)** | Encoder only | Patches as tokens |

### Parameter Count for a Full Transformer

For one encoder layer with $D = 512$, $H = 8$, $D_{qk} = D_v = 64$, $D_{ff} = 2048$:

| Component | Parameters |
|-----------|-----------|
| MHA ($W^Q, W^K, W^V, W^O$) | $4 \times 512^2 = 1,048,576$ |
| LayerNorm (x2) | $2 \times 2 \times 512 = 2,048$ |
| FFN ($W_1, b_1, W_2, b_2$) | $512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2,099,712$ |
| **Total per encoder layer** | **~3.15M** |

With 6 encoder + 6 decoder layers: ~44M parameters (not counting embeddings).

---

## Competition Connections

### What USAAIO Might Test

1. **Shape tracking through a full encoder block**
2. **Distinguishing the three types of attention in a decoder**: masked self-attention, cross-attention, and what mask each uses
3. **Parameter counting**: "How many parameters in a transformer with these specifications?"
4. **Pre-norm vs. post-norm**: Know both, implement both

### Practice Problem

For a transformer with $D=256, H=4, D_{qk}=D_v=64, D_{ff}=1024$, trace the shape of $x$ through one decoder block.

<details>
<summary>Solution</summary>

```
Input x:                      (B, L_dec, 256)
After LayerNorm1:             (B, L_dec, 256)
After masked self-attention:  (B, L_dec, 256)
After residual:               (B, L_dec, 256)

After LayerNorm2:             (B, L_dec, 256)
After cross-attention:        (B, L_dec, 256)  [attends to (B, L_enc, 256)]
After residual:               (B, L_dec, 256)

After LayerNorm3:             (B, L_dec, 256)
FFN W1:                       (B, L_dec, 1024)
FFN ReLU:                     (B, L_dec, 1024)
FFN W2:                       (B, L_dec, 256)
After residual:               (B, L_dec, 256)
```

The dimension stays constant at 256 throughout the block, by design.

</details>

---

### Key Takeaways

1. **Encoder block**: Self-attention + FFN, each with residual + LayerNorm
2. **Decoder block**: Masked self-attention + cross-attention + FFN
3. **Pre-norm** is modern standard: normalize BEFORE the sublayer
4. **FFN expands to 4x**, applies nonlinearity, projects back. Position-independent.
5. **Residual connections** are essential for training deep transformers
6. **The dimension $D$ is preserved** through every block — residual connections require matching dimensions

---

*Previous: [05 — Positional Encoding](05-positional-encoding.md) | Next: [07 — Grouped Query Attention](07-gqa.md)*

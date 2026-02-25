# Unit 09 — Transformers: Study Guide Overview

> **AI 500 | The single most important unit for USAAIO Round 2**

## Why This Unit Matters

The 2025 USAAIO Round 2 Problem 2 was worth **100 points** (out of the total exam) and tested 14 parts covering Multi-Head Attention (MHA), Grouped Query Attention (GQA), and Multi-Head Latent Attention (MLA). Students who mastered this unit had a decisive advantage.

This study guide covers transformers from first principles through competition-level mastery.

---

## Study Guide Roadmap

| # | Topic | Key Skills | Competition Relevance |
|---|-------|-----------|----------------------|
| 01 | Self-Attention | Intuition, Q/K/V, soft lookup | Foundation for everything |
| 02 | Scaled Dot-Product Attention | Scaling derivation, softmax stability | Tested directly in Round 2 |
| 03 | Multi-Head Attention (MHA) | Tensor reshaping, parallel heads, nn.Module | **Round 2 P2 Parts 1-5** |
| 04 | Cross-Attention & Masked Attention | Encoder-decoder, causal mask | Decoder understanding |
| 05 | Positional Encoding | Sinusoidal, learned, permutation invariance | Architecture completeness |
| 06 | Full Transformer Architecture | Encoder/decoder blocks, LayerNorm, FFN | System-level understanding |
| 07 | Grouped Query Attention (GQA) | Broadcasting, rank proofs, MHA $\subset$ GQA | **Round 2 P2 Parts 6-8** |
| 08 | Multi-Head Latent Attention (MLA) & KV-Cache | SVD decomposition, reduced matrices, cache analysis | **Round 2 P2 Parts 9-14** |

---

## Prerequisites

Before starting this unit, you should be comfortable with:

- **Linear algebra**: Matrix multiplication, rank, SVD (singular value decomposition)
- **PyTorch fundamentals**: `nn.Module`, `nn.Linear`, tensor operations
- **Tensor manipulation**: `reshape`, `permute`, `unsqueeze`, broadcasting rules
- **Calculus**: Softmax function and its derivatives
- **Probability**: Variance of sums of random variables

If any of these feel shaky, review Units 01 (Math Foundations) and 07 (Deep Learning) first.

---

## How to Use This Study Guide

### Recommended Study Order

1. **First pass** (Sections 01-06): Build understanding of the full transformer architecture.
2. **Deep dive** (Sections 03, 07, 08): Focus on MHA, GQA, and MLA — these are competition-critical.
3. **Practice**: Complete all exercises and assignments, especially the comprehensive Assignment 12.
4. **Speed drills**: Practice writing MHA/GQA/MLA implementations from memory until the tensor reshaping pattern is automatic.

### Study Approach: D-I-M-C

Each section follows the **Discovery-Intuition-Mastery-Competition** progression:

- **Discovery**: Historical context, the problem being solved, key papers
- **Intuition**: Visual and conceptual understanding, analogies
- **Mastery**: Full mathematical derivation, complete implementations
- **Competition**: USAAIO-style problems, proofs, edge cases

### Time Estimates

| Phase | Topics | Suggested Time |
|-------|--------|---------------|
| Foundation | Sections 01-02 | 4-6 hours |
| Core MHA | Section 03 | 6-8 hours |
| Architecture | Sections 04-06 | 6-8 hours |
| Competition | Sections 07-08 | 10-14 hours |
| Practice | Exercises + Assignments | 20-30 hours |
| **Total** | | **~50-60 hours** |

---

## Key Papers (Reference)

1. **Bahdanau et al. (2014)** — "Neural Machine Translation by Jointly Learning to Align and Translate" — introduced attention
2. **Vaswani et al. (2017)** — "Attention Is All You Need" — the transformer architecture
3. **Shazeer (2019)** — "Fast Transformer Decoding: One Write-Head is All You Need" — multi-query attention
4. **Ainslie et al. (2023)** — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — grouped query attention
5. **DeepSeek-AI (2024)** — "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" — multi-head latent attention

---

## The One Pattern to Rule Them All

If you remember nothing else, remember this tensor reshaping pattern:

```python
Q = self.W_Q(x)                                              # (B, L, H*D)
Q = Q.reshape(-1, L, self.H, self.D_qk).permute(0, 2, 1, 3) # (B, H, L, D)
# ... attention computation ...
O = O.permute(0, 2, 1, 3).reshape(-1, L, self.H * self.D_v)  # (B, L, H*Dv)
```

This is the heartbeat of multi-head attention. Practice it until it is muscle memory.

---

*Next: [01 — Self-Attention](01-self-attention.md)*

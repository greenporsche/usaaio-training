# 05 — Positional Encoding

> **Discovery — Intuition — Mastery — Competition**

---

## Discovery

### The Problem: Permutation Invariance

Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Consider what happens if we permute the input sequence. If $P$ is a permutation matrix:

$$Q' = PXW^Q = PQ, \quad K' = PK, \quad V' = PV$$

$$Q'K'^T = PQK^TP^T$$

$$\text{softmax}(PQK^TP^T) = P \cdot \text{softmax}(QK^T) \cdot P^T$$

$$\text{Output}' = P \cdot \text{softmax}(QK^T) \cdot P^T \cdot PV = P \cdot \text{softmax}(QK^T) \cdot V = P \cdot \text{Output}$$

This means: **self-attention is permutation equivariant**. If you shuffle the input tokens, the output is shuffled in exactly the same way. The model has NO concept of position.

But word order matters! "The dog bit the man" vs. "The man bit the dog" should have very different representations.

### The Solution: Add Position Information

Vaswani et al. (2017) proposed adding position-dependent vectors to the input embeddings:

$$\text{input}_i = \text{token\_embedding}_i + \text{positional\_encoding}_i$$

---

## Intuition

### Sinusoidal Encoding as a Clock

Think of positional encoding as a set of clocks running at different speeds:

```
Position:     0    1    2    3    4    5    6    7    ...
Dim 0 (fast): 0.0  0.84 0.91 0.14 -0.76 -0.96 -0.28 0.66  (sin, high freq)
Dim 1 (fast): 1.0  0.54 -0.42 -0.99 -0.65 0.28 0.96 0.75  (cos, high freq)
Dim 2 (med):  0.0  0.10 0.20 0.30 0.39  0.48  0.56 0.64  (sin, med freq)
Dim 3 (med):  1.0  0.99 0.98 0.95 0.92  0.88  0.83 0.77  (cos, med freq)
...
Dim d-1 (slow): 1.0  1.0  1.0  1.0  1.0   1.0   1.0  1.0  (cos, very low freq)
```

Different dimensions oscillate at different frequencies. Together, they create a unique "fingerprint" for each position, much like how binary numbers use bits at different place values.

---

## Mastery

### Sinusoidal Positional Encoding

For position $pos$ and dimension $i$:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

where $d$ is the model dimension and $i \in \{0, 1, \dots, d/2 - 1\}$.

### Frequency Spectrum

The wavelength for dimension pair $(2i, 2i+1)$:

$$\lambda_i = 2\pi \cdot 10000^{2i/d}$$

- $i = 0$: $\lambda = 2\pi \approx 6.28$ (highest frequency, shortest wavelength)
- $i = d/2 - 1$: $\lambda = 2\pi \cdot 10000$ (lowest frequency, longest wavelength)

This geometric progression of frequencies from $2\pi$ to $2\pi \cdot 10000$ allows the model to detect patterns at many different scales of position.

### Key Property: Relative Positions via Linear Transformations

For any fixed offset $k$, there exists a linear transformation $M_k$ such that:

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

**Proof sketch**: Using the trigonometric identity:

$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
$$\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

For each frequency $\omega_i = 1/10000^{2i/d}$:

$$\begin{pmatrix} \sin(\omega_i(pos + k)) \\ \cos(\omega_i(pos + k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}$$

This is a rotation matrix! The relative position $k$ corresponds to rotating each frequency component. This linear relationship allows attention to easily learn to attend to relative positions.

### Implementation

```python
import torch
import math

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Returns: (max_len, d_model) positional encoding matrix.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()       # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # (d_model/2,) — equivalent to 1/10000^(2i/d)

    pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

    return pe  # (max_len, d_model)
```

**Implementation note**: We compute $10000^{-2i/d}$ as $\exp(-2i \cdot \ln(10000)/d)$ for numerical stability.

### As an nn.Module

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = sinusoidal_positional_encoding(max_len, d_model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — for batch broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, L, D) with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]
```

### Learned Positional Embeddings

An alternative: make positional encodings learnable parameters.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        positions = torch.arange(x.size(1), device=x.device)  # (L,)
        return x + self.pe(positions)  # (L, D) broadcasts to (B, L, D)
```

### Comparison

| Property | Sinusoidal | Learned |
|----------|-----------|---------|
| Parameters | 0 (fixed) | $\text{max\_len} \times D$ |
| Extrapolation | Can generalize to longer sequences | Cannot (fixed max length) |
| Performance | Comparable | Slightly better on fixed-length tasks |
| Used in | Original Transformer, some modern models | BERT, GPT-2 |

### Modern Alternatives (Brief Overview)

- **Rotary Position Embeddings (RoPE)**: Encode relative positions by rotating Q and K vectors. Used in LLaMA, GPT-NeoX.
- **ALiBi**: Subtract a linear bias proportional to distance from attention logits. No parameters needed.
- **Relative position encodings**: Directly bias attention based on relative distance between tokens.

These are beyond USAAIO scope but important for understanding modern architectures.

---

## Competition Connections

### What Could Be Tested

1. **Compute sinusoidal PE** for specific positions and dimensions
2. **Prove** the relative position linear transformation property
3. **Explain** why attention is permutation equivariant without position encoding
4. **Implement** positional encoding in PyTorch

### Practice Problem

Given $d = 4$, compute $PE$ for positions 0, 1, 2:

<details>
<summary>Solution</summary>

$\omega_0 = 1/10000^{0/4} = 1$, $\omega_1 = 1/10000^{2/4} = 1/100 = 0.01$

Position 0: $[\sin(0), \cos(0), \sin(0), \cos(0)] = [0, 1, 0, 1]$

Position 1: $[\sin(1), \cos(1), \sin(0.01), \cos(0.01)] = [0.841, 0.540, 0.010, 1.000]$

Position 2: $[\sin(2), \cos(2), \sin(0.02), \cos(0.02)] = [0.909, -0.416, 0.020, 1.000]$

</details>

---

### Key Takeaways

1. **Without positional encoding, transformers are permutation equivariant** — word order is invisible.
2. **Sinusoidal encoding**: $\sin$ and $\cos$ at geometrically-spaced frequencies create unique position fingerprints.
3. **Relative positions become linear transformations** — the model can learn to attend to "3 positions back" via a learned linear map.
4. **Added to input embeddings**: $\text{input} = \text{token\_embedding} + \text{positional\_encoding}$.
5. **`register_buffer`** for non-learned tensors that should move to the correct device.

---

*Previous: [04 — Cross-Attention & Masked Attention](04-cross-attention-masked.md) | Next: [06 — Full Transformer Architecture](06-full-transformer.md)*

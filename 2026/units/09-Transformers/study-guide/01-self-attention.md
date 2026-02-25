# 01 — Self-Attention

> **Discovery — Intuition — Mastery — Competition**

---

## Discovery

### The Problem: Sequential Bottleneck

Before attention, sequence-to-sequence models relied on RNNs (recurrent neural networks). An encoder RNN compressed an entire input sequence into a single fixed-size hidden vector, and a decoder RNN generated the output from that vector.

**The bottleneck**: All information about a 100-word sentence had to squeeze through one vector. Long sentences lost information.

### The Breakthrough: Bahdanau et al. (2014)

Bahdanau, Cho, and Bengio proposed a simple but revolutionary idea: instead of compressing everything into one vector, let the decoder **look back** at the entire encoder sequence at each step, paying more attention to relevant parts.

This was the birth of the attention mechanism.

---

## Intuition

### Attention as Soft Dictionary Lookup

Think of attention as a **soft dictionary lookup**:

| Concept | Dictionary Analogy | Attention Analogy |
|---------|-------------------|-------------------|
| What you're looking for | The word you look up | **Query** ($Q$) |
| What each entry is labeled | Dictionary headwords | **Key** ($K$) |
| What each entry contains | Dictionary definitions | **Value** ($V$) |
| The lookup result | Definition of matching word | Weighted sum of values |

In a hard dictionary lookup, you find the exact match and return one definition. In **soft** (attention) lookup, you compute a similarity score with every key, then return a weighted blend of all values.

### Visualization

Consider a sequence of 4 tokens: ["The", "cat", "sat", "down"]

When processing "sat", the query asks: "Who is relevant to me?"

```
Query("sat") vs Key("The")  → low score  (0.05)
Query("sat") vs Key("cat")  → high score (0.70)
Query("sat") vs Key("sat")  → medium     (0.20)
Query("sat") vs Key("down") → low score  (0.05)
```

Output for "sat" = 0.05 * V("The") + 0.70 * V("cat") + 0.20 * V("sat") + 0.05 * V("down")

The word "sat" **attends** mostly to "cat" — learning that "cat" is the subject of the action.

### Self-Attention vs. Cross-Attention

- **Self-attention**: Q, K, V all come from the **same** sequence. Each token attends to all other tokens (including itself) in the same sequence.
- **Cross-attention**: Q comes from one sequence, K and V from another. (Covered in Section 04.)

---

## Mastery

### Mathematical Formulation

Given an input sequence $X \in \mathbb{R}^{L \times D}$ (L tokens, each of dimension D):

**Step 1: Create Q, K, V via learned linear projections**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q \in \mathbb{R}^{D \times D_{qk}}$, $W^K \in \mathbb{R}^{D \times D_{qk}}$, $W^V \in \mathbb{R}^{D \times D_v}$.

Note: $D_{qk}$ (query/key dim) and $D_v$ (value dim) can differ, but query and key dimensions **must** match because we compute their dot product.

**Step 2: Compute attention scores**

$$\text{score}(i, j) = q_i \cdot k_j = \sum_{d=1}^{D_{qk}} Q_{i,d} \cdot K_{j,d}$$

In matrix form: $\text{Scores} = QK^T \in \mathbb{R}^{L \times L}$

**Step 3: Normalize with softmax**

$$\alpha_{i,j} = \frac{\exp(\text{score}(i,j))}{\sum_{j'=1}^{L} \exp(\text{score}(i,j'))}$$

Each row of $\alpha$ sums to 1. Row $i$ gives the attention distribution for token $i$.

**Step 4: Weighted sum of values**

$$\text{Output}_i = \sum_{j=1}^{L} \alpha_{i,j} \cdot V_j$$

In matrix form: $\text{Output} = \alpha V \in \mathbb{R}^{L \times D_v}$

### Complete Formula (Single-Head)

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{D_{qk}}}\right) V$$

(The scaling by $\sqrt{D_{qk}}$ is explained in detail in Section 02.)

### Shape Tracking

```
X:       (L, D)
W^Q:     (D, D_qk)     → Q = X @ W^Q:     (L, D_qk)
W^K:     (D, D_qk)     → K = X @ W^K:     (L, D_qk)
W^V:     (D, D_v)      → V = X @ W^V:     (L, D_v)

Scores:  Q @ K^T:       (L, D_qk) @ (D_qk, L) = (L, L)
Alpha:   softmax(Scores): (L, L)
Output:  Alpha @ V:      (L, L) @ (L, D_v) = (L, D_v)
```

### PyTorch Implementation (Single-Head, No Batch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, D: int, D_qk: int, D_v: int):
        super().__init__()
        self.D_qk = D_qk
        self.W_Q = nn.Linear(D, D_qk, bias=False)
        self.W_K = nn.Linear(D, D_qk, bias=False)
        self.W_V = nn.Linear(D, D_v, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (L, D)
        Returns: (L, D_v)
        """
        Q = self.W_Q(X)                              # (L, D_qk)
        K = self.W_K(X)                              # (L, D_qk)
        V = self.W_V(X)                              # (L, D_v)

        scores = Q @ K.T / (self.D_qk ** 0.5)       # (L, L)
        alpha = F.softmax(scores, dim=-1)             # (L, L)
        output = alpha @ V                            # (L, D_v)

        return output
```

### Batched Implementation

```python
class SingleHeadAttentionBatched(nn.Module):
    def __init__(self, D: int, D_qk: int, D_v: int):
        super().__init__()
        self.D_qk = D_qk
        self.W_Q = nn.Linear(D, D_qk, bias=False)
        self.W_K = nn.Linear(D, D_qk, bias=False)
        self.W_V = nn.Linear(D, D_v, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, D)
        Returns: (B, L, D_v)
        """
        Q = self.W_Q(X)                              # (B, L, D_qk)
        K = self.W_K(X)                              # (B, L, D_qk)
        V = self.W_V(X)                              # (B, L, D_v)

        scores = Q @ K.mT / (self.D_qk ** 0.5)      # (B, L, L)
        alpha = F.softmax(scores, dim=-1)             # (B, L, L)
        output = alpha @ V                            # (B, L, D_v)

        return output
```

**Note**: We use `K.mT` (not `K.T`) for batched transpose — `.mT` transposes the last two dimensions while preserving batch dimensions.

---

## Competition Connections

### What USAAIO Tests

1. **Shape derivation**: Given dimensions, compute the shape of Q, K, V, scores, and output.
2. **Implementation**: Write attention in PyTorch without loops.
3. **Conceptual understanding**: Explain what attention weights represent, why softmax is used.

### Practice Problem

Given $X \in \mathbb{R}^{5 \times 8}$ (5 tokens, dim 8), $D_{qk} = 4$, $D_v = 6$:

1. What are the shapes of $W^Q$, $W^K$, $W^V$?
2. What is the shape of the attention matrix?
3. What is the shape of the output?
4. How many learnable parameters in total?

<details>
<summary>Answers</summary>

1. $W^Q: (8, 4)$, $W^K: (8, 4)$, $W^V: (8, 6)$
2. Attention matrix: $(5, 5)$ — each of the 5 tokens attends to all 5
3. Output: $(5, 6)$ — same number of tokens, value dimension
4. Parameters: $8 \times 4 + 8 \times 4 + 8 \times 6 = 32 + 32 + 48 = 112$

</details>

---

### Key Takeaways

1. **Self-attention lets every token look at every other token** — breaking the sequential bottleneck of RNNs.
2. **Q, K, V are learned linear projections** of the same input (in self-attention).
3. **Softmax produces a probability distribution** over all positions — attention weights sum to 1 per query.
4. **The output is a weighted blend of values**, where weights are determined by query-key similarity.
5. **Shape discipline is critical**: always track tensor shapes through every operation.

---

*Next: [02 — Scaled Dot-Product Attention](02-scaled-dot-product.md)*

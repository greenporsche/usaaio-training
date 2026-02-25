# Exercises: Novel Architectures

## Exercise 1: Implementing from a Diagram

**Difficulty:** Introductory

Consider the following architecture described in words (imagine a diagram):

**"Pre-Norm Transformer Block"**

```
Input x (B, L, d)
  │
  ├──→ LayerNorm → Self-Attention → Dropout ──→ (+) ──→ residual_1
  │                                              │
  └──────────────────────────────────────────────┘
  │
  ├──→ LayerNorm → FFN → Dropout ──→ (+) ──→ residual_2 = output
  │                                    │
  └────────────────────────────────────┘
```

Where FFN = Linear(d, 4d) → GELU → Linear(4d, d).

**Tasks:**

(a) Implement `PreNormTransformerBlock` as an `nn.Module`. The self-attention can be `nn.MultiheadAttention`.

(b) Verify that the output shape equals the input shape (required for residual connections).

(c) What is the difference between pre-norm and post-norm? Which line of code changes?

(d) How many parameters does this block have in terms of $d$? (Count the self-attention, FFN, and LayerNorm parameters.)

---

## Exercise 2: Multi-Head Attention from Scratch

**Difficulty:** Intermediate

Implement standard multi-head attention **without** using `nn.MultiheadAttention`.

**Specification:**
- Input: $X \in \mathbb{R}^{B \times L \times d}$
- Number of heads: $h$
- Head dimension: $d_k = d / h$
- Projections: $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$, $W_O \in \mathbb{R}^{d \times d}$

**Tasks:**

(a) Implement the `__init__` method. Use `nn.Linear` for all projections.

(b) Implement the `forward` method with these steps:
1. Project to Q, K, V: each is `(B, L, d)`
2. Reshape to `(B, h, L, d_k)` using `.view()` and `.transpose()`
3. Compute scaled dot-product attention per head
4. Concatenate heads back to `(B, L, d)`
5. Apply output projection

(c) Add an optional causal mask that prevents attending to future positions. How do you create the mask and apply it?

(d) What is the total FLOPs count for the forward pass in terms of $B$, $L$, $d$, and $h$?

---

## Exercise 3: Implementing a Novel Normalization

**Difficulty:** Intermediate

**Description: "Group-Scale Normalization"**

> "Divide the feature dimension $d$ into $G$ groups of size $d/G$. For each group, compute RMS normalization independently. Then apply a learnable affine transform per group: $\text{GSNorm}(x)_g = \gamma_g \cdot \frac{x_g}{\text{RMS}(x_g)} + \beta_g$ where $\gamma_g, \beta_g \in \mathbb{R}^{d/G}$."

**Tasks:**

(a) Implement `GroupScaleNorm` as an `nn.Module`.

(b) What happens when $G = 1$? What happens when $G = d$?

(c) Write a smoke test that verifies:
- Output shape matches input shape
- The RMS of each group in the output (before the affine transform) is approximately 1

(d) Compare the number of learnable parameters to `nn.LayerNorm(d)`.

---

## Exercise 4: Combining Novel Components

**Difficulty:** Advanced

Build a complete transformer block using **non-standard** components:

- **Attention:** Differential Attention (from study guide 04)
- **Normalization:** RMSNorm (from study guide 04)
- **FFN:** SwiGLU: $\text{SwiGLU}(x) = (\text{Swish}(xW_1)) \odot (xW_2)$ followed by $W_3$
  where $\text{Swish}(x) = x \cdot \sigma(x)$, and $W_1, W_2 \in \mathbb{R}^{d \times d_{ff}}$, $W_3 \in \mathbb{R}^{d_{ff} \times d}$
- **Connection:** Pre-norm residual

**Tasks:**

(a) Implement `SwiGLU` as an `nn.Module`.

(b) Implement `NovelTransformerBlock` composing all four components.

(c) Run a smoke test with $B=2$, $L=8$, $d=64$, $d_k=32$, $d_v=64$, $d_{ff}=128$.

(d) Count the total parameters and compare to a standard transformer block of the same dimensions.

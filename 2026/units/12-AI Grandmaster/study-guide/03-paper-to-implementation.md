# 03 — Paper-to-Implementation

> The **meta-skill** that separates USAAIO medalists from the rest. Round 2 teaches you new methods inside the exam — your speed of absorption determines your score.

---

## Discovery

### The Round 2 Reality

In Round 2, you are given 3 problems. Each problem:

1. Introduces a method you may not have seen before
2. Provides a paper-style description with equations and diagrams
3. Asks you to prove properties, implement the method, train it, and analyze results

You cannot memorize every method. What you can do is **master the process of going from description to implementation**. This is a trainable skill.

### The Core Loop

```
Read description → Identify key equation → Map to tensors → Implement → Verify shapes → Test
```

Every successful implementation follows this loop. Speed comes from practice, not from knowing more methods.

---

## Intuition

### Reading a Paper (or Exam Problem) Efficiently

**Do NOT read linearly.** Use this order:

1. **Abstract / Problem statement** — What is this method? What does it do?
2. **Figures / Diagrams** — Architecture diagrams tell you the data flow
3. **Key equation or algorithm** — The one equation that defines the method
4. **Input/output specification** — What goes in, what comes out, what shapes?
5. **Training procedure** — Loss function, optimizer, data requirements
6. **Experimental details** — Only if you need hyperparameters

In an exam context, the problem statement IS the abstract. The parts guide you through the method. Read ALL parts before starting — later parts often clarify earlier ones.

### The Notation-to-Code Bridge

Papers use mathematical notation. Code uses tensors. The bridge is **shape annotations**.

**Example:** A paper says:

> "Given input sequence $X \in \mathbb{R}^{B \times L \times d}$, compute queries $Q = XW_Q$ where $W_Q \in \mathbb{R}^{d \times d_k}$."

Translate immediately:

```python
# X: (B, L, d)
# W_Q: (d, d_k)
Q = X @ W_Q  # (B, L, d) @ (d, d_k) -> (B, L, d_k)
```

**Common notation mappings:**

| Paper Notation | PyTorch Code | Notes |
|---------------|-------------|-------|
| $X \in \mathbb{R}^{B \times L \times d}$ | `x: (B, L, d)` | Shape annotation |
| $W \in \mathbb{R}^{d \times k}$ | `nn.Linear(d, k)` or `nn.Parameter(torch.randn(d, k))` | Weight matrix |
| $\sigma(\cdot)$ | `torch.sigmoid(·)` or `F.relu(·)` | Activation (context-dependent) |
| $\text{softmax}(\cdot)$ | `F.softmax(·, dim=-1)` | Always specify `dim` |
| $\|\cdot\|_2$ | `torch.norm(·, dim=-1)` | L2 norm |
| $X^T$ | `x.transpose(-2, -1)` | Batch transpose |
| $\odot$ | `*` (elementwise) | Hadamard product |
| $\oplus$ | `torch.cat([·, ·], dim=-1)` | Concatenation (usually) |
| $\frac{\partial L}{\partial \theta}$ | `loss.backward(); param.grad` | Automatic |

---

## Mastery

### The Shape-First Method

Before writing any logic, write the shapes. This prevents 90% of bugs.

**Step 1:** Read the method description and extract all tensor shapes.

**Step 2:** Write the `__init__` with shape-annotated parameters.

**Step 3:** Write the `forward` as shape-only comments first.

**Step 4:** Fill in the code to match the shapes.

**Example: Implementing a novel "Gated Linear Attention" from description**

> "Given input $X \in \mathbb{R}^{B \times L \times d}$, compute:
> - $Q = XW_Q$, $K = XW_K$, $V = XW_V$ where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$
> - Apply feature maps: $\hat{Q} = \phi(Q)$, $\hat{K} = \phi(K)$ where $\phi(x) = \text{elu}(x) + 1$
> - Compute linear attention: $O = \hat{Q}(\hat{K}^T V) / (\hat{Q} \hat{K}^T \mathbf{1})$
> - Apply gating: $G = \sigma(XW_G)$ where $W_G \in \mathbb{R}^{d \times d_v}$
> - Output: $Y = G \odot O$"

**Shape-first implementation:**

```python
class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)  # (d, d_k)
        self.W_K = nn.Linear(d_model, d_k, bias=False)  # (d, d_k)
        self.W_V = nn.Linear(d_model, d_v, bias=False)  # (d, d_v)
        self.W_G = nn.Linear(d_model, d_v, bias=False)  # (d, d_v)

    def forward(self, x):
        # x: (B, L, d)
        Q = self.W_Q(x)    # (B, L, d_k)
        K = self.W_K(x)    # (B, L, d_k)
        V = self.W_V(x)    # (B, L, d_v)

        # Feature map: elu(x) + 1
        Q_hat = F.elu(Q) + 1  # (B, L, d_k)
        K_hat = F.elu(K) + 1  # (B, L, d_k)

        # Linear attention: Q_hat @ (K_hat^T @ V) -- associative reordering
        KV = K_hat.transpose(-2, -1) @ V  # (B, d_k, L) @ (B, L, d_v) -> (B, d_k, d_v)
        O = Q_hat @ KV                     # (B, L, d_k) @ (B, d_k, d_v) -> (B, L, d_v)

        # Normalization: Q_hat @ K_hat^T @ 1
        K_sum = K_hat.sum(dim=1, keepdim=True)  # (B, 1, d_k)
        norm = (Q_hat * K_sum).sum(dim=-1, keepdim=True)  # (B, L, 1)
        O = O / (norm + 1e-6)  # (B, L, d_v)

        # Gating
        G = torch.sigmoid(self.W_G(x))  # (B, L, d_v)
        Y = G * O                        # (B, L, d_v)

        return Y  # (B, L, d_v)
```

### The Smoke Test

After implementing any module, run a **shape check**:

```python
# Smoke test
B, L, d = 2, 10, 64
d_k, d_v = 32, 48
layer = GatedLinearAttention(d, d_k, d_v)
x = torch.randn(B, L, d)
y = layer(x)
print(f"Input: {x.shape}, Output: {y.shape}")
assert y.shape == (B, L, d_v), f"Expected (B, L, d_v) = ({B}, {L}, {d_v}), got {y.shape}"
print("Shape check passed!")
```

Always run this before moving on. A shape mismatch caught here saves 10 minutes of debugging later.

### Common Implementation Patterns

**Pattern 1: Multi-head splitting**
```python
# Split d_model into h heads of d_k each
# x: (B, L, d_model) -> (B, h, L, d_k)
x = x.view(B, L, h, d_k).transpose(1, 2)
```

**Pattern 2: Applying a mask**
```python
# mask: (B, 1, 1, L) or (1, 1, L, L)
scores = scores.masked_fill(mask == 0, float('-inf'))
```

**Pattern 3: Residual connection**
```python
x = x + self.sublayer(self.norm(x))  # Pre-norm
# or
x = self.norm(x + self.sublayer(x))  # Post-norm
```

**Pattern 4: Einsum for complex contractions**
```python
# When standard matmul doesn't capture the contraction pattern
output = torch.einsum('bhld,bhde->bhle', Q, KV)
```

**Pattern 5: Rearranging with einops-style reshape**
```python
# (B, L, h*d_k) -> (B, h, L, d_k)
x = x.reshape(B, L, h, d_k).permute(0, 2, 1, 3)
```

### Reading Speed Tips

**For the exam:**

1. **Underline tensor shapes** as you read — this is the most important information
2. **Draw the data flow** — boxes for operations, arrows for tensors with shape labels
3. **Identify the novel part** — most methods are 90% standard (Linear, softmax, etc.) and 10% new. Focus on the 10%.
4. **Use stated results** — if Part 3 says "the output of this layer is shape (B, L, d)", use that shape in Part 4 even if you couldn't solve Part 3

---

## Connection

### Paper-Reading Connects to Every Exam Problem

Every Round 2 problem follows this pattern:

```
Description of a new method
    → Part 1-2: Mathematical analysis (prove properties)
    → Part 3-5: Core implementation (build the key module)
    → Part 6-8: Training and data (set up the pipeline)
    → Part 9-10: Evaluation and analysis
```

Your paper-reading speed directly determines:
- How quickly you understand what to implement
- How accurately you translate equations to code
- Whether you can use Part N results to continue at Part N+1

### Practice Strategy

1. Pick a paper you haven't read (see suggestions below)
2. Set a 10-minute timer
3. Extract: (a) key equation, (b) architecture diagram, (c) input/output shapes
4. Implement the core module in PyTorch
5. Run the smoke test
6. Compare against a reference implementation

**Suggested papers for practice:**
- "Attention Is All You Need" (Vaswani et al., 2017) — you know the method; practice the reading process
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) — ViT, clear description
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020) — generative, equation-heavy
- "Physics-Informed Neural Networks" (Raissi et al., 2019) — directly relevant

---

## Summary

| Step | Action | Time Budget (Exam) |
|------|--------|-------------------|
| 1 | Read problem statement + all parts | 3-5 min |
| 2 | Identify key equation and shapes | 1-2 min |
| 3 | Write shape annotations | 1 min |
| 4 | Implement `__init__` | 2-3 min |
| 5 | Implement `forward` (shape-first) | 5-10 min |
| 6 | Smoke test | 1 min |

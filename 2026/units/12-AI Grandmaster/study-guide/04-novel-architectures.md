# 04 — Novel Architectures

> Given a description of a new architecture, implement it in PyTorch. This is the core skill tested in Round 2.

---

## Discovery

### Why Novel Architectures?

Round 2 does not test whether you have memorized specific architectures. It tests whether you can **implement an architecture you have never seen**, given a clear description.

This means the exam might present:
- A new attention variant with different complexity trade-offs
- A novel normalization scheme
- A custom pooling mechanism
- A new way to combine features across layers

The architecture will be described in paper-style notation with equations. Your job: turn those equations into a working PyTorch `nn.Module`.

### The Implementation Mindset

When you see a new architecture description, think:

1. **What are the learnable parameters?** → These go in `__init__`
2. **What is the forward computation?** → This goes in `forward`
3. **What are the shapes at each step?** → Write these as comments

---

## Intuition

### Anatomy of a PyTorch Module

Every architecture follows the same skeleton:

```python
class NovelArchitecture(nn.Module):
    def __init__(self, <hyperparameters>):
        super().__init__()
        # 1. Learnable parameters: nn.Linear, nn.Parameter, etc.
        # 2. Sub-modules: other nn.Modules
        # 3. Buffers: non-learnable tensors (nn.Buffer or register_buffer)

    def forward(self, x):
        # 1. Shape annotations as comments
        # 2. Step-by-step computation matching the paper
        # 3. Return output with documented shape
        return output
```

### Common Building Blocks

| Component | PyTorch | Shape Effect |
|-----------|---------|-------------|
| Linear projection | `nn.Linear(d_in, d_out)` | `(*, d_in) -> (*, d_out)` |
| Layer norm | `nn.LayerNorm(d)` | Shape unchanged |
| RMS norm | Custom (see below) | Shape unchanged |
| Softmax | `F.softmax(x, dim=-1)` | Shape unchanged |
| GELU | `F.gelu(x)` | Shape unchanged |
| Dropout | `nn.Dropout(p)` | Shape unchanged |
| Conv1D | `nn.Conv1d(c_in, c_out, k)` | `(B, c_in, L) -> (B, c_out, L')` |
| Embedding | `nn.Embedding(V, d)` | `(B, L) -> (B, L, d)` |

### RMS Norm (Common in Modern Architectures)

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (B, L, d)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight  # (B, L, d)
```

---

## Mastery

### Worked Example 1: Novel "Differential Attention"

**Paper description:**

> "Standard attention computes a single attention map. We propose Differential Attention, which computes two attention maps and takes their difference, reducing noise from irrelevant tokens.
>
> Given $X \in \mathbb{R}^{B \times L \times d}$:
> 1. Compute $Q_1, Q_2 = \text{split}(XW_Q)$ and $K_1, K_2 = \text{split}(XW_K)$ where $W_Q, W_K \in \mathbb{R}^{d \times 2d_k}$, split along the last dimension
> 2. $V = XW_V$ where $W_V \in \mathbb{R}^{d \times d_v}$
> 3. $A = \text{softmax}(Q_1 K_1^T / \sqrt{d_k}) - \lambda \cdot \text{softmax}(Q_2 K_2^T / \sqrt{d_k})$
> 4. $O = AV$
> 5. Output: $O \in \mathbb{R}^{B \times L \times d_v}$
>
> $\lambda$ is a learnable scalar initialized to 0.5."

**Implementation:**

```python
class DifferentialAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, 2 * d_k, bias=False)  # (d, 2*d_k)
        self.W_K = nn.Linear(d_model, 2 * d_k, bias=False)  # (d, 2*d_k)
        self.W_V = nn.Linear(d_model, d_v, bias=False)       # (d, d_v)
        self.lam = nn.Parameter(torch.tensor(0.5))            # learnable scalar

    def forward(self, x):
        # x: (B, L, d)
        B, L, _ = x.shape

        QQ = self.W_Q(x)  # (B, L, 2*d_k)
        KK = self.W_K(x)  # (B, L, 2*d_k)
        V = self.W_V(x)   # (B, L, d_v)

        # Split into two sets of queries and keys
        Q1, Q2 = QQ.chunk(2, dim=-1)  # each (B, L, d_k)
        K1, K2 = KK.chunk(2, dim=-1)  # each (B, L, d_k)

        # Two attention maps
        scale = self.d_k ** 0.5
        A1 = F.softmax(Q1 @ K1.transpose(-2, -1) / scale, dim=-1)  # (B, L, L)
        A2 = F.softmax(Q2 @ K2.transpose(-2, -1) / scale, dim=-1)  # (B, L, L)

        # Differential attention
        A = A1 - self.lam * A2  # (B, L, L)

        # Apply to values
        O = A @ V  # (B, L, L) @ (B, L, d_v) -> (B, L, d_v)

        return O  # (B, L, d_v)
```

### Worked Example 2: Novel "Mixture-of-Softmaxes Attention"

**Paper description:**

> "Standard softmax attention uses a single softmax, limiting the rank of the attention matrix. We use a mixture of $M$ softmax attention maps.
>
> For each mixture component $m$:
> - $A_m = \text{softmax}(Q W_{Qm}^T (K W_{Km}^T)^T / \sqrt{d_k})$
>
> The final attention is: $A = \sum_{m=1}^{M} \pi_m A_m$ where $\pi = \text{softmax}(Xw_\pi)$ and $w_\pi \in \mathbb{R}^{d \times M}$."

**Implementation:**

```python
class MixtureSoftmaxAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, M=4):
        super().__init__()
        self.M = M
        self.d_k = d_k
        # Per-component projections
        self.W_Qm = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(M)])
        self.W_Km = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(M)])
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        # Mixture weights
        self.w_pi = nn.Linear(d_model, M, bias=False)

    def forward(self, x):
        # x: (B, L, d)
        B, L, _ = x.shape
        V = self.W_V(x)  # (B, L, d_v)

        # Mixture weights: per-token, per-component
        pi = F.softmax(self.w_pi(x), dim=-1)  # (B, L, M)

        # Compute mixture of attention maps
        O = torch.zeros(B, L, V.shape[-1], device=x.device)
        scale = self.d_k ** 0.5

        for m in range(self.M):
            Qm = self.W_Qm[m](x)  # (B, L, d_k)
            Km = self.W_Km[m](x)  # (B, L, d_k)
            Am = F.softmax(Qm @ Km.transpose(-2, -1) / scale, dim=-1)  # (B, L, L)
            Om = Am @ V  # (B, L, d_v)
            O = O + pi[:, :, m:m+1] * Om  # weighted contribution

        return O  # (B, L, d_v)
```

### Worked Example 3: Custom Normalization — "ScaleNorm"

**Paper description:**

> "Replace LayerNorm with ScaleNorm: $\text{ScaleNorm}(x) = g \cdot \frac{x}{\|x\|_2}$ where $g$ is a learnable scalar initialized to $\sqrt{d}$."

```python
class ScaleNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(d_model ** 0.5))
        self.eps = eps

    def forward(self, x):
        # x: (B, L, d)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)  # (B, L, 1)
        return self.g * x / norm  # (B, L, d)
```

### Worked Example 4: Custom Pooling — "Attention Pooling"

**Paper description:**

> "Instead of mean pooling over the sequence, use a learnable query to attend to the sequence:
> $q \in \mathbb{R}^{d_k}$ (learnable), $K = XW_K$, $V = XW_V$
> $\text{pool}(X) = \text{softmax}(qK^T / \sqrt{d_k}) V$"

```python
class AttentionPooling(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_k))  # (d_k,)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_k ** 0.5

    def forward(self, x):
        # x: (B, L, d)
        K = self.W_K(x)  # (B, L, d_k)
        V = self.W_V(x)  # (B, L, d)

        # q: (d_k,) -> (1, 1, d_k) for broadcasting
        scores = (K @ self.q) / self.scale  # (B, L)
        weights = F.softmax(scores, dim=-1)  # (B, L)
        pooled = (weights.unsqueeze(-1) * V).sum(dim=1)  # (B, d)

        return pooled  # (B, d)
```

---

## Connection

### Architecture Implementation Is the Core Exam Skill

Round 2 problems follow this pattern:

1. **You are given** a description of a new architecture component
2. **You implement** it as an `nn.Module`
3. **You integrate** it into a larger model
4. **You train** the model on provided data
5. **You analyze** the results

Steps 2-3 require novel architecture implementation. Steps 4-5 use standard training patterns from earlier units.

### Common Exam Traps

1. **Forgetting `bias=False`** when the paper specifies no bias
2. **Wrong `dim` in softmax** — attention scores need `dim=-1` (over keys)
3. **Missing `scale` factor** — $1/\sqrt{d_k}$ is easy to forget
4. **Shape mismatch in residual** — output must match input shape for `x + sublayer(x)`
5. **Not using `nn.Parameter`** for learnable scalars — plain `torch.tensor` won't be trained

---

## Summary

| Principle | Rule |
|-----------|------|
| Shape-first | Write shapes as comments before code |
| Smoke test | Always verify with random input immediately |
| Parameters in `__init__` | All learnable weights must be `nn.Parameter` or `nn.Module` |
| Computation in `forward` | Step-by-step matching the paper equations |
| Assume nothing | Read the description carefully; don't assume standard patterns |

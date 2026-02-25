# 07 — Grouped Query Attention (GQA)

> **Discovery — Intuition — Mastery — Competition**
>
> **Competition-critical**: 2025 Round 2 Problem 2, Parts 6-8 (20 points) directly tested GQA.

---

## Discovery

### The Problem: KV-Cache Memory Explosion

During autoregressive generation, each layer stores the key and value tensors for all previous positions (the **KV-cache**). For standard MHA with $H$ heads:

$$\text{KV-cache per position per layer} = 2 \times H \times D_{qk} \text{ floats}$$

For a model with $D = 8192$, $H = 64$, $D_{qk} = 128$, and 80 layers at 4096 tokens:

$$2 \times 64 \times 128 \times 80 \times 4096 \times 2 \text{ bytes} \approx 10.7 \text{ GB}$$

This is a significant memory bottleneck, especially for long sequences and large batch sizes.

### The Solution: Share K/V Across Groups of Query Heads

**Multi-Query Attention (MQA)** (Shazeer, 2019): Use a SINGLE K/V head shared across all query heads. Reduces cache by $H\times$, but sometimes hurts quality.

**Grouped Query Attention (GQA)** (Ainslie et al., 2023): Compromise — use $G$ groups of K/V heads, where each group serves $H/G$ query heads. Balances quality and efficiency.

---

## Intuition

### The Spectrum from MHA to MQA

```
MHA (G=H):     Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8     ← 8 query heads
                |  |  |  |  |  |  |  |
               K1 K2 K3 K4 K5 K6 K7 K8      ← 8 KV heads (one each)

GQA (G=2):     Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8     ← 8 query heads
                \  |  |  /  \  |  |  /
                 K1=K2=K3=K4  K5=K6=K7=K8    ← 2 KV groups

MQA (G=1):     Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8     ← 8 query heads
                \  \  |  |  /  /  /  /
                 K1=K2=K3=K4=K5=K6=K7=K8     ← 1 KV head
```

Within each group, all query heads share the same K and V. The query projections remain independent, so different heads can still learn different attention patterns.

---

## Mastery

### GQA Formulation

Given:
- $H$ query heads, $G$ KV groups where $G | H$ (G divides H)
- Each group has $H/G$ query heads sharing one set of K/V

**Per-group projection matrices**:

$$W^K_g \in \mathbb{R}^{D \times D_{qk}}, \quad W^V_g \in \mathbb{R}^{D \times D_v}, \quad g = 1, \dots, G$$

**Per-head query projection** (all $H$ heads are independent):

$$W^Q_h \in \mathbb{R}^{D \times D_{qk}}, \quad h = 1, \dots, H$$

Head $h$ belongs to group $g = \lceil hG/H \rceil$ and uses $K_g, V_g$.

### Concatenated Projection Matrices for GQA

$$W^Q \in \mathbb{R}^{D \times (H \cdot D_{qk})}, \quad W^K \in \mathbb{R}^{D \times (G \cdot D_{qk})}, \quad W^V \in \mathbb{R}^{D \times (G \cdot D_v)}$$

Note: $W^K$ and $W^V$ have $G$ (not $H$) sets of columns.

### Implementation with Broadcasting (NO LOOPS)

The key technique is to reshape Q to expose the group structure, then use broadcasting to avoid explicit repetition of K/V.

```python
class MyGQA(nn.Module):
    """Grouped Query Attention — no loops, broadcasting only."""

    def __init__(self, D: int, H: int, G: int, D_qk: int, D_v: int):
        """
        D:    model dimension
        H:    number of query heads
        G:    number of KV groups (must divide H)
        D_qk: per-head query/key dimension
        D_v:  per-head value dimension
        """
        super().__init__()
        assert H % G == 0, "H must be divisible by G"
        self.H = H
        self.G = G
        self.D_qk = D_qk
        self.D_v = D_v
        self.heads_per_group = H // G

        self.W_Q = nn.Linear(D, H * D_qk, bias=False)    # H query heads
        self.W_K = nn.Linear(D, G * D_qk, bias=False)    # G key heads
        self.W_V = nn.Linear(D, G * D_v, bias=False)     # G value heads
        self.W_O = nn.Linear(H * D_v, D, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, _ = X.shape

        # Project
        Q = self.W_Q(X)  # (B, L, H*D_qk)
        K = self.W_K(X)  # (B, L, G*D_qk)
        V = self.W_V(X)  # (B, L, G*D_v)

        # Reshape Q: expose group structure
        # (B, L, H*D_qk) → (B, L, G, H//G, D_qk) → (B, G, H//G, L, D_qk)
        Q = Q.reshape(B, L, self.G, self.heads_per_group, self.D_qk)
        Q = Q.permute(0, 2, 3, 1, 4)                      # (B, G, H//G, L, D_qk)

        # Reshape K, V: one head per group
        # (B, L, G*D_qk) → (B, L, G, D_qk) → (B, G, L, D_qk) → (B, G, 1, L, D_qk)
        K = K.reshape(B, L, self.G, self.D_qk)
        K = K.permute(0, 2, 1, 3).unsqueeze(2)            # (B, G, 1, L, D_qk)

        V = V.reshape(B, L, self.G, self.D_v)
        V = V.permute(0, 2, 1, 3).unsqueeze(2)            # (B, G, 1, L, D_v)

        # Attention with broadcasting
        # Q: (B, G, H//G, L, D_qk)
        # K: (B, G, 1,    L, D_qk)  ← broadcasts across H//G
        logits = Q @ K.mT / (self.D_qk ** 0.5)            # (B, G, H//G, L, L)
        alpha = F.softmax(logits, dim=-1)                   # (B, G, H//G, L, L)
        O = alpha @ V                                       # (B, G, H//G, L, D_v)

        # Merge heads: (B, G, H//G, L, D_v) → (B, L, G, H//G, D_v) → (B, L, H*D_v)
        O = O.permute(0, 3, 1, 2, 4)                      # (B, L, G, H//G, D_v)
        O = O.reshape(B, L, self.H * self.D_v)            # (B, L, H*D_v)

        return self.W_O(O)                                 # (B, L, D)
```

### Shape Flow Summary

```
Input:           (B, L, D)

Q projected:     (B, L, H*D_qk)
Q reshaped:      (B, L, G, H//G, D_qk)
Q permuted:      (B, G, H//G, L, D_qk)

K projected:     (B, L, G*D_qk)
K reshaped:      (B, L, G, D_qk)
K permuted:      (B, G, L, D_qk)
K unsqueezed:    (B, G, 1, L, D_qk)     ← ready for broadcasting

logits:          (B, G, H//G, L, L)       ← broadcast happened here
alpha:           (B, G, H//G, L, L)
O:               (B, G, H//G, L, D_v)

O permuted:      (B, L, G, H//G, D_v)
O merged:        (B, L, H*D_v)
Output:          (B, L, D)
```

### Key Proof: Rank of Repeated GQA Weight Matrix

**Claim**: If we form $\tilde{W}^K$ by repeating $W^K_g$ for $H/G$ times (to make it look like MHA), then:

$$\text{rank}(\tilde{W}^K) = \text{rank}(W^K_g) \leq \min(D, D_{qk})$$

**Proof**:

$\tilde{W}^K = [W^K_g \;|\; W^K_g \;|\; \cdots \;|\; W^K_g] \in \mathbb{R}^{D \times (H/G \cdot D_{qk})}$

The column space of $\tilde{W}^K$ equals the column space of $W^K_g$, since every column of $\tilde{W}^K$ is a column of $W^K_g$. Therefore:

$$\text{rank}(\tilde{W}^K) = \text{rank}(W^K_g) \leq \min(D, D_{qk})$$

Repeating columns does not add any new linearly independent vectors. $\square$

**Competition insight**: This shows GQA has a rank constraint that MHA does not. In MHA, $W^K = [W^K_1 | \cdots | W^K_H]$ can have rank up to $\min(D, H \cdot D_{qk})$ since different heads have independent weight matrices.

### Key Proof: MHA is a Special Case of GQA

**Claim**: Multi-Head Attention is a special case of GQA with $G = H$.

**Proof**:

When $G = H$, each group contains exactly $H/G = 1$ query head. Therefore:
- Each query head has its own dedicated K/V head
- The GQA K/V projection $W^K \in \mathbb{R}^{D \times (G \cdot D_{qk})} = \mathbb{R}^{D \times (H \cdot D_{qk})}$

This is identical to MHA, where each head has independent $W^K_h$. $\square$

### KV-Cache Savings

| Method | KV-cache per position per layer | Relative to MHA |
|--------|-------------------------------|-----------------|
| MHA ($G=H$) | $2 \cdot H \cdot D_{qk}$ | $1\times$ |
| GQA ($G$ groups) | $2 \cdot G \cdot D_{qk}$ | $G/H$ |
| MQA ($G=1$) | $2 \cdot D_{qk}$ | $1/H$ |

---

## Competition Connections

### 2025 Round 2 Problem 2

- **Part 6** (5 pts): Prove rank of repeated GQA weight matrix
- **Part 7** (10 pts): Implement GQA from scratch, NO LOOPS
- **Part 8** (5 pts): Prove MHA is special case of GQA

### Practice Problem

Given $D=512, H=8, G=2, D_{qk}=64, D_v=64$:

1. How many query heads per group?
2. What are the shapes of $W^Q, W^K, W^V$?
3. What is the shape of Q, K, V after reshaping for the grouped computation?
4. What is the KV-cache reduction compared to MHA?

<details>
<summary>Answers</summary>

1. $H/G = 8/2 = 4$ query heads per group
2. $W^Q: (512, 512)$, $W^K: (512, 128)$, $W^V: (512, 128)$ — note K/V are smaller!
3. Q: $(B, 2, 4, L, 64)$, K: $(B, 2, 1, L, 64)$, V: $(B, 2, 1, L, 64)$
4. KV-cache: $2 \times 2 \times 64 = 256$ vs MHA's $2 \times 8 \times 64 = 1024$. Reduction: $4\times$.

</details>

---

### Key Takeaways

1. **GQA shares K/V across groups of query heads**, reducing memory by $H/G$.
2. **Broadcasting via unsqueeze** — the `1` dimension in K/V broadcasts to match `H//G` query heads. No loops needed.
3. **MHA = GQA with $G=H$**: a direct special case.
4. **Rank constraint**: Repeated GQA matrices have the same rank as one group's matrix.
5. **The reshape is 5D**: $(B, G, H/G, L, D)$ — this is the key structural difference from MHA's 4D reshape.

---

*Previous: [06 — Full Transformer](06-full-transformer.md) | Next: [08 — MLA & KV-Cache](08-mla-kv-cache.md)*

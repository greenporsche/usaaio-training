# 04 — Graph Attention Networks (GAT)

## Discovery

> **Velickovic et al. (2017) — letting the graph learn which neighbors matter.** GCN assigns fixed weights to neighbors based on degree: $\frac{1}{\sqrt{\hat{d}_v \hat{d}_u}}$. But in many graphs, not all neighbors are equally important. A citation might be tangential or central; a social connection might be a close friend or a distant acquaintance. GAT introduces **learned attention coefficients** that allow each node to dynamically weight its neighbors, adapting to the content of their features rather than just the graph structure.

---

## 1. Motivation: Beyond Fixed Normalization

GCN's normalization $\frac{1}{\sqrt{\hat{d}_v \hat{d}_u}}$ depends only on node degrees — it is **content-independent**. Two problems:

1. **Heterogeneous neighborhoods:** A paper might cite both highly relevant and barely relevant works. GCN weights them the same (modulo degree).
2. **Task-dependent relevance:** For different tasks, different neighbors might be important.

GAT solves this by computing **attention coefficients** $\alpha_{ij}$ that depend on the features of both nodes $i$ and $j$.

---

## 2. The GAT Mechanism

### Step 1: Linear Transformation

Apply a shared linear transformation to all nodes:

$$z_i = \mathbf{W} h_i, \quad z_i \in \mathbb{R}^{F'}$$

where $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is a learnable weight matrix.

### Step 2: Attention Coefficients

For each edge $(i, j)$, compute a raw attention score:

$$e_{ij} = \text{LeakyReLU}\!\left(\vec{a}^T [z_i \| z_j]\right)$$

where:
- $\|$ denotes concatenation
- $\vec{a} \in \mathbb{R}^{2F'}$ is a learnable attention vector
- LeakyReLU has negative slope $\alpha = 0.2$ (typically)

**Equivalently:** Split $\vec{a} = [\vec{a}_1 \| \vec{a}_2]$ where $\vec{a}_1, \vec{a}_2 \in \mathbb{R}^{F'}$:

$$e_{ij} = \text{LeakyReLU}\!\left(\vec{a}_1^T z_i + \vec{a}_2^T z_j\right)$$

### Step 3: Normalize with Softmax

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Now $\sum_{j \in \mathcal{N}(i)} \alpha_{ij} = 1$ — attention coefficients form a probability distribution over neighbors.

### Step 4: Weighted Aggregation

$$h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \, z_j\right) = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \, \mathbf{W} h_j\right)$$

---

## 3. Multi-Head Attention

Like Transformer multi-head attention, GAT uses $K$ independent attention heads:

$$h_i' = \Big\|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \, \mathbf{W}^k h_j\right)$$

Each head has its own $\mathbf{W}^k$ and $\vec{a}^k$. The outputs are **concatenated**, giving $h_i' \in \mathbb{R}^{K \cdot F'}$.

**Final layer:** Use averaging instead of concatenation:

$$h_i' = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \, \mathbf{W}^k h_j\right)$$

This produces $h_i' \in \mathbb{R}^{F'}$ (not $K \cdot F'$).

---

## 4. Worked Example

**Graph:** 3 nodes, edges: $(1,2), (1,3), (2,3)$ with self-loops.

**Features** ($F = 2$):

$$h_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad h_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad h_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

**Weight matrix** ($F' = 2$, identity for simplicity): $\mathbf{W} = I_2$

So $z_i = h_i$.

**Attention vector:** $\vec{a} = [0.2, 0.3, -0.1, 0.4]^T$

Split: $\vec{a}_1 = [0.2, 0.3]^T$, $\vec{a}_2 = [-0.1, 0.4]^T$

**Compute $e_{ij}$ for node 1** ($\mathcal{N}(1) = \{1, 2, 3\}$ with self-loops):

| $(i,j)$ | $\vec{a}_1^T z_i$ | $\vec{a}_2^T z_j$ | $e_{ij}$ (before LeakyReLU) |
|---|---|---|---|
| $(1,1)$ | $0.2$ | $-0.1$ | $0.1$ |
| $(1,2)$ | $0.2$ | $0.4$ | $0.6$ |
| $(1,3)$ | $0.2$ | $0.3$ | $0.5$ |

LeakyReLU (all positive, so unchanged): $e_{11} = 0.1$, $e_{12} = 0.6$, $e_{13} = 0.5$

**Softmax:**

$$\alpha_{11} = \frac{e^{0.1}}{e^{0.1} + e^{0.6} + e^{0.5}} = \frac{1.105}{1.105 + 1.822 + 1.649} = 0.241$$

$$\alpha_{12} = \frac{1.822}{4.576} = 0.398, \quad \alpha_{13} = \frac{1.649}{4.576} = 0.360$$

**Aggregate:**

$$h_1' = \sigma\!\left(0.241 \begin{pmatrix}1\\0\end{pmatrix} + 0.398 \begin{pmatrix}0\\1\end{pmatrix} + 0.360 \begin{pmatrix}1\\1\end{pmatrix}\right) = \sigma\!\left(\begin{pmatrix}0.601\\0.758\end{pmatrix}\right)$$

Node 1 attends more to node 2 (0.398) and node 3 (0.360) than to itself (0.241).

---

## 5. GAT vs. GCN

| Property | GCN | GAT |
|---|---|---|
| Neighbor weighting | Fixed ($\frac{1}{\sqrt{\hat{d}_i \hat{d}_j}}$) | Learned ($\alpha_{ij}$) |
| Parameters per layer | $F \times F'$ | $F \times F' + 2F'$ per head |
| Computational cost | $O(M \cdot F')$ | $O(M \cdot F' + N \cdot F')$ |
| Multi-head | No | Yes |
| Edge features | No (standard) | Can incorporate |
| When to prefer | Homogeneous graphs, limited data | Heterogeneous neighborhoods |

---

## 6. When Does Attention Help?

GAT outperforms GCN when:
- **Neighbors have varying relevance** — some edges are more informative than others
- **The graph is heterogeneous** — different node types or edge types
- **The task requires selectivity** — not all structural connections carry equal signal

GAT does NOT help when:
- The graph is very regular (e.g., grid, complete graph) — all neighbors equally relevant
- Training data is very limited — attention parameters need data to learn

---

## 7. Connection to Transformers

GAT attention and Transformer self-attention are closely related:

| Aspect | Transformer | GAT |
|---|---|---|
| Attention scope | All tokens | Graph neighbors only |
| Attention function | Scaled dot-product | Additive (LeakyReLU) |
| Positional info | Positional encoding | Graph structure |
| Complexity | $O(N^2)$ | $O(M)$ (sparse) |

A Transformer on a fully-connected graph is essentially a GAT variant. Graph Transformers (e.g., Graphormer) combine both ideas.

---

## 8. Implementation

### From Scratch

```python
class GATLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super().__init__()
        self.heads = heads
        self.W = torch.nn.Parameter(torch.randn(heads, in_features, out_features) * 0.01)
        self.a = torch.nn.Parameter(torch.randn(heads, 2 * out_features) * 0.01)

    def forward(self, h, adj):
        # h: [N, F_in], adj: [N, N] binary
        N = h.size(0)
        head_outputs = []

        for k in range(self.heads):
            z = h @ self.W[k]  # [N, F_out]

            # Attention scores for all pairs
            a1 = (z @ self.a[k, :z.size(1)]).unsqueeze(1)  # [N, 1]
            a2 = (z @ self.a[k, z.size(1):]).unsqueeze(0)  # [1, N]
            e = F.leaky_relu(a1 + a2, negative_slope=0.2)  # [N, N]

            # Mask non-neighbors
            e = e.masked_fill(adj == 0, float('-inf'))
            alpha = F.softmax(e, dim=1)  # [N, N]

            head_outputs.append(alpha @ z)  # [N, F_out]

        return torch.cat(head_outputs, dim=-1)  # [N, heads * F_out]
```

### With PyTorch Geometric

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## Investigate

1. In the worked example, compute the attention coefficients for node 2 (attending to its neighbors).
2. If all node features are identical ($h_i = h_j$ for all $i, j$), what happens to the GAT attention coefficients?
3. Why does GAT use LeakyReLU instead of ReLU in the attention mechanism?

---

## Master

1. Prove that if all attention coefficients are equal ($\alpha_{ij} = \frac{1}{|\mathcal{N}(i)|}$), GAT reduces to a form of GCN (mean normalization variant).
2. Implement multi-head GAT from scratch and verify that on a complete graph with identical features, all heads produce the same output.
3. Compare GCN and GAT on the Cora dataset — when does the attention mechanism provide measurable benefit?

---

## Connect

- **From GCN (Section 03):** GAT generalizes GCN's fixed normalization to learned, content-dependent weights.
- **From Transformers (Unit 10):** GAT attention is additive attention applied to graph-structured data.
- **To graph-level tasks (Section 05):** GAT produces node embeddings; combine with readout for graph-level tasks.

# Graph Neural Networks — Cheat Sheet

## Graph Basics

| Concept | Notation | Description |
|---|---|---|
| Graph | $G = (V, E)$ | Set of vertices $V$ and edges $E$ |
| Adjacency matrix | $A \in \{0,1\}^{N \times N}$ | $A_{ij} = 1$ if edge $(i,j) \in E$ |
| Degree matrix | $D_{ii} = \sum_j A_{ij}$ | Diagonal matrix of node degrees |
| Node features | $X \in \mathbb{R}^{N \times F}$ | $N$ nodes, $F$ features each |
| Edge features | $e_{ij} \in \mathbb{R}^{D}$ | Optional per-edge attributes |
| Neighborhood | $\mathcal{N}(v)$ | Set of nodes adjacent to $v$ |

## Self-Loop Augmentation

$$\hat{A} = A + I_N$$

$$\hat{D}_{ii} = \sum_j \hat{A}_{ij} = D_{ii} + 1$$

Adding self-loops ensures each node includes its own features during aggregation.

## Normalization Variants

| Type | Formula | Use Case |
|---|---|---|
| Row-normalized | $D^{-1}A$ | Random walk, asymmetric |
| Symmetric | $D^{-1/2}AD^{-1/2}$ | GCN, preserves symmetry |
| With self-loops | $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ | Standard GCN normalization |

Shorthand: $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$

## Message-Passing Framework

$$h_v^{(l+1)} = \text{UPDATE}\Big(h_v^{(l)},\; \text{AGGREGATE}\big(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\big)\Big)$$

Three steps per layer:
1. **Message**: compute messages from neighbors
2. **Aggregate**: combine messages (sum, mean, max)
3. **Update**: transform aggregated result with node's own state

## GCN Layer (Kipf & Welling, 2016)

$$H^{(l+1)} = \sigma\!\left(\tilde{A}\, H^{(l)}\, W^{(l)}\right)$$

- $H^{(l)} \in \mathbb{R}^{N \times F_l}$ — node embeddings at layer $l$
- $W^{(l)} \in \mathbb{R}^{F_l \times F_{l+1}}$ — learnable weight matrix
- $\sigma$ — nonlinearity (typically ReLU)
- $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ — normalized adjacency with self-loops

Per-node view:

$$h_v^{(l+1)} = \sigma\!\left(W^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{h_u^{(l)}}{\sqrt{\hat{d}_v \cdot \hat{d}_u}}\right)$$

## GAT Layer (Velickovic et al., 2017)

Attention coefficient:

$$e_{ij} = \text{LeakyReLU}\!\left(\vec{a}^T [\mathbf{W}h_i \| \mathbf{W}h_j]\right)$$

Normalized attention:

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Output:

$$h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, \mathbf{W} h_j\right)$$

Multi-head ($K$ heads):

$$h_i' = \Big\|_{k=1}^{K}\; \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k\, \mathbf{W}^k h_j\right)$$

Final layer — average instead of concatenate:

$$h_i' = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k\, \mathbf{W}^k h_j\right)$$

## Graph-Level Readout

$$h_G = \text{READOUT}\!\left(\{h_v^{(L)} : v \in V\}\right)$$

| Readout | Formula | Properties |
|---|---|---|
| Mean | $\frac{1}{|V|}\sum_v h_v$ | Size-invariant, loses magnitude |
| Sum | $\sum_v h_v$ | Injective (distinguishes graphs), size-dependent |
| Max | $\max_v h_v$ | Captures extremes, loses count info |

## Common Tasks

| Task | Level | Loss | Output |
|---|---|---|---|
| Node classification | Node | Cross-entropy | $\hat{y}_v = \text{softmax}(h_v^{(L)})$ |
| Link prediction | Edge | BCE | $\hat{y}_{uv} = \sigma(h_u^T h_v)$ |
| Graph classification | Graph | Cross-entropy | $\hat{y}_G = \text{MLP}(h_G)$ |

## Over-Smoothing

- After $k$ GCN layers, each node aggregates from its $k$-hop neighborhood
- Too many layers: all node embeddings converge to the same value
- Typical GNNs use 2-4 layers
- Mitigations: residual connections, JumpingKnowledge, DropEdge

## PyTorch Geometric Quick Reference

```python
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

# Data object
data.x          # [num_nodes, num_features]
data.edge_index # [2, num_edges] — COO format
data.y          # labels
data.batch      # graph assignment for batched graphs

# GCN layer
conv = GCNConv(in_channels, out_channels)
out = conv(x, edge_index)

# GAT layer
conv = GATConv(in_channels, out_channels, heads=8)
out = conv(x, edge_index)  # shape: [N, heads * out_channels]

# Graph-level pooling
graph_emb = global_mean_pool(x, batch)  # [num_graphs, features]
```

## Key Dimensions

| Symbol | Meaning | Typical Values |
|---|---|---|
| $N$ | Number of nodes | 10 - 100,000+ |
| $F$ | Input feature dim | 1 - 1,433 (Cora) |
| $H$ | Hidden dim | 16 - 256 |
| $L$ | Number of layers | 2 - 4 |
| $K$ | GAT attention heads | 4 - 8 |
| $C$ | Number of classes | 2 - 70 |

## Benchmark Datasets

| Dataset | Task | Nodes | Edges | Features | Classes |
|---|---|---|---|---|---|
| Cora | Node clf. | 2,708 | 10,556 | 1,433 | 7 |
| CiteSeer | Node clf. | 3,327 | 9,104 | 3,703 | 6 |
| PubMed | Node clf. | 19,717 | 88,648 | 500 | 3 |
| MUTAG | Graph clf. | ~18/graph | ~20/graph | 7 | 2 |
| PROTEINS | Graph clf. | ~39/graph | ~73/graph | 3 | 2 |

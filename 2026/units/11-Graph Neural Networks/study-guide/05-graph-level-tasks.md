# 05 — Graph-Level Tasks

## Discovery

> **From node embeddings to graph-level predictions.** GCN and GAT produce embeddings for individual nodes — but many real-world problems require predictions about entire graphs. Is this molecule toxic? Will this protein fold? Is this social network a bot ring? To make graph-level predictions, we need a **readout function** that collapses all node embeddings into a single vector representing the whole graph. The choice of readout has deep theoretical implications for what the GNN can and cannot distinguish.

---

## 1. Three Levels of Graph Tasks

| Task | Input | Output | Example |
|---|---|---|---|
| **Node classification** | Graph + node features | Label per node | Classify papers in a citation network |
| **Link prediction** | Graph + node features | Probability per node pair | Predict missing friendships |
| **Graph classification** | Entire graph | Label per graph | Predict molecular toxicity |

All three start with the same GNN backbone that produces node embeddings $\{h_v^{(L)}\}$. They differ in how those embeddings are used.

---

## 2. Node Classification

**Goal:** Predict a label $y_v$ for each node $v$.

**Pipeline:**
1. Run $L$ layers of GNN to get $h_v^{(L)}$ for all $v$
2. Apply a classifier: $\hat{y}_v = \text{softmax}(W_c \, h_v^{(L)} + b_c)$
3. Loss: cross-entropy on labeled nodes

**Semi-supervised setting** (Kipf & Welling):
- Only a few nodes have labels (e.g., 20 per class in Cora)
- The GNN propagates information through the graph structure
- Unlabeled nodes benefit from their neighbors' labels

```python
# Node classification with GCN
out = model(data.x, data.edge_index)  # [N, num_classes]
loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
```

---

## 3. Link Prediction

**Goal:** Predict whether an edge $(u, v)$ exists (or should exist).

**Pipeline:**
1. Run GNN to get node embeddings
2. Score each potential edge: $s_{uv} = f(h_u^{(L)}, h_v^{(L)})$
3. Loss: binary cross-entropy on positive (existing) and negative (sampled non-existing) edges

**Scoring functions:**

| Method | Formula | Notes |
|---|---|---|
| Dot product | $s_{uv} = h_u^T h_v$ | Simple, symmetric |
| Bilinear | $s_{uv} = h_u^T R h_v$ | Learnable relation matrix |
| MLP | $s_{uv} = \text{MLP}([h_u \| h_v])$ | Most expressive |
| Distance | $s_{uv} = -\|h_u - h_v\|$ | Metric-based |

**Negative sampling:** For each positive edge $(u, v) \in E$, sample a negative edge $(u, v')$ where $(u, v') \notin E$.

```python
# Link prediction scoring
pos_score = (h[edge_index[0]] * h[edge_index[1]]).sum(dim=1)
neg_score = (h[neg_edge_index[0]] * h[neg_edge_index[1]]).sum(dim=1)
loss = F.binary_cross_entropy_with_logits(
    torch.cat([pos_score, neg_score]),
    torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
)
```

---

## 4. Graph Classification

**Goal:** Predict a label $y_G$ for an entire graph $G$.

**Pipeline:**
1. Run GNN to get node embeddings $\{h_v^{(L)} : v \in V\}$
2. Apply **readout** to get graph embedding: $h_G = \text{READOUT}(\{h_v^{(L)}\})$
3. Apply classifier: $\hat{y}_G = \text{MLP}(h_G)$

This is the core new concept in this section.

---

## 5. Readout Functions

The readout function must be **permutation-invariant** — reordering nodes should not change the graph embedding.

### Mean Pooling

$$h_G = \frac{1}{|V|} \sum_{v \in V} h_v^{(L)}$$

- **Pro:** Size-invariant — doesn't depend on number of nodes
- **Con:** Loses information about graph size; cannot distinguish a graph with one node at feature $x$ from a graph with 100 nodes all at feature $x$

### Sum Pooling

$$h_G = \sum_{v \in V} h_v^{(L)}$$

- **Pro:** More expressive — can distinguish different graph sizes
- **Pro:** Proven to be **injective** for multisets (Xu et al., 2018)
- **Con:** Magnitude depends on graph size — may need normalization

### Max Pooling

$$h_G = \max_{v \in V} h_v^{(L)} \quad \text{(element-wise)}$$

- **Pro:** Captures the most salient features
- **Con:** Loses information about how many nodes have each feature

### Theoretical Ranking (Xu et al., 2018)

**Sum > Mean > Max** in terms of expressiveness.

Sum pooling can distinguish multisets that mean and max cannot:

| Multiset | Sum | Mean | Max |
|---|---|---|---|
| $\{1, 1, 2\}$ | 4 | 1.33 | 2 |
| $\{1, 2, 2\}$ | 5 | 1.67 | 2 |
| $\{1, 1, 1, 2\}$ | 5 | 1.25 | 2 |

Mean and max cannot distinguish $\{1, 2, 2\}$ from $\{1, 1, 1, 2\}$ (max gives 2 for both), but sum can.

---

## 6. Hierarchical Pooling

For large graphs, flat readout may lose structural information. **Hierarchical pooling** progressively coarsens the graph:

### DiffPool (Ying et al., 2018)

Learn a soft assignment matrix $S^{(l)} \in \mathbb{R}^{N_l \times N_{l+1}}$ that clusters nodes:

$$X^{(l+1)} = S^{(l)T} Z^{(l)}, \quad A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}$$

This creates a coarsened graph with fewer nodes, and the process repeats.

### TopKPooling

Select the top-$k$ nodes based on a learned score, drop the rest.

---

## 7. Batching Graphs in PyTorch Geometric

When training on multiple graphs, PyG batches them into a single disconnected graph:

```
Graph 1: nodes [0,1,2], edges [(0,1),(1,2)]
Graph 2: nodes [0,1],   edges [(0,1)]

Batched: nodes [0,1,2,3,4], edges [(0,1),(1,2),(3,4)]
batch =        [0,0,0,1,1]  ← which graph each node belongs to
```

The `batch` vector enables graph-level pooling:

```python
from torch_geometric.nn import global_mean_pool, global_add_pool

# Pool node embeddings to graph embeddings
graph_emb = global_mean_pool(node_emb, batch)  # [num_graphs, F]
```

---

## 8. Complete Graph Classification Model

```python
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Graph-level readout
        x = global_mean_pool(x, batch)  # [num_graphs, hidden]

        # Classification
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
```

**Training loop:**

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    out = model(data.x, data.edge_index, data.batch)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
```

---

## 9. Applications

| Domain | Graph Structure | Task | Readout |
|---|---|---|---|
| Drug discovery | Atoms = nodes, bonds = edges | Predict toxicity/activity | Sum/Mean → MLP |
| Materials science | Crystal structure as graph | Predict properties | Mean → MLP |
| Social networks | Users = nodes, connections = edges | Community detection | Node classification |
| Recommendation | Users + items as bipartite graph | Predict ratings | Link prediction |
| Program analysis | AST/CFG as graph | Bug detection | Graph classification |
| Physics simulation | Particles = nodes, interactions = edges | Predict next state | Node regression |

---

## 10. Evaluation Metrics

| Task | Metric | Notes |
|---|---|---|
| Node classification | Accuracy, F1 | Evaluate on test nodes |
| Link prediction | AUC-ROC, AP | Rank positive vs. negative edges |
| Graph classification | Accuracy, AUC | Evaluate on test graphs |

**Important:** For node classification, use the standard train/val/test split. For graph classification, use $k$-fold cross-validation since datasets are often small.

---

## Investigate

1. For a graph with 4 nodes and embeddings $[1, 0]$, $[0, 1]$, $[1, 1]$, $[0, 0]$, compute the graph embedding under mean, sum, and max readout.
2. Give an example of two different graphs that max pooling maps to the same embedding but sum pooling distinguishes.
3. Why does PyG batch graphs into a single disconnected graph instead of padding?

---

## Master

1. Implement a graph classification pipeline from scratch: GCN backbone + sum readout + MLP classifier.
2. Prove that sum readout is injective for multisets of node embeddings (under certain conditions on the embedding space).
3. Compare mean, sum, and max readout on the MUTAG dataset. Report accuracy and analyze which readout best captures molecular properties.

---

## Connect

- **From GCN/GAT (Sections 03-04):** These produce node embeddings — the input to readout functions.
- **From message passing (Section 02):** The number of GNN layers determines the receptive field, which affects what structural information the readout can capture.
- **To the Weisfeiler-Leman test (advanced):** The expressiveness of GNNs is bounded by the 1-WL graph isomorphism test. Sum aggregation + injective update matches 1-WL exactly (GIN).

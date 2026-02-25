# 01 — Graph Representations

## Discovery

> **Representing relationships as mathematical structures.** Every graph is a set of nodes and edges, but *how* you store it determines what operations are fast, what memory you need, and how it interfaces with neural network layers. The adjacency matrix — a simple square matrix of 0s and 1s — turns graph operations into linear algebra.

---

## 1. Graphs: The Basics

A **graph** $G = (V, E)$ consists of:
- $V = \{v_1, v_2, \ldots, v_N\}$ — a set of $N$ **nodes** (or vertices)
- $E \subseteq V \times V$ — a set of **edges** connecting pairs of nodes

Graphs can be:
- **Undirected**: $(v_i, v_j) \in E \implies (v_j, v_i) \in E$
- **Directed**: edges have direction
- **Weighted**: edges carry a weight $w_{ij} \in \mathbb{R}$
- **Attributed**: nodes and/or edges have feature vectors

---

## 2. Adjacency Matrix

The **adjacency matrix** $A \in \{0, 1\}^{N \times N}$ encodes all edges:

$$A_{ij} = \begin{cases} 1 & \text{if } (v_i, v_j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**Properties:**
- Undirected graphs: $A = A^T$ (symmetric)
- No self-loops: $A_{ii} = 0$
- Space: $O(N^2)$

**Example — Triangle graph:**

Nodes: $\{1, 2, 3\}$, Edges: $\{(1,2), (2,3), (1,3)\}$

$$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

**Powers of $A$:** The entry $(A^k)_{ij}$ counts the number of walks of length $k$ from node $i$ to node $j$. This is fundamental to understanding how GNNs propagate information.

---

## 3. Degree Matrix

The **degree matrix** $D \in \mathbb{R}^{N \times N}$ is diagonal:

$$D_{ii} = \sum_{j=1}^{N} A_{ij} = \deg(v_i)$$

All off-diagonal entries are zero. For our triangle graph:

$$D = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2 \end{pmatrix}$$

**Why it matters:** The degree matrix appears in every normalization scheme for GNNs. It prevents high-degree nodes from dominating aggregation.

---

## 4. Node Feature Matrix

Each node $v_i$ has a feature vector $x_i \in \mathbb{R}^F$. Stacking all $N$ feature vectors:

$$X = \begin{pmatrix} x_1^T \\ x_2^T \\ \vdots \\ x_N^T \end{pmatrix} \in \mathbb{R}^{N \times F}$$

**Examples of node features:**
- Citation networks: bag-of-words of the paper ($F = 1433$ for Cora)
- Molecular graphs: atom type, charge, degree (one-hot encoded)
- Social networks: user profile features

---

## 5. Edge List

An **edge list** stores edges as a list of pairs:

$$\text{edges} = [(i_1, j_1), (i_2, j_2), \ldots, (i_M, j_M)]$$

where $M = |E|$.

**PyTorch Geometric format** — COO (Coordinate) as a $2 \times M$ tensor:

```python
edge_index = torch.tensor([
    [0, 1, 1, 2, 0, 2],  # source nodes
    [1, 0, 2, 1, 2, 0],  # target nodes
], dtype=torch.long)
```

For undirected graphs, each edge appears twice (both directions).

**Space:** $O(M)$ — much better than $O(N^2)$ for sparse graphs.

---

## 6. Adjacency List

An **adjacency list** maps each node to its neighbors:

```
0 → [1, 2]
1 → [0, 2]
2 → [0, 1]
```

**Space:** $O(N + M)$
**Neighbor lookup:** $O(\deg(v))$ — fast for message passing

---

## 7. Sparse Representations

Real-world graphs are **sparse**: $M \ll N^2$. A social network with 1 million users might have 100 million edges — storing the full $10^6 \times 10^6$ adjacency matrix would require ~1 TB.

**Sparse matrix formats:**
- **COO (Coordinate):** Store $(row, col, value)$ triples — same as edge list with weights
- **CSR (Compressed Sparse Row):** Efficient for row slicing (neighbor access)
- **CSC (Compressed Sparse Column):** Efficient for column slicing

```python
import torch
from torch_sparse import SparseTensor

# COO to sparse
adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                   sparse_sizes=(num_nodes, num_nodes))
```

**Key insight:** GNN operations like $\tilde{A}X$ can be performed efficiently using sparse matrix multiplication, even for graphs with millions of nodes.

---

## 8. Self-Loops

Adding **self-loops** means each node is its own neighbor:

$$\hat{A} = A + I_N$$

This ensures that during message passing, a node's own features are included in the aggregation. The augmented degree matrix:

$$\hat{D}_{ii} = \sum_j \hat{A}_{ij} = D_{ii} + 1$$

For our triangle graph:

$$\hat{A} = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}, \quad \hat{D} = \begin{pmatrix} 3 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 3 \end{pmatrix}$$

---

## 9. Normalization

Raw aggregation $AX$ sums neighbor features, but nodes with many neighbors get disproportionately large values.

**Row normalization** (random walk):

$$D^{-1}A \implies (D^{-1}A)_{ij} = \frac{A_{ij}}{d_i}$$

Each row sums to 1 — this averages neighbor features.

**Symmetric normalization** (GCN):

$$D^{-1/2}AD^{-1/2} \implies (D^{-1/2}AD^{-1/2})_{ij} = \frac{A_{ij}}{\sqrt{d_i \cdot d_j}}$$

Preserves the symmetry of the adjacency matrix. With self-loops:

$$\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$$

This is the standard GCN normalization.

---

## 10. When to Use Which Representation

| Representation | Best For | Limitation |
|---|---|---|
| Adjacency matrix (dense) | Small graphs, hand computation, theory | $O(N^2)$ memory |
| Edge list / COO | GPU computation, PyG | No fast neighbor lookup |
| Adjacency list | CPU message passing, BFS/DFS | Not GPU-friendly |
| CSR sparse | Large-graph matrix operations | Complex to modify |

---

## Investigate

1. For a graph with $N = 1000$ nodes and average degree 10, compare the memory of a dense adjacency matrix vs. an edge list.
2. Compute $A^2$ for a path graph $1 - 2 - 3 - 4$. What do the entries tell you?
3. Why does symmetric normalization $D^{-1/2}AD^{-1/2}$ preserve eigenvalue bounds better than row normalization $D^{-1}A$?

---

## Master

1. Implement conversion between adjacency matrix, edge list, and adjacency list in Python.
2. Given a weighted graph, construct the normalized Laplacian $L = I - D^{-1/2}AD^{-1/2}$ and verify its eigenvalues are in $[0, 2]$.
3. Show that for an undirected graph, $\text{trace}(A^k)$ counts the number of closed walks of length $k$.

---

## Connect

- **To message passing (Section 02):** The operation $AX$ is the simplest form of neighbor aggregation — Section 02 generalizes this.
- **To GCN (Section 03):** The normalized adjacency $\tilde{A}$ is the propagation matrix in GCN.
- **To spectral theory:** The graph Laplacian $L = D - A$ connects graph structure to eigenvalue analysis, which motivates spectral GNNs.

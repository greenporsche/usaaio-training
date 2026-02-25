# 02 — Message Passing

## Discovery

> **The fundamental primitive of graph neural networks.** Every GNN — GCN, GAT, GraphSAGE, GIN — is a special case of the same idea: each node updates its representation by collecting (aggregating) information from its neighbors and combining it with its own state. This is the **message-passing** framework (Gilmer et al., 2017). Once you understand this, every GNN architecture becomes a choice of three functions: MESSAGE, AGGREGATE, UPDATE.

---

## 1. The Message-Passing Framework

At each layer $l$, every node $v$ performs three operations:

### Step 1: Message

Each neighbor $u \in \mathcal{N}(v)$ sends a **message**:

$$m_{u \to v}^{(l)} = \text{MSG}\!\left(h_u^{(l)}, h_v^{(l)}, e_{uv}\right)$$

The message can depend on the sender's features, the receiver's features, and edge attributes.

### Step 2: Aggregate

Node $v$ collects all incoming messages:

$$\bar{m}_v^{(l)} = \text{AGG}\!\left(\left\{m_{u \to v}^{(l)} : u \in \mathcal{N}(v)\right\}\right)$$

The aggregation must be **permutation-invariant** — the result should not depend on the ordering of neighbors. Common choices: sum, mean, max.

### Step 3: Update

Node $v$ updates its representation:

$$h_v^{(l+1)} = \text{UPDATE}\!\left(h_v^{(l)}, \bar{m}_v^{(l)}\right)$$

This combines the node's own state with the aggregated neighborhood information.

---

## 2. Why Permutation Invariance?

Graphs have no canonical node ordering. If we relabel the nodes, the GNN output should be the same. This means the AGGREGATE function must be a **set function** — invariant to the order of its inputs.

Valid aggregations:
- $\text{SUM}: \bar{m}_v = \sum_{u \in \mathcal{N}(v)} m_u$
- $\text{MEAN}: \bar{m}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} m_u$
- $\text{MAX}: \bar{m}_v = \max_{u \in \mathcal{N}(v)} m_u$ (element-wise)

**Not valid:** concatenation in a fixed order (depends on ordering).

---

## 3. Matrix Form

For a simple message-passing layer where MSG is the identity and AGG is sum:

$$\bar{M} = A \cdot H^{(l)}$$

Row $v$ of $AH^{(l)}$ is $\sum_{u \in \mathcal{N}(v)} h_u^{(l)}$ — the sum of neighbor features.

With self-loops and normalization:

$$\bar{M} = \tilde{A} \cdot H^{(l)}$$

This is exactly the GCN propagation (before applying the weight matrix).

---

## 4. Worked Example

Consider this 4-node graph:

```
    1
   / \
  2   3
   \ /
    4
```

Edges: $\{(1,2), (1,3), (2,4), (3,4)\}$. Node features (1D for simplicity):

$$h_1^{(0)} = 1, \quad h_2^{(0)} = 2, \quad h_3^{(0)} = 3, \quad h_4^{(0)} = 4$$

**Layer 1 with SUM aggregation (no self-loops, no weights):**

| Node $v$ | $\mathcal{N}(v)$ | $\bar{m}_v = \sum_{u \in \mathcal{N}(v)} h_u^{(0)}$ |
|---|---|---|
| 1 | $\{2, 3\}$ | $2 + 3 = 5$ |
| 2 | $\{1, 4\}$ | $1 + 4 = 5$ |
| 3 | $\{1, 4\}$ | $1 + 4 = 5$ |
| 4 | $\{2, 3\}$ | $2 + 3 = 5$ |

After UPDATE (if $\text{UPDATE}(h_v, \bar{m}_v) = \bar{m}_v$):

$$h_1^{(1)} = 5, \quad h_2^{(1)} = 5, \quad h_3^{(1)} = 5, \quad h_4^{(1)} = 5$$

All nodes have the same representation! This is an early sign of **over-smoothing**.

**With self-loops and MEAN aggregation:**

| Node $v$ | $\mathcal{N}(v) \cup \{v\}$ | $\bar{m}_v = \text{mean}$ |
|---|---|---|
| 1 | $\{1, 2, 3\}$ | $(1 + 2 + 3)/3 = 2.0$ |
| 2 | $\{2, 1, 4\}$ | $(2 + 1 + 4)/3 = 2.33$ |
| 3 | $\{3, 1, 4\}$ | $(3 + 1 + 4)/3 = 2.67$ |
| 4 | $\{4, 2, 3\}$ | $(4 + 2 + 3)/3 = 3.0$ |

Better — nodes retain distinct values because self-loops preserve their own features.

---

## 5. Receptive Field and $k$-Hop Neighborhoods

After $k$ layers of message passing, each node's representation depends on all nodes within $k$ hops.

- **1 layer:** direct neighbors
- **2 layers:** neighbors of neighbors
- **$k$ layers:** $k$-hop neighborhood

The **receptive field** of node $v$ after $k$ layers is:

$$\mathcal{N}^k(v) = \{u : d(u, v) \leq k\}$$

where $d(u, v)$ is the shortest-path distance.

**Example:** In our 4-node graph, after 2 layers, node 1's receptive field includes all nodes (diameter = 2). After just 1 layer, node 1 cannot "see" node 4.

---

## 6. Over-Smoothing

**Problem:** As the number of layers increases, node representations converge to the same value.

**Why it happens:** Each layer smooths features over neighborhoods. After enough layers, every node's receptive field covers the entire graph, and the averaging effect makes all representations identical.

**Formally:** Let $\tilde{A}$ be the normalized adjacency. Repeated application:

$$H^{(k)} = \tilde{A}^k X$$

As $k \to \infty$, $\tilde{A}^k$ converges to a rank-1 matrix (for connected graphs), meaning all rows become identical.

**Empirical evidence:** GCN accuracy on Cora peaks at 2-3 layers, then drops with more layers.

**Mitigations:**
- **Residual connections:** $h_v^{(l+1)} = h_v^{(l)} + \text{GNN}(h_v^{(l)}, \ldots)$
- **JumpingKnowledge:** Concatenate representations from all layers
- **DropEdge:** Randomly remove edges during training
- **PairNorm / NodeNorm:** Normalize node features to prevent convergence

---

## 7. Comparison of Aggregation Functions

| Aggregation | Formula | Strengths | Weaknesses |
|---|---|---|---|
| Sum | $\sum_u h_u$ | Injective (counts multiplicity) | Sensitive to degree |
| Mean | $\frac{1}{|\mathcal{N}|}\sum_u h_u$ | Degree-invariant | Cannot distinguish different-sized neighborhoods |
| Max | $\max_u h_u$ | Captures salient features | Loses information about distribution |

**Xu et al. (2018)** proved that SUM aggregation is the most expressive — it can distinguish graph structures that MEAN and MAX cannot. This led to the **Graph Isomorphism Network (GIN)**.

---

## 8. Message Passing as Matrix Multiplication

The elegance of the message-passing framework is that it can be expressed as sparse matrix operations:

```python
# Simple message passing: aggregate neighbor features
h_neighbors = torch.sparse.mm(adj, h)  # [N, F]

# With learned transformation
h_new = torch.sparse.mm(adj, h @ W)    # [N, F_out]

# With nonlinearity
h_new = relu(torch.sparse.mm(adj, h @ W))
```

This is why GNNs are efficient — sparse matrix multiplication is $O(M \cdot F)$, not $O(N^2 \cdot F)$.

---

## 9. Generalized Message Passing (MPNN)

Gilmer et al. (2017) formalized the **Message Passing Neural Network** framework:

$$m_v^{(l+1)} = \sum_{u \in \mathcal{N}(v)} M_l(h_v^{(l)}, h_u^{(l)}, e_{vu})$$

$$h_v^{(l+1)} = U_l(h_v^{(l)}, m_v^{(l+1)})$$

where $M_l$ and $U_l$ are learnable functions (typically MLPs).

This subsumes:
- **GCN:** $M_l(h_v, h_u, e) = \frac{1}{\sqrt{\hat{d}_v \hat{d}_u}} W h_u$, $U_l(h_v, m_v) = \sigma(m_v)$
- **GAT:** $M_l$ includes attention weighting
- **GIN:** $U_l(h_v, m_v) = \text{MLP}((1+\epsilon) h_v + m_v)$

---

## Investigate

1. Trace 2 rounds of message passing (SUM, with self-loops) on a 5-node path graph with features $[1, 2, 3, 4, 5]$.
2. Why must the aggregation function be permutation-invariant? Give a concrete example where a non-invariant function fails.
3. For a complete graph $K_n$, after 1 layer of MEAN aggregation with self-loops, what is each node's new feature?

---

## Master

1. Prove that after $k$ layers of message passing, node $v$'s representation depends only on the $k$-hop subgraph rooted at $v$.
2. For the normalized adjacency $\tilde{A}$ of a connected graph, show that $\lim_{k \to \infty} \tilde{A}^k$ has identical rows (hint: Perron-Frobenius theorem).
3. Implement a generic message-passing layer in PyTorch that takes MSG, AGG, and UPDATE as arguments.

---

## Connect

- **From graph representations (Section 01):** The adjacency matrix $A$ directly implements SUM aggregation: $AX$ sums neighbor features.
- **To GCN (Section 03):** GCN is message passing with symmetric normalization and a learnable weight matrix.
- **To GAT (Section 04):** GAT replaces fixed normalization with learned attention weights.
- **To GIN (advanced):** The most expressive message-passing GNN uses SUM aggregation with an MLP update.

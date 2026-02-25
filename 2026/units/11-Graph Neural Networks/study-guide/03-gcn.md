# 03 — Graph Convolutional Networks (GCN)

## Discovery

> **Kipf & Welling (2016) — bridging spectral and spatial graph processing.** The Graph Convolutional Network is arguably the most important GNN architecture. It starts from a principled spectral theory — filtering signals in the graph frequency domain — and through a series of clever approximations arrives at an elegantly simple formula: multiply by the normalized adjacency, apply a weight matrix, activate. The derivation from spectral to spatial is a masterclass in simplification.

---

## 1. Motivation: Convolution on Graphs

On images, a convolution slides a filter over a regular grid. On graphs, there is no grid — nodes have varying numbers of neighbors in no particular spatial order.

**Two approaches:**
- **Spectral:** Define convolution in the frequency domain using the graph Fourier transform
- **Spatial:** Define convolution as aggregation over local neighborhoods

GCN starts spectral and ends spatial.

---

## 2. Spectral Graph Theory Foundations

### The Graph Laplacian

$$L = D - A$$

**Normalized Laplacian:**

$$L_{\text{norm}} = I - D^{-1/2}AD^{-1/2} = D^{-1/2}LD^{-1/2}$$

$L_{\text{norm}}$ is symmetric positive semi-definite, so it has a complete set of real eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_N \leq 2$.

### Eigendecomposition

$$L_{\text{norm}} = U \Lambda U^T$$

where:
- $U = [u_1, u_2, \ldots, u_N]$ — orthonormal eigenvectors (the **graph Fourier basis**)
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ — eigenvalues (the **graph frequencies**)

### Graph Fourier Transform

For a signal $x \in \mathbb{R}^N$ on the graph:

$$\hat{x} = U^T x \quad \text{(transform to frequency domain)}$$

$$x = U \hat{x} \quad \text{(inverse transform)}$$

Small eigenvalues = smooth signals (similar values on connected nodes).
Large eigenvalues = high-frequency signals (different values on connected nodes).

---

## 3. Spectral Convolution

Convolution in the vertex domain corresponds to multiplication in the spectral domain:

$$g_\theta \star x = U \, g_\theta(\Lambda) \, U^T x$$

where $g_\theta(\Lambda) = \text{diag}(g_\theta(\lambda_1), \ldots, g_\theta(\lambda_N))$ is a spectral filter.

**Problem:** This requires:
1. Computing the eigendecomposition — $O(N^3)$
2. Storing $N$ filter parameters — not transferable between graphs
3. Dense matrix multiplication with $U$ — $O(N^2)$

---

## 4. ChebNet: Chebyshev Approximation

**Key idea:** Approximate the spectral filter with a $K$-th order Chebyshev polynomial:

$$g_{\theta'}(\Lambda) \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{\Lambda})$$

where $\tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I$ rescales eigenvalues to $[-1, 1]$, and $T_k$ are Chebyshev polynomials.

Since $T_k(\tilde{\Lambda})$ can be computed from $T_k(\tilde{L}_{\text{norm}})$ without eigendecomposition:

$$g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L}_{\text{norm}}) \, x$$

**Properties:**
- $K$ parameters instead of $N$
- $K$-localized: only aggregates from $K$-hop neighbors
- $O(K \cdot M)$ computation (sparse matrix-vector products)

---

## 5. From ChebNet to GCN: The Key Simplification

Kipf & Welling make two simplifications:

### Simplification 1: $K = 1$ (first-order)

$$g_{\theta'} \star x \approx \theta'_0 T_0(\tilde{L}) \, x + \theta'_1 T_1(\tilde{L}) \, x = \theta'_0 x + \theta'_1 \tilde{L} x$$

With $\lambda_{\max} \approx 2$ (approximate for normalized Laplacian):

$$\tilde{L} = \frac{2}{\lambda_{\max}}L_{\text{norm}} - I \approx L_{\text{norm}} - I = -(D^{-1/2}AD^{-1/2})$$

So:

$$g_{\theta'} \star x \approx \theta'_0 x - \theta'_1 D^{-1/2}AD^{-1/2} x$$

### Simplification 2: Single parameter

Set $\theta = \theta'_0 = -\theta'_1$ to reduce to one parameter:

$$g_\theta \star x \approx \theta \left(I + D^{-1/2}AD^{-1/2}\right) x$$

Note that $I + D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0, 2]$, which can cause numerical instability with repeated application.

### Renormalization Trick

Replace $I + D^{-1/2}AD^{-1/2}$ with $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ where $\hat{A} = A + I$ and $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$:

$$g_\theta \star x \approx \theta \, \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} \, x$$

This is the **renormalization trick** — it adds self-loops and renormalizes, constraining eigenvalues to $[0, 1]$.

---

## 6. The GCN Propagation Rule

Generalizing to $F$-dimensional input and $F'$-dimensional output with a weight matrix $W \in \mathbb{R}^{F \times F'}$:

$$\boxed{H^{(l+1)} = \sigma\!\left(\tilde{A} \, H^{(l)} \, W^{(l)}\right)}$$

where:

$$\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}, \quad \hat{A} = A + I, \quad \hat{D}_{ii} = \sum_j \hat{A}_{ij}$$

**Per-node form:**

$$h_v^{(l+1)} = \sigma\!\left(W^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{h_u^{(l)}}{\sqrt{\hat{d}_v \cdot \hat{d}_u}}\right)$$

where $\hat{d}_v = d_v + 1$ is the degree with self-loop.

---

## 7. Complete Worked Example

**Graph:** Triangle with nodes $\{1, 2, 3\}$, all connected.

$$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}$$

**Step 1: Add self-loops**

$$\hat{A} = A + I = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$

**Step 2: Compute $\hat{D}$**

All rows sum to 3: $\hat{D} = 3I$

**Step 3: Compute $\tilde{A}$**

$$\hat{D}^{-1/2} = \frac{1}{\sqrt{3}}I$$

$$\tilde{A} = \frac{1}{\sqrt{3}}I \cdot \hat{A} \cdot \frac{1}{\sqrt{3}}I = \frac{1}{3}\hat{A} = \begin{pmatrix} 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \end{pmatrix}$$

**Step 4: Node features** (1D for simplicity)

$$X = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$$

**Step 5: Propagate** (without weight matrix for clarity)

$$\tilde{A}X = \begin{pmatrix} 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \\ 1/3 & 1/3 & 1/3 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \\ 2 \end{pmatrix}$$

All nodes get the mean — makes sense for a complete graph.

**For a non-uniform graph**, nodes would retain more distinct values. The normalization $\frac{1}{\sqrt{\hat{d}_i \hat{d}_j}}$ balances contributions based on the degrees of both sender and receiver.

---

## 8. Two-Layer GCN for Node Classification

The standard GCN for semi-supervised node classification:

$$Z = \text{softmax}\!\left(\tilde{A} \; \text{ReLU}\!\left(\tilde{A} X W^{(0)}\right) W^{(1)}\right)$$

- $X \in \mathbb{R}^{N \times F}$ — input features
- $W^{(0)} \in \mathbb{R}^{F \times H}$ — first layer weights
- $W^{(1)} \in \mathbb{R}^{H \times C}$ — second layer weights
- $Z \in \mathbb{R}^{N \times C}$ — class predictions for all nodes

**Loss:** Cross-entropy on labeled nodes only:

$$\mathcal{L} = -\sum_{v \in V_L} \sum_{c=1}^{C} y_{vc} \ln Z_{vc}$$

where $V_L$ is the (small) set of labeled nodes. This is **semi-supervised** — we train on a few labels but propagate information through the graph structure to all nodes.

---

## 9. Why Symmetric Normalization?

Three normalization options:

| Normalization | Formula | Effect |
|---|---|---|
| None | $\hat{A}$ | High-degree nodes dominate; scale explodes |
| Row (random walk) | $\hat{D}^{-1}\hat{A}$ | Averages neighbors; asymmetric |
| Symmetric | $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ | Balanced; symmetric; stable eigenvalues |

Symmetric normalization:
- Preserves the symmetry of $\hat{A}$ (important for spectral analysis)
- Weights the contribution of neighbor $u$ to node $v$ by $\frac{1}{\sqrt{\hat{d}_v \cdot \hat{d}_u}}$
- Both high-degree senders and high-degree receivers are downweighted

---

## 10. Implementation

### From Scratch

```python
import torch
import torch.nn.functional as F

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.01)

    def forward(self, A_hat, H):
        # A_hat: normalized adjacency [N, N]
        # H: node features [N, F_in]
        return A_hat @ H @ self.W  # [N, F_out]

def compute_normalized_adj(A):
    A_hat = A + torch.eye(A.size(0))
    D_hat = torch.diag(A_hat.sum(dim=1))
    D_hat_inv_sqrt = torch.diag(1.0 / torch.sqrt(A_hat.sum(dim=1)))
    return D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
```

### With PyTorch Geometric

```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

---

## Investigate

1. For a star graph (one center connected to 4 leaves), compute $\tilde{A}$ and one round of GCN propagation by hand.
2. Why does GCN use $K=1$ Chebyshev approximation rather than higher orders? What is the trade-off?
3. In the two-layer GCN, how many hops of information does each node aggregate?

---

## Master

1. Derive the full path from spectral convolution $g_\theta \star x = U g_\theta(\Lambda) U^T x$ to the GCN propagation rule, clearly stating each approximation.
2. Show that the eigenvalues of $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ are in $[0, 1]$ for graphs with self-loops.
3. Implement a 2-layer GCN from scratch (no PyG) and train it on a small graph with 20 labeled nodes.

---

## Connect

- **From message passing (Section 02):** GCN is message passing with MSG = identity, AGG = normalized sum, UPDATE = linear transform + activation.
- **To GAT (Section 04):** GAT replaces the fixed $\frac{1}{\sqrt{\hat{d}_v \hat{d}_u}}$ normalization with *learned* attention weights.
- **To graph-level tasks (Section 05):** GCN produces node embeddings; graph-level tasks require an additional readout step to pool these into a single vector.

# Exercises — 03 GCN

---

## Exercise 1: GCN Propagation by Hand

Consider a triangle graph with nodes $\{0, 1, 2\}$, all pairwise connected.

Node features (2D):

$$X = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$$

Weight matrix: $W = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ (identity, for simplicity).

**(a)** Compute $\hat{A} = A + I$ and $\hat{D}$.

**(b)** Compute $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$.

**(c)** Compute $H^{(1)} = \text{ReLU}(\tilde{A}XW)$.

**(d)** Now use $W = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ and recompute $H^{(1)}$. How does the weight matrix affect the output?

---

## Exercise 2: Star Graph GCN

Consider a star graph: node 0 (center) connected to nodes 1, 2, 3, 4 (leaves). No edges between leaves.

Node features (1D): $x_0 = 10, x_1 = 1, x_2 = 2, x_3 = 3, x_4 = 4$.

**(a)** Compute $\hat{D}$ for each node (with self-loops).

**(b)** Compute the GCN normalization factor $\frac{1}{\sqrt{\hat{d}_i \hat{d}_j}}$ for the edge between node 0 and node 1.

**(c)** Compute one GCN layer output for node 0 (without weight matrix): $h_0^{(1)} = \sum_{j \in \mathcal{N}(0) \cup \{0\}} \frac{x_j}{\sqrt{\hat{d}_0 \hat{d}_j}}$.

**(d)** Compute one GCN layer output for node 1: $h_1^{(1)} = \sum_{j \in \mathcal{N}(1) \cup \{1\}} \frac{x_j}{\sqrt{\hat{d}_1 \hat{d}_j}}$.

**(e)** Compare with row normalization (MEAN). Which normalization gives more weight to the center node's own feature when computing node 1's output?

---

## Exercise 3: Spectral to Spatial Derivation

**(a)** Write the spectral convolution formula: $g_\theta \star x = \ldots$

**(b)** The $K=1$ Chebyshev approximation with $\lambda_{\max} \approx 2$ gives:

$$g_\theta \star x \approx \theta_0 x + \theta_1 D^{-1/2}AD^{-1/2}x$$

Set $\theta = \theta_0 = -\theta_1$ and simplify.

**(c)** Explain the renormalization trick: why replace $I + D^{-1/2}AD^{-1/2}$ with $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$?

**(d)** The eigenvalues of $I + D^{-1/2}AD^{-1/2}$ are in $[0, 2]$. Why is this problematic for deep networks? How does the renormalization trick help?

---

## Exercise 4: Two-Layer GCN

Using the graph from Exercise 1 (triangle) with the same features $X$:

$$W^{(0)} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \quad (2 \times 1), \quad W^{(1)} = \begin{pmatrix} 1 & -1 \end{pmatrix} \quad (1 \times 2)$$

**(a)** Compute $H^{(1)} = \text{ReLU}(\tilde{A}XW^{(0)})$. What are the dimensions?

**(b)** Compute $H^{(2)} = \tilde{A}H^{(1)}W^{(1)}$ (no activation on last layer).

**(c)** If this is a 2-class classification problem, apply softmax to $H^{(2)}$ row-wise to get predictions for each node.

**(d)** If the true labels are $y_0 = 0, y_1 = 1, y_2 = 0$, compute the cross-entropy loss.

---

## Exercise 5: GCN Properties

**(a)** A GCN with $L$ layers has a receptive field of $L$ hops. For the Cora dataset (diameter $\approx 19$), how many layers would you need for full graph coverage? Why is this a bad idea?

**(b)** Compute the number of parameters in a 2-layer GCN with:
- Input features: $F = 1433$ (Cora)
- Hidden dimension: $H = 16$
- Output classes: $C = 7$
Include bias terms.

**(c)** GCN applies the same weight matrix $W$ to all nodes. Why is this a form of **weight sharing**, similar to CNNs? What would happen if each node had its own weight matrix?

**(d)** Consider adding a skip connection: $H^{(l+1)} = \sigma(\tilde{A}H^{(l)}W^{(l)}) + H^{(l)}$. What constraint must the dimensions satisfy? How does this help with over-smoothing?

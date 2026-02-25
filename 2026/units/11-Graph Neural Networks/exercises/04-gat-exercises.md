# Exercises — 04 GAT

---

## Exercise 1: Attention Coefficient Computation

Three nodes in a fully connected graph (with self-loops). Node features ($F = 2$):

$$h_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad h_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad h_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

Weight matrix ($F' = 2$): $\mathbf{W} = I_2$ (identity).

Attention vector: $\vec{a} = [0.5, -0.5, 0.3, 0.3]^T$, so $\vec{a}_1 = [0.5, -0.5]^T$, $\vec{a}_2 = [0.3, 0.3]^T$.

**(a)** Compute $z_i = \mathbf{W}h_i$ for all nodes.

**(b)** For node 1, compute the raw attention scores $e_{11}$, $e_{12}$, $e_{13}$ (before and after LeakyReLU with negative slope 0.2).

**(c)** Apply softmax to get $\alpha_{11}$, $\alpha_{12}$, $\alpha_{13}$.

**(d)** Compute the output $h_1' = \sum_j \alpha_{1j} z_j$ (no activation).

---

## Exercise 2: Attention with Non-Identity Weights

Same graph and attention vector as Exercise 1, but now:

$$\mathbf{W} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$$

**(a)** Compute $z_i = \mathbf{W}h_i$ for all nodes.

**(b)** Recompute attention coefficients $\alpha_{1j}$ for node 1.

**(c)** How did the attention distribution change compared to Exercise 1? Why?

**(d)** Compute the output $h_1'$ and compare with Exercise 1(d).

---

## Exercise 3: Multi-Head Attention

Using the graph and features from Exercise 1, consider 2 attention heads:

**Head 1:** $\mathbf{W}^1 = I_2$, $\vec{a}^1 = [0.5, -0.5, 0.3, 0.3]^T$ (same as Exercise 1)

**Head 2:** $\mathbf{W}^2 = I_2$, $\vec{a}^2 = [-0.3, 0.3, 0.5, -0.5]^T$

**(a)** Compute the attention coefficients for node 1 under head 2: $\alpha_{11}^2, \alpha_{12}^2, \alpha_{13}^2$.

**(b)** Compute the output of head 2 for node 1: $h_1^{'(2)}$.

**(c)** For an intermediate layer, the multi-head output is the concatenation: $h_1' = h_1^{'(1)} \| h_1^{'(2)}$. What is the dimension of $h_1'$?

**(d)** For the final layer, the output is the average: $h_1' = \frac{1}{2}(h_1^{'(1)} + h_1^{'(2)})$. Compute this.

---

## Exercise 4: GAT vs. GCN Comparison

Consider a graph with 4 nodes arranged as:

```
1 --- 2
|     |
3 --- 4
```

(A cycle of length 4.) All nodes have degree 2.

**(a)** In GCN, what is the normalization factor $\frac{1}{\sqrt{\hat{d}_i \hat{d}_j}}$ for any edge in this graph? (With self-loops.)

**(b)** In GCN, all edges get the same weight. In GAT, suppose the learned attention gives $\alpha_{12} = 0.6$ and $\alpha_{13} = 0.1$ (with $\alpha_{11} = 0.3$ for the self-loop). What does this mean about node 1's relationship with its neighbors?

**(c)** Give a concrete scenario (in terms of node features and the task) where GAT's ability to assign different weights to different neighbors would outperform GCN.

**(d)** Now suppose all node features are identical: $h_i = h$ for all $i$. What are the GAT attention coefficients? What does GAT reduce to?

---

## Exercise 5: Complexity Analysis

**(a)** For a single GAT head with input dimension $F$ and output dimension $F'$:
- How many parameters are in $\mathbf{W}$?
- How many parameters are in $\vec{a}$?
- Total parameters per head?

**(b)** A GAT layer with $K = 8$ heads, $F = 64$ input, $F' = 8$ output per head:
- Total parameters?
- Output dimension (intermediate layer with concatenation)?
- Output dimension (final layer with averaging)?

**(c)** Compare the computational cost of GCN vs. GAT for a graph with $N$ nodes, $M$ edges, input dimension $F$, output dimension $F'$:
- GCN: $O(?)$ for the forward pass
- GAT (single head): $O(?)$ for the forward pass
- What is the additional cost of attention?

**(d)** For the Cora dataset ($N = 2708$, $M = 10556$, $F = 1433$), a GAT with 8 heads and $F' = 8$: how many total parameters in the first layer? Compare to a GCN layer with the same output dimension ($8 \times 8 = 64$).

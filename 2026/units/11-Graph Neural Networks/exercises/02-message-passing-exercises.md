# Exercises — 02 Message Passing

---

## Exercise 1: Basic Message Passing

Consider this graph:

```
    1
   / \
  2   3
  |   |
  4   5
```

Edges: $\{(1,2), (1,3), (2,4), (3,5)\}$ (undirected). Node features (1D): $h_1=1, h_2=2, h_3=3, h_4=4, h_5=5$.

**(a)** Perform one round of message passing with **SUM** aggregation (no self-loops). Write the new feature for each node.

**(b)** Perform one round with **MEAN** aggregation (no self-loops).

**(c)** Now add self-loops and repeat part (b) with MEAN aggregation.

**(d)** After 2 rounds of SUM aggregation with self-loops, what is node 1's feature? What is its receptive field?

---

## Exercise 2: Receptive Field

Using the same graph from Exercise 1:

**(a)** Draw the 1-hop receptive field of node 4 (which nodes influence $h_4^{(1)}$?).

**(b)** Draw the 2-hop receptive field of node 4.

**(c)** After how many layers can node 4 "see" node 5? What is the shortest path between them?

**(d)** For a graph with diameter $d$, what is the minimum number of message-passing layers needed for every node to receive information from every other node?

---

## Exercise 3: Over-Smoothing Demonstration

Consider a cycle graph with 6 nodes: $1 - 2 - 3 - 4 - 5 - 6 - 1$.

Node features: $h_1=0, h_2=1, h_3=0, h_4=1, h_5=0, h_6=1$.

**(a)** Compute features after 1 round of MEAN aggregation with self-loops.

**(b)** Compute features after 2 rounds of MEAN aggregation with self-loops (starting from original features, using results of part (a) as input to round 2).

**(c)** What value do the features converge to as the number of rounds $\to \infty$? Why?

**(d)** How does this illustrate the over-smoothing problem?

---

## Exercise 4: Matrix Form of Message Passing

For the graph:

```
Nodes: {0, 1, 2, 3}
Edges: {(0,1), (0,2), (1,2), (2,3)}
```

Node features:

$$X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}$$

**(a)** Write the adjacency matrix $A$ and compute $AX$. Verify that row $i$ of $AX$ equals the sum of features of node $i$'s neighbors.

**(b)** Compute $\hat{A} = A + I$ and $\hat{A}X$. What changed compared to part (a)?

**(c)** Compute $\hat{D}^{-1}\hat{A}X$ (row-normalized, with self-loops). Verify each node's new feature is the mean of its neighbors' features (including itself).

**(d)** If we apply a weight matrix $W = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$, compute $\hat{D}^{-1}\hat{A}XW$. What is the effect of $W$?

---

## Exercise 5: Aggregation Function Comparison

Consider this graph where node 0 has three different neighborhoods in three different graphs:

**Graph A:** Node 0 connected to nodes with features $\{[1,0], [1,0], [0,1]\}$

**Graph B:** Node 0 connected to nodes with features $\{[1,0], [0,1], [0,1]\}$

**Graph C:** Node 0 connected to nodes with features $\{[1,0], [0,1]\}$

**(a)** Compute the aggregated message to node 0 under SUM, MEAN, and MAX for each graph.

**(b)** Which aggregation functions can distinguish Graph A from Graph B? Which cannot?

**(c)** Which aggregation functions can distinguish Graph B from Graph C? Which cannot?

**(d)** Rank the three aggregation functions by expressiveness based on your answers. Does this match the theoretical result from Xu et al. (2018)?

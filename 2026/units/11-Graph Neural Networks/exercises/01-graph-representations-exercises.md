# Exercises — 01 Graph Representations

---

## Exercise 1: Adjacency Matrix Construction

Given the following undirected graph:

```
Nodes: {A, B, C, D, E}
Edges: {(A,B), (A,C), (B,C), (B,D), (D,E)}
```

**(a)** Write the adjacency matrix $A$ (use alphabetical ordering for rows/columns).

**(b)** Compute the degree matrix $D$.

**(c)** Compute $\hat{A} = A + I$ and $\hat{D}$.

**(d)** Verify that each row of $\hat{A}$ sums to the corresponding diagonal entry of $\hat{D}$.

---

## Exercise 2: Powers of the Adjacency Matrix

Consider the path graph: $1 - 2 - 3 - 4 - 5$.

**(a)** Write the adjacency matrix $A$.

**(b)** Compute $A^2$ by hand. What does $(A^2)_{13}$ represent?

**(c)** Compute $(A^2)_{15}$. Explain why this value makes sense in terms of graph connectivity.

**(d)** Without computing $A^3$, determine $(A^3)_{14}$. How many walks of length 3 go from node 1 to node 4?

---

## Exercise 3: Normalization

For the graph in Exercise 1:

**(a)** Compute the row-normalized adjacency $D^{-1}A$. Verify each row sums to 1.

**(b)** Compute the symmetric normalized adjacency $D^{-1/2}AD^{-1/2}$.

**(c)** Compute $\tilde{A} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ (with self-loops).

**(d)** Given node features $x_A = 1, x_B = 3, x_C = 5, x_D = 2, x_E = 4$, compute $\tilde{A}x$. Interpret the result for node A.

---

## Exercise 4: Representation Conversion

Given this edge list for a directed graph:

```
[(0,1), (0,2), (1,2), (2,0), (2,3), (3,3)]
```

**(a)** Draw the graph (including the self-loop on node 3).

**(b)** Write the adjacency matrix. Is it symmetric? Why or why not?

**(c)** Convert to an adjacency list representation.

**(d)** Write the edge list in PyTorch Geometric COO format (as a $2 \times M$ array of source and target nodes).

**(e)** If you wanted to treat this as undirected, what edges would you add?

---

## Exercise 5: Sparse vs. Dense Analysis

**(a)** A social network has $N = 10{,}000$ nodes and $M = 50{,}000$ edges. Compare the memory (in number of stored values) for:
- Dense adjacency matrix
- Edge list (COO format)
- CSR format (store row pointers, column indices, values)

**(b)** The Cora citation network has $N = 2{,}708$ nodes and $M = 5{,}278$ edges (before making undirected). What is the **sparsity** (fraction of zero entries in $A$)?

**(c)** For the GCN propagation $\tilde{A}X$ where $X \in \mathbb{R}^{N \times F}$:
- What is the computational cost using dense matrix multiplication?
- What is the cost using sparse matrix multiplication (in terms of $M$ and $F$)?

**(d)** At what edge density (ratio $M/N^2$) does sparse representation become less efficient than dense? Assume COO format stores 2 integers per edge and dense stores 1 value per entry.

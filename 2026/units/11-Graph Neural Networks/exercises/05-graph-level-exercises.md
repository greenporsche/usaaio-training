# Exercises — 05 Graph-Level Tasks

---

## Exercise 1: Readout Functions

A graph has 4 nodes with the following final-layer embeddings ($F = 3$):

$$h_1 = \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix}, \; h_2 = \begin{pmatrix} 0 \\ 3 \\ 1 \end{pmatrix}, \; h_3 = \begin{pmatrix} 1 \\ 1 \\ 4 \end{pmatrix}, \; h_4 = \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix}$$

**(a)** Compute the graph embedding $h_G$ under mean pooling.

**(b)** Compute $h_G$ under sum pooling.

**(c)** Compute $h_G$ under max pooling (element-wise).

**(d)** Notice that $h_1 = h_4$. Which readout functions can detect that this feature vector appears twice vs. once? Explain.

---

## Exercise 2: Distinguishing Graphs

Consider two graphs:

**Graph A:** 3 nodes with embeddings $\{[1, 0], [1, 0], [0, 1]\}$

**Graph B:** 2 nodes with embeddings $\{[1, 0], [0, 1]\}$

**(a)** Compute mean, sum, and max readout for both graphs.

**(b)** Which readout functions map Graph A and Graph B to the same embedding?

**(c)** Give a real-world scenario where Graph A and Graph B represent genuinely different objects that should be distinguished.

**(d)** Now consider **Graph C:** 4 nodes with embeddings $\{[1, 0], [1, 0], [0, 1], [0, 1]\}$. Which readout functions distinguish Graph C from both A and B?

---

## Exercise 3: Link Prediction Scoring

Given node embeddings after GNN:

$$h_A = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix}, \; h_B = \begin{pmatrix} 2 \\ 1 \\ 0 \end{pmatrix}, \; h_C = \begin{pmatrix} 0 \\ 0 \\ 3 \end{pmatrix}, \; h_D = \begin{pmatrix} 1 \\ 2 \\ 0.1 \end{pmatrix}$$

**(a)** Compute the dot-product score $s_{AB} = h_A^T h_B$ and $s_{AC} = h_A^T h_C$.

**(b)** Compute the Euclidean distance between $h_A$ and $h_B$, and between $h_A$ and $h_C$.

**(c)** Which pair is more likely to have an edge, based on each scoring method?

**(d)** Compute $s_{AD} = h_A^T h_D$. Compare $s_{AD}$ to $s_{AB}$. Which link would the model predict as more likely?

**(e)** Apply sigmoid to $s_{AB}$ and $s_{AC}$ to get probabilities. If the threshold is 0.5, which edges are predicted as positive?

---

## Exercise 4: Batching Graphs

Three graphs are batched together:

- **Graph 0:** 2 nodes, features $[[1,0], [0,1]]$, edges $[(0,1)]$
- **Graph 1:** 3 nodes, features $[[1,1], [2,0], [0,2]]$, edges $[(0,1), (1,2)]$
- **Graph 2:** 2 nodes, features $[[3,0], [0,3]]$, edges $[(0,1)]$

**(a)** Write the batched node feature matrix $X$ (7 rows total).

**(b)** Write the batched `edge_index` tensor (reindex nodes so they are globally unique).

**(c)** Write the `batch` vector that maps each node to its graph.

**(d)** Using the `batch` vector, compute `global_mean_pool` manually. Verify you get 3 graph embeddings of dimension 2.

**(e)** Compute `global_add_pool` (sum pooling). Compare to mean pooling — which graph has the largest norm?

---

## Exercise 5: Full Pipeline Design

You are designing a GNN for molecular property prediction (graph classification). Each molecule is a graph where atoms are nodes and bonds are edges.

**(a)** Node features: atomic number, degree, charge, aromatic (boolean). What is the input feature dimension $F$ if you one-hot encode atomic number (assume 10 types) and degree (assume max 4)?

**(b)** You use a 3-layer GCN with hidden dimension 64, followed by sum readout and a 2-layer MLP with hidden dimension 32 for binary classification. Draw the architecture and label all dimensions.

**(c)** How many trainable parameters does this model have? (Include biases.)

**(d)** The training set has 5,000 molecules. You use batch size 32. How many graphs are in one batch? How does PyG handle variable-size graphs in a single batch?

**(e)** After training, your model achieves 80% accuracy on the test set with mean readout, but 85% with sum readout. Hypothesize why sum readout performs better for molecular property prediction.

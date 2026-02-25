# Unit 11: Graph Neural Networks — Study Guide Overview

## AI 510-GNN

### What This Unit Covers

Graph Neural Networks (GNNs) extend deep learning to non-Euclidean data: social networks, molecules, knowledge graphs, citation networks, and any domain where entities have relationships. Unlike images (grids) or text (sequences), graphs have irregular structure — each node can have a different number of neighbors, and there is no canonical ordering.

This unit builds from the ground up: graph representations, the message-passing framework, two foundational architectures (GCN and GAT), and three core tasks (node classification, link prediction, graph classification).

### Prerequisites

- Linear algebra: matrix multiplication, eigenvalues, diagonalization
- Deep learning fundamentals: backpropagation, SGD, softmax, cross-entropy
- PyTorch basics: tensors, `nn.Module`, autograd
- Attention mechanisms (from Unit 10 / Transformers) are helpful for GAT

### Study Guide Structure

| Section | Topic | Key Concept |
|---|---|---|
| 01 | Graph Representations | Adjacency matrix, degree matrix, feature matrix |
| 02 | Message Passing | AGGREGATE-UPDATE framework |
| 03 | Graph Convolutional Networks | Spectral-to-spatial derivation |
| 04 | Graph Attention Networks | Learned attention over neighbors |
| 05 | Graph-Level Tasks | Readout, pooling, graph classification |

### Learning Objectives

By the end of this unit, you should be able to:

1. **Represent** graphs as adjacency matrices, edge lists, and node feature matrices
2. **Explain** the message-passing framework and how it generalizes graph operations
3. **Derive** the GCN propagation rule from spectral graph theory principles
4. **Implement** GCN and GAT layers from scratch using matrix operations
5. **Apply** GNNs to node classification, link prediction, and graph classification
6. **Compare** readout functions and understand their theoretical properties
7. **Use** PyTorch Geometric to build and train GNN models
8. **Identify** the over-smoothing problem and common mitigations

### How to Study

1. **Read** each study guide section in order — they build on each other
2. **Work** the exercises by hand first (especially small-graph computations)
3. **Code** the assignments — start from scratch before using PyG
4. **Connect** concepts: message-passing unifies GCN and GAT

### Key Papers

- Kipf & Welling (2016). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- Velickovic et al. (2017). *Graph Attention Networks.* ICLR 2018.
- Xu et al. (2018). *How Powerful are Graph Neural Networks?* ICLR 2019.
- Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML 2017.

### Connections to Other Units

- **Unit 8 (Deep Learning Fundamentals)**: GNNs are neural networks — same training loop, loss functions, optimizers
- **Unit 10 (Transformers)**: GAT uses the same attention mechanism; transformers on graphs are an active area
- **Unit 12 (Reinforcement Learning)**: GNNs can encode state spaces with relational structure

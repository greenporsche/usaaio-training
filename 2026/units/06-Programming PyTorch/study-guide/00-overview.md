# Unit 06: Programming PyTorch — Study Guide Overview

**Course**: AI 310 — Coding for AI 2: PyTorch
**USAAIO Relevance**: Essential for Round 2 (all coding problems use PyTorch). Critical for Units 07–12.

---

## What This Unit Covers

PyTorch is the framework you will use to implement every deep learning model in USAAIO. Unlike scikit-learn (which provides high-level `.fit()` calls), PyTorch gives you fine-grained control over tensors, gradients, and training — exactly what competition problems demand.

This unit teaches you to think in PyTorch: create and manipulate tensors, compute gradients automatically, build custom neural network modules, load data efficiently, and run complete training pipelines. By the end, you should be able to read a problem description and translate it into working PyTorch code under time pressure.

## Roadmap

| # | Topic | Key Ideas |
|---|---|---|
| 01 | Tensors | Creation, dtypes, devices, operations, NumPy interop, broadcasting |
| 02 | Autograd | Computation graphs, backward(), grad, higher-order derivatives |
| 03 | nn.Module | Custom layers, forward(), parameters, Sequential, ModuleList |
| 04 | Datasets & DataLoaders | Dataset protocol, batching, shuffling, custom collate |
| 05 | Loss Functions | MSE, CrossEntropy, BCE, custom losses, reduction |
| 06 | Optimizers | SGD, Adam, learning rate scheduling, weight decay |
| 07 | Training Loops | Full pipeline, GPU training, checkpointing, evaluation |

## Prerequisites

- **NumPy** (AI 210): Broadcasting, vectorized operations, indexing — PyTorch tensors mirror NumPy arrays
- **Linear Algebra** (AI 200): Matrix multiplication, transpose, shapes — you will manipulate tensor dimensions constantly
- **Calculus** (AI 210): Derivatives, chain rule — autograd automates this, but you must understand what it computes
- **Python** (AI 100/110): Classes, `__init__`, `__len__`, `__getitem__` — nn.Module and Dataset rely on OOP patterns

## How to Use This Guide

Each topic follows the **D-I-M-C** pattern:

1. **Discovery** — Why this concept exists, historical context, Socratic questions
2. **Intuition** — Visual explanations, ASCII diagrams, mental models
3. **Math** — Formal definitions and derivations where applicable
4. **Code** — Working PyTorch implementations with shape annotations

### Study Strategy

1. **First pass**: Read Discovery + Intuition for each topic to build mental models
2. **Second pass**: Work through Code sections, running each example yourself
3. **Third pass**: Implement from memory — close the guide and write the code
4. **Practice**: Complete the exercises for each topic
5. **Apply**: Work through the assignment notebooks (USAAIO Round 2 format)

## Connections to Other Units

```
AI 200 (Linear Algebra)  ──→ Tensor shapes, matrix operations
AI 210 (Calculus/NumPy)  ──→ Gradients (autograd), broadcasting
         ↓
    ┌────────────┐
    │  AI 310    │ ← YOU ARE HERE
    │  PyTorch   │
    └────────────┘
         ↓
AI 400 (ML2)   ──→ Implement clustering, PCA in PyTorch
AI 410 (DL1)   ──→ CNNs, backpropagation, ResNets
AI 420 (DL2)   ──→ RNNs, LSTMs, sequence models
AI 430 (NLP)   ──→ Tokenization, embeddings, language models
AI 440 (Transformers) ──→ Attention, MHA from scratch
```

## USAAIO Exam Patterns

On the USAAIO exam, PyTorch topics appear directly in **Round 2** coding problems:

- **Build modules from scratch**: No `nn.MultiheadAttention` — implement attention using `nn.Linear`, `reshape`, `permute`
- **Use autograd for physics**: `torch.autograd.grad` with `create_graph=True` for PDE-constrained optimization (PINNs)
- **Shape manipulation under pressure**: Reshape tensors between `(B, L, H*D)` and `(B, H, L, D)` without errors
- **Custom loss functions**: Combine multiple loss terms with weighting
- **Training from scratch**: No `model.fit()` — write the full forward/backward/update loop

### What 2025 Round 2 Tested

| Problem | PyTorch Skills Required |
|---|---|
| PINNs (Problem 1) | `autograd.grad`, `create_graph=True`, custom compound loss |
| Multi-Head Attention (Problem 2) | `nn.Linear`, tensor reshaping, `nn.Module` subclassing |
| GNN (Problem 3) | Custom `forward()`, message passing, `nn.Parameter` |

**Bottom line**: Every Round 2 problem is a PyTorch problem. Master this unit completely.

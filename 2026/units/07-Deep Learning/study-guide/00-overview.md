# Unit 07: Deep Learning — Study Guide Overview

**Course**: AI 410 — Deep Learning and Computer Vision 1
**USAAIO Relevance**: Core unit for Round 1 (theory) and Round 2 (implementation). Backpropagation, CNNs, and architecture analysis are tested every year. Round 2 coding problems require building layers from scratch.

---

## What This Unit Covers

Deep learning is the engine behind modern AI — from image recognition to language generation. This unit takes you inside the engine: how neural networks compute (forward propagation), how they learn (backpropagation), what makes them powerful (depth, nonlinearity, architecture design), and how to leverage existing models (transfer learning).

Unlike Unit 06 (PyTorch), which taught you the tool, this unit teaches you the theory and mathematics behind what the tool computes. You will understand every line of a training loop at the mathematical level, not just the API level.

By the end of this unit, you should be able to:
- Derive the forward and backward pass for any feedforward network by hand
- Compute output shapes and parameter counts for any architecture
- Explain why batch normalization and skip connections enable training of deep networks
- Implement CNNs, ResNet blocks, and transfer learning pipelines from scratch

## Roadmap

| # | Topic | Key Ideas |
|---|---|---|
| 01 | MLPs & Universal Approximation | Perceptrons, hidden layers, XOR problem, universal approximation theorem |
| 02 | Forward Propagation | Layer-by-layer computation, matrix view, shape tracking |
| 03 | Backpropagation | Chain rule, computational graphs, gradient flow, derivation by hand |
| 04 | Activation Functions | ReLU, sigmoid, tanh, GELU, vanishing/dying gradients |
| 05 | Batch Norm & Dropout | Normalization, regularization, training vs inference |
| 06 | CNNs | Convolution, kernels, stride, padding, pooling, parameter counting |
| 07 | Architectures | VGG, ResNet, GoogLeNet, design principles |
| 08 | Transfer Learning | Feature extraction, fine-tuning, pretrained models |

## Prerequisites

- **PyTorch** (AI 310): Tensors, autograd, nn.Module, training loops — you will implement everything in PyTorch
- **Linear Algebra** (AI 200): Matrix multiplication, transpose, shapes — the core of forward/backward pass
- **Calculus** (AI 210): Chain rule, partial derivatives, gradients — backpropagation is applied calculus
- **ML Fundamentals** (AI 300): Overfitting, regularization, train/test split — deep learning inherits all of these

## How to Use This Guide

Each topic follows the **D-I-M-C** pattern:

1. **Discovery** — Why this concept exists, historical context, Socratic questions
2. **Intuition** — Visual explanations, ASCII diagrams, mental models
3. **Math** — Formal definitions and derivations
4. **Code** — Working PyTorch implementations with shape annotations

### Study Strategy

1. **First pass**: Read Discovery + Intuition for each topic to build mental models
2. **Second pass**: Work through Math sections, derive everything by hand on paper
3. **Third pass**: Implement from scratch in Code sections
4. **Practice**: Complete exercises (many require hand computation — no code allowed)
5. **Apply**: Work through assignment notebooks (USAAIO Round 2 format)

## Connections to Other Units

```
AI 200 (Linear Algebra)  ──→ Matrix operations in forward/backward pass
AI 210 (Calculus/NumPy)  ──→ Chain rule (backpropagation), gradient computation
AI 300 (ML1)             ──→ Overfitting, regularization, loss functions
AI 310 (PyTorch)         ──→ Implementation framework
         ↓
    ┌────────────┐
    │  AI 410    │ ← YOU ARE HERE
    │  Deep      │
    │  Learning  │
    └────────────┘
         ↓
AI 420 (DL2)   ──→ RNNs, LSTMs, sequence models (extends backprop to time)
AI 430 (NLP)   ──→ Tokenization, embeddings, language models
AI 440 (Transformers) ──→ Attention (replaces CNNs for many tasks)
AI 450 (CV2)   ──→ Advanced vision: detection, segmentation, generation
```

## USAAIO Exam Patterns

### Round 1 (Theory)

- **Compute forward pass by hand**: Given weights, biases, activation, compute output for a small network
- **Compute backward pass by hand**: Given loss, compute gradients $\frac{\partial L}{\partial W}$ step by step
- **Parameter counting**: How many learnable parameters in a given architecture?
- **Output shape computation**: What shape does a tensor have after passing through Conv2d, BatchNorm, Pool, Flatten, Linear?
- **Identify problems**: Vanishing gradients, dying ReLU, overfitting — what causes them and how to fix

### Round 2 (Coding)

| Problem Type | Skills Required |
|---|---|
| Build layer from scratch | Implement Conv2d, BatchNorm, etc. using only `torch.Tensor` operations |
| Architecture analysis | Compute parameter counts, FLOPs, receptive fields programmatically |
| Transfer learning | Load pretrained model, modify head, freeze/unfreeze layers |
| PINNs | Use `autograd.grad` with `create_graph=True` for PDE constraints |
| Custom training | Full training pipeline with augmentation, scheduling, evaluation |

### What 2025 Round 2 Tested

| Problem | Deep Learning Skills Required |
|---|---|
| PINNs (Problem 1) | Forward pass design, autograd for second derivatives, compound loss |
| Multi-Head Attention (Problem 2) | Linear layers, tensor reshaping, understanding of architecture blocks |
| GNN (Problem 3) | Custom message passing (analogous to convolution), nn.Parameter |

**Bottom line**: Deep learning is the most heavily tested topic on USAAIO. Master both the theory (for Round 1) and implementation (for Round 2).

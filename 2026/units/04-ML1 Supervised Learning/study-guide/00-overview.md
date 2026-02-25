# Unit 04: ML1 Supervised Learning — Study Guide Overview

**Course**: AI 300 — Machine Learning 1
**USAAIO Relevance**: Core unit for Round 1 (High Honor Roll) and Round 2

---

## What This Unit Covers

Supervised learning is the foundation of modern machine learning. Given labeled data $(x_i, y_i)$, we learn a function $\hat{f}$ that maps inputs to outputs. This unit covers the classical algorithms that every ML practitioner must understand deeply — not just how to call `sklearn.fit()`, but the math, geometry, and code from first principles.

## Roadmap

| # | Topic | Key Ideas |
|---|---|---|
| 01 | Linear Regression | Normal equation, gradient descent, projection |
| 02 | Bias-Variance & Regularization | Decomposition, Ridge, Lasso, elastic net |
| 03 | Kernel Methods | Feature maps, kernel trick, kernel ridge regression |
| 04 | kNN & Cross-Validation | Distance metrics, k-fold CV, LOOCV |
| 05 | Logistic Regression | Sigmoid, MLE, cross-entropy, softmax |
| 06 | Classification Metrics | Precision, recall, F1, ROC-AUC |
| 07 | Loss Functions | MSE, CE, hinge, convexity |

## Prerequisites

- **Linear algebra** (AI 200): Matrix multiplication, inverse, eigenvalues, projections
- **Calculus & Optimization** (AI 210): Gradients, chain rule, convexity, gradient descent
- **Probability** (AI 210): Expectation, variance, Bayes' rule, MLE
- **NumPy** (AI 100/110): Broadcasting, vectorized operations, no-loop programming

## How to Use This Guide

Each topic follows the **D-I-M-C** pattern:

1. **Discovery** — Historical context, motivating questions, misconception traps
2. **Intuition** — Visual and geometric explanations, ASCII diagrams
3. **Math** — Rigorous derivations with no skipped steps
4. **Code** — NumPy from-scratch implementations + PyTorch equivalents

### Study Strategy

1. **First pass**: Read Discovery + Intuition sections for each topic
2. **Second pass**: Work through Math derivations with pen and paper
3. **Third pass**: Implement Code sections from memory (close the guide)
4. **Practice**: Complete the exercises for each topic
5. **Apply**: Work through the assignment notebooks

## Connections to Other Units

```
AI 200 (Linear Algebra) ──→ Normal equation, projections, PSD matrices
AI 210 (Optimization)   ──→ Gradient descent, convexity, MLE
         ↓
    ┌────────────┐
    │  AI 300    │ ← YOU ARE HERE
    │  ML1       │
    └────────────┘
         ↓
AI 400 (ML2) ──→ Decision trees, ensembles, EM, PCA
AI 410 (DL1) ──→ Neural networks, backpropagation
```

## USAAIO Exam Patterns

On the USAAIO exam, ML1 topics typically appear as:

- **Derive the gradient** of a loss function (logistic regression, ridge regression)
- **Compute a prediction** given weights and input (forward pass)
- **Identify the correct regularization** for a given scenario
- **Evaluate metrics** from a confusion matrix
- **Select hyperparameters** using cross-validation results
- **Analyze bias-variance** tradeoffs for model selection
- **Implement from scratch** in NumPy (no sklearn, no loops)

# Unit 02 Study Guide — Mathematical Foundations for AI (AI 200)

## Overview

This study guide covers the mathematical backbone of modern AI/ML, organized into eight core topics. Mastering these foundations is essential for USAAIO Round 1 Honor Roll and a prerequisite for all subsequent units.

---

## How to Use This Guide

Each topic file follows the **D-I-M-C** (Discovery-Intuition-Math-Code) format:

1. **Discovery** — Historical framing and Socratic exploration. Read actively, try to answer the questions before moving on.
2. **Intuition** — Geometric and visual understanding. This is where concepts "click."
3. **Math** — Rigorous definitions, derivations, and proofs. Nothing is skipped.
4. **Code** — NumPy from-scratch implementations, then PyTorch equivalents. Shape annotations on every operation.

---

## Topic Map & Dependencies

```
01 Vector Spaces ──────┐
                       ├──> 03 Eigenvalues/Eigenvectors ──> 04 SVD ──> 05 Projections & PCA
02 Matrix Operations ──┘                                       │
                                                               │
06 Probability & Statistics ───────────────────────────────────┤
                                                               │
07 Multivariable Calculus ─────────────────────────────────────┼──> 08 Convex Optimization
                                                               │
                                                     [Unit 03+]┘
```

**Recommended order**: Follow the numbering 01 through 08. Topics 01-02 can be studied in parallel. Topic 06 is somewhat independent and can be studied alongside 03-05.

---

## Topics

| # | File | Topic | USAAIO Weight |
|---|------|-------|---------------|
| 01 | `01-vector-spaces.md` | Vector spaces, basis, Gram-Schmidt | Medium |
| 02 | `02-matrix-operations.md` | Matrix multiply, inverse, determinant | Medium |
| 03 | `03-eigenvalues-eigenvectors.md` | Eigendecomposition, spectral theorem | **High** |
| 04 | `04-svd-decompositions.md` | SVD, truncated SVD, Eckart-Young | **High** |
| 05 | `05-projections-pca.md` | Projection, PCA (both formulations) | **Very High** |
| 06 | `06-probability-statistics.md` | Bayes, distributions, MLE | **High** |
| 07 | `07-multivariable-calculus.md` | Gradient, Jacobian, Hessian, chain rule | **High** |
| 08 | `08-convex-optimization.md` | Gradient descent, convexity, KKT | Medium-High |

---

## Companion Materials

- **Cheat Sheet**: `../cheat-sheet.md` — dense 2-page formula reference
- **Exercises**: `../exercises/` — 50 competition-style problems with solutions
- **Assignments**: `../assignments/` — 10 Jupyter notebooks (Easy to Contest difficulty)
- **Existing Assignment**: `../assignments/AI 200, linalg, eigendecomposition, assignment.md` — BeaverEdge eigendecomposition problem set

---

## External Resources

- [Mathematics for Machine Learning](https://mml-book.github.io/) — Deisenroth, Faisal, Ong (free textbook)
- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — visual intuition
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — matrix calculus reference

---

## Time Estimates

| Activity | Suggested Time |
|----------|---------------|
| Study guide (all 8 topics) | 15-20 hours |
| Exercises (50 problems) | 8-10 hours |
| Assignments (10 notebooks) | 12-15 hours |
| **Total** | **35-45 hours** |

Start with the study guide, do exercises for each topic as you go, then tackle assignments.

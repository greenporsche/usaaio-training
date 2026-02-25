# Unit 05: ML2 Unsupervised Learning — Overview

**Course Level**: AI 400 (BeaverEdge)
**Estimated Study Time**: 20 hours
**USAAIO Relevance**: Core topics for Round 1 (multiple choice) and Round 2 (implementation)

---

## What This Unit Covers

This unit bridges classical machine learning and the deeper world of unsupervised and ensemble methods. You will learn algorithms that appear **directly** on USAAIO problems — particularly PCA, K-means, and SVMs — plus the ensemble methods (random forests, boosting) that form the backbone of modern ML pipelines.

Despite the unit name "Unsupervised Learning," we also cover supervised ensemble methods (decision trees, random forests, boosting, SVMs) because the BeaverEdge AI 400 course bundles them together, and USAAIO tests them in the same section.

---

## Prerequisites

Before starting this unit, you should be comfortable with:

- **Linear algebra**: Matrix multiplication, eigenvalues/eigenvectors, SVD (Unit 02)
- **Calculus**: Gradients, chain rule, optimization (Unit 02)
- **Probability**: Distributions, Bayes' theorem (Unit 02)
- **Python/NumPy**: Vectorized operations, broadcasting (Unit 03)
- **Supervised ML basics**: Linear regression, logistic regression, gradient descent (Unit 04)

---

## Topic Roadmap

| # | Topic | Key Concepts | USAAIO Weight |
|---|-------|-------------|---------------|
| 01 | Decision Trees | Gini, entropy, information gain, pruning | Medium |
| 02 | Random Forests | Bagging, feature subsampling, OOB error | Medium |
| 03 | Boosting | AdaBoost, gradient boosting, learning rate | Medium |
| 04 | PCA | Eigendecomposition, variance explained, scree plot | **High** |
| 05 | Dimensionality Reduction | t-SNE, UMAP, visualization | Low–Medium |
| 06 | K-Means | Lloyd's algorithm, k-means++, elbow method | **High** |
| 07 | SVMs | Max margin, kernel trick, dual formulation | **High** |

---

## How to Use This Study Guide

1. **Read each topic in order** — they build on each other. Decision trees lead to forests lead to boosting. PCA leads to dimensionality reduction.
2. **Follow the D-I-M-C structure** in each file:
   - **Discovery** — Problem framing and motivation
   - **Intuition** — Visual/geometric understanding
   - **Math** — Formal derivations
   - **Code** — NumPy from-scratch + PyTorch equivalents
3. **Do the exercises** after each topic (5 per topic, timed 2–5 min each).
4. **Complete the assignment notebooks** for hands-on implementation practice.

---

## Connections to Other Units

- **Unit 04 (ML1)** provides the supervised learning foundation; this unit extends it with ensemble methods and introduces unsupervised paradigms.
- **Unit 06 (Deep Learning)** will use many concepts from here: dimensionality reduction for embeddings, ensemble ideas for model selection, kernel methods as inspiration for neural network feature spaces.

---

## Key Resources

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Reference implementations
- [DataCamp: Unsupervised Learning in Python](https://app.datacamp.com/learn/courses/unsupervised-learning-in-python)
- *An Introduction to Statistical Learning* (ISLR), Chapters 8–12
- *The Elements of Statistical Learning* (ESL), Chapters 9–14 (advanced)

# Practice Variations — 2026 USAAIO Round 1

Practice problem sets generated as variations on the [official 2026 Round 1 sample problems](../../../usaaio-official/2026/round-1-sample-problems/). Each set takes an original problem and produces exhaustive variations that drill the same core skills from different angles, with different numbers, dimensions, or constraints.

## Problem sets

| # | Original Problem | Variation Focus | Unit(s) |
|---|-----------------|-----------------|---------|
| 01 | PCA & Projections | Vector normalization, dot products, projection formula, residuals | 02, 05 |
| 02 | Permutation Matrices | Matrix representation of transformations, outer-product decomposition | 02 |
| 03 | Regularization & Bias-Variance | L1 vs L2 effects, bias-variance tradeoff reasoning | 04 |
| 04 | NumPy Array Manipulation | Array creation, squeeze/expand_dims/swapaxes, boolean indexing, flatten | 03 |
| 05 | Kernel Methods | Feature maps from kernels, matrix rank, SVD, trace/determinant, vectorized NumPy | 02, 04 |
| 06 | Derivatives | Activation function derivatives, chain rule, gradient of loss functions | 02 |
| 07 | SVM / Separating Hyperplanes | Hyperplane equation, maximum margin, unit normal vectors | 04 |
| 08 | Classification Metrics | Confusion matrix, accuracy, precision, recall, F1-score | 04 |
| 09 | Data Preprocessing | pandas exploration, min-max normalization, one-hot encoding, binning, train/test split | 03, 04 |
| 10 | Custom PyTorch Modules | `nn.Module` subclassing, activation functions, custom backward passes | 06 |
| 11 | MLP Geometry | Half-plane representations, Boolean logic with neurons, manual weight setting | 07 |
| 12 | Pretrained Models (ResNet) | Parameter counting, output shapes, layer analysis, transfer learning | 07 |
| 13 | End-to-End ML (kNN) | Full ML pipeline — preprocessing, feature engineering, scikit-learn, macro-F1 | 04 |
| 14 | End-to-End ML (Unsupervised) | Companion to #13 — clustering, dimensionality reduction, anomaly detection | 05 |

## File format

Each problem set exists in two formats:

- **`.md`** — Markdown source with solutions in collapsible `<details>` blocks and Python code fences
- **`.ipynb`** — Jupyter notebook generated from the markdown, with code blocks converted to executable cells and Unicode math converted to LaTeX

## Converting markdown to notebooks

The `convert_to_notebooks.py` script handles the `.md` → `.ipynb` conversion, including LaTeX math conversion, code cell extraction, and notebook formatting.

Convert a single file:

```sh
uv run python convert_to_notebooks.py problem-01-variations.md
```

Convert all problem sets:

```sh
uv run python convert_to_notebooks.py
```

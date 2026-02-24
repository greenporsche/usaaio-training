# USAAIO Problem Analysis

> Analysis of problem patterns to inform targeted preparation

---

## 2026 Round 1 Exam Structure

**From Official Instructions (January 30, 2026, 12:00-3:00pm EST)**:

| Aspect          | Details                                              |
| --------------- | ---------------------------------------------------- |
| **Problems**    | 9 problems total                                     |
| **Total Score** | 300 points                                           |
| **Duration**    | 3 hours                                              |
| **Format**      | Jupyter notebooks (.ipynb) via Google Colab          |
| **Submission**  | One .ipynb file per problem using official templates |

### Problem Types

- **Multi-part problems**: Each problem has multiple parts (e.g., Problem 1 has 5 parts)
- **Coding tasks**: Must follow restrictions (no unauthorized imports)
- **Non-coding (math) tasks**: Must typeset equations in LaTeX (no handwritten photos)
- **Multiple choice**: Exactly one correct answer, 0 points for wrong/blank

### Template Structure

- Problems 1-8: Structured with labeled parts (`Problem X, Part Y`)
- Problem 9: **Open-ended** with free-form solution + required report section

### Key Constraints

- **NO AI tools** (ChatGPT, Gemini, Colab AI features disabled)
- **NO search engines** or external resources
- **NO calculators** (use code cells instead)
- **NO past/sample problems** during exam
- Screen + face recording required

---

## Syllabus Split: Round 1 vs Round 2 - https://www.usaaio.org/2026-usa-na-aio

**CRITICAL DISTINCTION**:

| Round       | Units Covered | Topics                                                                      |
| ----------- | ------------- | --------------------------------------------------------------------------- |
| **Round 1** | Units 01-07   | Markdown, Math, NumPy, ML1, ML2, PyTorch, Deep Learning (MLP, CNN)          |
| **Round 2** | Units 01-12   | Everything above + NLP, Transformers, Computer Vision/GenAI, GNNs, Advanced |

This is **confirmed by the problem sets**:

- **2025 Round 1**: Fibonacci/eigendecomposition, Neural network basics, Data science (Titanic)
- **2025 Round 2**: Physics-Informed Neural Networks (PDEs), **Multi-Head Attention/Transformers**, **CLIP (multimodal)**
- **2026 Round 1 Sample**: PCA, matrices, regularization, NumPy, kernels, metrics, CNNs, ResNet

**Round 2 introduces**: Transformers (Unit 09), attention mechanisms, multimodal AI (CLIP), advanced architecttic patterns.

---

## Executive Summary

After analyzing the 2025 and 2026 USAAIO problems, clear patterns emerge:

### Round 1 Topic Distribution

| Category              | Weight | Primary Units             |
| --------------------- | ------ | ------------------------- |
| Linear Algebra        | ~30%   | Unit 02 (AI 200)          |
| NumPy Programming     | ~25%   | Unit 03 (AI 210)          |
| ML Fundamentals       | ~20%   | Unit 04 (AI 300)          |
| Deep Learning/PyTorch | ~20%   | Units 06-07 (AI 310, 410) |
| Data Analysis         | ~5%    | Unit 03-04                |

### Round 2 Additional Topics

| Category                                 | Primary Units          |
| ---------------------------------------- | ---------------------- |
| Attention Mechanisms                     | Unit 09 (Transformers) |
| Multi-Head Attention variants (GQA, MLA) | Unit 09                |
| Multimodal AI (CLIP)                     | Unit 10                |
| Physics-Informed Neural Networks         | Unit 07 (Advanced)     |

**Key Insight**: Problems heavily emphasize **mathematical derivation + from-scratch implementation**. High-level APIs are often forbidden.

---

## 2026 Round 1 Sample Problems - Detailed Analysis

### Problem 1: PCA & Projections

| Aspect             | Details                                                                     |
| ------------------ | --------------------------------------------------------------------------- |
| **Unit**           | 02 (Mathematical Foundations) + 05 (Unsupervised Learning)                  |
| **Concepts**       | Unit vectors, projection, residuals, PCA                                    |
| **Skills Tested**  | Vector normalization, dot product, projection formula                       |
| **Difficulty**     | Medium                                                                      |
| **Type**           | Multiple Choice (computation)                                               |
| **What They Want** | Can you compute projections by hand? Do you understand the geometry of PCA? |

**Atomic Skills**:

- Normalize a vector to unit length
- Compute projection: $\text{proj}_{\hat{e}}(\mathbf{x}) = (\mathbf{x} \cdot \hat{e})\hat{e}$
- Compute residual: $\mathbf{r} = \mathbf{x} - \text{proj}_{\hat{e}}(\mathbf{x})$

---

### Problem 2: Permutation Matrices

| Aspect             | Details                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| **Unit**           | 02 (Mathematical Foundations)                                                                          |
| **Concepts**       | Linear transformations, matrix representation, outer products                                          |
| **Skills Tested**  | Recognizing matrix operations, matrix decomposition                                                    |
| **Difficulty**     | Medium                                                                                                 |
| **Type**           | Multiple Choice + Short Answer                                                                         |
| **What They Want** | Do you understand how matrices encode transformations? Can you decompose matrices into basis elements? |

**Atomic Skills**:

- Identify permutation operations
- Construct matrix from transformation definition
- Decompose using outer products: $A = \sum_i \mathbf{e}^{(f(i))} \mathbf{e}^{(i)T}$

---

### Problem 3: Regularization & Bias-Variance

| Aspect             | Details                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| **Unit**           | 04 (ML1 - Supervised Learning)                                                 |
| **Concepts**       | L1 vs L2 regularization, bias-variance tradeoff                                |
| **Skills Tested**  | Conceptual understanding of regularization effects                             |
| **Difficulty**     | Easy-Medium                                                                    |
| **Type**           | Multiple Choice (conceptual)                                                   |
| **What They Want** | Do you understand WHY L1 induces sparsity? Can you reason about bias-variance? |

**Key Misconception Tested**:

- L1 produces sparse solutions (geometric: diamond constraint)
- L2 produces small but non-zero weights (geometric: circular constraint)
- Complexity ↑ → Bias ↓, Variance ↑

---

### Problem 4: NumPy Array Manipulation

| Aspect             | Details                                                   |
| ------------------ | --------------------------------------------------------- |
| **Unit**           | 03 (AI Programming in Python)                             |
| **Concepts**       | Array creation, shape manipulation, indexing              |
| **Skills Tested**  | squeeze, expand_dims, swapaxes, boolean indexing, flatten |
| **Difficulty**     | Easy                                                      |
| **Type**           | Coding                                                    |
| **What They Want** | Can you manipulate array shapes fluently without loops?   |

**Atomic Skills**:

- `np.random.seed()`, `np.random.randn()`
- `np.squeeze()` - remove dim of length 1
- `np.expand_dims()` - insert new axis
- `np.swapaxes()` - transpose specific axes
- Boolean indexing: `arr[arr > 1] = 100`
- `arr.flatten()`

---

### Problem 5: Kernel Methods (CRITICAL - Multi-Part)

| Aspect             | Details                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| **Unit**           | 04 (ML1) + 02 (Math)                                                          |
| **Concepts**       | Kernel functions, feature maps, kernel matrix properties, SVD                 |
| **Skills Tested**  | Deriving feature maps from kernels, matrix rank, trace/determinant            |
| **Difficulty**     | Hard                                                                          |
| **Type**           | Short Answer (derivation) + Coding                                            |
| **What They Want** | Deep understanding of kernel trick, linear algebra, vectorized implementation |

**Atomic Skills**:

- Expand kernel function to find φ(x)
- Understand: rank(K) ≤ dim(feature space)
- trace(K) = sum of squared singular values of Φ
- det(K) = product of eigenvalues
- **Vectorized kernel computation without loops**

**This is a flagship problem type** - combines theory + implementation.

---

### Problem 6: Derivatives

| Aspect             | Details                                                     |
| ------------------ | ----------------------------------------------------------- |
| **Unit**           | 02 (Mathematical Foundations)                               |
| **Concepts**       | Activation function derivatives, gradient of loss functions |
| **Skills Tested**  | Chain rule, matrix calculus                                 |
| **Difficulty**     | Medium                                                      |
| **Type**           | Short Answer                                                |
| **What They Want** | Can you derive gradients needed for backpropagation?        |

**Key Formulas**:

- $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$
- $\nabla_\theta \sum_n (y^{(n)} - \theta^T x^{(n)})^2 = -2\sum_n (y^{(n)} - \theta^T x^{(n)}) x^{(n)}$

---

### Problem 7: SVM / Separating Hyperplane

| Aspect             | Details                                                     |
| ------------------ | ----------------------------------------------------------- |
| **Unit**           | 04 (ML1 - Supervised Learning)                              |
| **Concepts**       | Hyperplane equation, maximum margin, unit normal vectors    |
| **Skills Tested**  | Geometric intuition, computing midpoint and normal          |
| **Difficulty**     | Medium                                                      |
| **Type**           | Short Answer + Coding                                       |
| **What They Want** | Do you understand SVM geometry? Can you implement the math? |

**Key Insight**: For two points, separating hyperplane passes through midpoint with normal along the line connecting them.

---

### Problem 8: Classification Metrics

| Aspect             | Details                                                   |
| ------------------ | --------------------------------------------------------- |
| **Unit**           | 04 (ML1)                                                  |
| **Concepts**       | Confusion matrix, accuracy, precision, recall, F1         |
| **Skills Tested**  | Computing metrics from confusion matrix                   |
| **Difficulty**     | Easy                                                      |
| **Type**           | Short Answer (computation)                                |
| **What They Want** | Do you know the formulas? Can you compute them correctly? |

**Formulas**:

- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

### Problem 9: Data Preprocessing (LONG)

| Aspect             | Details                                                        |
| ------------------ | -------------------------------------------------------------- |
| **Unit**           | 03 (NumPy) + 04 (ML1)                                          |
| **Concepts**       | Pandas basics, normalization, one-hot encoding, data splitting |
| **Skills Tested**  | Data wrangling FROM SCRATCH                                    |
| **Difficulty**     | Medium (tedious)                                               |
| **Type**           | Coding                                                         |
| **What They Want** | Can you preprocess data without sklearn?                       |

**Critical Constraint**: "Not allowed to use any existing normalization/encoding function"

**From-Scratch Skills**:

- Min-max normalization: $(x - x_{min}) / (x_{max} - x_{min})$
- One-hot encoding with pandas/numpy
- Train-test split with random seed

---

### Problem 10: Custom PyTorch Module (ReLU)

| Aspect             | Details                                    |
| ------------------ | ------------------------------------------ |
| **Unit**           | 06 (Programming PyTorch)                   |
| **Concepts**       | nn.Module, forward method                  |
| **Skills Tested**  | Building custom layers in PyTorch          |
| **Difficulty**     | Easy                                       |
| **Type**           | Coding                                     |
| **What They Want** | Do you understand PyTorch's module system? |

```python
class My_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.max(torch.zeros_like(x), x)
```

---

### Problem 11: MLP for Geometric Classification (NO TRAINING)

| Aspect             | Details                                                       |
| ------------------ | ------------------------------------------------------------- |
| **Unit**           | 07 (Deep Learning)                                            |
| **Concepts**       | MLP architecture, geometric interpretation of neural networks |
| **Skills Tested**  | Manually setting weights to achieve desired behavior          |
| **Difficulty**     | Hard                                                          |
| **Type**           | Coding                                                        |
| **What They Want** | Do you deeply understand how MLPs work geometrically?         |

**Key Insight**: Triangle = intersection of 3 half-planes. Each hidden neuron encodes one half-plane inequality.

---

### Problem 12: Pretrained Model Analysis (ResNet)

| Aspect             | Details                                                  |
| ------------------ | -------------------------------------------------------- |
| **Unit**           | 07 (Deep Learning)                                       |
| **Concepts**       | CNN architectures, parameter counting, transfer learning |
| **Skills Tested**  | Understanding model structure, freezing parameters       |
| **Difficulty**     | Medium                                                   |
| **Type**           | Coding + Short Answer                                    |
| **What They Want** | Can you navigate and modify pretrained models?           |

**Skills**:

- Count parameters: `sum(p.numel() for p in model.parameters())`
- Understand layer shapes
- Freeze parameters: `param.requires_grad = False`
- Add classification head

---

### Problem 13: End-to-End ML (kNN with constraints)

| Aspect             | Details                                               |
| ------------------ | ----------------------------------------------------- |
| **Unit**           | 04 (ML1)                                              |
| **Concepts**       | Full ML pipeline, kNN, feature engineering            |
| **Skills Tested**  | End-to-end problem solving with constraints           |
| **Difficulty**     | Medium-Hard                                           |
| **Type**           | Open-Ended Coding                                     |
| **What They Want** | Can you design a complete solution under constraints? |

**Evaluation**: F1-macro on hidden test set

---

## 2025 Round 1 Problems - Pattern Analysis

### Problem 1: Fibonacci & Eigendecomposition

| Aspect            | Details                                                        |
| ----------------- | -------------------------------------------------------------- |
| **Unit**          | 02 (Mathematical Foundations)                                  |
| **Concepts**      | Recurrence relations, eigenvalues, spectral decomposition      |
| **Skills Tested** | Matrix formulation, eigendecomposition, closed-form derivation |
| **Difficulty**    | Hard                                                           |
| **Type**          | Multi-part (derivation + coding)                               |

**This is a flagship problem**: Takes a simple recurrence and goes DEEP into linear algebra.

**Skills Chain**:

1. Write recurrence as matrix equation
2. Prove eigenvectors of symmetric matrices are orthogonal
3. Compute eigenvalues (characteristic polynomial)
4. Build orthonormal eigenvector matrix Q
5. Derive closed-form using spectral decomposition
6. Implement in NumPy (no loops!)

---

### Problem 2: Neural Network Fundamentals

| Aspect            | Details                                                    |
| ----------------- | ---------------------------------------------------------- |
| **Unit**          | 07 (Deep Learning) + 02 (Math)                             |
| **Concepts**      | Affine transformation, gradients, symmetric networks, rank |
| **Skills Tested** | Gradient computation, PyTorch module building              |
| **Difficulty**    | Medium-Hard                                                |
| **Type**          | Multi-part                                                 |

**Progression**:

1. Basic matrix multiplication
2. Derive gradients ∂y/∂x, ∂y/∂W, ∂y/∂b
3. Build affine layer from scratch (NumPy)
4. Build in PyTorch
5. Analyze tied-weight networks
6. Build ReLU module
7. Build MLP and train

---

### Problem 3: Data Science Pipeline (Titanic)

| Aspect            | Details                                            |
| ----------------- | -------------------------------------------------- |
| **Unit**          | 03-04                                              |
| **Concepts**      | Data loading, exploration, preprocessing, modeling |
| **Skills Tested** | Full data science workflow                         |
| **Difficulty**    | Medium                                             |
| **Type**          | Coding                                             |

---

## 2025 Round 2 Problems - Advanced Topics (Units 08-12)

### Round 2 Problem 1: Physics-Informed Neural Networks (PINN)

| Aspect            | Details                                                               |
| ----------------- | --------------------------------------------------------------------- |
| **Unit**          | 07 (Deep Learning) - Advanced Application                             |
| **Concepts**      | PDEs, automatic differentiation, custom loss functions, heat equation |
| **Skills Tested** | Advanced PyTorch, physics modeling, boundary/initial conditions       |
| **Difficulty**    | Very Hard                                                             |
| **Type**          | Multi-part (derivation + coding)                                      |

**Key Insight**: This is an advanced application of deep learning to solve differential equations. Requires understanding of:

- Partial differential equations (heat equation)
- Automatic differentiation for computing derivatives
- Custom loss functions combining PDE residual + boundary conditions

---

### Round 2 Problem 2: Multi-Head Attention & Variants (GQA, MLA)

| Aspect            | Details                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| **Unit**          | 09 (Transformers)                                                        |
| **Concepts**      | Multi-head attention, Group Query Attention, Multi-head Latent Attention |
| **Skills Tested** | Attention mechanism math, matrix decomposition, efficient implementation |
| **Difficulty**    | Very Hard                                                                |
| **Type**          | Multi-part (14 parts!)                                                   |

**This is a flagship Round 2 problem** - covers:

1. Query/Key/Value projection matrices and their shapes
2. Softmax attention scores with scaling
3. Building MHA from scratch in PyTorch (no loops!)
4. Group Query Attention (GQA) - fewer K/V heads
5. Multi-head Latent Attention (MLA) - low-rank decomposition
6. Proving GQA ⊂ MLA (GQA is special case)
7. KV-cache efficiency analysis

**Atomic Skills**:

- Tensor reshaping for multi-head: `(B, L, H*D) → (B, H, L, D)`
- Scaled dot-product attention: `softmax(QK^T / √d) V`
- Matrix rank and SVD
- Memory efficiency analysis

---

### Round 2 Problem 3: CLIP (Contrastive Language-Image Pre-Training)

| Aspect            | Details                                                            |
| ----------------- | ------------------------------------------------------------------ |
| **Unit**          | 10 (Computer Vision & Generative AI)                               |
| **Concepts**      | Multimodal learning, contrastive loss, image-text alignment        |
| **Skills Tested** | Understanding CLIP architecture, implementing contrastive learning |
| **Difficulty**    | Hard                                                               |
| **Type**          | Multi-part                                                         |

**Key Insight**: Tests understanding of:

- How vision and language encoders are trained jointly
- Contrastive loss functions
- Zero-shot classification using CLIP

---

## Round 1 vs Round 2: Preparation Strategy

### For Round 1 Only (Units 01-07)

Focus on:

1. **Linear algebra fundamentals** (eigendecomposition, SVD, projections)
2. **NumPy mastery** (vectorization, broadcasting, no loops)
3. **ML basics** (regression, regularization, kernels, metrics)
4. **PyTorch basics** (nn.Module, custom layers, training loops)
5. **CNN architectures** (ResNet, transfer learning)

### Additional for Round 2 (Units 08-12)

Add:

1. **Attention mechanisms** - MHA, GQA, MLA from scratch
2. **Transformer architecture** - encoder/decoder, positional encoding
3. **Multimodal AI** - CLIP, vision-language models
4. **Advanced applications** - PINNs, GNNs

---

## Key Patterns & Preparation Priorities

### Pattern 1: Math-First, Code-Second

Almost every problem requires mathematical understanding BEFORE coding. The code is just implementing the math.

**Preparation**: Practice deriving formulas by hand, then implementing.

### Pattern 2: From-Scratch Implementation

High-level APIs (sklearn, etc.) are often forbidden. You must implement:

- Normalization
- One-hot encoding
- Kernel matrices
- Basic layers (Linear, ReLU)
- Eigendecomposition-based solutions

**Preparation**: Practice NumPy vectorization extensively.

### Pattern 3: No Loops Allowed

Many coding tasks explicitly forbid loops. You must use:

- Broadcasting
- Outer products (`np.outer`)
- Matrix operations
- Boolean indexing

**Preparation**: Rewrite any loop-based solution to use vectorization.

### Pattern 4: Multi-Part Scaffolding

Problems build from simple to complex:

1. Compute something small by hand
2. Generalize to formula
3. Prove a property
4. Implement in code
5. Verify/apply

**Preparation**: Practice full problem arcs, not isolated skills.

### Pattern 5: Linear Algebra is Foundational

Nearly every problem touches linear algebra:

- Eigenvalues/eigenvectors
- SVD
- Matrix rank
- Projections
- Orthogonality

**Preparation**: Unit 02 is the most critical foundation.

---

## Recommended Preparation Order

### Phase 1: Foundations (40 hours)

1. **Unit 02**: Linear algebra (eigendecomposition, SVD, projections)
2. **Unit 03**: NumPy mastery (broadcasting, vectorization, no loops)

### Phase 2: Core ML (30 hours)

3. **Unit 04**: ML1 (regression, regularization, kernels, metrics)
4. **Unit 05**: ML2 (PCA, clustering)

### Phase 3: Deep Learning (30 hours)

5. **Unit 06**: PyTorch basics (tensors, modules, autograd)
6. **Unit 07**: Deep learning (MLP, backprop, CNNs)

### Phase 4: Practice (20 hours)

7. Solve all past problems under timed conditions
8. Focus on multi-part problems that combine concepts

---

## Concept → Problem Mapping

| Concept                | 2025     | 2026          |
| ---------------------- | -------- | ------------- |
| Eigendecomposition     | P1       | P2            |
| Matrix rank            | P2.8     | P5.3          |
| Projections/PCA        | -        | P1            |
| Kernels                | -        | P5            |
| Gradients              | P2.2     | P6            |
| L1/L2 Regularization   | -        | P3            |
| Bias-Variance          | -        | P3            |
| NumPy arrays           | P1.9-10  | P4, P5.5      |
| Pandas preprocessing   | P3       | P9            |
| Classification metrics | -        | P8            |
| PyTorch modules        | P2.5+    | P10, P11, P12 |
| MLP architecture       | P2.11-13 | P11           |
| CNN/Transfer learning  | -        | P12           |
| Full ML pipeline       | P3       | P13           |

---

## High-Yield Study Topics

Based on frequency and point value:

1. **Eigendecomposition** - Appears in flagship problems both years
2. **Vectorized NumPy** - Required for almost every coding task
3. **Kernel methods** - Deep understanding needed (feature maps, rank)
4. **PyTorch nn.Module** - Custom layer implementation
5. **Matrix calculus** - Gradients for backprop
6. **Classification metrics** - Easy points if you know formulas
7. **From-scratch preprocessing** - Normalization, one-hot encoding

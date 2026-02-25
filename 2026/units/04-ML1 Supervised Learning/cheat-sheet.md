# ML1 Supervised Learning — Cheat Sheet

> Quick reference for USAAIO 2026 | AI 300

---

## Linear Regression

| Item | Formula |
|---|---|
| Model | $\hat{y} = Xw + b$ |
| MSE Loss | $\mathcal{L}(w) = \frac{1}{n}\|Xw - y\|^2$ |
| Normal Equation | $\hat{w} = (X^TX)^{-1}X^Ty$ |
| Gradient | $\nabla_w \mathcal{L} = \frac{2}{n}X^T(Xw - y)$ |
| Geometric View | $\hat{y} = X(X^TX)^{-1}X^Ty$ is the projection of $y$ onto $\text{col}(X)$ |

---

## Bias-Variance Decomposition

$$E\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(f(x) - E[\hat{f}(x)]\right)^2}_{\text{Bias}^2} + \underbrace{E\left[\left(\hat{f}(x) - E[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

| Complexity | Bias | Variance | Total Error |
|---|---|---|---|
| Too simple (underfit) | High | Low | High |
| Just right | Medium | Medium | Lowest |
| Too complex (overfit) | Low | High | High |

---

## Regularization

| Method | Objective | Closed-Form | Effect |
|---|---|---|---|
| **Ridge** (L2) | $\|Xw-y\|^2 + \lambda\|w\|_2^2$ | $\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$ | Shrinks all coefficients |
| **Lasso** (L1) | $\|Xw-y\|^2 + \lambda\|w\|_1$ | No closed-form (use coordinate descent) | Sparsity (some $w_j = 0$) |
| **Elastic Net** | $\|Xw-y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$ | No closed-form | Sparse + grouped selection |

---

## Kernel Methods

| Kernel | $K(x_i, x_j)$ | Hyperparameters |
|---|---|---|
| Linear | $x_i^T x_j$ | None |
| Polynomial | $(x_i^T x_j + c)^d$ | $c \geq 0$, degree $d$ |
| RBF (Gaussian) | $\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$ | $\sigma$ (bandwidth) |

**Kernel trick**: $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ — compute inner products in high-dimensional feature space without explicit mapping.

**Kernel Ridge Regression**: $\hat{\alpha} = (K + \lambda I)^{-1}y$, prediction: $\hat{y}(x) = \sum_i \alpha_i K(x_i, x)$

**Mercer's condition**: $K$ must be symmetric positive semi-definite.

---

## k-Nearest Neighbors (kNN)

| Item | Detail |
|---|---|
| Prediction (classification) | $\hat{y} = \text{mode}\{y_j : x_j \in N_k(x)\}$ |
| Prediction (regression) | $\hat{y} = \frac{1}{k}\sum_{x_j \in N_k(x)} y_j$ |
| Common distances | Euclidean: $\|x-x'\|_2$, Manhattan: $\|x-x'\|_1$, Minkowski: $\|x-x'\|_p$ |
| Curse of dimensionality | Volume of unit ball $\to 0$ as $d \to \infty$; need $n$ exponential in $d$ |

**k selection**: Small $k$ = low bias, high variance. Large $k$ = high bias, low variance.

---

## Cross-Validation

| Method | Formula | Notes |
|---|---|---|
| k-fold CV | $\text{CV}_k = \frac{1}{k}\sum_{i=1}^{k} \mathcal{L}(f^{(-i)}, D_i)$ | Standard: $k = 5$ or $10$ |
| LOOCV | $\text{CV}_n = \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f^{(-i)}, (x_i, y_i))$ | Low bias, high variance, expensive |

---

## Logistic Regression

| Item | Formula |
|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| Model | $P(y=1|x) = \sigma(w^Tx + b)$ |
| Log-odds | $\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$ |
| NLL Loss | $\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$ |
| Gradient | $\nabla_w \mathcal{L} = \frac{1}{n}X^T(\hat{y} - y)$ |
| Decision boundary | $w^Tx + b = 0$ (hyperplane) |

**Key property**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

---

## Softmax (Multiclass)

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

| Item | Formula |
|---|---|
| Cross-entropy loss | $\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic}\log\hat{y}_{ic}$ |
| Gradient w.r.t. logits | $\nabla_{z}\mathcal{L} = \hat{y} - y$ (one-hot) |
| Log-sum-exp trick | $\log\sum e^{z_i} = m + \log\sum e^{z_i - m}$, where $m = \max(z)$ |

---

## Classification Metrics

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

| Metric | Formula | When to use |
|---|---|---|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Balanced classes only |
| Precision | $\frac{TP}{TP + FP}$ | Cost of FP is high |
| Recall (Sensitivity) | $\frac{TP}{TP + FN}$ | Cost of FN is high |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | Balance P and R |
| Specificity | $\frac{TN}{TN + FP}$ | |
| ROC-AUC | Area under TPR vs FPR curve | Threshold-independent evaluation |

---

## Loss Functions

| Loss | Formula | Use Case | Convex? |
|---|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Regression | Yes |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Robust regression | Yes (not smooth) |
| Huber | MSE if $|r| \leq \delta$, MAE otherwise | Robust regression | Yes |
| Binary CE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Binary classification | Yes |
| Hinge | $\max(0, 1 - y_i \cdot f(x_i))$ | SVM | Yes (not smooth) |
| Multiclass CE | $-\sum_c y_c \log\hat{y}_c$ | Multiclass classification | Yes |

---

## SVM (Support Vector Machine)

| Item | Formula |
|---|---|
| Decision function | $f(x) = w^Tx + b$ |
| Margin | $\frac{2}{\|w\|}$ |
| Hard-margin objective | $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^Tx_i + b) \geq 1$ |
| Soft-margin objective | $\min \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$ s.t. $y_i(w^Tx_i+b) \geq 1-\xi_i$ |
| Kernel SVM | $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$ |

---

## Quick Recipes

```
Linear Regression:  X^T X w = X^T y  →  solve linear system
Ridge:              (X^T X + λI) w = X^T y  →  always invertible
Gradient Descent:   w ← w - η · ∇L(w)  →  repeat until convergence
kNN:                sort by distance, take k nearest, vote/average
Logistic:           iterate  w ← w - η · X^T(σ(Xw) - y)/n
```

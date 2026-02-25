# AI 200 — Mathematical Foundations for AI: Cheat Sheet

> Dense reference for USAAIO 2026 Round 1 & Round 2. Keep this handy during practice.

---

## Symbol Definitions

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}, \mathbf{v}$ | Column vectors (bold lowercase) |
| $A, B, M$ | Matrices (uppercase) |
| $\mathbb{R}^{n}$ | $n$-dimensional real vector space |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ or $\mathbf{u}^\top \mathbf{v}$ | Inner (dot) product |
| $\|\mathbf{x}\|$ | Euclidean norm $\sqrt{\mathbf{x}^\top \mathbf{x}}$ |
| $I_n$ | $n \times n$ identity matrix |
| $\text{tr}(A)$ | Trace: sum of diagonal entries |
| $\det(A)$ or $|A|$ | Determinant |
| $\nabla f$ | Gradient vector |
| $\mathbf{J}$ | Jacobian matrix |
| $\mathbf{H}$ | Hessian matrix |
| $\mathbb{E}[X]$ | Expected value |
| $\text{Var}(X)$ | Variance |
| $\text{Cov}(X, Y)$ | Covariance |

---

## 1. Linear Algebra Core

### Vector Spaces & Basis
- **Vector space axioms**: closure under addition & scalar multiplication, associativity, commutativity, additive identity/inverse, distributivity, scalar identity
- **Linearly independent**: $c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_i = 0 \ \forall i$
- **Basis**: linearly independent spanning set; **dimension** = number of basis vectors
- **Rank**: $\text{rank}(A) = \dim(\text{col}(A)) = \dim(\text{row}(A))$
- **Gram-Schmidt**: $\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \frac{\langle \mathbf{v}_k, \mathbf{u}_j \rangle}{\langle \mathbf{u}_j, \mathbf{u}_j \rangle} \mathbf{u}_j$, then normalize $\mathbf{e}_k = \mathbf{u}_k / \|\mathbf{u}_k\|$

### Matrix Operations
- $(AB)^\top = B^\top A^\top$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$ (cyclic property)
- $\det(AB) = \det(A)\det(B)$; $\det(A^{-1}) = 1/\det(A)$
- $\det(cA) = c^n \det(A)$ for $A \in \mathbb{R}^{n \times n}$

---

## 2. Eigenvalues & Eigenvectors

$$A\mathbf{x} = \lambda \mathbf{x} \quad \Longleftrightarrow \quad \det(A - \lambda I) = 0$$

- **Eigendecomposition** (diagonalizable $A$): $A = Q\Lambda Q^{-1}$
  - $Q = [\mathbf{q}_0 \cdots \mathbf{q}_{N-1}]$ (right eigenvectors as columns)
  - $\Lambda = \text{diag}(\lambda_0, \ldots, \lambda_{N-1})$
- **Spectral theorem** (symmetric $A = A^\top$): $A = Q\Lambda Q^\top$ with $Q$ orthogonal
- **Properties**: $\text{tr}(A) = \sum \lambda_i$; $\det(A) = \prod \lambda_i$
- **Matrix powers**: $A^k = Q\Lambda^k Q^{-1}$
- **Outer product form**: $A = \sum_{i} \lambda_i \mathbf{q}_i \mathbf{p}_i^\top$ where $P = Q^{-\top}$ (left eigenvectors)

---

## 3. SVD — Singular Value Decomposition

$$A = U\Sigma V^\top$$

| Component | Size | Properties |
|-----------|------|------------|
| $U$ | $m \times m$ | Orthogonal; columns = left singular vectors (eigenvectors of $AA^\top$) |
| $\Sigma$ | $m \times n$ | Diagonal; $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $V$ | $n \times n$ | Orthogonal; columns = right singular vectors (eigenvectors of $A^\top A$) |

- **Relation to eigenvalues**: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$
- **Truncated SVD (rank-$k$)**: $A_k = U_k \Sigma_k V_k^\top$ — best rank-$k$ approximation (Eckart-Young)
- **Frobenius error**: $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$

---

## 4. Projections & PCA

- **Projection of $\mathbf{b}$ onto $\mathbf{a}$**: $\text{proj}_\mathbf{a} \mathbf{b} = \frac{\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top \mathbf{a}} \mathbf{a}$
- **Projection matrix onto column space of $A$**: $P = A(A^\top A)^{-1}A^\top$; $P^2 = P$, $P^\top = P$

### PCA Algorithm
1. Center data: $\bar{X} = X - \mu$
2. Covariance matrix: $C = \frac{1}{N-1}\bar{X}^\top \bar{X}$
3. Eigendecompose $C$: $C = Q\Lambda Q^\top$
4. Top-$k$ principal components: columns of $Q$ with largest eigenvalues
5. Project: $Z = \bar{X} Q_k$ — shape $(N, k)$
6. Reconstruct: $\hat{X} = Z Q_k^\top + \mu$
7. Variance explained: $\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$

**Equivalently via SVD**: $\bar{X} = U\Sigma V^\top \implies$ principal directions $= V_k$, scores $= U_k \Sigma_k$

---

## 5. Probability & Statistics

### Key Rules
- **Bayes' theorem**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- **Law of total probability**: $P(B) = \sum_i P(B|A_i)P(A_i)$
- **Chain rule**: $P(A \cap B) = P(A|B)P(B)$

### Distributions

| Distribution | PMF/PDF | $\mathbb{E}[X]$ | $\text{Var}(X)$ |
|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Categorical($\mathbf{p}$) | $\prod p_k^{x_k}$ | — | — |
| Gaussian($\mu, \sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Multivariate Gaussian | $\frac{1}{\sqrt{(2\pi)^d|\Sigma|}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ | $\boldsymbol{\mu}$ | $\Sigma$ |

### Key Identities
- $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$
- $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$
- **Covariance matrix**: $\Sigma_{ij} = \text{Cov}(X_i, X_j)$; always positive semi-definite
- **MLE for Gaussian**: $\hat{\mu} = \frac{1}{N}\sum x_i$, $\hat{\sigma}^2 = \frac{1}{N}\sum (x_i - \hat{\mu})^2$

---

## 6. Multivariable Calculus

### Gradient, Jacobian, Hessian

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \quad \mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \quad \mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

### Matrix Calculus Identities
| Expression | Derivative w.r.t. $\mathbf{x}$ |
|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$; if $A$ symmetric: $2A\mathbf{x}$ |
| $\|\mathbf{x} - \mathbf{b}\|^2$ | $2(\mathbf{x} - \mathbf{b})$ |
| $\text{tr}(A^\top B)$ w.r.t. $A$ | $B$ |

### Chain Rule (vectors)
$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{x}}^\top \frac{\partial L}{\partial \mathbf{z}} = \mathbf{J}^\top \nabla_\mathbf{z} L$$

---

## 7. Convex Optimization

### Convexity
- **Convex set**: $\forall \mathbf{x}, \mathbf{y} \in S, \ \theta \in [0,1]: \theta\mathbf{x} + (1-\theta)\mathbf{y} \in S$
- **Convex function**: $f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$
- **Twice-differentiable test**: $f$ convex $\iff$ $\mathbf{H} \succeq 0$ (Hessian is PSD)

### Gradient Descent
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

- **Learning rate** $\eta$: too large $\to$ diverge; too small $\to$ slow convergence
- **Momentum**: $\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)$; $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_{t+1}$

### Constrained Optimization
- **Lagrangian**: $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$
- **KKT conditions** (necessary for optimality):
  1. Stationarity: $\nabla_\mathbf{x} \mathcal{L} = 0$
  2. Primal feasibility: $g_i(\mathbf{x}) \leq 0$, $h_j(\mathbf{x}) = 0$
  3. Dual feasibility: $\lambda_i \geq 0$
  4. Complementary slackness: $\lambda_i g_i(\mathbf{x}) = 0$

---

## Tensor Shape Quick Reference

| Operation | Input Shapes | Output Shape |
|---|---|---|
| Matrix multiply $AB$ | $(m, n) \times (n, p)$ | $(m, p)$ |
| Outer product $\mathbf{u}\mathbf{v}^\top$ | $(m,) \times (n,)$ | $(m, n)$ |
| Dot product $\mathbf{u}^\top \mathbf{v}$ | $(n,) \times (n,)$ | scalar |
| Batch matmul | $(B, m, n) \times (B, n, p)$ | $(B, m, p)$ |
| Covariance matrix | data $(N, d)$ | $(d, d)$ |
| PCA projection $XQ_k$ | $(N, d) \times (d, k)$ | $(N, k)$ |
| SVD: $U$ | $(m, n) \to U$ | $(m, m)$ |
| SVD: $\Sigma$ | $(m, n) \to \Sigma$ | $(m, n)$ |
| SVD: $V^\top$ | $(m, n) \to V^\top$ | $(n, n)$ |

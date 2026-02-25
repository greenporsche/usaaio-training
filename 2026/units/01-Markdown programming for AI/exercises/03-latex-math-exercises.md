# LaTeX Math Exercises

**10 exercises** | Covers: fractions, Greek letters, matrices, summations, aligned equations, USAAIO formulas

---

## Exercise 1: Basic Fractions and Subscripts

**Target time**: 2 minutes

Typeset the following formula for min-max normalization:

> x-prime-sub-i equals x-sub-i minus x-min, divided by x-max minus x-min

The rendered result should look like:

$$
x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}
$$

<details>
<summary>Solution</summary>

```latex
$$
x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}
$$
```

Key points:
- `x'_i` uses the prime (apostrophe) for the normalized version
- `x_{\min}` uses `\min` inside a subscript brace group
- `\frac{numerator}{denominator}` for the fraction

</details>

---

## Exercise 2: Greek Letters in Context

**Target time**: 2 minutes

Typeset the gradient descent update rule:

> theta at time t+1 equals theta at time t minus eta times the gradient of L with respect to theta

The rendered result should look like:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

<details>
<summary>Solution</summary>

```latex
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$
```

Key points:
- `\theta_{t+1}` -- brace group needed for multi-character subscript
- `\eta` -- Greek letter for learning rate
- `\nabla_\theta` -- gradient operator subscripted with the variable

</details>

---

## Exercise 3: Matrix Notation

**Target time**: 3 minutes

Typeset the following 3x3 matrix equation:

> A times vector x equals vector b, where A is [[2, 1, 0], [1, 3, 1], [0, 1, 2]], x is [x1, x2, x3] (column), and b is [5, 10, 7] (column)

Use square bracket matrices (bmatrix) and bold for vectors.

<details>
<summary>Solution</summary>

```latex
$$
\mathbf{A}\mathbf{x} = \mathbf{b} \quad \text{where} \quad
\begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
=
\begin{bmatrix} 5 \\ 10 \\ 7 \end{bmatrix}
$$
```

Key points:
- `\mathbf{}` for bold vectors/matrices
- `\\` separates rows in the matrix
- `&` separates columns
- Column vectors have `\\` between entries (no `&`)

</details>

---

## Exercise 4: Softmax Function

**Target time**: 2 minutes

Typeset the softmax function with the condition:

> softmax of z-sub-i equals e-to-the-z-sub-i divided by the sum from j=1 to K of e-to-the-z-sub-j, for i = 1, ..., K

<details>
<summary>Solution</summary>

```latex
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \ldots, K
$$
```

Key points:
- `\text{softmax}` -- use `\text{}` for words in math mode
- `e^{z_i}` -- superscript with brace group
- `\sum_{j=1}^{K}` -- summation with lower and upper bounds
- `\ldots` -- horizontal ellipsis (three dots)
- `\quad` -- wide space before the condition

</details>

---

## Exercise 5: Cross-Entropy Loss (Binary)

**Target time**: 3 minutes

Typeset the binary cross-entropy loss function:

> L equals negative one over n times the sum from i=1 to n of [y-sub-i times log of y-hat-sub-i plus (1 minus y-sub-i) times log of (1 minus y-hat-sub-i)]

Use `\left[` and `\right]` for auto-sized brackets.

<details>
<summary>Solution</summary>

```latex
$$
L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$
```

Key points:
- `\hat{y}_i` -- hat accent followed by subscript
- `\left[` and `\right]` -- auto-sizing brackets
- The negative sign is outside the fraction
- `\log` (not `log`) -- uses the proper math operator font

</details>

---

## Exercise 6: Eigendecomposition

**Target time**: 3 minutes

Typeset the full eigendecomposition statement:

> A times v equals lambda times v, which implies A equals V Lambda V-inverse, where V = [v1 | v2 | ... | vn] and Lambda = diag(lambda-1, ..., lambda-n)

Use the `\Rightarrow` arrow and proper matrix notation.

<details>
<summary>Solution</summary>

```latex
$$
A\mathbf{v} = \lambda\mathbf{v} \quad \Rightarrow \quad A = V\Lambda V^{-1}
$$

where $V = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n]$ and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.
```

Key points:
- `\Rightarrow` for the implication arrow
- `V^{-1}` for the inverse
- `\Lambda` (capital) for the eigenvalue matrix
- `\lambda` (lowercase) for individual eigenvalues
- `\mid` for the vertical bar separating column vectors
- `\cdots` for centered ellipsis (vs `\ldots` for baseline)

</details>

---

## Exercise 7: Scaled Dot-Product Attention

**Target time**: 3 minutes

Typeset the attention formula from the "Attention Is All You Need" paper:

> Attention(Q, K, V) = softmax(Q times K-transpose divided by square root of d-sub-k) times V

Include the dimensions: Q is in R^{n x d_k}, K is in R^{m x d_k}, V is in R^{m x d_v}.

<details>
<summary>Solution</summary>

```latex
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$.
```

Key points:
- `\!\left(` -- negative thin space before the parenthesis tightens the gap after softmax
- `\sqrt{d_k}` -- square root command
- `K^T` -- transpose superscript
- `\mathbb{R}` -- blackboard bold for the reals
- `\times` -- multiplication sign in dimensions (not `x`)
- `\in` -- set membership

</details>

---

## Exercise 8: Multi-Line Derivation with Alignment

**Target time**: 4 minutes

Typeset the following logistic regression gradient derivation as an aligned equation:

Line 1: partial L / partial w_j = partial / partial w_j of (-sum from i=1 to n of [y_i log(sigma) + (1-y_i) log(1-sigma)])
Line 2: = -sum from i=1 to n of [y_i / sigma - (1-y_i) / (1-sigma)] times partial sigma / partial w_j
Line 3: = sum from i=1 to n of (sigma_i - y_i) times x_{ij}

Where sigma stands for the sigmoid output (use `\hat{y}_i` for readability).

<details>
<summary>Solution</summary>

```latex
$$
\begin{aligned}
\frac{\partial L}{\partial w_j}
  &= \frac{\partial}{\partial w_j} \left( -\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right] \right) \\
  &= -\sum_{i=1}^{n} \left[ \frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i} \right] \frac{\partial \hat{y}_i}{\partial w_j} \\
  &= \sum_{i=1}^{n} (\hat{y}_i - y_i)\, x_{ij}
\end{aligned}
$$
```

Key points:
- `\begin{aligned}...\end{aligned}` for multi-line alignment
- `&=` -- the `&` marks the alignment point (all `=` signs line up)
- `\\` -- line break within aligned
- Each line continues the derivation from the `=` sign
- `\,` thin space before $x_{ij}$ for readability

</details>

---

## Exercise 9: Gaussian Distribution

**Target time**: 3 minutes

Typeset the probability density function of the multivariate Gaussian distribution:

> f(x) = 1 / ((2*pi)^(d/2) * |Sigma|^(1/2)) * exp(-1/2 * (x - mu)^T * Sigma^(-1) * (x - mu))

Use bold for vectors, proper sizing, and `\det` or `|\Sigma|` for the determinant.

<details>
<summary>Solution</summary>

```latex
$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}} \exp\!\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$
```

Key points:
- `\mathbf{x}` for vector x, `\boldsymbol{\mu}` for vector mu (Greek bold)
- `(2\pi)^{d/2}` -- brace group for the fractional exponent
- `|\Sigma|^{1/2}` -- determinant using vertical bars
- `\Sigma^{-1}` -- matrix inverse
- `\exp\!\left(` -- exponential with auto-sized parentheses
- `(\mathbf{x} - \boldsymbol{\mu})^T` -- transpose of the difference vector

</details>

---

## Exercise 10: Complete USAAIO Formula Sheet

**Target time**: 5 minutes

Typeset ALL of the following formulas as a clean, labeled reference sheet. Use display math for each, with a bold label before each formula:

1. Sigmoid: sigma(z) = 1 / (1 + e^(-z))
2. ReLU: ReLU(z) = max(0, z)
3. Precision: TP / (TP + FP)
4. Recall: TP / (TP + FN)
5. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
6. Cosine Similarity: cos(a, b) = (a dot b) / (||a|| * ||b||)
7. L2 Regularization term: (lambda / 2) * sum of w_j^2

<details>
<summary>Solution</summary>

```markdown
**1. Sigmoid:**
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**2. ReLU:**
$$
\text{ReLU}(z) = \max(0, z)
$$

**3. Precision:**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**4. Recall:**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**5. F1-Score:**
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**6. Cosine Similarity:**
$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

**7. L2 Regularization:**
$$
R = \frac{\lambda}{2}\sum_{j=1}^{p} w_j^2
$$
```

Key points:
- `\sigma(z)` -- sigma as a function name
- `\max(0, z)` -- use `\max` for the math operator
- `\cdot` for dot product and scalar multiplication
- `\|\mathbf{a}\|` for norm (double pipe with backslash)
- `\frac{\lambda}{2}` -- lambda is the regularization strength

</details>

---

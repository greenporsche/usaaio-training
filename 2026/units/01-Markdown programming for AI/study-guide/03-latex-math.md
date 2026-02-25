# LaTeX Math for USAAIO

**Prerequisites**: `01-text-formatting.md`, `02-code-cells.md`
**USAAIO Relevance**: This is the single most important skill in Unit 01. Roughly 30-40% of Round 1 questions require you to typeset mathematical formulas. You will write loss functions, gradient derivations, attention mechanisms, eigendecompositions, and more -- all in LaTeX.

---

## Discovery

It is 1977. You are Donald Knuth, a computer scientist at Stanford. You have just received the proofs for Volume 2 of *The Art of Computer Programming* and you are horrified. The typesetter botched your mathematical formulas. Fractions are misaligned, subscripts are in the wrong place, and your beautiful equations look like garbage.

You decide: enough. You will build your own typesetting system. Over the next decade, you create **TeX** -- a system that gives you *complete control* over how mathematics is rendered.

**Try this**: Look at the following equation and imagine typing it as plain text:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

How would you represent fractions? Square roots? Superscripts? Matrices?

**Question**: Why can't we just use regular text for math? Consider: does "1/2" mean the same thing as $\frac{1}{2}$? What about "xi" versus $x_i$ versus $\xi$?

**Misconception trap**: LaTeX in Google Colab uses **MathJax** (a JavaScript renderer), not a full LaTeX installation. This means some advanced LaTeX packages (like `tikz` for diagrams) are not available. Stick to standard math commands.

---

## Intuition

What you just discovered is the fundamental problem LaTeX solves: **mathematical notation is inherently two-dimensional** (fractions stack vertically, matrices are grids, superscripts float up), but text is one-dimensional (a stream of characters). LaTeX bridges this gap with a simple idea: **commands describe structure, the renderer handles layout.**

### The Two Modes

| Mode | Syntax | Where It Appears |
|---|---|---|
| **Inline** | `$...$` | Within a sentence: "The loss is $L = 0.5$" |
| **Display** | `$$...$$` | Centered on its own line, larger |

**Rule of thumb**: Use inline for simple expressions ($x_i$, $n = 100$). Use display for anything with fractions, summations, or multi-line derivations.

### How LaTeX Commands Work

Every LaTeX command starts with a backslash: `\frac`, `\sum`, `\alpha`.

Commands that take arguments use curly braces: `\frac{numerator}{denominator}`.

Commands can be nested: `\frac{\partial L}{\partial \theta}` produces $\frac{\partial L}{\partial \theta}$.

### Visual Mental Model

Think of LaTeX as a tree:

```
\frac{...}{...}
  |         |
  |         +-- denominator (below the line)
  |
  +-- numerator (above the line)
```

```
\sum_{...}^{...}
      |       |
      |       +-- upper bound (top)
      |
      +-- lower bound (bottom)
```

### What Goes Wrong Without LaTeX?

| Plain Text | LaTeX | Ambiguity |
|---|---|---|
| `x_ij` | $x_{ij}$ | Is it $x_{ij}$ or $x_i \cdot j$? |
| `1/n sum xi` | $\frac{1}{n}\sum x_i$ | Where does the sum end? |
| `||x||` | $\|x\|$ | Are those pipes, bars, or norms? |
| `R^n` | $\mathbb{R}^n$ | Is R a variable or the reals? |

---

## Math

This section is your comprehensive reference for all LaTeX math commands used in USAAIO.

### Fractions

| Command | Result |
|---|---|
| `\frac{a}{b}` | $\frac{a}{b}$ |
| `\frac{1}{n}\sum_{i=1}^{n}` | $\frac{1}{n}\sum_{i=1}^{n}$ |
| `\frac{\partial L}{\partial w}` | $\frac{\partial L}{\partial w}$ |

Nested fractions:
```latex
\frac{1}{1 + \frac{1}{x}}
```
Produces: $\frac{1}{1 + \frac{1}{x}}$

*Reasoning required*: For inline fractions that look too small, use `\dfrac` (display-style fraction) instead of `\frac`.

### Greek Letters

**Lowercase** (used constantly in USAAIO):

| Command | Letter | Common Meaning |
|---|---|---|
| `\alpha` | $\alpha$ | Learning rate, significance level |
| `\beta` | $\beta$ | Coefficients, momentum |
| `\gamma` | $\gamma$ | Discount factor, regularization |
| `\delta` | $\delta$ | Small change, Kronecker delta |
| `\epsilon` | $\epsilon$ | Small positive value, noise |
| `\eta` | $\eta$ | Learning rate |
| `\theta` | $\theta$ | Model parameters |
| `\lambda` | $\lambda$ | Eigenvalue, regularization strength |
| `\mu` | $\mu$ | Mean |
| `\sigma` | $\sigma$ | Standard deviation, sigmoid |
| `\pi` | $\pi$ | 3.14159..., policy |
| `\phi` | $\phi$ | Feature map, activation |
| `\nabla` | $\nabla$ | Gradient operator |
| `\partial` | $\partial$ | Partial derivative |

**Uppercase** (less frequent but important):

| Command | Letter | Common Meaning |
|---|---|---|
| `\Sigma` | $\Sigma$ | Covariance matrix, summation |
| `\Lambda` | $\Lambda$ | Diagonal eigenvalue matrix |
| `\Theta` | $\Theta$ | Parameter set |
| `\Omega` | $\Omega$ | Sample space |

### Subscripts and Superscripts

| Command | Result | Note |
|---|---|---|
| `x_i` | $x_i$ | Single character subscript |
| `x_{ij}` | $x_{ij}$ | Multi-character needs braces |
| `x^2` | $x^2$ | Single character superscript |
| `x^{n+1}` | $x^{n+1}$ | Multi-character needs braces |
| `x_i^2` | $x_i^2$ | Both |
| `x_i^{(j)}` | $x_i^{(j)}$ | Parenthesized superscript |
| `\theta^{(t)}` | $\theta^{(t)}$ | Iteration number |

### Decorators (Hats, Bars, etc.)

| Command | Result | Meaning |
|---|---|---|
| `\hat{y}` | $\hat{y}$ | Predicted value, estimator |
| `\bar{x}` | $\bar{x}$ | Mean |
| `\tilde{x}` | $\tilde{x}$ | Approximation, transform |
| `\dot{x}` | $\dot{x}$ | Time derivative |
| `\vec{x}` | $\vec{x}$ | Vector (arrow notation) |
| `\mathbf{x}` | $\mathbf{x}$ | Vector (bold notation, preferred) |
| `\mathbb{R}` | $\mathbb{R}$ | Real numbers |
| `\mathcal{L}` | $\mathcal{L}$ | Loss function (calligraphic) |

### Vectors and Matrices

**Vector notation** (USAAIO typically uses bold):
```latex
\mathbf{x} \in \mathbb{R}^n
```
Produces: $\mathbf{x} \in \mathbb{R}^n$

**Transpose**:
```latex
\mathbf{x}^T \quad \text{or} \quad \mathbf{x}^\top
```

**Norms**:

| Command | Result | Name |
|---|---|---|
| `\|\mathbf{x}\|` | $\|\mathbf{x}\|$ | Generic norm |
| `\|\mathbf{x}\|_2` | $\|\mathbf{x}\|_2$ | L2 (Euclidean) norm |
| `\|\mathbf{x}\|_1` | $\|\mathbf{x}\|_1$ | L1 (Manhattan) norm |
| `\|\mathbf{x}\|_\infty` | $\|\mathbf{x}\|_\infty$ | Infinity norm |

**Matrix with square brackets** (`bmatrix`):
```latex
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
```

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
$$

**Matrix with parentheses** (`pmatrix`):
```latex
\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
```

$$
\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

**Larger matrix** (with dots):
```latex
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
```

$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

### Summations and Products

```latex
\sum_{i=1}^{n} x_i          % sum from 1 to n
\prod_{j=1}^{m} p_j         % product from 1 to m
\int_a^b f(x)\,dx           % definite integral
\int_{-\infty}^{\infty}     % improper integral
\lim_{n \to \infty} a_n     % limit
```

*Reasoning not required*: In display mode (`$$`), limits appear above and below. In inline mode (`$`), they appear to the side. This is automatic.

### Set Notation

| Command | Result | Meaning |
|---|---|---|
| `x \in S` | $x \in S$ | x is an element of S |
| `x \notin S` | $x \notin S$ | x is not in S |
| `A \subset B` | $A \subset B$ | A is a subset of B |
| `A \cup B` | $A \cup B$ | Union |
| `A \cap B` | $A \cap B$ | Intersection |
| `\emptyset` | $\emptyset$ | Empty set |
| `\forall x` | $\forall x$ | For all x |
| `\exists x` | $\exists x$ | There exists x |
| `\mathbb{R}^n` | $\mathbb{R}^n$ | n-dimensional real space |
| `\{1, 2, 3\}` | $\{1, 2, 3\}$ | Explicit set |

### Calculus

**Derivatives**:
```latex
\frac{dy}{dx}                    % ordinary derivative
\frac{\partial L}{\partial w}    % partial derivative
\nabla_\theta L                  % gradient with respect to theta
\frac{\partial^2 f}{\partial x^2}  % second partial
```

**Integrals**:
```latex
\int_a^b f(x)\,dx
\iint_D f(x,y)\,dA
```

### Auto-Sizing Delimiters

When expressions are tall (fractions, matrices), use `\left` and `\right` to auto-size parentheses:

```latex
\left( \frac{a}{b} \right)        % parentheses
\left[ \frac{a}{b} \right]        % brackets
\left\{ \frac{a}{b} \right\}      % braces
\left| \frac{a}{b} \right|        % absolute value
```

Compare:
- Without: $(\frac{a}{b})$ -- parentheses too small
- With: $\left(\frac{a}{b}\right)$ -- parentheses scale to content

### Aligned Equations

For multi-line derivations, use `aligned` with `&` marking the alignment point:

```latex
$$
\begin{aligned}
L &= -\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right] \\
\frac{\partial L}{\partial w_j} &= -\sum_{i=1}^{n} \left[ y_i \frac{1}{\hat{y}_i} - (1-y_i)\frac{1}{1-\hat{y}_i} \right] \frac{\partial \hat{y}_i}{\partial w_j} \\
&= \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}
\end{aligned}
$$
```

$$
\begin{aligned}
L &= -\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right] \\
\frac{\partial L}{\partial w_j} &= -\sum_{i=1}^{n} \left[ y_i \frac{1}{\hat{y}_i} - (1-y_i)\frac{1}{1-\hat{y}_i} \right] \frac{\partial \hat{y}_i}{\partial w_j} \\
&= \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}
\end{aligned}
$$

### Text Inside Math

Use `\text{}` for words within math mode:

```latex
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\text{where } z_i \text{ is the } i\text{-th logit}
```

Other text commands:
- `\text{...}` -- normal text
- `\textbf{...}` -- bold text
- `\quad` -- wide space
- `\,` -- thin space
- `\!` -- negative thin space

### Spacing

| Command | Width | Use |
|---|---|---|
| `\,` | thin | Before $dx$ in integrals: `\int f(x)\,dx` |
| `\;` | medium | Separating conditions |
| `\quad` | wide | Between equation and condition |
| `\qquad` | very wide | Large separation |

---

## Code

### USAAIO Formula Reference

Below are actual formulas that appear in USAAIO problems. Practice typesetting each one.

#### Softmax Function

```latex
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \ldots, K
$$
```

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \ldots, K
$$

#### Cross-Entropy Loss (Binary)

```latex
$$
L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$
```

$$
L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$

#### Cross-Entropy Loss (Multi-class)

```latex
$$
L = -\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$
```

$$
L = -\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

#### Mean Squared Error

```latex
$$
L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
```

$$
L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

#### Gradient Descent Update

```latex
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$
```

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

#### Scaled Dot-Product Attention

```latex
$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$.

#### Multi-Head Attention

```latex
$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$
```

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

#### Eigendecomposition

```latex
$$
A\mathbf{v} = \lambda\mathbf{v} \quad \Rightarrow \quad A = V\Lambda V^{-1}
$$
```

$$
A\mathbf{v} = \lambda\mathbf{v} \quad \Rightarrow \quad A = V\Lambda V^{-1}
$$

Where $V = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_n]$ and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

#### Singular Value Decomposition (SVD)

```latex
$$
A = U\Sigma V^T
$$
```

$$
A = U\Sigma V^T
$$

Where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, $V \in \mathbb{R}^{n \times n}$.

#### PCA (Covariance Matrix)

```latex
$$
C = \frac{1}{n-1}X^T X \quad \text{where } X \text{ is centered}
$$
```

$$
C = \frac{1}{n-1}X^T X \quad \text{where } X \text{ is centered}
$$

#### Gaussian (Normal) Distribution

```latex
$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$
```

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

#### F1-Score

```latex
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$
```

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

#### Backpropagation Chain Rule

```latex
$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}
$$
```

$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}
$$

#### Convolution (2D)

```latex
$$
(f * g)(i, j) = \sum_{m}\sum_{n} f(m, n) \cdot g(i - m, j - n)
$$
```

$$
(f * g)(i, j) = \sum_{m}\sum_{n} f(m, n) \cdot g(i - m, j - n)
$$

---

## Resources

- [Detexify](https://detexify.kirelabs.org/classify.html) -- draw a symbol, get the LaTeX command
- [LaTeX/Mathematics (Wikibooks)](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
- [MathJax Documentation](https://docs.mathjax.org/en/latest/)
- [KaTeX Supported Functions](https://katex.org/docs/supported) -- similar to MathJax, good reference
- [Overleaf LaTeX Math Reference](https://www.overleaf.com/learn/latex/Mathematical_expressions)

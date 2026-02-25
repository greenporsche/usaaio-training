# Markdown & LaTeX Cheat Sheet for USAAIO

## Text Formatting

| Syntax | Result |
|---|---|
| `# Heading 1` ... `###### Heading 6` | Headings (6 levels) |
| `**bold**` | **bold** |
| `*italic*` or `_italic_` | *italic* |
| `***bold italic***` | ***bold italic*** |
| `~~strikethrough~~` | ~~strikethrough~~ |
| `` `inline code` `` | `inline code` |
| `[text](url)` | Hyperlink |
| `![alt](url)` | Image |

## Lists

```markdown
- Unordered item          1. Ordered item
  - Nested item              1. Sub-item
    - Deep nested               1. Deep sub-item
```

## Blockquotes & Rules

```markdown
> Blockquote text
> > Nested blockquote

---                        <!-- horizontal rule -->
```

## Tables

```markdown
| Left | Center | Right |
|:-----|:------:|------:|
| L    |   C    |     R |
```

## Code Blocks

````markdown
```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
```
````

Inline: `` `variable_name` ``

---

## LaTeX Math

### Inline vs Display

| Mode | Syntax | Use |
|---|---|---|
| Inline | `$E = mc^2$` | Within text |
| Display | `$$E = mc^2$$` | Centered, own line |

### Fractions & Operations

| Syntax | Renders |
|---|---|
| `\frac{a}{b}` | $\frac{a}{b}$ |
| `\frac{\partial L}{\partial w}` | $\frac{\partial L}{\partial w}$ |
| `\sqrt{x}`, `\sqrt[3]{x}` | $\sqrt{x}$, $\sqrt[3]{x}$ |
| `a \cdot b`, `a \times b` | $a \cdot b$, $a \times b$ |

### Greek Letters

| Lowercase | Uppercase |
|---|---|
| `\alpha` $\alpha$, `\beta` $\beta$, `\gamma` $\gamma$ | `\Gamma` $\Gamma$, `\Delta` $\Delta$ |
| `\delta` $\delta$, `\epsilon` $\epsilon$, `\lambda` $\lambda$ | `\Lambda` $\Lambda$, `\Sigma` $\Sigma$ |
| `\sigma` $\sigma$, `\theta` $\theta$, `\mu` $\mu$ | `\Theta` $\Theta$, `\Omega` $\Omega$ |
| `\nabla` $\nabla$, `\partial` $\partial$, `\pi` $\pi$ | `\Pi` $\Pi$, `\Phi` $\Phi$ |

### Subscripts & Superscripts

| Syntax | Renders |
|---|---|
| `x_i`, `x_{ij}` | $x_i$, $x_{ij}$ |
| `x^2`, `x^{n+1}` | $x^2$, $x^{n+1}$ |
| `x_i^{(j)}` | $x_i^{(j)}$ |
| `\hat{y}`, `\bar{x}`, `\tilde{x}` | $\hat{y}$, $\bar{x}$, $\tilde{x}$ |

### Vectors & Matrices

```latex
\mathbf{x}            % bold vector
\hat{x}               % unit vector
\|x\|, \|x\|_2       % norms
\mathbf{x}^T          % transpose
```

Matrix (square brackets):
```latex
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
```

Matrix (parentheses):
```latex
\begin{pmatrix} a & b \\ c & d \end{pmatrix}
```

### Summations & Products

| Syntax | Renders |
|---|---|
| `\sum_{i=1}^{n} x_i` | $\sum_{i=1}^{n} x_i$ |
| `\prod_{i=1}^{n} x_i` | $\prod_{i=1}^{n} x_i$ |
| `\int_a^b f(x)\,dx` | $\int_a^b f(x)\,dx$ |
| `\lim_{n \to \infty}` | $\lim_{n \to \infty}$ |

### Set Notation

| Syntax | Renders |
|---|---|
| `\in`, `\notin`, `\subset` | $\in$, $\notin$, $\subset$ |
| `\cup`, `\cap` | $\cup$, $\cap$ |
| `\mathbb{R}`, `\mathbb{R}^n` | $\mathbb{R}$, $\mathbb{R}^n$ |
| `\forall`, `\exists` | $\forall$, $\exists$ |
| `\{`, `\}` | $\{$, $\}$ |

### Aligned Equations

```latex
$$
\begin{aligned}
L &= -\sum_{i=1}^{n} y_i \log(\hat{y}_i) \\
  &= -\left[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \right]
\end{aligned}
$$
```

---

## Common USAAIO Formulas

**Softmax:**
```latex
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
```

**Attention:**
```latex
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**Cross-Entropy Loss:**
```latex
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
```

**MSE Loss:**
```latex
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```

**Gradient Descent:**
```latex
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
```

**Eigendecomposition:**
```latex
A\mathbf{v} = \lambda\mathbf{v} \quad\Rightarrow\quad A = V\Lambda V^{-1}
```

**PCA (covariance):**
```latex
C = \frac{1}{n-1}X^T X
```

**F1-Score:**
```latex
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

---

## Gotchas

- `$` for inline math, `$$` for display math -- never mix
- Braces `{}` group multi-character sub/superscripts: `x_{10}` not `x_10`
- Backslash before Greek: `\alpha` not `alpha`
- Use `\left(` and `\right)` for auto-sizing parentheses around tall expressions
- Newlines in aligned equations: `\\` (double backslash)
- Ampersand `&` aligns columns in `aligned`, `bmatrix`, tables
- Use `\text{}` for words inside math: `\text{softmax}` not `softmax`
- Use `\,` for thin space in integrals: `\int f(x)\,dx`
- Google Colab uses MathJax -- some LaTeX packages unavailable

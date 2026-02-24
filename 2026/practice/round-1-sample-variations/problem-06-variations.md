# Problem 6 Variations: Derivatives (EXHAUSTIVE)

> Original: Compute d(tanh x)/dx and ∇_θ of MSE loss
> Core Skills: Activation function derivatives, chain rule, gradient of vector expressions, loss function gradients

---

## USAAIO ALIGNMENT

### Direct Match to Problem 6
The original USAAIO Problem 6 asks exactly:
- **Part 6.1**: d(tanh x)/dx
- **Part 6.2**: ∇_θ Σ(y^(n) - θᵀx^(n))²

**Priority variations for Round 1 prep**: A1, A2, A3, E1, E3, G1, G2 (these match the exact format)

### Cross-Problem Connections
| USAAIO Problem | Derivative Skills Needed |
|----------------|-------------------------|
| Problem 1 (PCA) | Gradient of projection, ∂(xᵀê)/∂x |
| Problem 5 (Kernels) | Gradient of kernel expressions |
| Problem 7 (SVM) | Gradient of margin, hinge loss |
| Problem 10 (PyTorch) | Implementing forward/backward |
| Problem 11 (MLP) | Chain rule through layers |

### Round 2 Extensions
Problems requiring deeper derivative knowledge:
- Backpropagation through custom layers
- Second-order optimization (Hessians)
- Implicit differentiation in meta-learning

---

## DIFFICULTY RATING SYSTEM

| Level | Description | Time (est.) | Prerequisites |
|-------|-------------|-------------|---------------|
| ⭐ | Direct application of derivative rules | 2-5 min | Basic calculus |
| ⭐⭐ | Chain rule, product rule combinations | 5-10 min | Multivariable calculus |
| ⭐⭐⭐ | Matrix calculus, Jacobians | 10-20 min | Linear algebra + calculus |
| ⭐⭐⭐⭐ | Proofs, convergence analysis | 20-30 min | Real analysis basics |
| ⭐⭐⭐⭐⭐ | Research-level derivations | 30+ min | Graduate-level math |

---

## CATEGORY A: Different Activation Functions

### Variation A1: Sigmoid Derivative ⭐
**USAAIO Priority: HIGH** — Same format as original Problem 6
Compute the following derivative:

**Part 6.1**: d(σ(x))/dx where σ(x) = 1/(1 + e^(-x))

**Part 6.2**: Express the derivative in terms of σ(x) itself.

<details>
<summary>Solution A1</summary>

**6.1**: Using the quotient rule:
σ(x) = (1 + e^(-x))^(-1)

dσ/dx = -1 · (1 + e^(-x))^(-2) · (-e^(-x))
      = e^(-x) / (1 + e^(-x))²

**6.2**: Note that σ(x) = 1/(1 + e^(-x)), so:
- 1 - σ(x) = e^(-x)/(1 + e^(-x))
- σ(x)(1 - σ(x)) = [1/(1 + e^(-x))] · [e^(-x)/(1 + e^(-x))]
                  = e^(-x)/(1 + e^(-x))²

**Answer: dσ/dx = σ(x)(1 - σ(x))**

*Key Insight*: The sigmoid derivative has this elegant form, which is computationally efficient since σ(x) is already computed during forward pass.
</details>

### Variation A2: ReLU and Variants ⭐
**USAAIO Priority: HIGH** — Piecewise functions, subgradients

Compute the derivatives of the following functions:

**Part 6.1**: ReLU(x) = max(0, x)

**Part 6.2**: Leaky ReLU: f(x) = x if x > 0, αx if x ≤ 0 (where α = 0.01)

**Part 6.3**: ELU: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0

<details>
<summary>Solution A2</summary>

**6.1**: ReLU derivative:
d(ReLU)/dx = {1 if x > 0, 0 if x < 0, undefined at x = 0}

In practice: d(ReLU)/dx = 1_{x > 0} (indicator function)

**6.2**: Leaky ReLU derivative:
df/dx = {1 if x > 0, α if x ≤ 0} = {1 if x > 0, 0.01 if x ≤ 0}

**6.3**: ELU derivative:
df/dx = {1 if x > 0, αe^x if x ≤ 0}

Note: At x = 0, ELU derivative is α (continuous from left), making it smoother than ReLU.
</details>

### Variation A3: Softplus ⭐⭐
**USAAIO Priority: MEDIUM** — Shows sigmoid-softplus relationship

Compute the derivative of softplus:

**Part 6.1**: f(x) = ln(1 + e^x). Compute df/dx.

**Part 6.2**: Show that softplus is a smooth approximation to ReLU.

**Part 6.3**: What is d²f/dx²?

<details>
<summary>Solution A3</summary>

**6.1**: Using chain rule:
df/dx = e^x / (1 + e^x) = 1/(1 + e^(-x)) = σ(x)

**Answer: df/dx = σ(x)** (the sigmoid function!)

**6.2**:
- As x → -∞: softplus(x) → 0 (like ReLU)
- As x → +∞: softplus(x) ≈ x (like ReLU)
- But softplus is smooth everywhere (unlike ReLU at x=0)

**6.3**: d²f/dx² = dσ/dx = σ(x)(1 - σ(x))

*Key Insight*: The derivative of softplus is sigmoid, and the derivative of sigmoid is σ(1-σ). These functions form a beautiful chain!
</details>

### Variation A4: GELU (Gaussian Error Linear Unit) ⭐⭐⭐
**USAAIO Priority: LOW** — Advanced, used in transformers (Round 2)

The GELU function is defined as:
GELU(x) = x · Φ(x)

where Φ(x) is the CDF of the standard normal distribution.

**Part 6.1**: Write Φ(x) in terms of the error function erf.

**Part 6.2**: Compute d(GELU)/dx. Use the fact that Φ'(x) = φ(x) = (1/√(2π))e^(-x²/2).

<details>
<summary>Solution A4</summary>

**6.1**: Φ(x) = (1/2)(1 + erf(x/√2))

**6.2**: Using the product rule:
d(GELU)/dx = d(x · Φ(x))/dx
           = Φ(x) + x · Φ'(x)
           = Φ(x) + x · φ(x)
           = Φ(x) + x · (1/√(2π))e^(-x²/2)

**Answer: d(GELU)/dx = Φ(x) + xφ(x)**

*Key Insight*: GELU is used in BERT and GPT models. Unlike ReLU, it's smooth and has a probabilistic interpretation.
</details>

### Variation A5: Swish (SiLU) ⭐⭐
**USAAIO Priority: MEDIUM** — Product rule application

The Swish function is: f(x) = x · σ(x) where σ is sigmoid.

**Part 6.1**: Compute df/dx.

**Part 6.2**: Show that df/dx = f(x) + σ(x)(1 - f(x)).

<details>
<summary>Solution A5</summary>

**6.1**: Using product rule:
df/dx = σ(x) + x · σ'(x)
      = σ(x) + x · σ(x)(1 - σ(x))
      = σ(x)[1 + x(1 - σ(x))]
      = σ(x)[1 + x - xσ(x)]

**6.2**: Let's verify:
f(x) + σ(x)(1 - f(x)) = xσ(x) + σ(x)(1 - xσ(x))
                       = xσ(x) + σ(x) - xσ(x)²
                       = σ(x)(x + 1 - xσ(x))
                       = σ(x)[1 + x(1 - σ(x))] ✓

**Answer: df/dx = σ(x)(1 + x - xσ(x)) = f(x) + σ(x)(1 - f(x))**
</details>

---

## CATEGORY B: Different Loss Functions

### Variation B1: Cross-Entropy Loss ⭐⭐
**USAAIO Priority: HIGH** — Core classification loss

Let L = -Σₙ [y^(n) log(ŷ^(n)) + (1-y^(n)) log(1-ŷ^(n))] where ŷ^(n) = σ(θᵀx^(n)).

**Part 6.1**: Compute ∂L/∂ŷ^(n).

**Part 6.2**: Compute ∇_θ L.

<details>
<summary>Solution B1</summary>

**6.1**:
∂L/∂ŷ^(n) = -y^(n)/ŷ^(n) + (1-y^(n))/(1-ŷ^(n))
           = (-y^(n)(1-ŷ^(n)) + (1-y^(n))ŷ^(n)) / (ŷ^(n)(1-ŷ^(n)))
           = (ŷ^(n) - y^(n)) / (ŷ^(n)(1-ŷ^(n)))

**6.2**: Using chain rule:
∇_θ L = Σₙ (∂L/∂ŷ^(n)) · (∂ŷ^(n)/∂θ)

Since ŷ^(n) = σ(θᵀx^(n)):
∂ŷ^(n)/∂θ = σ(θᵀx^(n))(1 - σ(θᵀx^(n))) · x^(n) = ŷ^(n)(1-ŷ^(n)) · x^(n)

Therefore:
∇_θ L = Σₙ [(ŷ^(n) - y^(n)) / (ŷ^(n)(1-ŷ^(n)))] · [ŷ^(n)(1-ŷ^(n)) · x^(n)]
      = **Σₙ (ŷ^(n) - y^(n)) x^(n)**

*Key Insight*: The gradient has the same elegant form as MSE with linear regression! This is why cross-entropy with sigmoid is so nice.
</details>

### Variation B2: Hinge Loss ⭐⭐
**USAAIO Priority: HIGH** — Connects to Problem 7 (SVM)

Let L = Σₙ max(0, 1 - y^(n) θᵀx^(n)) where y^(n) ∈ {-1, +1}.

**Part 6.1**: Compute the subgradient ∂L/∂θ for a single sample.

**Part 6.2**: When is the gradient zero?

<details>
<summary>Solution B2</summary>

**6.1**: For a single sample:
L_n = max(0, 1 - y^(n) θᵀx^(n))

If y^(n) θᵀx^(n) ≥ 1: ∂L_n/∂θ = 0
If y^(n) θᵀx^(n) < 1: ∂L_n/∂θ = -y^(n) x^(n)

**6.2**: The gradient is zero when:
- y^(n) θᵀx^(n) ≥ 1 (sample is correctly classified with sufficient margin)

This is the margin condition in SVM: correctly classified points far from the decision boundary contribute zero gradient.
</details>

### Variation B3: Huber Loss ⭐⭐
**USAAIO Priority: MEDIUM** — Robust regression, piecewise

The Huber loss is:
L_δ(r) = {(1/2)r² if |r| ≤ δ, δ(|r| - δ/2) if |r| > δ}

where r = y - θᵀx.

**Part 6.1**: Compute dL_δ/dr.

**Part 6.2**: Show that Huber loss is differentiable everywhere.

**Part 6.3**: Compute ∇_θ Σₙ L_δ(y^(n) - θᵀx^(n)).

<details>
<summary>Solution B3</summary>

**6.1**:
dL_δ/dr = {r if |r| ≤ δ, δ·sign(r) if |r| > δ}

**6.2**: At r = ±δ:
- From inside: derivative = ±δ
- From outside: derivative = δ·sign(±δ) = ±δ ✓

The derivatives match, so Huber loss is C¹ (continuously differentiable).

**6.3**: Let r^(n) = y^(n) - θᵀx^(n), then ∂r^(n)/∂θ = -x^(n)

∇_θ L = Σₙ (dL_δ/dr^(n)) · (-x^(n))
      = -Σₙ clip(r^(n), -δ, δ) · x^(n)

where clip(r, -δ, δ) = max(-δ, min(δ, r)).

*Key Insight*: Huber loss is quadratic for small errors but linear for large errors, making it robust to outliers.
</details>

### Variation B4: Softmax Cross-Entropy ⭐⭐⭐
**USAAIO Priority: HIGH** — Multi-class classification essential

Let z = Wᵀx ∈ ℝᴷ (K classes), and p = softmax(z).

**Part 6.1**: Compute ∂pᵢ/∂zⱼ for i ≠ j and i = j.

**Part 6.2**: Let L = -Σₖ yₖ log(pₖ) (one-hot y). Compute ∂L/∂zⱼ.

<details>
<summary>Solution B4</summary>

**6.1**: pᵢ = exp(zᵢ) / Σₖ exp(zₖ)

For i = j:
∂pᵢ/∂zᵢ = [exp(zᵢ) · Σₖ exp(zₖ) - exp(zᵢ) · exp(zᵢ)] / (Σₖ exp(zₖ))²
        = pᵢ - pᵢ² = **pᵢ(1 - pᵢ)**

For i ≠ j:
∂pᵢ/∂zⱼ = [0 - exp(zᵢ) · exp(zⱼ)] / (Σₖ exp(zₖ))²
        = **-pᵢpⱼ**

**6.2**:
∂L/∂zⱼ = -Σᵢ yᵢ · (1/pᵢ) · ∂pᵢ/∂zⱼ
       = -yⱼ(1/pⱼ)·pⱼ(1-pⱼ) - Σᵢ≠ⱼ yᵢ(1/pᵢ)·(-pᵢpⱼ)
       = -yⱼ(1-pⱼ) + Σᵢ≠ⱼ yᵢpⱼ
       = -yⱼ + yⱼpⱼ + Σᵢ≠ⱼ yᵢpⱼ
       = -yⱼ + pⱼ(Σᵢ yᵢ)
       = -yⱼ + pⱼ (since Σᵢ yᵢ = 1 for one-hot)

**Answer: ∂L/∂z = p - y**

*Key Insight*: The gradient is simply prediction minus target, same beautiful form as in binary classification!
</details>

### Variation B5: KL Divergence ⭐⭐⭐
**USAAIO Priority: MEDIUM** — Information theory, distillation

Let D_KL(p||q) = Σₖ pₖ log(pₖ/qₖ) where q = softmax(θ).

**Part 6.1**: Compute ∂D_KL/∂θⱼ (treating p as fixed).

**Part 6.2**: Show that minimizing KL divergence is equivalent to minimizing cross-entropy.

<details>
<summary>Solution B5</summary>

**6.1**: D_KL = Σₖ pₖ log(pₖ) - Σₖ pₖ log(qₖ)

The first term is constant (entropy of p). For the second term:
∂(-Σₖ pₖ log(qₖ))/∂θⱼ = -Σₖ pₖ · (1/qₖ) · ∂qₖ/∂θⱼ

Using ∂qᵢ/∂θⱼ = qᵢ(δᵢⱼ - qⱼ):
= -Σₖ pₖ · (1/qₖ) · qₖ(δₖⱼ - qⱼ)
= -Σₖ pₖ(δₖⱼ - qⱼ)
= -pⱼ + qⱼ(Σₖ pₖ)
= **qⱼ - pⱼ**

**6.2**: D_KL(p||q) = H(p,q) - H(p)

Since H(p) is constant, argmin D_KL = argmin H(p,q) (cross-entropy).
</details>

---

## CATEGORY C: Chain Rule Applications

### Variation C1: Nested Activations ⭐⭐
**USAAIO Priority: HIGH** — Chain rule is fundamental

Let f(x) = tanh(σ(x)) where σ is sigmoid.

**Part 6.1**: Compute df/dx using the chain rule.

**Part 6.2**: What is f(0) and f'(0)?

<details>
<summary>Solution C1</summary>

**6.1**:
Let u = σ(x), then f = tanh(u).

df/dx = (d(tanh(u))/du) · (du/dx)
      = (1 - tanh²(u)) · σ(x)(1 - σ(x))
      = (1 - tanh²(σ(x))) · σ(x)(1 - σ(x))

**6.2**:
f(0) = tanh(σ(0)) = tanh(1/2) ≈ 0.462

f'(0) = (1 - tanh²(1/2)) · (1/2)(1/2)
      = (1 - 0.462²) · 0.25
      ≈ 0.786 · 0.25 ≈ 0.197
</details>

### Variation C2: Composition with Linear ⭐⭐
**USAAIO Priority: HIGH** — Logistic regression gradient

Let f(x) = σ(wᵀx + b) where w ∈ ℝᵈ, b ∈ ℝ.

**Part 6.1**: Compute ∂f/∂w.

**Part 6.2**: Compute ∂f/∂b.

**Part 6.3**: Compute ∂f/∂x.

<details>
<summary>Solution C2</summary>

**6.1**: Let z = wᵀx + b, then f = σ(z).
∂f/∂w = (∂f/∂z) · (∂z/∂w) = σ(z)(1-σ(z)) · x = **f(1-f)x**

**6.2**:
∂f/∂b = (∂f/∂z) · (∂z/∂b) = σ(z)(1-σ(z)) · 1 = **f(1-f)**

**6.3**:
∂f/∂x = (∂f/∂z) · (∂z/∂x) = σ(z)(1-σ(z)) · w = **f(1-f)w**
</details>

### Variation C3: Two-Layer Network ⭐⭐⭐
**USAAIO Priority: HIGH** — Connects to Problem 11 (MLP)

Let f(x) = w₂ᵀ σ(W₁x) where W₁ ∈ ℝʰˣᵈ, w₂ ∈ ℝʰ, and σ is applied elementwise.

**Part 6.1**: Compute ∂f/∂w₂.

**Part 6.2**: Compute ∂f/∂W₁.

<details>
<summary>Solution C3</summary>

**6.1**: Let h = σ(W₁x) ∈ ℝʰ.
f = w₂ᵀh = Σᵢ w₂ᵢhᵢ

∂f/∂w₂ = h = **σ(W₁x)**

**6.2**: Let z = W₁x, then h = σ(z).
∂f/∂zᵢ = w₂ᵢ · σ'(zᵢ) = w₂ᵢ · σ(zᵢ)(1-σ(zᵢ))

Using notation: let δ = w₂ ⊙ σ'(z) ∈ ℝʰ (elementwise product).

Then: ∂f/∂W₁ = δxᵀ (outer product, dimension h × d)

**Answer: ∂f/∂W₁ = [w₂ ⊙ σ(W₁x) ⊙ (1-σ(W₁x))] xᵀ**
</details>

### Variation C4: Batch Normalization ⭐⭐⭐⭐
**USAAIO Priority: MEDIUM** — Advanced normalization

Let y = γ(x - μ)/σ + β where μ = E[x], σ² = Var[x].

For a mini-batch of size n:
- μ_B = (1/n)Σᵢ xᵢ
- σ_B² = (1/n)Σᵢ (xᵢ - μ_B)²
- x̂ᵢ = (xᵢ - μ_B)/√(σ_B² + ε)
- yᵢ = γx̂ᵢ + β

**Part 6.1**: Compute ∂L/∂γ and ∂L/∂β given ∂L/∂yᵢ.

**Part 6.2**: Compute ∂L/∂x̂ᵢ.

<details>
<summary>Solution C4</summary>

**6.1**:
∂L/∂γ = Σᵢ (∂L/∂yᵢ) · x̂ᵢ
∂L/∂β = Σᵢ (∂L/∂yᵢ)

**6.2**:
∂L/∂x̂ᵢ = (∂L/∂yᵢ) · γ

*Key Insight*: The full backprop through batch norm is more complex because μ_B and σ_B depend on all xᵢ. The complete derivation involves careful application of the chain rule through these dependencies.
</details>

### Variation C5: Attention Score ⭐⭐⭐⭐
**USAAIO Priority: LOW** — Transformers (Round 2, AI 500)

Let α = softmax(QKᵀ/√d) and output = αV.

**Part 6.1**: Compute ∂α/∂Q (treating K as constant).

**Part 6.2**: What is the intuition behind dividing by √d?

<details>
<summary>Solution C5</summary>

**6.1**: Let S = QKᵀ/√d, then α = softmax(S).

∂αᵢⱼ/∂Sₖₗ follows the softmax derivative pattern (see B4).

∂Sᵢⱼ/∂Qᵢₘ = Kⱼₘ/√d

The full gradient combines these via chain rule.

**6.2**: Without /√d, when d is large:
- Each entry of QKᵀ is a sum of d terms
- Variance grows like d
- Softmax becomes very peaked (nearly one-hot)
- Gradients become very small

Dividing by √d keeps the variance of QKᵀ entries around 1, maintaining healthy gradients.

*Key Insight*: This is the "scaled dot-product attention" in Transformers. The √d factor is crucial for stable training!
</details>

---

## CATEGORY D: Matrix/Vector Calculus

### Variation D1: Gradient of Quadratic Form ⭐⭐⭐
**USAAIO Priority: HIGH** — Foundation for optimization

Let f(x) = xᵀAx where A ∈ ℝⁿˣⁿ.

**Part 6.1**: Compute ∇_x f when A is symmetric.

**Part 6.2**: Compute ∇_x f for general A.

**Part 6.3**: Compute ∇_x(xᵀAx + bᵀx + c).

<details>
<summary>Solution D1</summary>

**6.1**: For symmetric A:
∇_x(xᵀAx) = 2Ax

**6.2**: For general A:
∇_x(xᵀAx) = (A + Aᵀ)x

**6.3**:
∇_x(xᵀAx + bᵀx + c) = (A + Aᵀ)x + b

For symmetric A: = **2Ax + b**

*Key Insight*: This is the gradient form that appears in least squares: ∇_θ ||Xθ - y||² = 2XᵀXθ - 2Xᵀy.
</details>

### Variation D2: Gradient of Trace ⭐⭐⭐
**USAAIO Priority: MEDIUM** — Matrix calculus identity

Let f(W) = tr(WᵀAW) where W, A ∈ ℝⁿˣⁿ.

**Part 6.1**: Compute ∂f/∂W.

**Part 6.2**: Compute ∂tr(AB)/∂A.

<details>
<summary>Solution D2</summary>

**6.1**: Using trace properties:
tr(WᵀAW) = tr(AWWᵀ) (cyclic property, if dimensions match; here assume square)

∂tr(WᵀAW)/∂W = (A + Aᵀ)W

For symmetric A: = **2AW**

**6.2**:
∂tr(AB)/∂A = Bᵀ

*Key Insight*: Matrix calculus identities like ∂tr(AB)/∂A = Bᵀ are fundamental for deriving neural network gradients.
</details>

### Variation D3: Gradient with respect to Matrix ⭐⭐⭐
**USAAIO Priority: HIGH** — Multi-output regression

Let L = ||Y - XW||²_F (Frobenius norm) where X ∈ ℝⁿˣᵈ, W ∈ ℝᵈˣᵏ, Y ∈ ℝⁿˣᵏ.

**Part 6.1**: Expand ||Y - XW||²_F in terms of trace.

**Part 6.2**: Compute ∂L/∂W.

**Part 6.3**: Set the gradient to zero and solve for W.

<details>
<summary>Solution D3</summary>

**6.1**: ||A||²_F = tr(AᵀA)

L = tr((Y - XW)ᵀ(Y - XW))
  = tr(YᵀY - YᵀXW - WᵀXᵀY + WᵀXᵀXW)

**6.2**:
∂L/∂W = -2XᵀY + 2XᵀXW = **2Xᵀ(XW - Y)**

**6.3**: Setting to zero:
XᵀXW = XᵀY
W = **(XᵀX)⁻¹XᵀY** (assuming XᵀX is invertible)

*Key Insight*: This is the normal equation for linear regression! The gradient derivation uses matrix calculus rules.
</details>

### Variation D4: Jacobian of Softmax ⭐⭐⭐
**USAAIO Priority: HIGH** — Essential for classification

Let p = softmax(z) where z ∈ ℝᴷ.

**Part 6.1**: Write the Jacobian matrix J = ∂p/∂z.

**Part 6.2**: Show that J = diag(p) - ppᵀ.

**Part 6.3**: Verify that J1 = 0 (where 1 is a vector of ones).

<details>
<summary>Solution D4</summary>

**6.1**: From B4, we know:
Jᵢⱼ = ∂pᵢ/∂zⱼ = {pᵢ(1-pᵢ) if i=j, -pᵢpⱼ if i≠j}

**6.2**:
diag(p) = [[p₁,0,...],[0,p₂,...],...] (diagonal matrix)
ppᵀ = [[p₁p₁,p₁p₂,...],[p₂p₁,p₂p₂,...],...]

[diag(p) - ppᵀ]ᵢⱼ = {pᵢ - pᵢ² = pᵢ(1-pᵢ) if i=j, 0 - pᵢpⱼ = -pᵢpⱼ if i≠j} ✓

**6.3**: J1 = diag(p)·1 - ppᵀ·1 = p - p(pᵀ1) = p - p·1 = 0 ✓

(Since Σpᵢ = 1, we have pᵀ1 = 1)

*Key Insight*: J1 = 0 means softmax output is insensitive to uniform shifts in z (adding constant to all logits doesn't change probabilities).
</details>

### Variation D5: Gradient of Log-Determinant ⭐⭐⭐⭐
**USAAIO Priority: LOW** — Gaussian models (advanced)

Let f(A) = log det(A) for positive definite A.

**Part 6.1**: Show that ∂f/∂A = A⁻ᵀ = (A⁻¹)ᵀ.

**Part 6.2**: For symmetric A, what is ∂f/∂A?

<details>
<summary>Solution D5</summary>

**6.1**: Using the identity d(log det A) = tr(A⁻¹ dA):

For element Aᵢⱼ:
∂(log det A)/∂Aᵢⱼ = tr(A⁻¹ · ∂A/∂Aᵢⱼ) = tr(A⁻¹ · Eᵢⱼ) = (A⁻¹)ⱼᵢ = (A⁻ᵀ)ᵢⱼ

**Answer: ∂f/∂A = A⁻ᵀ**

**6.2**: For symmetric A, A⁻ᵀ = A⁻¹, so **∂f/∂A = A⁻¹**.

*Key Insight*: This appears in Gaussian models where we optimize over covariance matrices.
</details>

---

## CATEGORY E: MSE Loss Variations

### Variation E1: MSE with Bias ⭐⭐
**USAAIO Priority: HIGH** — Direct extension of Problem 6.2

∇_θ Σₙ (y^(n) - θᵀx^(n) - b)² where b is a bias term.

**Part 6.1**: Compute ∇_θ L.

**Part 6.2**: Compute ∂L/∂b.

**Part 6.3**: Write both gradients in matrix form.

<details>
<summary>Solution E1</summary>

**6.1**: Let rₙ = y^(n) - θᵀx^(n) - b (residual).
∇_θ L = Σₙ 2rₙ · (-x^(n)) = **-2 Σₙ (y^(n) - θᵀx^(n) - b) x^(n)**

**6.2**:
∂L/∂b = Σₙ 2rₙ · (-1) = **-2 Σₙ (y^(n) - θᵀx^(n) - b)**

**6.3**: Let X ∈ ℝᴺˣᵈ (rows are x^(n)ᵀ), y ∈ ℝᴺ, 1 ∈ ℝᴺ (ones vector).

∇_θ L = -2Xᵀ(y - Xθ - b·1)
∂L/∂b = -2 · 1ᵀ(y - Xθ - b·1)
</details>

### Variation E2: Weighted MSE ⭐⭐
**USAAIO Priority: MEDIUM** — Imbalanced data handling

L = Σₙ wₙ(y^(n) - θᵀx^(n))² where wₙ > 0 are weights.

**Part 6.1**: Compute ∇_θ L.

**Part 6.2**: Write in matrix form with W = diag(w₁, ..., wₙ).

<details>
<summary>Solution E2</summary>

**6.1**:
∇_θ L = Σₙ 2wₙ(y^(n) - θᵀx^(n))(-x^(n))
      = **-2 Σₙ wₙ(y^(n) - θᵀx^(n)) x^(n)**

**6.2**:
∇_θ L = -2Xᵀ W (y - Xθ)

Setting to zero: XᵀWXθ = XᵀWy
θ = **(XᵀWX)⁻¹XᵀWy** (weighted least squares solution)

*Key Insight*: Weighted least squares is useful when different observations have different reliabilities/variances.
</details>

### Variation E3: Ridge Regression ⭐⭐
**USAAIO Priority: HIGH** — Connects to Problem 3 (Regularization)

L = Σₙ (y^(n) - θᵀx^(n))² + λ||θ||²

**Part 6.1**: Compute ∇_θ L.

**Part 6.2**: Solve for the optimal θ.

**Part 6.3**: What happens as λ → 0 and λ → ∞?

<details>
<summary>Solution E3</summary>

**6.1**:
∇_θ L = -2 Σₙ (y^(n) - θᵀx^(n)) x^(n) + 2λθ
      = **-2Xᵀ(y - Xθ) + 2λθ**

**6.2**: Setting to zero:
Xᵀy - XᵀXθ + λθ = 0
(XᵀX + λI)θ = Xᵀy
θ = **(XᵀX + λI)⁻¹Xᵀy**

**6.3**:
- λ → 0: θ → (XᵀX)⁻¹Xᵀy (ordinary least squares)
- λ → ∞: θ → 0 (shrinks weights to zero)

*Key Insight*: The +λI term ensures the matrix is always invertible, even if XᵀX is singular!
</details>

### Variation E4: Multi-Output Regression ⭐⭐⭐
**USAAIO Priority: MEDIUM** — Matrix form extension

L = ||Y - XW||²_F where Y ∈ ℝᴺˣᴷ, W ∈ ℝᵈˣᴷ.

**Part 6.1**: Compute ∂L/∂W.

**Part 6.2**: Solve for optimal W.

<details>
<summary>Solution E4</summary>

**6.1**: This is D3, repeated for reference:
∂L/∂W = **2Xᵀ(XW - Y)**

**6.2**: Setting to zero:
XᵀXW = XᵀY
W = **(XᵀX)⁻¹XᵀY**

Note: Each column of W is independently the OLS solution for that output dimension.
</details>

### Variation E5: Constrained Optimization ⭐⭐⭐⭐
**USAAIO Priority: LOW** — Lagrangian methods (advanced)

Minimize L = ||y - Xθ||² subject to ||θ|| = 1.

**Part 6.1**: Write the Lagrangian.

**Part 6.2**: Compute ∇_θ of the Lagrangian.

**Part 6.3**: What equation must θ satisfy at the optimum?

<details>
<summary>Solution E5</summary>

**6.1**:
L(θ, λ) = ||y - Xθ||² + λ(||θ||² - 1)
        = yᵀy - 2yᵀXθ + θᵀXᵀXθ + λ(θᵀθ - 1)

**6.2**:
∇_θ L = -2Xᵀy + 2XᵀXθ + 2λθ = 0
(XᵀX + λI)θ = Xᵀy

**6.3**: θ must satisfy:
1. (XᵀX + λI)θ = Xᵀy
2. ||θ|| = 1

This is related to ridge regression, but with λ chosen to satisfy the norm constraint rather than being a hyperparameter.
</details>

---

## CATEGORY F: Proofs & Theory

### Variation F1: Prove tanh Derivative ⭐⭐
**USAAIO Priority: HIGH** — Exact match to Problem 6.1 with proof

**Part 6.1**: Starting from tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ), prove that d(tanh x)/dx = 1 - tanh²(x).

**Part 6.2**: Show that tanh(x) = 2σ(2x) - 1 where σ is sigmoid.

<details>
<summary>Solution F1</summary>

**6.1**: Using quotient rule:
d(tanh x)/dx = [(eˣ + e⁻ˣ)(eˣ + e⁻ˣ) - (eˣ - e⁻ˣ)(eˣ - e⁻ˣ)] / (eˣ + e⁻ˣ)²

Numerator: (eˣ + e⁻ˣ)² - (eˣ - e⁻ˣ)²
= [e²ˣ + 2 + e⁻²ˣ] - [e²ˣ - 2 + e⁻²ˣ] = 4

So: d(tanh x)/dx = 4/(eˣ + e⁻ˣ)²

Now, tanh²(x) = (eˣ - e⁻ˣ)²/(eˣ + e⁻ˣ)²

1 - tanh²(x) = [(eˣ + e⁻ˣ)² - (eˣ - e⁻ˣ)²]/(eˣ + e⁻ˣ)²
             = 4/(eˣ + e⁻ˣ)² ✓

**6.2**:
2σ(2x) - 1 = 2/(1 + e⁻²ˣ) - 1
           = (2 - 1 - e⁻²ˣ)/(1 + e⁻²ˣ)
           = (1 - e⁻²ˣ)/(1 + e⁻²ˣ)

Multiply top and bottom by eˣ:
= (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) = tanh(x) ✓
</details>

### Variation F2: Convexity of MSE ⭐⭐⭐
**USAAIO Priority: MEDIUM** — Optimization theory

**Part 6.1**: Show that L(θ) = ||y - Xθ||² is convex in θ.

**Part 6.2**: Compute the Hessian and show it's positive semi-definite.

<details>
<summary>Solution F2</summary>

**6.1**: L(θ) = (y - Xθ)ᵀ(y - Xθ) = yᵀy - 2yᵀXθ + θᵀXᵀXθ

This is a quadratic function in θ. The Hessian determines convexity.

**6.2**:
∇_θ L = -2Xᵀy + 2XᵀXθ
∇²_θ L = 2XᵀX

For any vector v: vᵀ(2XᵀX)v = 2||Xv||² ≥ 0

So the Hessian is positive semi-definite, proving L is convex.

*Key Insight*: Convexity guarantees that any local minimum is a global minimum, which is why linear regression with MSE is so well-behaved!
</details>

### Variation F3: Gradient Descent Convergence ⭐⭐⭐⭐
**USAAIO Priority: MEDIUM** — Deep theory (Round 2)

For L(θ) = (1/2)||y - Xθ||², the gradient descent update is:
θ_{t+1} = θ_t - η∇L(θ_t)

**Part 6.1**: Write the update explicitly.

**Part 6.2**: Show that convergence requires η < 2/λ_max where λ_max is the largest eigenvalue of XᵀX.

<details>
<summary>Solution F3</summary>

**6.1**: ∇L = XᵀXθ - Xᵀy

θ_{t+1} = θ_t - η(XᵀXθ_t - Xᵀy)
        = (I - ηXᵀX)θ_t + ηXᵀy

**6.2**: Let θ* = (XᵀX)⁻¹Xᵀy be the optimum. Define error e_t = θ_t - θ*.

e_{t+1} = (I - ηXᵀX)e_t

For convergence, we need ||(I - ηXᵀX)||₂ < 1.

The eigenvalues of (I - ηXᵀX) are (1 - ηλᵢ) where λᵢ are eigenvalues of XᵀX.

For |1 - ηλᵢ| < 1:
-1 < 1 - ηλᵢ < 1
0 < ηλᵢ < 2
η < 2/λᵢ

This must hold for all i, so **η < 2/λ_max**.

*Key Insight*: The learning rate must be small enough to not overshoot, bounded by the curvature (largest eigenvalue) of the loss surface.
</details>

### Variation F4: Chain Rule for Backprop ⭐⭐⭐⭐
**USAAIO Priority: HIGH** — Foundation for all deep learning

For a neural network f(x) = f_L(f_{L-1}(...f_1(x)...)), prove that:

∂L/∂θ_l = (∂L/∂a_L)(∂a_L/∂a_{L-1})...(∂a_{l+1}/∂a_l)(∂a_l/∂θ_l)

where a_l is the activation of layer l.

<details>
<summary>Solution F4</summary>

**Proof by chain rule:**

Let L depend on θ_l only through the chain: θ_l → a_l → a_{l+1} → ... → a_L → L

By the chain rule:
∂L/∂θ_l = (∂L/∂a_L)(∂a_L/∂θ_l)

But a_L depends on θ_l through a_{L-1}:
∂a_L/∂θ_l = (∂a_L/∂a_{L-1})(∂a_{L-1}/∂θ_l)

Continuing this recursively:
∂a_l/∂θ_l is the direct dependence.

Combining:
∂L/∂θ_l = (∂L/∂a_L)(∂a_L/∂a_{L-1})...(∂a_{l+1}/∂a_l)(∂a_l/∂θ_l)

**Key notation**: Define δ_l = ∂L/∂a_l (the "error signal" at layer l).

Then: δ_l = (∂a_{l+1}/∂a_l)ᵀ δ_{l+1}

This is the backpropagation equation!

*Key Insight*: Backpropagation is simply the chain rule applied systematically. The δ_l terms are computed backward from the output.
</details>

### Variation F5: Vanishing Gradients ⭐⭐⭐⭐
**USAAIO Priority: HIGH** — Critical deep learning insight

Consider a deep network with tanh activations: a_l = tanh(W_l a_{l-1}).

**Part 6.1**: Show that |tanh'(x)| ≤ 1 for all x.

**Part 6.2**: If ||W_l|| < 1 for all l, show that gradients vanish exponentially with depth.

<details>
<summary>Solution F5</summary>

**6.1**: tanh'(x) = 1 - tanh²(x)

Since -1 < tanh(x) < 1, we have 0 < 1 - tanh²(x) < 1.

At x = 0: tanh'(0) = 1 (maximum).
As |x| → ∞: tanh'(x) → 0.

So **|tanh'(x)| ≤ 1** for all x. ✓

**6.2**: The gradient flow from layer L to layer l is:
∂a_L/∂a_l = Π_{k=l+1}^{L} diag(tanh'(z_k)) W_k

Taking norms:
||∂a_L/∂a_l|| ≤ Π_{k=l+1}^{L} ||diag(tanh'(z_k))|| · ||W_k||
              ≤ Π_{k=l+1}^{L} 1 · ||W_k||
              < Π_{k=l+1}^{L} 1 = 1^{L-l}

But if ||W_k|| < c < 1:
||∂a_L/∂a_l|| < c^{L-l} → 0 exponentially as (L-l) grows.

*Key Insight*: This is the vanishing gradient problem! Solutions include:
- ReLU (derivative is 0 or 1, not exponentially small)
- Residual connections (skip connections)
- Careful initialization
- Batch normalization
</details>

---

## CATEGORY G: Coding Implementations

### Variation G1: Implement Activation Derivatives ⭐⭐
**USAAIO Priority: HIGH** — Connects to Problem 10 (PyTorch modules)

```python
import numpy as np

def sigmoid(x):
    """Implement sigmoid and its derivative."""
    pass

def sigmoid_derivative(x):
    """Return derivative of sigmoid at x."""
    pass

def tanh_derivative(x):
    """Return derivative of tanh at x."""
    pass

def relu_derivative(x):
    """Return derivative of ReLU at x."""
    pass

# Test
x = np.array([-2, -1, 0, 1, 2])
```

<details>
<summary>Solution G1</summary>

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu_derivative(x):
    return (x > 0).astype(float)

# Test
x = np.array([-2, -1, 0, 1, 2])
print("sigmoid':", sigmoid_derivative(x))
# [0.105, 0.197, 0.25, 0.197, 0.105]

print("tanh':", tanh_derivative(x))
# [0.071, 0.420, 1.0, 0.420, 0.071]

print("relu':", relu_derivative(x))
# [0, 0, 0, 1, 1]  (x=0 is treated as negative here)
```
</details>

### Variation G2: Implement MSE Gradient ⭐⭐
**USAAIO Priority: HIGH** — Direct implementation of Problem 6.2

```python
import numpy as np

def mse_gradient(X, y, theta):
    """
    Compute gradient of MSE loss.

    Args:
        X: (N, d) feature matrix
        y: (N,) target vector
        theta: (d,) weight vector

    Returns:
        gradient: (d,) gradient vector
    """
    pass

# Test
np.random.seed(42)
N, d = 100, 5
X = np.random.randn(N, d)
theta_true = np.array([1, -2, 3, -4, 5])
y = X @ theta_true + 0.1 * np.random.randn(N)
theta = np.zeros(d)
```

<details>
<summary>Solution G2</summary>

```python
import numpy as np

def mse_gradient(X, y, theta):
    """
    Compute gradient of MSE loss: L = ||y - Xθ||²
    Gradient: ∇L = -2Xᵀ(y - Xθ) = 2Xᵀ(Xθ - y)
    """
    N = X.shape[0]
    residual = X @ theta - y  # (N,)
    gradient = (2 / N) * X.T @ residual  # (d,) - normalized by N
    return gradient

# Test
np.random.seed(42)
N, d = 100, 5
X = np.random.randn(N, d)
theta_true = np.array([1, -2, 3, -4, 5])
y = X @ theta_true + 0.1 * np.random.randn(N)
theta = np.zeros(d)

# Gradient at theta=0
grad = mse_gradient(X, y, theta)
print("Gradient:", grad)

# Verify numerically
eps = 1e-5
numerical_grad = np.zeros(d)
for i in range(d):
    theta_plus = theta.copy()
    theta_plus[i] += eps
    theta_minus = theta.copy()
    theta_minus[i] -= eps
    loss_plus = np.mean((y - X @ theta_plus)**2)
    loss_minus = np.mean((y - X @ theta_minus)**2)
    numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

print("Numerical grad:", numerical_grad)
print("Difference:", np.max(np.abs(grad - numerical_grad)))  # Should be ~0
```
</details>

### Variation G3: Implement Softmax Gradient ⭐⭐⭐
**USAAIO Priority: HIGH** — Multi-class classification

```python
import numpy as np

def softmax(z):
    """Compute softmax of vector z."""
    pass

def softmax_jacobian(z):
    """
    Compute the Jacobian matrix of softmax.
    J[i,j] = ∂softmax(z)[i] / ∂z[j]
    """
    pass

# Test
z = np.array([1.0, 2.0, 3.0])
```

<details>
<summary>Solution G3</summary>

```python
import numpy as np

def softmax(z):
    """Compute softmax with numerical stability."""
    z_shifted = z - np.max(z)  # For numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def softmax_jacobian(z):
    """
    Compute Jacobian: J = diag(p) - ppᵀ
    J[i,j] = p[i](δ[i,j] - p[j])
    """
    p = softmax(z)
    K = len(p)
    J = np.diag(p) - np.outer(p, p)
    return J

# Test
z = np.array([1.0, 2.0, 3.0])
p = softmax(z)
J = softmax_jacobian(z)

print("softmax(z):", p)
print("Jacobian:\n", J)

# Verify: each row sums to 0 (J @ 1 = 0)
print("Row sums:", J.sum(axis=1))  # Should be ~0

# Verify numerically
eps = 1e-5
J_numerical = np.zeros((3, 3))
for j in range(3):
    z_plus = z.copy()
    z_plus[j] += eps
    z_minus = z.copy()
    z_minus[j] -= eps
    J_numerical[:, j] = (softmax(z_plus) - softmax(z_minus)) / (2 * eps)

print("Max difference:", np.max(np.abs(J - J_numerical)))  # Should be ~0
```
</details>

### Variation G4: Implement Cross-Entropy Gradient ⭐⭐⭐
**USAAIO Priority: HIGH** — Classification loss

```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy loss.
    y_true: (N,) binary labels
    y_pred: (N,) predicted probabilities
    """
    pass

def cross_entropy_gradient(X, y_true, theta):
    """
    Gradient of cross-entropy loss for logistic regression.
    y_pred = sigmoid(X @ theta)
    """
    pass
```

<details>
<summary>Solution G4</summary>

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy: L = -Σ[y log(p) + (1-y) log(1-p)]
    """
    eps = 1e-15  # Numerical stability
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_gradient(X, y_true, theta):
    """
    Gradient: ∇L = (1/N) Xᵀ(ŷ - y)
    where ŷ = sigmoid(Xθ)
    """
    N = X.shape[0]
    y_pred = sigmoid(X @ theta)
    gradient = (1 / N) * X.T @ (y_pred - y_true)
    return gradient

# Test
np.random.seed(42)
N, d = 100, 5
X = np.random.randn(N, d)
theta_true = np.array([1, -2, 3, -4, 5])
probs = sigmoid(X @ theta_true)
y = (probs > 0.5).astype(float)

theta = np.zeros(d)
grad = cross_entropy_gradient(X, y, theta)
print("Gradient:", grad)

# Verify numerically
eps = 1e-5
numerical_grad = np.zeros(d)
for i in range(d):
    theta_plus = theta.copy()
    theta_plus[i] += eps
    theta_minus = theta.copy()
    theta_minus[i] -= eps
    loss_plus = cross_entropy_loss(y, sigmoid(X @ theta_plus))
    loss_minus = cross_entropy_loss(y, sigmoid(X @ theta_minus))
    numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

print("Numerical grad:", numerical_grad)
print("Max difference:", np.max(np.abs(grad - numerical_grad)))
```
</details>

### Variation G5: Implement Gradient Descent ⭐⭐
**USAAIO Priority: HIGH** — Core optimization

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Gradient descent for linear regression.
    Returns: theta, loss_history
    """
    pass

def gradient_descent_with_momentum(X, y, learning_rate=0.01, momentum=0.9, n_iterations=1000):
    """
    Gradient descent with momentum.
    """
    pass
```

<details>
<summary>Solution G5</summary>

```python
import numpy as np

def mse_loss(X, y, theta):
    return np.mean((y - X @ theta)**2)

def mse_gradient(X, y, theta):
    N = X.shape[0]
    return (2 / N) * X.T @ (X @ theta - y)

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """Vanilla gradient descent."""
    N, d = X.shape
    theta = np.zeros(d)
    loss_history = []

    for i in range(n_iterations):
        loss = mse_loss(X, y, theta)
        loss_history.append(loss)
        grad = mse_gradient(X, y, theta)
        theta = theta - learning_rate * grad

    return theta, loss_history

def gradient_descent_with_momentum(X, y, learning_rate=0.01, momentum=0.9, n_iterations=1000):
    """Gradient descent with momentum."""
    N, d = X.shape
    theta = np.zeros(d)
    velocity = np.zeros(d)
    loss_history = []

    for i in range(n_iterations):
        loss = mse_loss(X, y, theta)
        loss_history.append(loss)
        grad = mse_gradient(X, y, theta)
        velocity = momentum * velocity - learning_rate * grad
        theta = theta + velocity

    return theta, loss_history

# Test
np.random.seed(42)
N, d = 100, 5
X = np.random.randn(N, d)
theta_true = np.array([1, -2, 3, -4, 5])
y = X @ theta_true + 0.1 * np.random.randn(N)

theta_gd, loss_gd = gradient_descent(X, y, learning_rate=0.1, n_iterations=100)
theta_mom, loss_mom = gradient_descent_with_momentum(X, y, learning_rate=0.1, n_iterations=100)

print("True theta:", theta_true)
print("GD theta:", theta_gd)
print("Momentum theta:", theta_mom)
print("GD final loss:", loss_gd[-1])
print("Momentum final loss:", loss_mom[-1])
```
</details>

### Variation G6: Numerical Gradient Check ⭐⭐
**USAAIO Priority: MEDIUM** — Debugging skill

```python
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient of function f at point x.
    Uses central difference: (f(x+eps) - f(x-eps)) / (2*eps)
    """
    pass

def gradient_check(analytical_grad_fn, numerical_loss_fn, x, eps=1e-5):
    """
    Compare analytical gradient to numerical gradient.
    Returns relative error.
    """
    pass
```

<details>
<summary>Solution G6</summary>

```python
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using central differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def gradient_check(analytical_grad_fn, numerical_loss_fn, x, eps=1e-5):
    """
    Compare analytical and numerical gradients.
    Returns relative error: ||g_a - g_n|| / (||g_a|| + ||g_n||)
    """
    g_analytical = analytical_grad_fn(x)
    g_numerical = numerical_gradient(numerical_loss_fn, x, eps)

    numerator = np.linalg.norm(g_analytical - g_numerical)
    denominator = np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical)

    if denominator < 1e-10:
        return 0.0 if numerator < 1e-10 else float('inf')

    return numerator / denominator

# Test with MSE loss
np.random.seed(42)
N, d = 50, 5
X = np.random.randn(N, d)
y = np.random.randn(N)

def loss_fn(theta):
    return np.mean((y - X @ theta)**2)

def grad_fn(theta):
    return (2 / N) * X.T @ (X @ theta - y)

theta = np.random.randn(d)
rel_error = gradient_check(grad_fn, loss_fn, theta)
print(f"Relative error: {rel_error:.2e}")  # Should be < 1e-7
```
</details>

---

## CATEGORY H: Edge Cases & Special Situations

### Variation H1: Gradient at Extreme Values ⭐⭐
**USAAIO Priority: MEDIUM** — Numerical stability
**Part 6.1**: What is tanh'(x) as x → ∞?

**Part 6.2**: What is σ'(x) as x → ∞?

**Part 6.3**: What numerical issues can arise when computing these derivatives?

<details>
<summary>Solution H1</summary>

**6.1**: As x → ∞:
tanh(x) → 1, so tanh'(x) = 1 - tanh²(x) → 1 - 1 = **0**

**6.2**: As x → ∞:
σ(x) → 1, so σ'(x) = σ(x)(1-σ(x)) → 1 · 0 = **0**

**6.3**: Numerical issues:
- **Overflow**: e^x can overflow for large x (x > 709 for float64)
- **Underflow**: e^(-x) becomes 0 for large x
- **Loss of precision**: 1 - σ(x) ≈ 0 when x is large

Solutions:
```python
def sigmoid_stable(x):
    # For x >= 0: σ(x) = 1/(1+e^(-x))
    # For x < 0: σ(x) = e^x/(1+e^x) (avoid overflow)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
```
</details>

### Variation H2: Non-Differentiable Points ⭐⭐
**USAAIO Priority: HIGH** — ReLU subgradient understanding

**Part 6.1**: At what point is ReLU non-differentiable?

**Part 6.2**: How do frameworks like PyTorch handle this?

**Part 6.3**: Does this cause problems in practice?

<details>
<summary>Solution H2</summary>

**6.1**: ReLU is non-differentiable at x = 0 (left derivative is 0, right derivative is 1).

**6.2**: PyTorch uses a **subgradient**: at x = 0, it typically returns 0 (treating 0 as belonging to the "negative" region).

The subgradient ∂ReLU(0) is the interval [0, 1], and any value in this interval is valid.

**6.3**: In practice, this rarely causes problems because:
- The probability of hitting exactly x = 0 with floating point is essentially 0
- Even if it happens, using any subgradient works fine for optimization
- The loss function is still convex/smooth enough for convergence

*Key Insight*: "Almost everywhere differentiable" is sufficient for gradient-based optimization!
</details>

### Variation H3: Zero Gradient Situations ⭐⭐
**USAAIO Priority: MEDIUM** — Critical points analysis

**Part 6.1**: When is the MSE gradient exactly zero?

**Part 6.2**: When is the gradient of ridge regression zero?

**Part 6.3**: Can the sigmoid derivative ever be exactly 1?

<details>
<summary>Solution H3</summary>

**6.1**: ∇L = 2Xᵀ(Xθ - y) = 0

This happens when Xθ = y (perfect fit) or when X^T(Xθ - y) = 0 (residual orthogonal to columns of X).

**6.2**: For ridge: (XᵀX + λI)θ = Xᵀy
Gradient is zero when θ = (XᵀX + λI)⁻¹Xᵀy (unique solution always exists for λ > 0).

**6.3**: σ'(x) = σ(x)(1-σ(x))

Maximum at x = 0: σ(0) = 0.5, so σ'(0) = 0.5 × 0.5 = **0.25**

The sigmoid derivative is always ≤ 0.25, never equal to 1!

*Key Insight*: This is why sigmoid causes vanishing gradients—even at its maximum, the derivative multiplies gradients by at most 0.25 per layer.
</details>

### Variation H4: Complex-Valued Derivatives ⭐⭐⭐⭐⭐
**USAAIO Priority: LOW** — Advanced (Wirtinger calculus)

For complex z = x + iy:

**Part 6.1**: What is d|z|²/dz* (Wirtinger derivative)?

**Part 6.2**: How does this relate to real-valued optimization?

<details>
<summary>Solution H4</summary>

**6.1**: |z|² = zz*

Using Wirtinger calculus: ∂(zz*)/∂z* = z

**Answer: d|z|²/dz* = z**

**6.2**: For a real-valued function f of complex z, the gradient descent update is:
z_{t+1} = z_t - η · ∂f/∂z*

This ensures the update direction is correct for minimizing the real-valued loss.

In PyTorch, complex gradients are handled automatically using Wirtinger derivatives.
</details>

### Variation H5: Higher-Order Derivatives ⭐⭐⭐
**USAAIO Priority: LOW** — Curvature, Hessians

**Part 6.1**: Compute d²(tanh x)/dx².

**Part 6.2**: Compute d²(σ(x))/dx².

**Part 6.3**: When are these second derivatives zero?

<details>
<summary>Solution H5</summary>

**6.1**: Let f = tanh(x), f' = 1 - tanh²(x) = 1 - f²

f'' = d(1 - f²)/dx = -2f · f' = -2f(1 - f²) = **-2tanh(x)(1 - tanh²(x))**

Or: f'' = -2tanh(x)sech²(x)

**6.2**: Let g = σ(x), g' = g(1-g)

g'' = d(g(1-g))/dx = g'(1-g) + g(-g') = g'(1 - 2g)
    = **σ(x)(1-σ(x))(1 - 2σ(x))**

**6.3**:
- tanh'': zero when x = 0 (since tanh(0) = 0)
- σ'': zero when σ(x) = 0.5, i.e., x = 0

At x = 0, both functions have inflection points (second derivative changes sign).

*Key Insight*: Second derivatives tell us about the curvature of the function, which is important for second-order optimization methods like Newton's method.
</details>

---

## KEY FORMULAS SUMMARY

| Activation | Function | Derivative |
|------------|----------|------------|
| Sigmoid | σ(x) = 1/(1+e^(-x)) | σ(x)(1-σ(x)) |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | 1 - tanh²(x) |
| ReLU | max(0, x) | 1_{x>0} |
| Leaky ReLU | max(αx, x) | {1 if x>0, α if x≤0} |
| Softplus | log(1 + e^x) | σ(x) |
| Swish | xσ(x) | σ(x)(1 + x(1-σ(x))) |
| GELU | xΦ(x) | Φ(x) + xφ(x) |

| Loss Function | Gradient w.r.t. θ |
|---------------|-------------------|
| MSE: Σ(y - θᵀx)² | -2Xᵀ(y - Xθ) |
| Cross-Entropy (binary) | Xᵀ(ŷ - y) |
| Cross-Entropy (multi) | Xᵀ(p - y) |
| Ridge: MSE + λ‖θ‖² | -2Xᵀ(y - Xθ) + 2λθ |
| Hinge | -yx if yx < 1, else 0 |

| Matrix Calculus | Result |
|-----------------|--------|
| ∇_x(xᵀAx) | (A + Aᵀ)x |
| ∂tr(AB)/∂A | Bᵀ |
| ∂log det(A)/∂A | A^(-T) |
| Softmax Jacobian | diag(p) - ppᵀ |

---

## ATOMIC SKILLS CHECKLIST

- [ ] Compute derivatives of common activation functions (sigmoid, tanh, ReLU)
- [ ] Apply chain rule to nested functions
- [ ] Compute gradient of MSE loss with respect to parameters
- [ ] Compute gradient of cross-entropy loss
- [ ] Use matrix calculus notation (gradient w.r.t. vectors and matrices)
- [ ] Derive normal equations from gradient = 0
- [ ] Understand vanishing gradient problem
- [ ] Implement numerical gradient checking
- [ ] Compute Softmax Jacobian
- [ ] Apply product rule to compound functions (e.g., Swish)

---

## COMMON MISCONCEPTIONS

1. **Confusing ∂/∂x with ∇**: ∂/∂x is a partial derivative (scalar), ∇ is the gradient (vector)

2. **Forgetting the chain rule**: When computing ∂L/∂θ, you must trace through all intermediate variables

3. **Sigmoid derivative ≤ 0.25**: The maximum of σ(x)(1-σ(x)) is 0.25, not 1. This causes vanishing gradients.

4. **tanh vs sigmoid relationship**: tanh(x) = 2σ(2x) - 1, so tanh'(x) = 4σ(2x)(1-σ(2x))

5. **ReLU at x=0**: The derivative is undefined but we use a subgradient (typically 0)

6. **Matrix calculus conventions**: Different conventions exist for ∂f/∂X. Be consistent with numerator vs denominator layout.

7. **Softmax gradient simplification**: ∂L/∂z = p - y is only true for cross-entropy loss, not arbitrary losses.

---

## CATEGORY I: Additional Variations (NEW)

*These variations address gaps identified in the comprehensive review.*

### Variation I1: Layer Normalization ⭐⭐⭐⭐
**USAAIO Priority: MEDIUM** — Essential for transformers (Round 2)

Layer Normalization normalizes across features (not batch):
y = γ · (x - μ)/√(σ² + ε) + β

where μ = (1/d)Σᵢxᵢ and σ² = (1/d)Σᵢ(xᵢ - μ)² are computed per sample.

**Part 6.1**: Compute ∂y/∂γ and ∂y/∂β.

**Part 6.2**: Compute ∂y/∂x (the tricky part).

**Part 6.3**: Why is LayerNorm preferred over BatchNorm in transformers?

<details>
<summary>Solution I1</summary>

**6.1**:
∂y/∂γ = (x - μ)/√(σ² + ε) = x̂ (the normalized input)
∂y/∂β = 1

**6.2**: This is complex because μ and σ² both depend on x.

Let x̂ = (x - μ)/√(σ² + ε). Then y = γx̂ + β.

∂L/∂x = γ · ∂L/∂x̂ · ∂x̂/∂x

The full derivation involves:
∂x̂ᵢ/∂xⱼ = (1/√(σ² + ε)) · [δᵢⱼ - 1/d - x̂ᵢx̂ⱼ/d]

**6.3**: LayerNorm is preferred because:
- Works with variable sequence lengths
- No batch statistics needed at inference
- Independent of batch size (important for small batches)
- Doesn't require running statistics

*Key Insight*: BatchNorm and LayerNorm have identical forward passes but different gradient flows. LayerNorm's gradient is local to each sample.
</details>

### Variation I2: Residual Connection Gradients ⭐⭐⭐
**USAAIO Priority: HIGH** — Critical for deep networks

Consider a residual block: y = x + F(x) where F is some transformation.

**Part 6.1**: Compute ∂y/∂x.

**Part 6.2**: For a deep network with L residual blocks, what is ∂yₗ/∂x₀?

**Part 6.3**: How does this solve the vanishing gradient problem?

<details>
<summary>Solution I2</summary>

**6.1**: y = x + F(x)
∂y/∂x = I + ∂F/∂x

**6.2**: For L blocks: yₗ = yₗ₋₁ + F(yₗ₋₁)
∂yₗ/∂x₀ = Πₖ₌₁ᴸ (I + ∂Fₖ/∂yₖ₋₁)

Expanding: = I + Σₖ ∂Fₖ/∂yₖ₋₁ + (higher order terms)

**6.3**: The gradient has an **identity shortcut**:
∂yₗ/∂x₀ = I + (other terms)

Even if all ∂Fₖ/∂y → 0 (vanishing), the identity term I ensures gradients flow!

*Key Insight*: The "1" in (I + ∂F/∂x) is the secret sauce of ResNets. It guarantees a gradient path that doesn't vanish, no matter how deep the network.

**Historical Note**: He et al. (2015) didn't initially understand why ResNets worked so well. The gradient flow explanation came later.
</details>

### Variation I3: Adam Optimizer Update ⭐⭐⭐
**USAAIO Priority: HIGH** — Most common optimizer in practice

Adam maintains two moving averages:
- m_t = β₁m_{t-1} + (1-β₁)g_t (first moment)
- v_t = β₂v_{t-1} + (1-β₂)g_t² (second moment)

**Part 6.1**: Write the bias-corrected estimates m̂_t and v̂_t.

**Part 6.2**: Write the full Adam update rule.

**Part 6.3**: Show that Adam reduces to SGD with momentum when β₂ → 1.

<details>
<summary>Solution I3</summary>

**6.1**: Bias correction compensates for initialization at zero:
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

**6.2**: Full update:
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)

Or expanded:
θ_{t+1} = θ_t - η · [m_t/(1-β₁ᵗ)] / [√(v_t/(1-β₂ᵗ)) + ε]

**6.3**: As β₂ → 1:
- v_t → g₀² (constant, the first gradient squared)
- The update becomes θ_{t+1} = θ_t - η · m̂_t / C
- This is momentum with adaptive learning rate

*Key Insight*: Adam combines momentum (m_t) with per-parameter learning rates (1/√v_t). It's like giving each parameter its own tuned learning rate based on historical gradients.
</details>

### Variation I4: RMSprop ⭐⭐
**USAAIO Priority: MEDIUM** — Precursor to Adam

RMSprop uses:
v_t = ρv_{t-1} + (1-ρ)g_t²

**Part 6.1**: Write the RMSprop update rule.

**Part 6.2**: Why does dividing by √v_t help?

**Part 6.3**: Compare to Adagrad.

<details>
<summary>Solution I4</summary>

**6.1**:
θ_{t+1} = θ_t - η · g_t / (√v_t + ε)

**6.2**: Dividing by √v_t:
- For parameters with large gradients: v_t large → smaller effective learning rate
- For parameters with small gradients: v_t small → larger effective learning rate
- This adapts learning rates per parameter automatically

**6.3**: Adagrad: v_t = Σₛ₌₁ᵗ gₛ²
- Adagrad accumulates forever → learning rate goes to 0
- RMSprop uses exponential moving average → forgets old gradients
- RMSprop maintains reasonable learning rates throughout training

*Key Insight*: Geoffrey Hinton proposed RMSprop in a Coursera lecture (unpublished!) to fix Adagrad's dying learning rate problem. It was later formalized in Adam.
</details>

### Variation I5: SELU (Self-Normalizing) ⭐⭐⭐
**USAAIO Priority: LOW** — Specialized activation

SELU(x) = λ · {x if x > 0, α(eˣ - 1) if x ≤ 0}
where λ ≈ 1.0507, α ≈ 1.6733

**Part 6.1**: Compute d(SELU)/dx.

**Part 6.2**: What special property does SELU have?

<details>
<summary>Solution I5</summary>

**6.1**:
d(SELU)/dx = λ · {1 if x > 0, αeˣ if x ≤ 0}

At x = 0⁻: derivative = λα ≈ 1.76

**6.2**: SELU is **self-normalizing**:
- If inputs have mean 0 and variance 1
- After applying SELU, outputs also have mean ≈ 0 and variance ≈ 1
- This property propagates through layers without explicit normalization!

*Key Insight*: The specific values of λ and α are carefully chosen (via fixed-point theory) to achieve the self-normalizing property. Klambauer et al. (2017) proved this mathematically.
</details>

### Variation I6: Focal Loss ⭐⭐⭐
**USAAIO Priority: LOW** — Object detection, imbalanced classes

Focal loss modifies cross-entropy:
FL(p_t) = -αₜ(1 - p_t)^γ log(p_t)

where p_t = p if y=1, else 1-p.

**Part 6.1**: Compute ∂FL/∂p.

**Part 6.2**: What happens when γ = 0?

**Part 6.3**: How does focal loss help with class imbalance?

<details>
<summary>Solution I6</summary>

**6.1**: Let q = 1 - p_t (confidence in wrong class).
FL = -αₜ q^γ log(p_t)

∂FL/∂p_t = -αₜ [γq^{γ-1}(-1)log(p_t) + q^γ(1/p_t)]
         = αₜ q^{γ-1} [γ log(p_t) + q/p_t · (1-p_t)]

For the derivative w.r.t. logit z (where p = σ(z)):
∂FL/∂z = αₜ · (1-p_t)^γ · (γ log(p_t)(p_t - y) + (y - p_t))

**6.2**: When γ = 0:
FL = -αₜ log(p_t) = standard weighted cross-entropy

**6.3**: The (1-p_t)^γ factor:
- Easy examples (p_t → 1): factor → 0, contribution ≈ 0
- Hard examples (p_t → 0): factor → 1, full contribution
- This down-weights easy examples automatically!

*Key Insight*: In object detection, most boxes are easy negatives (background). Focal loss focuses training on hard examples without explicit hard negative mining.
</details>

### Variation I7: Gradient Through Argmax ⭐⭐⭐⭐
**USAAIO Priority: LOW** — Reinforcement learning, NLP

The argmax function is non-differentiable. Consider the Gumbel-Softmax trick:
y = softmax((log(π) + g) / τ)
where g ~ Gumbel(0,1) and τ is temperature.

**Part 6.1**: What is ∂y/∂log(π)?

**Part 6.2**: What happens as τ → 0?

<details>
<summary>Solution I7</summary>

**6.1**: This is just softmax with shifted inputs.
Let z = (log(π) + g) / τ
∂y/∂z = diag(y) - yyᵀ (softmax Jacobian)
∂z/∂log(π) = 1/τ · I

So: ∂y/∂log(π) = (1/τ) · [diag(y) - yyᵀ]

**6.2**: As τ → 0:
- softmax becomes harder (approaches argmax)
- Gradients become larger (1/τ factor)
- In the limit, y → one-hot (non-differentiable)

*Key Insight*: Gumbel-Softmax allows backpropagation through discrete sampling. Temperature τ trades off between discrete samples (τ→0) and differentiability (τ large).
</details>

---

## VARIATION SUMMARY BY PRIORITY

### Highest Priority for USAAIO Round 1
These variations directly match Problem 6 format or appear in other problems:

| Variation | Topic | Difficulty | Why Important |
|-----------|-------|------------|---------------|
| A1 | Sigmoid derivative | ⭐ | Same format as Problem 6.1 |
| A2 | ReLU variants | ⭐ | Common in neural networks |
| B1 | Cross-entropy | ⭐⭐ | Classification loss |
| B2 | Hinge loss | ⭐⭐ | SVM (Problem 7) |
| C2 | Linear composition | ⭐⭐ | Logistic regression |
| C3 | Two-layer network | ⭐⭐⭐ | Problem 11 (MLP) |
| E1 | MSE with bias | ⭐⭐ | Direct extension of 6.2 |
| E3 | Ridge regression | ⭐⭐ | Problem 3 (Regularization) |
| F1 | tanh proof | ⭐⭐ | Exact Problem 6.1 |
| G1-G2 | Code implementations | ⭐⭐ | Problem 10 (PyTorch) |
| I2 | Residual gradients | ⭐⭐⭐ | Deep learning theory |
| I3 | Adam optimizer | ⭐⭐⭐ | Practical importance |

### Medium Priority (Round 1 Extension / Round 2)
| Variation | Topic | Difficulty |
|-----------|-------|------------|
| A3, A5 | Softplus, Swish | ⭐⭐ |
| B3, B4 | Huber, Softmax CE | ⭐⭐-⭐⭐⭐ |
| D1, D3, D4 | Matrix calculus | ⭐⭐⭐ |
| F2, F4, F5 | Theory | ⭐⭐⭐-⭐⭐⭐⭐ |
| G3-G6 | Advanced code | ⭐⭐-⭐⭐⭐ |
| I1, I4 | LayerNorm, RMSprop | ⭐⭐⭐-⭐⭐⭐⭐ |

### Lower Priority (Advanced / Research-Level)
| Variation | Topic | Difficulty |
|-----------|-------|------------|
| A4 | GELU | ⭐⭐⭐ |
| C4, C5 | BatchNorm, Attention | ⭐⭐⭐⭐ |
| D5 | Log-determinant | ⭐⭐⭐⭐ |
| E5 | Constrained optimization | ⭐⭐⭐⭐ |
| H4 | Wirtinger derivatives | ⭐⭐⭐⭐⭐ |
| I5-I7 | SELU, Focal, Gumbel | ⭐⭐⭐-⭐⭐⭐⭐ |

---

## STUDY PATH RECOMMENDATION

### Phase 1: Foundation (Week 1)
Complete these first to build core skills:
1. A1 (Sigmoid) → A2 (ReLU) → A3 (Softplus)
2. F1 (tanh proof) — this IS Problem 6.1
3. E1 (MSE with bias) — extends Problem 6.2
4. G1-G2 (Code implementations)

### Phase 2: Core Extensions (Week 2)
1. B1 (Cross-entropy) → B4 (Softmax CE)
2. C1-C3 (Chain rule applications)
3. E3 (Ridge regression)
4. G5-G6 (GD and gradient checking)

### Phase 3: Advanced Topics (Week 3-4)
1. D1, D3, D4 (Matrix calculus)
2. F2-F5 (Theory and proofs)
3. I2-I3 (Residuals, Adam)
4. H1-H3 (Edge cases)

### Phase 4: Competition Mastery (Optional)
1. Remaining B variations (Huber, Hinge, KL)
2. C4-C5 (BatchNorm, Attention)
3. I1, I4-I7 (LayerNorm, advanced topics)

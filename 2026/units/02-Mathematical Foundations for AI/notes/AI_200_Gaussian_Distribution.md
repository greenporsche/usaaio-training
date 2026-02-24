# AI 200 Sample Lecture Note

**Beaver-Edge AI Institute**

---

## 5. Gaussian Distribution

### Theorem 13. (Single-variable Gaussian)

Let **X ~ N(μ, σ²)**. Then its PDF is:

**f(x | μ, σ²) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))**

*Proof. Not required. ■*

---

### Theorem 14. (Multivariate Gaussian)

Let **X ~ N(μ, Σ) ∈ ℝ^d**. Then its PDF is:

**f(x | μ, Σ) = (1/√((2π)^d |Σ|)) · exp(-½(x-μ)^T Σ^(-1) (x-μ))**

*Proof. Not required. ■*

---

### Theorem 15.

Let **X^(0), ..., X^(N-1) ∈ ℝ^(d×1)** be jointly normal. Define **Y = Σ(n=0 to N-1) c_n X^(n) ∈ ℝ^(d×1)**.

Then:

1. **Y is a normal random vector.**

2. **E[Y] = Σ(n=0 to N-1) c_n E[X^(n)]**

3. **Var[Y] = Σ(n=0 to N-1) c_n² Var[X^(n)] + 2 Σ(0≤n<m≤N-1) Cov[X^(n), X^(m)]**

*Proof. The proof of Part 1 is not required. The proofs of other parts are omitted. ■*

---

### Theorem 16. (Marginal Gaussians are Gaussians)

Let **X ∈ ℝ^(d₁×1)**, **Y ∈ ℝ^(d₂×1)**, and

**[X; Y] ~ N([μ_X; μ_Y], [Σ_XX, Σ_XY; Σ_YX, Σ_YY])**

where **Σ_ij = Cov[i, j]** for i, j ∈ {X, Y}.

Then:

**X ~ N(μ_X, Σ_XX)**

*Proof. Not required. ■*

---

### Theorem 17. (Conditional Gaussians are Gaussians)

Let **[X; Y] ~ N([μ_X; μ_Y], [Σ_XX, Σ_XY; Σ_YX, Σ_YY])**

where **Σ_ij = Cov[i, j]** for i, j ∈ {X, Y}.

Then conditional on **Y = y**, the conditional distribution of X is Gaussian:

**X | (Y = y) ~ N(μ_X|y, Σ_X|y)**

---

*Copyright © Beaver-Edge AI Institute. All Rights Reserved. No part of this document may be copied or reproduced without the written permission of Beaver-Edge AI Institute.*

*Page 8*

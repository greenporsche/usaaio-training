# Problem 1 Variations: PCA & Projections (EXHAUSTIVE)

> Original: Unit vectors, projections onto principal components, residuals
> Core Skills: Vector normalization, dot products, projection formula, residual computation

---

## CATEGORY A: Different Vector Values (Same Structure)

### Variation A1
Let d = 3. Consider vector **v** = [2, 1, -2]^T. Let **ê** be in the same direction as **v**.

**Part 1.1**: Write ê = (1/a)[b, c, d]^T where a is a positive integer. What is a + b + c + d?
**Part 1.2**: Let x = [1, 2, 3]^T. Compute the scalar projection x · ê.
**Part 1.3**: Compute the residual r = x - (x · ê)ê. What is ||r||²?

<details>
<summary>Solution A1</summary>

**1.1**: ||v|| = √(4+1+4) = 3, so ê = (1/3)[2,1,-2]^T. a=3, b=2, c=1, d=-2. **Answer: 4**

**1.2**: x · ê = (1/3)(2+2-6) = -2/3. **Answer: -2/3**

**1.3**: proj = (-2/3)(1/3)[2,1,-2]^T = (-2/9)[2,1,-2]^T = [-4/9, -2/9, 4/9]^T
r = [1,2,3]^T - [-4/9,-2/9,4/9]^T = [13/9, 20/9, 23/9]^T
||r||² = (169+400+529)/81 = 1098/81 = 122/9. **Answer: 122/9**
</details>

### Variation A2
Let d = 3. Consider vector **v** = [1, -1, 1]^T.

**Part 1.1**: Write ê in the form (1/√a)[b,c,d]^T where a is not a perfect square. What is a + b + c + d?
**Part 1.2**: Let x = [3, 0, 3]^T. Compute the projection of x onto ê.
**Part 1.3**: Verify that r ⊥ ê by computing r · ê.

<details>
<summary>Solution A2</summary>

**1.1**: ||v|| = √3, so ê = (1/√3)[1,-1,1]^T. a=3, b=1, c=-1, d=1. **Answer: 4**

**1.2**: x · ê = (1/√3)(3+0+3) = 6/√3 = 2√3
proj = 2√3 · (1/√3)[1,-1,1]^T = 2[1,-1,1]^T = [2,-2,2]^T. **Answer: [2,-2,2]^T**

**1.3**: r = [3,0,3]^T - [2,-2,2]^T = [1,2,1]^T
r · ê = (1/√3)(1-2+1) = 0. **Answer: 0 ✓**
</details>

### Variation A3
Let d = 3. Consider vector **v** = [3, 4, 0]^T.

**Part 1.1**: Compute ||v|| and write ê.
**Part 1.2**: Let x = [6, 8, 5]^T. Compute the scalar projection and vector projection.
**Part 1.3**: What is the angle θ between x and ê? (Give cos θ)

<details>
<summary>Solution A3</summary>

**1.1**: ||v|| = √(9+16+0) = 5, ê = (1/5)[3,4,0]^T = [0.6, 0.8, 0]^T

**1.2**: Scalar: x · ê = (1/5)(18+32+0) = 10
Vector: proj = 10 · [0.6,0.8,0]^T = [6,8,0]^T

**1.3**: cos θ = (x · ê)/(||x||) = 10/√(36+64+25) = 10/√125 = 10/(5√5) = 2/√5 = 2√5/5
</details>

### Variation A4
Let d = 3. Consider vector **v** = [1, 2, 2]^T.

**Part 1.1**: Normalize v to get ê.
**Part 1.2**: Let x = [4, -1, 2]^T. Find the component of x parallel to ê.
**Part 1.3**: Find the component of x perpendicular to ê.
**Part 1.4**: Verify: ||x||² = ||x_parallel||² + ||x_perp||²

<details>
<summary>Solution A4</summary>

**1.1**: ||v|| = √(1+4+4) = 3, ê = (1/3)[1,2,2]^T

**1.2**: x · ê = (1/3)(4-2+4) = 2
x_parallel = 2ê = (2/3)[1,2,2]^T = [2/3, 4/3, 4/3]^T

**1.3**: x_perp = x - x_parallel = [4-2/3, -1-4/3, 2-4/3]^T = [10/3, -7/3, 2/3]^T

**1.4**: ||x||² = 16+1+4 = 21
||x_parallel||² = (4/9)(1+4+4) = 4
||x_perp||² = (1/9)(100+49+4) = 153/9 = 17
4 + 17 = 21 ✓
</details>

### Variation A5
Let d = 3. Consider vector **v** = [1, 0, -1]^T.

**Part 1.1**: Find ê.
**Part 1.2**: Let x = [a, b, c]^T be arbitrary. Write the projection formula.
**Part 1.3**: For what values of x is the projection zero?

<details>
<summary>Solution A5</summary>

**1.1**: ||v|| = √2, ê = (1/√2)[1,0,-1]^T

**1.2**: x · ê = (1/√2)(a - c)
proj = ((a-c)/√2) · (1/√2)[1,0,-1]^T = ((a-c)/2)[1,0,-1]^T

**1.3**: Projection is zero when a - c = 0, i.e., when a = c.
Any vector of form [a, b, a]^T has zero projection onto ê.
</details>

---

## CATEGORY B: Higher Dimensions

### Variation B1 (d = 4)
Let **v** = [1, 1, 1, 1]^T ∈ ℝ⁴.

**Part 1.1**: Find the unit vector ê.
**Part 1.2**: Let x = [4, 0, 0, 0]^T. Compute the projection.
**Part 1.3**: Compute ||r||.

<details>
<summary>Solution B1</summary>

**1.1**: ||v|| = 2, ê = (1/2)[1,1,1,1]^T

**1.2**: x · ê = (1/2)(4) = 2
proj = 2 · (1/2)[1,1,1,1]^T = [1,1,1,1]^T

**1.3**: r = [4,0,0,0]^T - [1,1,1,1]^T = [3,-1,-1,-1]^T
||r|| = √(9+1+1+1) = √12 = 2√3
</details>

### Variation B2 (d = 4)
Let **v** = [1, -1, 1, -1]^T ∈ ℝ⁴.

**Part 1.1**: Find ê and verify ||ê|| = 1.
**Part 1.2**: Let x = [2, 2, 2, 2]^T. What is x · ê?
**Part 1.3**: Interpret the result geometrically.

<details>
<summary>Solution B2</summary>

**1.1**: ||v|| = 2, ê = (1/2)[1,-1,1,-1]^T
||ê||² = (1/4)(1+1+1+1) = 1 ✓

**1.2**: x · ê = (1/2)(2-2+2-2) = 0

**1.3**: x is orthogonal to ê. The projection is zero, meaning x lies entirely in the hyperplane perpendicular to ê.
</details>

### Variation B3 (d = 5)
Let **v** = [1, 2, 0, -2, 1]^T ∈ ℝ⁵.

**Part 1.1**: Compute ||v||² and ||v||.
**Part 1.2**: Let x = [1, 1, 1, 1, 1]^T. Compute x · v.
**Part 1.3**: Compute the scalar projection of x onto v.

<details>
<summary>Solution B3</summary>

**1.1**: ||v||² = 1+4+0+4+1 = 10, ||v|| = √10

**1.2**: x · v = 1+2+0-2+1 = 2

**1.3**: Scalar projection = (x · v)/||v|| = 2/√10 = √10/5
</details>

### Variation B4 (d = 2, simpler)
Let **v** = [3, 4]^T ∈ ℝ².

**Part 1.1**: Find ê.
**Part 1.2**: Let x = [5, 0]^T. Find the projection.
**Part 1.3**: Find the distance from x to the line spanned by v.

<details>
<summary>Solution B4</summary>

**1.1**: ||v|| = 5, ê = [3/5, 4/5]^T = [0.6, 0.8]^T

**1.2**: x · ê = 3
proj = 3[0.6, 0.8]^T = [1.8, 2.4]^T

**1.3**: r = [5,0]^T - [1.8, 2.4]^T = [3.2, -2.4]^T
distance = ||r|| = √(10.24 + 5.76) = √16 = 4
</details>

---

## CATEGORY C: Multiple Principal Components

### Variation C1 (Two orthogonal PCs)
Let **ê₁** = (1/√2)[1, 1, 0]^T and **ê₂** = (1/√2)[1, -1, 0]^T.

**Part 1.1**: Verify ê₁ ⊥ ê₂.
**Part 1.2**: Let x = [3, 1, 5]^T. Project x onto span{ê₁, ê₂}.
**Part 1.3**: What is the residual? Interpret geometrically.

<details>
<summary>Solution C1</summary>

**1.1**: ê₁ · ê₂ = (1/2)(1-1+0) = 0 ✓

**1.2**:
x · ê₁ = (1/√2)(3+1) = 2√2
x · ê₂ = (1/√2)(3-1) = √2
proj = 2√2 · ê₁ + √2 · ê₂ = 2[1,1,0]^T + [1,-1,0]^T = [3, 1, 0]^T

**1.3**: r = [3,1,5]^T - [3,1,0]^T = [0,0,5]^T
The residual is purely in the z-direction, which is orthogonal to the xy-plane spanned by ê₁ and ê₂.
</details>

### Variation C2 (Three orthonormal PCs in ℝ³)
Let **ê₁** = [1,0,0]^T, **ê₂** = [0,1,0]^T, **ê₃** = [0,0,1]^T (standard basis).

**Part 1.1**: For x = [a,b,c]^T, what is the projection onto span{ê₁, ê₂}?
**Part 1.2**: What is the residual?
**Part 1.3**: If we project onto all three, what is the residual?

<details>
<summary>Solution C2</summary>

**1.1**: proj = (x·ê₁)ê₁ + (x·ê₂)ê₂ = aê₁ + bê₂ = [a,b,0]^T

**1.2**: r = [a,b,c]^T - [a,b,0]^T = [0,0,c]^T

**1.3**: proj = aê₁ + bê₂ + cê₃ = [a,b,c]^T = x
Residual = 0 (the three vectors span all of ℝ³)
</details>

### Variation C3 (Non-standard orthonormal basis)
Let **ê₁** = (1/√3)[1,1,1]^T, **ê₂** = (1/√2)[1,-1,0]^T, **ê₃** = (1/√6)[1,1,-2]^T.

**Part 1.1**: Verify this is an orthonormal basis.
**Part 1.2**: Express x = [1,0,0]^T in this basis.
**Part 1.3**: Verify your answer by reconstruction.

<details>
<summary>Solution C3</summary>

**1.1**:
- ||ê₁||² = (1/3)(3) = 1 ✓
- ||ê₂||² = (1/2)(2) = 1 ✓
- ||ê₃||² = (1/6)(6) = 1 ✓
- ê₁·ê₂ = (1/√6)(1-1+0) = 0 ✓
- ê₁·ê₃ = (1/√18)(1+1-2) = 0 ✓
- ê₂·ê₃ = (1/√12)(1-1+0) = 0 ✓

**1.2**:
c₁ = x·ê₁ = 1/√3
c₂ = x·ê₂ = 1/√2
c₃ = x·ê₃ = 1/√6

**1.3**:
x = c₁ê₁ + c₂ê₂ + c₃ê₃
= (1/√3)(1/√3)[1,1,1]^T + (1/√2)(1/√2)[1,-1,0]^T + (1/√6)(1/√6)[1,1,-2]^T
= (1/3)[1,1,1]^T + (1/2)[1,-1,0]^T + (1/6)[1,1,-2]^T
= [1/3+1/2+1/6, 1/3-1/2+1/6, 1/3+0-1/3]^T
= [1, 0, 0]^T ✓
</details>

### Variation C4 (Explained Variance)
A dataset X ∈ ℝ^(100×3) has been centered (mean subtracted). The first two principal components are:
- **ê₁** with eigenvalue λ₁ = 50
- **ê₂** with eigenvalue λ₂ = 30
- **ê₃** with eigenvalue λ₃ = 20

**Part 1.1**: What is the total variance in the data?
**Part 1.2**: What fraction of variance is explained by the first PC alone?
**Part 1.3**: What fraction is explained by the first two PCs together?
**Part 1.4**: If we project all data onto span{ê₁, ê₂}, what is the average reconstruction error per sample?

<details>
<summary>Solution C4</summary>

**1.1**: Total variance = λ₁ + λ₂ + λ₃ = 50 + 30 + 20 = 100

**1.2**: Fraction by PC1 = λ₁/total = 50/100 = **0.5 or 50%**

**1.3**: Fraction by PC1+PC2 = (λ₁+λ₂)/total = 80/100 = **0.8 or 80%**

**1.4**: Reconstruction error = variance not captured = λ₃ = 20
Average per sample = 20/100 = **0.2**

Key insight: Eigenvalues in PCA represent variance along each principal direction.
</details>

### Variation C5 (Incremental PCA)
You have projected x = [10, 6, 3]^T onto ê₁ = (1/√2)[1, 1, 0]^T, obtaining:
- Projection onto PC1: proj₁ = [8, 8, 0]^T
- Residual: r₁ = [2, -2, 3]^T

Now you add a second PC: ê₂ = (1/√2)[1, -1, 0]^T.

**Part 1.1**: What additional variance does PC2 capture from x?
**Part 1.2**: What is the new residual after projecting onto both PCs?
**Part 1.3**: Is there any benefit to adding a third PC ê₃ = [0, 0, 1]^T?

<details>
<summary>Solution C5</summary>

**1.1**:
r₁ · ê₂ = (1/√2)(2 - (-2) + 0) = 4/√2 = 2√2
Additional projection = (r₁ · ê₂)ê₂ = 2√2 · (1/√2)[1,-1,0]^T = 2[1,-1,0]^T = [2,-2,0]^T
Additional variance captured = ||[2,-2,0]||² = 8

**1.2**:
proj₂ = proj₁ + [2,-2,0]^T = [8,8,0]^T + [2,-2,0]^T = [10,6,0]^T
New residual r₂ = x - proj₂ = [10,6,3]^T - [10,6,0]^T = [0,0,3]^T

**1.3**:
r₂ · ê₃ = 3, so proj onto ê₃ is 3[0,0,1]^T = [0,0,3]^T
After PC3: residual = 0
Yes! PC3 captures the remaining variance (||r₂||² = 9)
</details>

### Variation C6 (Choosing Number of PCs)
A 1000-dimensional dataset has eigenvalues: λ₁=500, λ₂=200, λ₃=100, λ₄=50, and λᵢ≤10 for i≥5.
Total variance = 1000.

**Part 1.1**: How many PCs are needed to capture 80% of the variance?
**Part 1.2**: How many PCs are needed to capture 95% of the variance?
**Part 1.3**: What is the "elbow" in the scree plot, and what does it suggest?

<details>
<summary>Solution C6</summary>

**1.1**:
- PC1: 500/1000 = 50%
- PC1+PC2: (500+200)/1000 = 70%
- PC1+PC2+PC3: (500+200+100)/1000 = 80% ✓
**Answer: 3 PCs**

**1.2**:
- PC1-4: (500+200+100+50)/1000 = 85%
- Remaining variance = 150, spread across ≤996 PCs with λᵢ≤10
- Need enough to add 100 more (95%-85%=10% of 1000)
- With max λᵢ=10, need at least 10 more PCs
**Answer: At least 14 PCs** (upper bound estimate; exact count depends on individual λᵢ values)

**1.3**: The elbow is at PC4→PC5 where eigenvalues drop from 50 to ≤10. This suggests 4 PCs capture the "signal" while remaining PCs capture "noise."
</details>

### Variation C7 (PCA Reconstruction)
Given orthonormal PCs ê₁ = [1,0,0]^T, ê₂ = [0,1,0]^T and original point x = [3, 4, 5]^T.

**Part 1.1**: Reconstruct x using only PC1. What is the reconstruction error?
**Part 1.2**: Reconstruct x using PC1 and PC2. What is the reconstruction error?
**Part 1.3**: Write the general formula for reconstruction using k PCs.

<details>
<summary>Solution C7</summary>

**1.1**:
x̂₁ = (x·ê₁)ê₁ = 3[1,0,0]^T = [3,0,0]^T
Error = ||x - x̂₁||² = ||[0,4,5]||² = 16 + 25 = **41**

**1.2**:
x̂₂ = (x·ê₁)ê₁ + (x·ê₂)ê₂ = [3,0,0]^T + [0,4,0]^T = [3,4,0]^T
Error = ||x - x̂₂||² = ||[0,0,5]||² = **25**

**1.3**:
x̂ₖ = Σᵢ₌₁ᵏ (x·êᵢ)êᵢ

Or in matrix form: x̂ₖ = Uₖ Uₖᵀ x where Uₖ = [ê₁ | ê₂ | ... | êₖ]
</details>

---

## CATEGORY D: Projection Properties & Proofs

### Variation D1 (Idempotence)
Let P be the projection matrix onto ê, where P = êêᵀ.

**Part 1.1**: For ê = (1/√2)[1,1]^T, write P explicitly.
**Part 1.2**: Compute P² and verify P² = P.
**Part 1.3**: What are the eigenvalues of P?

<details>
<summary>Solution D1</summary>

**1.1**: P = êêᵀ = (1/2)[1,1]^T[1,1] = (1/2)[[1,1],[1,1]]

**1.2**: P² = (1/4)[[1,1],[1,1]][[1,1],[1,1]] = (1/4)[[2,2],[2,2]] = (1/2)[[1,1],[1,1]] = P ✓

**1.3**: P is idempotent, so eigenvalues are 0 and 1.
- λ=1: eigenvector is ê (projection preserves vectors in its range)
- λ=0: eigenvector is any vector ⊥ ê
</details>

### Variation D2 (Symmetry)
**Part 1.1**: Prove that P = êêᵀ is symmetric.
**Part 1.2**: What does symmetry imply about the eigenvalues?
**Part 1.3**: For P projecting onto a k-dimensional subspace in ℝⁿ, what is trace(P)?

<details>
<summary>Solution D2</summary>

**1.1**: Pᵀ = (êêᵀ)ᵀ = (êᵀ)ᵀ(ê)ᵀ = êêᵀ = P ✓

**1.2**: Symmetric matrices have real eigenvalues and orthogonal eigenvectors.

**1.3**: trace(P) = sum of eigenvalues = k (there are k eigenvalues equal to 1, and n-k equal to 0)
</details>

### Variation D3 (Orthogonal complement)
Let P project onto span{ê} and Q = I - P project onto the orthogonal complement.

**Part 1.1**: Verify that PQ = 0.
**Part 1.2**: Verify that P + Q = I.
**Part 1.3**: For any x, verify ||x||² = ||Px||² + ||Qx||².

<details>
<summary>Solution D3</summary>

**1.1**: PQ = P(I-P) = P - P² = P - P = 0 ✓

**1.2**: P + Q = P + (I-P) = I ✓

**1.3**: ||x||² = xᵀx = xᵀ(P+Q)ᵀ(P+Q)x = xᵀ(P²+PQ+QP+Q²)x = xᵀ(P+Q)x = xᵀPx + xᵀQx = ||Px||² + ||Qx||² ✓
(using P²=P, Q²=Q, PQ=QP=0)
</details>

---

## CATEGORY E: Computational Variations (Different Numbers)

### Variation E1
v = [2, -3, 6]^T, x = [1, 1, 1]^T

<details>
<summary>Solution E1</summary>
||v|| = √(4+9+36) = 7
ê = (1/7)[2,-3,6]^T
x·ê = (1/7)(2-3+6) = 5/7
proj = (5/7)(1/7)[2,-3,6]^T = (5/49)[2,-3,6]^T = [10/49, -15/49, 30/49]^T
r = [1,1,1]^T - [10/49, -15/49, 30/49]^T = [39/49, 64/49, 19/49]^T
</details>

### Variation E2
v = [1, 2, -2, 4]^T, x = [5, 0, 0, 0]^T

<details>
<summary>Solution E2</summary>
||v|| = √(1+4+4+16) = 5
ê = (1/5)[1,2,-2,4]^T
x·ê = (1/5)(5) = 1
proj = (1/5)[1,2,-2,4]^T = [0.2, 0.4, -0.4, 0.8]^T
r = [5,0,0,0]^T - [0.2,0.4,-0.4,0.8]^T = [4.8, -0.4, 0.4, -0.8]^T
||r||² = 23.04 + 0.16 + 0.16 + 0.64 = 24
</details>

### Variation E3
v = [1, 1, 1, 1, 1]^T, x = [5, 4, 3, 2, 1]^T

<details>
<summary>Solution E3</summary>
||v|| = √5
ê = (1/√5)[1,1,1,1,1]^T
x·ê = (1/√5)(5+4+3+2+1) = 15/√5 = 3√5
proj = 3√5 · (1/√5)[1,1,1,1,1]^T = 3[1,1,1,1,1]^T = [3,3,3,3,3]^T
r = [5,4,3,2,1]^T - [3,3,3,3,3]^T = [2,1,0,-1,-2]^T
||r||² = 4+1+0+1+4 = 10
</details>

### Variation E4
v = [1, -1, 0]^T, x = [a, a, b]^T (parametric)

<details>
<summary>Solution E4</summary>
||v|| = √2
ê = (1/√2)[1,-1,0]^T
x·ê = (1/√2)(a - a) = 0
proj = 0 (for any a, b!)
r = x (the entire vector is the residual)

Key insight: Any vector of form [a,a,b]^T is orthogonal to [1,-1,0]^T.
</details>

### Variation E5
v = [cos θ, sin θ]^T for arbitrary θ, x = [1, 0]^T

<details>
<summary>Solution E5</summary>
||v|| = √(cos²θ + sin²θ) = 1 (already unit)
ê = [cos θ, sin θ]^T
x·ê = cos θ
proj = cos θ · [cos θ, sin θ]^T = [cos²θ, cos θ sin θ]^T
r = [1,0]^T - [cos²θ, cos θ sin θ]^T = [sin²θ, -cos θ sin θ]^T
||r|| = |sin θ|
</details>

---

## CATEGORY F: Word Problems / Applications

### Variation F1 (Physics: Force decomposition)
A force F = [10, 6, -8]^T Newtons acts on an object. The object can only move along the direction v = [1, 2, 2]^T.

**Part 1.1**: What component of F does useful work (parallel to motion)?
**Part 1.2**: What component is wasted (perpendicular to motion)?
**Part 1.3**: If the object moves 5 meters along v, how much work is done?

<details>
<summary>Solution F1</summary>

**1.1**: ||v|| = 3, ê = (1/3)[1,2,2]^T
F·ê = (1/3)(10+12-16) = 2
F_parallel = 2ê = (2/3)[1,2,2]^T

**1.2**: F_perp = F - F_parallel = [10,6,-8]^T - [2/3, 4/3, 4/3]^T = [28/3, 14/3, -28/3]^T

**1.3**: Work = F_parallel · displacement = 2 · 5 = 10 Joules
(or: F · (5ê) = 5(F·ê) = 5·2 = 10 J)
</details>

### Variation F2 (Statistics: Regression)
In simple linear regression, we project y onto the column space of X = [1, x]^T.

**Part 1.1**: If x = [1, 2, 3]^T and y = [2, 4, 5]^T, find the projection of y onto x.
**Part 1.2**: What is the residual?
**Part 1.3**: Verify the residual is orthogonal to x.

<details>
<summary>Solution F2</summary>

**1.1**: ||x|| = √14
ê = (1/√14)[1,2,3]^T
y·ê = (1/√14)(2+8+15) = 25/√14
proj = (25/√14)(1/√14)[1,2,3]^T = (25/14)[1,2,3]^T = [25/14, 50/14, 75/14]^T

**1.2**: r = [2,4,5]^T - [25/14, 50/14, 75/14]^T = [3/14, 6/14, -5/14]^T = (1/14)[3,6,-5]^T

**1.3**: r·x = (1/14)(3+12-15) = 0 ✓
</details>

### Variation F3 (Graphics: Shadow projection)
A light source is directly above (along z-axis). An object at point P = [3, 4, 5]^T casts a shadow on the xy-plane.

**Part 1.1**: What is the shadow point (projection onto z=0 plane)?
**Part 1.2**: Express this as a projection onto span{[1,0,0]^T, [0,1,0]^T}.
**Part 1.3**: What is the "height" (residual)?

<details>
<summary>Solution F3</summary>

**1.1**: Shadow = [3, 4, 0]^T (just drop the z-coordinate)

**1.2**: Let ê₁ = [1,0,0]^T, ê₂ = [0,1,0]^T
proj = (P·ê₁)ê₁ + (P·ê₂)ê₂ = 3ê₁ + 4ê₂ = [3,4,0]^T ✓

**1.3**: r = P - proj = [0,0,5]^T, ||r|| = 5 (the height above the plane)
</details>

### Variation F4 (PCA for Dimensionality Reduction)
You have 3D face scan data with 1000 points. After PCA, the first 2 PCs explain 95% of variance.
A new face x = [0.5, 0.3, 0.1]^T (in PC coordinates: c₁=0.5, c₂=0.3, c₃=0.1) needs compression.

**Part 1.1**: What is the 2D compressed representation?
**Part 1.2**: Reconstruct the face from the 2D representation (assume PCs are standard basis).
**Part 1.3**: What is the compression ratio (original dims / compressed dims)?

<details>
<summary>Solution F4</summary>

**1.1**: 2D representation = [c₁, c₂] = **[0.5, 0.3]** (drop PC3 coefficient)

**1.2**: Reconstructed = c₁ê₁ + c₂ê₂ = 0.5[1,0,0]^T + 0.3[0,1,0]^T = **[0.5, 0.3, 0]^T**

**1.3**: Compression ratio = 3/2 = **1.5x**

For real applications: original might be 10000D → 50D, giving 200x compression!
</details>

### Variation F5 (Whitening Transformation)
Data X has covariance matrix Σ with eigendecomposition Σ = UΛUᵀ where:
- U = [[1/√2, 1/√2], [1/√2, -1/√2]] (eigenvectors as columns)
- Λ = diag(4, 1) (eigenvalues)

**Part 1.1**: What is the whitening transformation W?
**Part 1.2**: Apply W to x = [2, 0]^T to get the whitened vector.
**Part 1.3**: Verify: the transformed data has identity covariance (for this single point, show ||Wx|| relates to original variance correctly).

<details>
<summary>Solution F5</summary>

**1.1**: Whitening transformation: W = Λ^(-1/2) Uᵀ

Λ^(-1/2) = diag(1/2, 1)
W = [[1/2, 0], [0, 1]] · [[1/√2, 1/√2], [1/√2, -1/√2]]
W = [[1/(2√2), 1/(2√2)], [1/√2, -1/√2]]

**1.2**:
Wx = [[1/(2√2), 1/(2√2)], [1/√2, -1/√2]] · [2, 0]^T
= [2/(2√2), 2/√2]^T = [1/√2, √2]^T ≈ [0.707, 1.414]^T

**1.3**:
Original x in PC coords: Uᵀx = [√2, √2]^T
Scaled by eigenvalues: variance along PC1 is 4, along PC2 is 1
After whitening: [√2/2, √2·1]^T = [1/√2, √2]^T ✓
Both components now have unit variance contribution.
</details>

### Variation F6 (Noise Filtering via PCA)
A signal s = [4, 4, 4]^T is corrupted by noise n to give observed x = [4.2, 3.8, 4.1]^T.
The signal lies in direction v = [1, 1, 1]^T.

**Part 1.1**: Project x onto the signal direction to denoise it.
**Part 1.2**: What is the estimated noise (residual)?
**Part 1.3**: What SNR (signal-to-noise ratio in dB) does this correspond to?

<details>
<summary>Solution F6</summary>

**1.1**:
||v|| = √3, ê = (1/√3)[1,1,1]^T
x·ê = (1/√3)(4.2 + 3.8 + 4.1) = 12.1/√3 ≈ 6.987
proj = 6.987 · (1/√3)[1,1,1]^T ≈ 4.033[1,1,1]^T = **[4.033, 4.033, 4.033]^T**

**1.2**:
noise = x - proj = [4.2-4.033, 3.8-4.033, 4.1-4.033]^T ≈ **[0.167, -0.233, 0.067]^T**

**1.3**:
Signal power = ||proj||² ≈ 3 · 4.033² ≈ 48.8
Noise power = ||noise||² ≈ 0.028 + 0.054 + 0.004 ≈ 0.086
SNR = 10·log₁₀(48.8/0.086) ≈ 10·log₁₀(567) ≈ **27.5 dB**
</details>

### Variation F7 (Data Visualization)
High-dimensional word embeddings in ℝ^(300) need to be visualized in 2D. The first two PCs are ê₁, ê₂.
Three words have coordinates:
- "king": x₁ = [...] with x₁·ê₁ = 2.5, x₁·ê₂ = 1.0
- "queen": x₂ = [...] with x₂·ê₁ = 2.3, x₂·ê₂ = 1.8
- "man": x₃ = [...] with x₃·ê₁ = 1.0, x₃·ê₂ = 0.5

**Part 1.1**: Plot the 2D projections (give coordinates).
**Part 1.2**: In 2D, which two words are closest?
**Part 1.3**: What information is lost in this visualization?

<details>
<summary>Solution F7</summary>

**1.1**: 2D coordinates:
- "king": (2.5, 1.0)
- "queen": (2.3, 1.8)
- "man": (1.0, 0.5)

**1.2**:
d(king, queen) = √((2.5-2.3)² + (1.0-1.8)²) = √(0.04 + 0.64) = √0.68 ≈ 0.825
d(king, man) = √((2.5-1.0)² + (1.0-0.5)²) = √(2.25 + 0.25) = √2.5 ≈ 1.58
d(queen, man) = √((2.3-1.0)² + (1.8-0.5)²) = √(1.69 + 1.69) = √3.38 ≈ 1.84
**Closest: king and queen**

**1.3**: Lost information:
- Distances in the remaining 298 dimensions
- The relationship "king - man + woman = queen" might not be visible in 2D
- Words that are similar in high-D but different in PC1-PC2 appear far apart
</details>

---

## CATEGORY G: Edge Cases & Tricky Situations

### Variation G1 (Zero projection)
v = [1, 0]^T, x = [0, 5]^T

**Part 1.1**: Compute the projection.
**Part 1.2**: What does this mean geometrically?

<details>
<summary>Solution G1</summary>

**1.1**: ê = [1,0]^T, x·ê = 0, proj = 0

**1.2**: x is perpendicular to v. The projection is zero because x has no component in the direction of v.
</details>

### Variation G2 (Full projection)
v = [2, 4]^T, x = [1, 2]^T

**Part 1.1**: Compute the projection.
**Part 1.2**: What is the residual?
**Part 1.3**: What does this mean?

<details>
<summary>Solution G2</summary>

**1.1**: ||v|| = √20 = 2√5, ê = (1/2√5)[2,4]^T = (1/√5)[1,2]^T
x·ê = (1/√5)(1+4) = √5
proj = √5 · (1/√5)[1,2]^T = [1,2]^T = x

**1.2**: r = x - proj = 0

**1.3**: x is parallel to v (x = (1/2)v). The projection equals x itself.
</details>

### Variation G3 (Negative scalar projection)
v = [1, 0]^T, x = [-3, 4]^T

**Part 1.1**: Compute the scalar projection.
**Part 1.2**: Compute the vector projection.
**Part 1.3**: Interpret the negative sign.

<details>
<summary>Solution G3</summary>

**1.1**: ê = [1,0]^T, scalar proj = x·ê = -3

**1.2**: vector proj = -3[1,0]^T = [-3, 0]^T

**1.3**: The negative sign means x points "backwards" relative to ê. The angle between x and ê is obtuse (> 90°).
</details>

### Variation G4 (Projection onto zero vector - undefined!)
**Part 1.1**: What happens if we try to project x = [1, 2]^T onto v = [0, 0]^T?
**Part 1.2**: Why is this mathematically undefined?
**Part 1.3**: How would you handle this case in code?

<details>
<summary>Solution G4</summary>

**1.1**: **The projection is undefined.**
We would need ê = v/||v|| = [0,0]/0, which involves division by zero.

**1.2**: Geometrically: The zero vector doesn't define a direction to project onto.
Algebraically: The projection formula (x·v/||v||²)v becomes (x·0/0)·0 = 0/0, which is indeterminate.

**1.3**: In code:
```python
def safe_project(x, v):
    norm_sq = np.dot(v, v)
    if norm_sq < 1e-10:  # numerical tolerance
        return np.zeros_like(x)  # or raise an error
    return (np.dot(x, v) / norm_sq) * v
```
</details>

### Variation G5 (Nearly parallel vectors - numerical stability)
v = [1, 0]^T, x = [1000000, 0.001]^T

**Part 1.1**: Compute the projection analytically.
**Part 1.2**: What is ||r||/||x||?
**Part 1.3**: What numerical issues might arise when computing this in floating point?

<details>
<summary>Solution G5</summary>

**1.1**:
ê = [1, 0]^T
x·ê = 1000000
proj = 1000000[1,0]^T = [1000000, 0]^T

**1.2**:
r = x - proj = [0, 0.001]^T
||r|| = 0.001
||x|| = √(10¹² + 10⁻⁶) ≈ 10⁶
||r||/||x|| ≈ 10⁻⁹ (residual is 1 billionth of the original!)

**1.3**: Numerical issues:
- When computing ||x||², we add 10¹² + 10⁻⁶. The small term is lost to floating-point rounding.
- Subtractive cancellation: x - proj involves subtracting nearly equal large numbers
- The tiny residual might round to exactly 0 due to limited precision
- Solution: Use numerically stable orthogonalization (e.g., modified Gram-Schmidt)
</details>

### Variation G6 (Very high dimensions)
In ℝ^1000, let v = [1, 1, 1, ..., 1]^T (all ones).

**Part 1.1**: What is ||v||?
**Part 1.2**: Let x = eᵢ (the i-th standard basis vector). What is the projection of x onto v?
**Part 1.3**: What is ||r||² for this x?
**Part 1.4**: What happens to the fraction ||proj||²/||x||² as dimension d → ∞?

<details>
<summary>Solution G6</summary>

**1.1**: ||v|| = √1000 ≈ 31.62

**1.2**:
ê = (1/√1000)[1,1,...,1]^T
eᵢ · ê = 1/√1000
proj = (1/√1000) · (1/√1000)[1,1,...,1]^T = (1/1000)[1,1,...,1]^T

**1.3**:
||proj||² = (1/1000)² · 1000 = 1/1000
||x||² = 1
||r||² = ||x||² - ||proj||² = 1 - 1/1000 = **999/1000**

**1.4**:
||proj||²/||x||² = 1/d → 0 as d → ∞

Key insight: In high dimensions, almost all of a random vector is orthogonal to any fixed direction! This is the "curse of dimensionality."
</details>

### Variation G7 (Projection of projection)
Let P be the projection matrix onto ê.

**Part 1.1**: What is P(Px)? Simplify.
**Part 1.2**: What is Pⁿx for any positive integer n?
**Part 1.3**: Why does this property make sense geometrically?

<details>
<summary>Solution G7</summary>

**1.1**: P(Px) = P²x = Px (since P² = P for projection matrices)

**1.2**: Pⁿx = Px for all n ≥ 1

**1.3**: Once you project a vector onto a subspace, it's already in that subspace. Projecting again doesn't change it. It's like asking "what's the shadow of a shadow?" - it's the same shadow.
</details>

### Variation G8 (Projection preserves linear combinations)
Let x = 3a + 2b where a = [1, 0, 1]^T and b = [0, 1, 0]^T. Project onto ê = (1/√2)[1, 0, 1]^T.

**Part 1.1**: Compute proj(x) directly.
**Part 1.2**: Compute 3·proj(a) + 2·proj(b).
**Part 1.3**: Are they equal? What property does this demonstrate?

<details>
<summary>Solution G8</summary>

**1.1**:
x = [3, 2, 3]^T
x · ê = (1/√2)(3 + 0 + 3) = 6/√2 = 3√2
proj(x) = 3√2 · (1/√2)[1,0,1]^T = 3[1,0,1]^T = [3, 0, 3]^T

**1.2**:
a · ê = (1/√2)(1 + 0 + 1) = √2, proj(a) = √2 · (1/√2)[1,0,1]^T = [1,0,1]^T
b · ê = (1/√2)(0 + 0 + 0) = 0, proj(b) = [0,0,0]^T
3·proj(a) + 2·proj(b) = 3[1,0,1]^T + 0 = [3, 0, 3]^T

**1.3**: Yes, they're equal! This demonstrates that projection is a **linear operator**:
P(αx + βy) = αP(x) + βP(y)
</details>

---

## CATEGORY H: Coding Implementations (NumPy/PyTorch)

### Variation H1 (Basic Projection Function)
Implement a function to project a vector onto a direction.

**Part 1.1**: Write a NumPy function `project(x, v)` that returns the projection of x onto v.
**Part 1.2**: Test with x = [3, 4] and v = [1, 0]. Expected output: [3, 0].
**Part 1.3**: Handle the edge case where v is the zero vector.

<details>
<summary>Solution H1</summary>

```python
import numpy as np

def project(x, v):
    """Project vector x onto direction v."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    # Handle zero vector edge case
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-10:
        return np.zeros_like(x)

    # Projection formula: (x·v / ||v||²) * v
    scalar_proj = np.dot(x, v) / v_norm_sq
    return scalar_proj * v

# Test
x = np.array([3, 4])
v = np.array([1, 0])
print(project(x, v))  # [3. 0.]

# Edge case
print(project(x, [0, 0]))  # [0. 0.]
```
</details>

### Variation H2 (Projection Matrix)
**Part 1.1**: Write a function to compute the projection matrix P = vvᵀ/(vᵀv) for a given v.
**Part 1.2**: Verify P² = P (idempotence) numerically.
**Part 1.3**: Verify Pᵀ = P (symmetry) numerically.

<details>
<summary>Solution H2</summary>

```python
import numpy as np

def projection_matrix(v):
    """Return the projection matrix onto the span of v."""
    v = np.asarray(v, dtype=float).reshape(-1, 1)  # Column vector
    v_norm_sq = (v.T @ v)[0, 0]
    if v_norm_sq < 1e-10:
        return np.zeros((len(v), len(v)))
    return (v @ v.T) / v_norm_sq

# Test
v = np.array([1, 1])
P = projection_matrix(v)
print("P =\n", P)
# [[0.5 0.5]
#  [0.5 0.5]]

# Verify idempotence: P² = P
P_squared = P @ P
print("P² = P?", np.allclose(P_squared, P))  # True

# Verify symmetry: Pᵀ = P
print("Pᵀ = P?", np.allclose(P.T, P))  # True
```
</details>

### Variation H3 (Multi-PC Projection)
**Part 1.1**: Given orthonormal vectors U = [u₁, u₂, ...] as columns, write a function to project x onto their span.
**Part 1.2**: Use the formula: proj = U @ Uᵀ @ x (vectorized, no loops).
**Part 1.3**: Test with standard basis vectors projecting onto xy-plane.

<details>
<summary>Solution H3</summary>

```python
import numpy as np

def project_onto_subspace(x, U):
    """
    Project x onto the subspace spanned by orthonormal columns of U.

    Args:
        x: vector to project (n,)
        U: matrix with orthonormal columns (n, k) where k <= n

    Returns:
        projection of x onto col(U)
    """
    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)

    # Projection: P = UUᵀ, so Px = U(Uᵀx)
    return U @ (U.T @ x)

# Test: project onto xy-plane
U = np.array([[1, 0],
              [0, 1],
              [0, 0]], dtype=float)  # e1 and e2 as columns

x = np.array([3, 4, 5])
proj = project_onto_subspace(x, U)
print(proj)  # [3. 4. 0.]

# Verify residual is orthogonal to subspace
r = x - proj
print("Residual orthogonal to U?", np.allclose(U.T @ r, 0))  # True
```
</details>

### Variation H4 (Gram-Schmidt Orthogonalization)
Given vectors v₁ = [1, 1, 0] and v₂ = [1, 0, 1], use projection to create an orthonormal basis.

**Part 1.1**: Normalize v₁ to get u₁.
**Part 1.2**: Project v₂ onto u₁ and subtract to get w₂ = v₂ - proj.
**Part 1.3**: Normalize w₂ to get u₂. Verify u₁ ⊥ u₂.

<details>
<summary>Solution H4</summary>

```python
import numpy as np

def gram_schmidt(vectors):
    """Orthonormalize a list of vectors using Gram-Schmidt."""
    vectors = [np.asarray(v, dtype=float) for v in vectors]
    orthonormal = []

    for v in vectors:
        # Subtract projections onto all previous orthonormal vectors
        w = v.copy()
        for u in orthonormal:
            w = w - np.dot(v, u) * u

        # Normalize if non-zero
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            orthonormal.append(w / norm)

    return orthonormal

# Test
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])

# Step by step:
u1 = v1 / np.linalg.norm(v1)
print("u1 =", u1)  # [0.707, 0.707, 0]

proj_v2_on_u1 = np.dot(v2, u1) * u1
w2 = v2 - proj_v2_on_u1
u2 = w2 / np.linalg.norm(w2)
print("u2 =", u2)  # [0.408, -0.408, 0.816]

print("u1 · u2 =", np.dot(u1, u2))  # ~0 (orthogonal)

# Using function:
basis = gram_schmidt([v1, v2])
print("Orthonormal basis:", basis)
```
</details>

### Variation H5 (PCA from Scratch)
Implement PCA projection without using sklearn.

**Part 1.1**: Given data matrix X (n samples × d features), center it.
**Part 1.2**: Compute covariance matrix and its eigenvectors.
**Part 1.3**: Project data onto top k principal components.

<details>
<summary>Solution H5</summary>

```python
import numpy as np

def pca_project(X, k):
    """
    Project data X onto its top k principal components.

    Args:
        X: data matrix (n_samples, n_features)
        k: number of principal components

    Returns:
        X_proj: projected data (n_samples, k)
        components: principal component directions (k, n_features)
        explained_var: variance explained by each PC
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Step 1: Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Select top k components
    components = eigenvectors[:, :k].T  # (k, d)

    # Step 5: Project
    X_proj = X_centered @ components.T  # (n, k)

    explained_var = eigenvalues[:k] / eigenvalues.sum()

    return X_proj, components, explained_var

# Test with simple data
np.random.seed(42)
X = np.random.randn(100, 3)
X[:, 0] = X[:, 0] * 5  # First feature has high variance

X_proj, components, explained = pca_project(X, k=2)
print(f"Projected shape: {X_proj.shape}")  # (100, 2)
print(f"Explained variance: {explained}")  # First PC explains most
```
</details>

### Variation H6 (PyTorch Implementation)
Implement projection using PyTorch tensors with gradient support.

**Part 1.1**: Create a differentiable projection function.
**Part 1.2**: Compute gradients of ||proj(x, v)||² with respect to x.
**Part 1.3**: Verify the gradient analytically.

<details>
<summary>Solution H6</summary>

```python
import torch

def project_torch(x, v):
    """
    Differentiable projection of x onto v.
    """
    v_norm_sq = torch.dot(v, v)
    if v_norm_sq < 1e-10:
        return torch.zeros_like(x)
    scalar_proj = torch.dot(x, v) / v_norm_sq
    return scalar_proj * v

# Test with gradients
x = torch.tensor([3.0, 4.0], requires_grad=True)
v = torch.tensor([1.0, 0.0])  # Project onto x-axis

proj = project_torch(x, v)
print("Projection:", proj)  # tensor([3., 0.])

# Compute gradient of ||proj||² w.r.t. x
loss = torch.sum(proj ** 2)  # ||proj||² = 9
loss.backward()

print("Gradient d(||proj||²)/dx:", x.grad)  # tensor([6., 0.])

# Analytical verification:
# proj = (x·v/||v||²)v = x₁[1,0] for v = [1,0]
# ||proj||² = x₁²
# d(x₁²)/dx = [2x₁, 0] = [6, 0] ✓
```
</details>

### Variation H7 (Vectorized Batch Projection)
Project multiple vectors simultaneously (batch operation).

**Part 1.1**: Given X (n × d) and v (d,), project all rows of X onto v.
**Part 1.2**: Implement without loops using broadcasting.
**Part 1.3**: Compare performance with loop-based version.

<details>
<summary>Solution H7</summary>

```python
import numpy as np
import time

def project_batch_loop(X, v):
    """Project each row of X onto v using a loop."""
    v_norm_sq = np.dot(v, v)
    result = np.zeros_like(X)
    for i in range(len(X)):
        scalar = np.dot(X[i], v) / v_norm_sq
        result[i] = scalar * v
    return result

def project_batch_vectorized(X, v):
    """Project each row of X onto v using vectorization."""
    v = v.reshape(1, -1)  # (1, d)
    v_norm_sq = np.sum(v ** 2)
    scalars = (X @ v.T) / v_norm_sq  # (n, 1)
    return scalars * v  # (n, d) via broadcasting

# Test
np.random.seed(42)
X = np.random.randn(10000, 100)  # 10k vectors in R^100
v = np.random.randn(100)

# Correctness check
proj_loop = project_batch_loop(X, v)
proj_vec = project_batch_vectorized(X, v)
print("Results match:", np.allclose(proj_loop, proj_vec))  # True

# Performance comparison
start = time.time()
for _ in range(10):
    _ = project_batch_loop(X, v)
print(f"Loop: {time.time() - start:.3f}s")

start = time.time()
for _ in range(10):
    _ = project_batch_vectorized(X, v)
print(f"Vectorized: {time.time() - start:.3f}s")
# Vectorized is typically 10-100x faster!
```
</details>

---

## CATEGORY I: Proofs & Derivations

### Variation I1 (Derive the Projection Formula)
**Part 1.1**: Starting from "minimize ||x - αv||²", derive that α = (x·v)/(v·v).
**Part 1.2**: Why does minimizing distance give us the projection?
**Part 1.3**: Show this is equivalent to requiring r ⊥ v.

<details>
<summary>Solution I1</summary>

**1.1**: We want to minimize f(α) = ||x - αv||²

f(α) = (x - αv)·(x - αv) = x·x - 2α(x·v) + α²(v·v)

Take derivative and set to zero:
df/dα = -2(x·v) + 2α(v·v) = 0
α = (x·v)/(v·v) ✓

Second derivative: d²f/dα² = 2(v·v) > 0 (assuming v≠0), so this is indeed a minimum.

**1.2**: The projection onto a line is defined as the closest point on that line. "Closest" means minimum distance, hence minimizing ||x - proj||.

**1.3**:
r = x - αv is orthogonal to v when r·v = 0:
r·v = (x - αv)·v = x·v - α(v·v) = 0
α = (x·v)/(v·v)

This is the same formula! The geometric requirement (orthogonality) and the optimization requirement (minimum distance) give the same answer—because the shortest path to a line is perpendicular to it.
</details>

### Variation I2 (Prove Pythagorean Theorem for Projections)
**Part 1.1**: Let proj = (x·ê)ê and r = x - proj. Prove ||x||² = ||proj||² + ||r||².
**Part 1.2**: What does this say about the relationship between total, explained, and unexplained variance?
**Part 1.3**: Generalize to k orthogonal projections.

<details>
<summary>Solution I2</summary>

**1.1**:
||x||² = x·x = (proj + r)·(proj + r)
= proj·proj + 2(proj·r) + r·r
= ||proj||² + 2(proj·r) + ||r||²

Now, proj·r = [(x·ê)ê]·r = (x·ê)(ê·r)

Since r = x - (x·ê)ê, we have:
ê·r = ê·x - (x·ê)(ê·ê) = x·ê - x·ê = 0

Therefore: proj·r = 0, and:
||x||² = ||proj||² + ||r||² ✓

**1.2**: In statistics:
- ||x||² ↔ total variance
- ||proj||² ↔ explained variance (by the model/PC)
- ||r||² ↔ residual/unexplained variance

The Pythagorean theorem says: Total = Explained + Unexplained

**1.3**: For orthonormal {ê₁, ..., êₖ}:
||x||² = Σᵢ (x·êᵢ)² + ||r||²

where r = x - Σᵢ (x·êᵢ)êᵢ

Each term (x·êᵢ)² represents variance along the i-th direction.
</details>

### Variation I3 (Prove Projection Matrix Properties)
Let P = UUᵀ where U has orthonormal columns.

**Part 1.1**: Prove P² = P (idempotence).
**Part 1.2**: Prove Pᵀ = P (symmetry).
**Part 1.3**: Prove rank(P) = number of columns in U.

<details>
<summary>Solution I3</summary>

**1.1**: P² = (UUᵀ)(UUᵀ) = U(UᵀU)Uᵀ

Since U has orthonormal columns: UᵀU = I

Therefore: P² = UIUᵀ = UUᵀ = P ✓

**1.2**: Pᵀ = (UUᵀ)ᵀ = (Uᵀ)ᵀUᵀ = UUᵀ = P ✓

**1.3**:
- rank(P) = rank(UUᵀ)
- For any matrices A, B: rank(AB) ≤ min(rank(A), rank(B))
- rank(P) ≤ rank(U) = k (number of columns, since columns are linearly independent)
- Also: Pu = UUᵀu for each column u of U gives Pu = u (since UᵀU = I means Uᵀu = eᵢ for i-th column)
- So the k columns of U are eigenvectors of P with eigenvalue 1
- Since P is symmetric, remaining n-k eigenvectors have eigenvalue 0
- rank(P) = number of non-zero eigenvalues = k ✓
</details>

### Variation I4 (Prove Best Approximation Property)
**Part 1.1**: Prove that for any vector y in the subspace W, ||x - proj_W(x)|| ≤ ||x - y||.
**Part 1.2**: When does equality hold?
**Part 1.3**: Why is this called the "best approximation theorem"?

<details>
<summary>Solution I4</summary>

**1.1**: Let p = proj_W(x) and r = x - p (residual).

For any y ∈ W:
||x - y||² = ||(x - p) + (p - y)||² = ||r + (p - y)||²

Since r ⊥ W and (p - y) ∈ W, we have r ⊥ (p - y), so:
||r + (p - y)||² = ||r||² + ||p - y||² (Pythagorean theorem)

Therefore:
||x - y||² = ||x - p||² + ||p - y||² ≥ ||x - p||² ✓

Taking square roots: ||x - y|| ≥ ||x - p||

**1.2**: Equality holds when ||p - y|| = 0, i.e., when y = p = proj_W(x).

**1.3**: The projection is the "best approximation" of x by vectors in W, where "best" means minimum Euclidean distance. Among all vectors in W, the projection is uniquely closest to x.

This is fundamental to:
- Least squares regression (best linear fit)
- PCA (best low-rank approximation)
- Signal processing (best approximation in subspace)
</details>

### Variation I5 (Derive Normal Equations via Projection)
In linear regression, we want to find β that minimizes ||y - Xβ||².

**Part 1.1**: Interpret this as projecting y onto the column space of X.
**Part 1.2**: Show that the residual r = y - Xβ must satisfy Xᵀr = 0.
**Part 1.3**: Derive the normal equations: XᵀXβ = Xᵀy.

<details>
<summary>Solution I5</summary>

**1.1**: The column space of X, col(X), is all vectors of form Xβ for some β. Finding β to minimize ||y - Xβ|| means finding the point Xβ* in col(X) closest to y—this is the projection of y onto col(X).

**1.2**: The residual r = y - Xβ* must be orthogonal to col(X). Since col(X) is spanned by the columns of X, r must be orthogonal to each column:

xⱼᵀr = 0 for all columns xⱼ

In matrix form: Xᵀr = 0

**1.3**: Substituting r = y - Xβ:
Xᵀ(y - Xβ) = 0
Xᵀy - XᵀXβ = 0
**XᵀXβ = Xᵀy** ✓

If XᵀX is invertible: β = (XᵀX)⁻¹Xᵀy

This is the projection formula in disguise:
- P = X(XᵀX)⁻¹Xᵀ is the projection matrix onto col(X)
- ŷ = Py = X(XᵀX)⁻¹Xᵀy = Xβ
</details>

### Variation I6 (Eigenvalue Proof)
**Part 1.1**: Prove that the eigenvalues of a projection matrix are 0 and 1 only.
**Part 1.2**: How many eigenvalues equal 1?
**Part 1.3**: What are the corresponding eigenvectors?

<details>
<summary>Solution I6</summary>

**1.1**: Let Pv = λv for some eigenvector v ≠ 0.

Apply P again:
P²v = P(λv) = λPv = λ²v

But P² = P (idempotence), so:
Pv = λ²v

Therefore: λv = λ²v, which gives (λ - λ²)v = 0.

Since v ≠ 0: λ - λ² = 0, so λ(1 - λ) = 0.

Therefore: **λ = 0 or λ = 1** ✓

**1.2**: The number of eigenvalues equal to 1 equals rank(P) = dim(range(P)) = dimension of the subspace being projected onto.

For projection onto a k-dimensional subspace in ℝⁿ:
- k eigenvalues equal to 1
- n-k eigenvalues equal to 0

**1.3**:
- λ = 1: eigenvectors are any vectors in the range of P (the subspace). These are fixed by projection.
- λ = 0: eigenvectors are any vectors in the null space of P (orthogonal complement). These are annihilated by projection.
</details>

---

## CATEGORY J: Advanced Topics

### Variation J1 (Oblique Projection)
An **oblique projection** projects onto subspace W along a direction that is NOT perpendicular to W.

Let W = span{[1, 0]^T} and project along direction d = [1, 1]^T (not perpendicular to W).

**Part 1.1**: For x = [0, 2]^T, find the oblique projection onto W along d.
**Part 1.2**: Compare with the orthogonal projection.
**Part 1.3**: Is the oblique projection matrix idempotent?

<details>
<summary>Solution J1</summary>

**1.1**: The oblique projection of x finds the unique point p ∈ W such that (x - p) is parallel to d.

p = α[1, 0]^T for some α (since p ∈ W)
x - p = [0, 2]^T - [α, 0]^T = [-α, 2]^T

For this to be parallel to d = [1, 1]^T:
[-α, 2] = β[1, 1] for some β
-α = β and 2 = β, so β = 2, α = -2

p = -2[1, 0]^T = **[-2, 0]^T**

**1.2**: Orthogonal projection:
ê = [1, 0]^T
x · ê = 0
proj_orthogonal = 0 · [1, 0]^T = **[0, 0]^T**

The oblique projection is [-2, 0]^T while the orthogonal projection is [0, 0]^T!

**1.3**: Yes, oblique projections are still idempotent (P² = P). Once you project onto W, applying the projection again doesn't change anything—the point is already in W.

However, oblique projections are NOT symmetric (P ≠ Pᵀ) in general.
</details>

### Variation J2 (Projection onto Affine Subspace)
Project x = [3, 3, 3]^T onto the plane x + y + z = 6 (not through origin!).

**Part 1.1**: Find a point p₀ on the plane and the normal vector n.
**Part 1.2**: Compute the projection.
**Part 1.3**: Why can't we use the standard P = UUᵀ formula directly?

<details>
<summary>Solution J2</summary>

**1.1**:
- Point on plane: p₀ = [6, 0, 0]^T (setting y=z=0)
- Normal vector: n = [1, 1, 1]^T (coefficients of x+y+z=6)

**1.2**:
First, translate so the plane passes through origin:
x' = x - p₀ = [3, 3, 3]^T - [6, 0, 0]^T = [-3, 3, 3]^T

Project onto the plane (= reject from normal direction):
||n|| = √3, ê = (1/√3)[1, 1, 1]^T
x' · ê = (1/√3)(-3 + 3 + 3) = 3/√3 = √3
proj_onto_n = √3 · (1/√3)[1, 1, 1]^T = [1, 1, 1]^T

Projection onto plane (orthogonal complement of n):
x'_plane = x' - proj_onto_n = [-3, 3, 3]^T - [1, 1, 1]^T = [-4, 2, 2]^T

Translate back:
proj = x'_plane + p₀ = [-4, 2, 2]^T + [6, 0, 0]^T = **[2, 2, 2]^T**

Verify: 2 + 2 + 2 = 6 ✓ (on the plane)

**1.3**: P = UUᵀ only works for linear subspaces (through origin). An affine subspace (like x+y+z=6) requires translation. The formula becomes:
proj_affine(x) = p₀ + P_linear(x - p₀)
</details>

### Variation J3 (Projection onto Convex Set)
Project x = [3, 3]^T onto the unit ball B = {y : ||y|| ≤ 1}.

**Part 1.1**: Is x inside B?
**Part 1.2**: Find the projection.
**Part 1.3**: Give the general formula for projection onto a ball of radius r centered at origin.

<details>
<summary>Solution J3</summary>

**1.1**: ||x|| = √(9 + 9) = √18 ≈ 4.24 > 1
No, x is outside B.

**1.2**: The projection onto a convex set is the closest point in the set. For a ball, this is the point on the boundary in the direction of x:

proj = x / ||x|| = [3, 3]^T / √18 = (1/√18)[3, 3]^T = **[1/√2, 1/√2]^T ≈ [0.707, 0.707]^T**

**1.3**: General formula for ball of radius r:

proj_B(x) =
- x if ||x|| ≤ r (inside ball: no change)
- r · x/||x|| if ||x|| > r (outside: project to boundary)

Or compactly: proj_B(x) = min(1, r/||x||) · x

This is used in gradient clipping for neural networks!
</details>

### Variation J4 (Householder Reflection)
A Householder reflection reflects vectors across a hyperplane with normal n.

**Part 1.1**: The Householder matrix is H = I - 2nnᵀ (for unit n). Show H² = I.
**Part 1.2**: Show H = I - 2P where P is the projection onto span{n}.
**Part 1.3**: For n = (1/√2)[1, 1]^T, apply H to x = [1, 0]^T.

<details>
<summary>Solution J4</summary>

**1.1**:
H² = (I - 2nnᵀ)(I - 2nnᵀ)
= I - 2nnᵀ - 2nnᵀ + 4nnᵀnnᵀ
= I - 4nnᵀ + 4n(nᵀn)nᵀ
= I - 4nnᵀ + 4nnᵀ (since nᵀn = 1 for unit n)
= I ✓

**1.2**:
P = nnᵀ (projection onto span{n})
I - 2P = I - 2nnᵀ = H ✓

Interpretation: H = I - 2P means "subtract twice the projection onto n", which reflects across the hyperplane perpendicular to n.

**1.3**:
n = (1/√2)[1, 1]^T
nnᵀ = (1/2)[[1, 1], [1, 1]]
H = I - 2(1/2)[[1, 1], [1, 1]] = [[1,0],[0,1]] - [[1,1],[1,1]] = [[0, -1], [-1, 0]]

Hx = [[0, -1], [-1, 0]] · [1, 0]^T = **[0, -1]^T**

Geometric check: x = [1, 0] reflected across the line y = x (perpendicular to n) gives [0, 1]... wait, we got [0, -1].

Actually, the hyperplane perpendicular to [1,1] is the line x + y = 0 (through origin), which is y = -x. Reflecting [1, 0] across y = -x gives [0, -1] ✓
</details>

### Variation J5 (QR Decomposition via Projections)
Use Gram-Schmidt to decompose A = [[1, 1], [1, 2], [0, 1]] into A = QR.

**Part 1.1**: Find Q (orthonormal columns).
**Part 1.2**: Find R (upper triangular).
**Part 1.3**: Verify A = QR.

<details>
<summary>Solution J5</summary>

**1.1**: Apply Gram-Schmidt to columns of A.

a₁ = [1, 1, 0]^T
||a₁|| = √2
q₁ = (1/√2)[1, 1, 0]^T

a₂ = [1, 2, 1]^T
proj = (a₂ · q₁)q₁ = (1/√2)(1+2+0) · (1/√2)[1,1,0]^T = (3/2)[1, 1, 0]^T
w₂ = a₂ - proj = [1, 2, 1]^T - [3/2, 3/2, 0]^T = [-1/2, 1/2, 1]^T
||w₂|| = √(1/4 + 1/4 + 1) = √(3/2) = √6/2
q₂ = (2/√6)[-1/2, 1/2, 1]^T = (1/√6)[-1, 1, 2]^T

Q = [[1/√2, -1/√6], [1/√2, 1/√6], [0, 2/√6]]

**1.2**: R = QᵀA (since A = QR implies QᵀA = QᵀQR = R for orthogonal Q)

r₁₁ = q₁ · a₁ = √2
r₁₂ = q₁ · a₂ = (1/√2)(1+2+0) = 3/√2
r₂₂ = q₂ · a₂ = (1/√6)(-1+2+2) = 3/√6 = √6/2

R = [[√2, 3/√2], [0, √6/2]]

**1.3**: Verify QR = A (numerical check omitted for brevity, but can be verified by matrix multiplication)
</details>

---

## KEY FORMULAS SUMMARY

### Basic Projection

| Concept | Formula |
|---------|---------|
| Unit vector | ê = v / \|\|v\|\| |
| Scalar projection | x · ê |
| Vector projection | (x · ê)ê |
| Projection matrix (1D) | P = êêᵀ |
| Residual | r = x - Px = (I - êêᵀ)x |
| Pythagorean theorem | \|\|x\|\|² = \|\|proj\|\|² + \|\|r\|\|² |
| Orthogonality check | r · ê = 0 |

### Multi-Component Projection

| Concept | Formula |
|---------|---------|
| Multi-PC projection | Σᵢ (x · êᵢ)êᵢ |
| Projection matrix (subspace) | P = UUᵀ (U has orthonormal columns) |
| Reconstruction | x̂ₖ = Uₖ Uₖᵀ x |
| Explained variance ratio | Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ |

### Projection Matrix Properties

| Property | Condition |
|----------|-----------|
| Idempotence | P² = P |
| Symmetry | Pᵀ = P |
| Eigenvalues | λ ∈ {0, 1} only |
| Rank | rank(P) = dim(subspace) = trace(P) |
| Complement | I - P projects onto orthogonal complement |

### Advanced Formulas

| Concept | Formula |
|---------|---------|
| Normal equations | XᵀXβ = Xᵀy |
| Least squares projection | P = X(XᵀX)⁻¹Xᵀ |
| Householder reflection | H = I - 2nnᵀ |
| Affine projection | p₀ + P_linear(x - p₀) |
| Ball projection | min(1, r/\|\|x\|\|) · x |

### NumPy Quick Reference

```python
# Basic projection
def project(x, v):
    return (np.dot(x, v) / np.dot(v, v)) * v

# Projection matrix
P = np.outer(v, v) / np.dot(v, v)

# Subspace projection (U has orthonormal columns)
proj = U @ (U.T @ x)

# PCA projection
X_centered = X - X.mean(axis=0)
_, _, Vt = np.linalg.svd(X_centered)
X_proj = X_centered @ Vt[:k].T
```

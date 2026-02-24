# Problem 5 Variations: Kernel Methods (EXHAUSTIVE)

> **Original Problem**: Feature maps from kernels, kernel matrix rank, SVD relations, vectorized kernel computation
> **Core Skills**: Kernel trick, feature space, matrix rank, SVD, trace/determinant, vectorized NumPy
> **Units**: 02 (Mathematical Foundations) + 04 (ML1 - Supervised Learning)

---

## ORIGINAL PROBLEM (Reference)

**Setup**: Dataset with N samples {x^(n) ∈ ℝ}_(n=0 to N-1), N ≥ 1000. Feature function φ(x^(n)) ∈ ℝ^d.

Kernel function: **κᵢⱼ = φ(x^(i))ᵀ · φ(x^(j))**

Kernel matrix: **K = [κᵢⱼ]** for i, j = 0 to N-1

**Part 5.1**: κᵢⱼ = 1 + x^(i)x^(j) + (x^(i)x^(j))². Compute φ(x).
**Part 5.2**: κᵢⱼ = (1 + x^(i)x^(j) + 2(x^(i)x^(j))²)². Compute φ(x).
**Part 5.3**: For the kernel in 5.2, compute rank(K).
**Part 5.4**: Given Φ = UΣVᵀ (SVD), write trace(K) and det(K) in terms of SVD.
**Part 5.5**: Implement kernel matrix computation without loops or np.linalg.

<details>
<summary>Original Solutions</summary>

**5.1**: κᵢⱼ = 1 + xᵢxⱼ + (xᵢxⱼ)² = 1·1 + xᵢ·xⱼ + xᵢ²·xⱼ²

So **φ(x) = [1, x, x²]ᵀ** ∈ ℝ³

**5.2**: Let g(x) = 1 + x + 2x². Then κᵢⱼ = g(xᵢxⱼ)².

Expand: g(xᵢxⱼ)² = (1 + xᵢxⱼ + 2(xᵢxⱼ)²)²
= 1 + 2xᵢxⱼ + 5(xᵢxⱼ)² + 4(xᵢxⱼ)³ + 4(xᵢxⱼ)⁴

So **φ(x) = [1, √2·x, √5·x², 2x³, 2x⁴]ᵀ** ∈ ℝ⁵

**5.3**: rank(K) = rank(ΦΦᵀ) ≤ min(N, d) = min(N, 5) = **5** (since N ≥ 1000)

**5.4**: K = ΦΦᵀ = (UΣVᵀ)(VΣᵀUᵀ) = UΣΣᵀUᵀ = U(ΣΣᵀ)Uᵀ

- **trace(K) = trace(ΣΣᵀ) = Σᵢ σᵢ²** (sum of squared singular values)
- **det(K) = det(ΣΣᵀ) = (∏ᵢ σᵢ)²** but K is N×N and Σ is N×d...

  Actually, det(K) = 0 if N > d (rank deficient). If N ≤ d, det(K) = ∏ᵢ σᵢ².

**5.5**:
```python
def kernel_matrix(x):
    # x has shape (N,)
    outer = np.outer(x, x)  # (N, N) matrix of xᵢxⱼ
    return (1 + outer + 2 * outer**2)**2
```

</details>

---

## CATEGORY A: Different Kernel Functions (Same Structure)

### Variation A1: Linear Kernel

**κᵢⱼ = x^(i)x^(j)**

**Part 5.1**: Compute φ(x).
**Part 5.2**: What is the rank of K for N = 1000 samples from a continuous distribution?
**Part 5.3**: Implement without loops.

<details>
<summary>Solution A1</summary>

**5.1**: κᵢⱼ = xᵢ · xⱼ, so **φ(x) = x** (scalar, or [x] as 1D vector)

**5.2**: rank(K) = rank(xxᵀ) where x is the N-vector of all samples.
- xxᵀ is an outer product of a vector with itself
- **rank(K) = 1** (outer product of vector with itself has rank 1)

**5.3**:
```python
def kernel_matrix(x):
    return np.outer(x, x)
```

**Key insight**: The linear kernel in 1D maps to a 1D feature space, so the kernel matrix has rank 1 regardless of N.
</details>

### Variation A2: Polynomial Kernel (Degree 2)

**κᵢⱼ = (1 + x^(i)x^(j))²**

**Part 5.1**: Expand the kernel and find φ(x).
**Part 5.2**: What is dim(φ(x))?
**Part 5.3**: Compute rank(K) for N ≥ 1000.

<details>
<summary>Solution A2</summary>

**5.1**: (1 + xᵢxⱼ)² = 1 + 2xᵢxⱼ + (xᵢxⱼ)²
= 1·1 + (√2·xᵢ)(√2·xⱼ) + xᵢ²·xⱼ²

So **φ(x) = [1, √2·x, x²]ᵀ**

**5.2**: dim(φ(x)) = **3**

**5.3**: rank(K) ≤ min(N, 3) = **3**

```python
def kernel_matrix(x):
    outer = np.outer(x, x)
    return (1 + outer)**2
```

**Key insight**: The polynomial kernel (1+xy)^d maps to a feature space of dimension (d+1) for scalar inputs.
</details>

### Variation A3: Polynomial Kernel (Degree 3)

**κᵢⱼ = (1 + x^(i)x^(j))³**

**Part 5.1**: Expand and find φ(x).
**Part 5.2**: Compute rank(K).
**Part 5.3**: Implement without loops.

<details>
<summary>Solution A3</summary>

**5.1**: (1 + xᵢxⱼ)³ = 1 + 3xᵢxⱼ + 3(xᵢxⱼ)² + (xᵢxⱼ)³

Match to φ(xᵢ)ᵀφ(xⱼ):
- 1 = 1·1
- 3xᵢxⱼ = (√3·xᵢ)(√3·xⱼ)
- 3(xᵢxⱼ)² = (√3·xᵢ²)(√3·xⱼ²)
- (xᵢxⱼ)³ = xᵢ³·xⱼ³

**φ(x) = [1, √3·x, √3·x², x³]ᵀ** ∈ ℝ⁴

**5.2**: rank(K) = **4** (for N ≥ 4)

**5.3**:
```python
def kernel_matrix(x):
    outer = np.outer(x, x)
    return (1 + outer)**3
```
</details>

### Variation A4: Inhomogeneous Polynomial

**κᵢⱼ = (c + x^(i)x^(j))² where c = 4**

**Part 5.1**: Find φ(x).
**Part 5.2**: How does changing c affect the feature map?
**Part 5.3**: For what value of c does the constant term in φ(x) equal 1?

<details>
<summary>Solution A4</summary>

**5.1**: (4 + xᵢxⱼ)² = 16 + 8xᵢxⱼ + (xᵢxⱼ)²

**φ(x) = [4, 2√2·x, x²]ᵀ** (or equivalently [4, √8·x, x²]ᵀ)

**5.2**: Changing c:
- Scales the constant feature
- Scales the linear feature by √c
- Doesn't affect the quadratic term

For (c + xy)²: φ(x) = [√c², √(2c)·x, x²]ᵀ = [c, √(2c)·x, x²]ᵀ

**5.3**: For constant term = 1, we need c = 1.

Then (1 + xy)² → φ(x) = [1, √2·x, x²]ᵀ

**Key insight**: The parameter c in polynomial kernels controls the relative weighting of lower-degree vs higher-degree terms.
</details>

### Variation A5: Sum of Kernels

**κᵢⱼ = (x^(i)x^(j)) + (x^(i)x^(j))² + (x^(i)x^(j))³**

**Part 5.1**: Find φ(x).
**Part 5.2**: Compute rank(K) for N = 1000.
**Part 5.3**: Compare to κ'ᵢⱼ = 1 + xᵢxⱼ + (xᵢxⱼ)² + (xᵢxⱼ)³. How does adding the constant change things?

<details>
<summary>Solution A5</summary>

**5.1**: κᵢⱼ = xᵢxⱼ + xᵢ²xⱼ² + xᵢ³xⱼ³

**φ(x) = [x, x², x³]ᵀ** ∈ ℝ³

**5.2**: rank(K) = min(N, 3) = **3**

**5.3**: With constant: κ'ᵢⱼ = 1 + xᵢxⱼ + xᵢ²xⱼ² + xᵢ³xⱼ³

φ'(x) = [1, x, x², x³]ᵀ ∈ ℝ⁴

Adding the constant:
- Increases feature dimension by 1
- Increases rank by 1 (to 4)
- Adds a "bias" feature that's the same for all samples

**Key insight**: The constant term in a kernel corresponds to a bias feature in the feature space.
</details>

---

## CATEGORY B: Multivariate Kernels (Vector Inputs)

### Variation B1: Linear Kernel in ℝ²

Let **x^(n) ∈ ℝ²**. Define **κᵢⱼ = x^(i)ᵀx^(j)**.

**Part 5.1**: What is φ(x)?
**Part 5.2**: What is the maximum rank of K for any N?
**Part 5.3**: Implement without loops.

<details>
<summary>Solution B1</summary>

**5.1**: κᵢⱼ = x₁⁽ⁱ⁾x₁⁽ʲ⁾ + x₂⁽ⁱ⁾x₂⁽ʲ⁾

**φ(x) = x = [x₁, x₂]ᵀ** (identity map)

**5.2**: rank(K) = rank(XXᵀ) where X is N×2 data matrix.
rank(K) ≤ min(N, 2) = **2** (for N ≥ 2)

**5.3**:
```python
def kernel_matrix(X):
    # X has shape (N, 2)
    return X @ X.T  # (N, N)
```

**Key insight**: For linear kernels in ℝᵈ, the kernel matrix rank is at most d (the input dimension).
</details>

### Variation B2: Polynomial Kernel in ℝ²

Let **x^(n) ∈ ℝ²**. Define **κᵢⱼ = (1 + x^(i)ᵀx^(j))²**.

**Part 5.1**: Expand to find the explicit feature map.
**Part 5.2**: What is dim(φ(x))?
**Part 5.3**: Verify: does φ(x)ᵀφ(y) = κ(x,y)?

<details>
<summary>Solution B2</summary>

**5.1**: Let u = xᵀy = x₁y₁ + x₂y₂

(1 + u)² = 1 + 2u + u²
= 1 + 2(x₁y₁ + x₂y₂) + (x₁y₁ + x₂y₂)²
= 1 + 2x₁y₁ + 2x₂y₂ + x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂²

Match coefficients:
**φ(x) = [1, √2·x₁, √2·x₂, x₁², √2·x₁x₂, x₂²]ᵀ**

**5.2**: dim(φ(x)) = **6**

General formula for polynomial kernel (1+xᵀy)² in ℝᵈ: dim = C(d+2, 2) = (d+1)(d+2)/2
For d=2: (3)(4)/2 = 6 ✓

**5.3**: Verification:
φ(x)ᵀφ(y) = 1 + 2x₁y₁ + 2x₂y₂ + x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂²
= 1 + 2(x₁y₁ + x₂y₂) + (x₁y₁ + x₂y₂)²
= (1 + xᵀy)² ✓

```python
def kernel_matrix(X):
    # X has shape (N, 2)
    linear = X @ X.T  # (N, N) matrix of xᵢᵀxⱼ
    return (1 + linear)**2
```
</details>

### Variation B3: RBF-like Polynomial Approximation

Let **x^(n) ∈ ℝ**. Consider **κᵢⱼ = exp(x^(i)x^(j)) ≈ 1 + xy + (xy)²/2 + (xy)³/6** (Taylor approx).

**Part 5.1**: Find the approximate feature map.
**Part 5.2**: Why is this only an approximation to the true RBF feature space?
**Part 5.3**: Implement the approximation.

<details>
<summary>Solution B3</summary>

**5.1**: κᵢⱼ ≈ 1 + xᵢxⱼ + (xᵢxⱼ)²/2 + (xᵢxⱼ)³/6

Need φ(x)ᵀφ(y) to give these coefficients:
- 1 = 1·1
- xy = x·y
- (xy)²/2 = (x²/√2)(y²/√2)
- (xy)³/6 = (x³/√6)(y³/√6)

**φ(x) = [1, x, x²/√2, x³/√6]ᵀ**

**5.2**: The true RBF kernel exp(xy) has an infinite Taylor series:
exp(xy) = Σₙ (xy)ⁿ/n!

The true feature space is **infinite-dimensional**. Any finite truncation is an approximation.

**5.3**:
```python
def approx_kernel_matrix(x):
    outer = np.outer(x, x)
    # Taylor expansion of exp(outer) to degree 3
    return 1 + outer + outer**2/2 + outer**3/6

def true_kernel_matrix(x):
    return np.exp(np.outer(x, x))
```

**Key insight**: The RBF/Gaussian kernel corresponds to an infinite-dimensional feature space, which is why it's so powerful—but also why we use the kernel trick instead of explicit features.
</details>

### Variation B4: Product Kernel

Let **x = [x₁, x₂]ᵀ ∈ ℝ²**. Define **κ(x,y) = (1 + x₁y₁)·(1 + x₂y₂)**.

**Part 5.1**: Expand and find φ(x).
**Part 5.2**: Compare to κ'(x,y) = (1 + x₁y₁ + x₂y₂)². Which has higher feature dimension?
**Part 5.3**: Implement both kernels.

<details>
<summary>Solution B4</summary>

**5.1**: (1 + x₁y₁)(1 + x₂y₂) = 1 + x₁y₁ + x₂y₂ + x₁x₂y₁y₂

**φ(x) = [1, x₁, x₂, x₁x₂]ᵀ** ∈ ℝ⁴

**5.2**:
- Product kernel: dim = 4
- (1 + xᵀy)²: dim = 6 (from B2)

The product kernel has **lower dimension** because it only includes cross-terms up to degree 1 in each variable, not squared terms like x₁².

**5.3**:
```python
def product_kernel(X):
    # X has shape (N, 2)
    k1 = 1 + np.outer(X[:, 0], X[:, 0])  # (1 + x₁y₁)
    k2 = 1 + np.outer(X[:, 1], X[:, 1])  # (1 + x₂y₂)
    return k1 * k2

def polynomial_kernel(X):
    linear = X @ X.T
    return (1 + linear)**2
```

**Key insight**: Product kernels decompose into independent contributions from each input dimension—useful when features have different interpretations.
</details>

---

## CATEGORY C: Kernel Matrix Properties

### Variation C1: Positive Semi-Definiteness

**Part 5.1**: Prove that any kernel matrix K = ΦΦᵀ is positive semi-definite.
**Part 5.2**: Show that eigenvalues of K are non-negative.
**Part 5.3**: For κᵢⱼ = xᵢxⱼ + xᵢ²xⱼ², verify K is PSD for x = [1, 2, 3].

<details>
<summary>Solution C1</summary>

**5.1**: For any vector v ∈ ℝᴺ:
vᵀKv = vᵀ(ΦΦᵀ)v = (Φᵀv)ᵀ(Φᵀv) = ||Φᵀv||² ≥ 0

Since vᵀKv ≥ 0 for all v, K is **positive semi-definite**. ∎

**5.2**: If Kv = λv for some eigenvalue λ:
λ||v||² = λvᵀv = vᵀKv ≥ 0

Since ||v||² > 0 (eigenvectors are nonzero), we have **λ ≥ 0**. ∎

**5.3**: For x = [1, 2, 3]:
φ(x) = [x, x²]ᵀ, so:
- φ(1) = [1, 1]
- φ(2) = [2, 4]
- φ(3) = [3, 9]

Φ = [[1, 1], [2, 4], [3, 9]]

K = ΦΦᵀ:
```
K[0,0] = 1·1 + 1·1 = 2
K[0,1] = 1·2 + 1·4 = 6
K[0,2] = 1·3 + 1·9 = 12
K[1,1] = 2·2 + 4·4 = 20
K[1,2] = 2·3 + 4·9 = 42
K[2,2] = 3·3 + 9·9 = 90
```

K = [[2, 6, 12], [6, 20, 42], [12, 42, 90]]

Eigenvalues: λ ≈ [0.057, 1.87, 110.07] — all positive ✓

```python
import numpy as np
x = np.array([1, 2, 3])
outer = np.outer(x, x)
K = outer + outer**2
print(np.linalg.eigvalsh(K))  # All non-negative
```
</details>

### Variation C2: Rank Analysis

Consider κᵢⱼ = 1 + xᵢxⱼ for x with N = 5 samples: x = [1, 1, 2, 2, 3].

**Part 5.1**: Compute the feature matrix Φ.
**Part 5.2**: Compute K = ΦΦᵀ.
**Part 5.3**: What is rank(K)? Why isn't it 2?

<details>
<summary>Solution C2</summary>

**5.1**: φ(x) = [1, x]ᵀ, so:
```
Φ = [[1, 1],
     [1, 1],
     [1, 2],
     [1, 2],
     [1, 3]]
```
Shape: (5, 2)

**5.2**: K = ΦΦᵀ (5×5 matrix):
```
K[i,j] = 1 + xᵢxⱼ

K = [[2, 2, 3, 3, 4],
     [2, 2, 3, 3, 4],
     [3, 3, 5, 5, 7],
     [3, 3, 5, 5, 7],
     [4, 4, 7, 7, 10]]
```

**5.3**: rank(K) = rank(ΦΦᵀ) = rank(Φ) = **2**

But wait—rows 0,1 are identical, and rows 2,3 are identical!
So K only has 3 distinct rows. However, rank is still 2 because:
- The column space of Φ is 2D (span of [1,1,1,1,1] and [1,1,2,2,3])
- rank(ΦΦᵀ) = rank(Φ) = 2

The duplicate rows don't reduce rank beyond what's determined by the feature dimension.

```python
x = np.array([1, 1, 2, 2, 3])
K = 1 + np.outer(x, x)
print(f"Rank: {np.linalg.matrix_rank(K)}")  # 2
```

**Key insight**: Duplicate data points create duplicate rows in K, but don't affect rank. The rank is bounded by the feature space dimension.
</details>

### Variation C3: Trace and Determinant

For κᵢⱼ = xᵢxⱼ and x = [1, 2, 3]:

**Part 5.1**: Compute K.
**Part 5.2**: Compute trace(K) directly and via ||x||².
**Part 5.3**: Explain why det(K) = 0.

<details>
<summary>Solution C3</summary>

**5.1**: K = xxᵀ (outer product)
```
K = [[1, 2, 3],
     [2, 4, 6],
     [3, 6, 9]]
```

**5.2**:
- Direct: trace(K) = 1 + 4 + 9 = **14**
- Via norm: trace(xxᵀ) = xᵀx = ||x||² = 1 + 4 + 9 = **14** ✓

General: trace(K) = Σᵢ κᵢᵢ = Σᵢ φ(xᵢ)ᵀφ(xᵢ) = Σᵢ ||φ(xᵢ)||²

**5.3**: rank(K) = rank(xxᵀ) = 1 (since K is outer product of vector with itself)

For a 3×3 matrix with rank 1, there are 2 zero eigenvalues.
det(K) = product of eigenvalues = λ₁ · 0 · 0 = **0**

```python
x = np.array([1, 2, 3])
K = np.outer(x, x)
print(f"Trace: {np.trace(K)}")  # 14
print(f"Det: {np.linalg.det(K)}")  # 0.0 (or very small)
print(f"Eigenvalues: {np.linalg.eigvalsh(K)}")  # [0, 0, 14]
```

**Key insight**: trace(xxᵀ) = ||x||², and det(xxᵀ) = 0 for any vector x (rank-1 matrix).
</details>

### Variation C4: SVD Relationship

Let Φ = UΣVᵀ where Φ is N×d.

**Part 5.1**: Express K = ΦΦᵀ in terms of U and Σ.
**Part 5.2**: What are the eigenvalues of K in terms of singular values of Φ?
**Part 5.3**: If Σ = diag(3, 2, 1) and N = 5, d = 3, what is trace(K)?

<details>
<summary>Solution C4</summary>

**5.1**: K = ΦΦᵀ = (UΣVᵀ)(VΣᵀUᵀ) = UΣ(VᵀV)ΣᵀUᵀ = U(ΣΣᵀ)Uᵀ

Since VᵀV = I (V is orthogonal).

**K = U(ΣΣᵀ)Uᵀ** — this is the eigendecomposition of K!

**5.2**: The eigenvalues of K are the diagonal entries of ΣΣᵀ.
If Σ has singular values σ₁, ..., σᵣ, then:
**Eigenvalues of K = {σ₁², σ₂², ..., σᵣ², 0, 0, ..., 0}**
(with N-r zeros if N > d)

**5.3**: trace(K) = trace(ΣΣᵀ) = σ₁² + σ₂² + σ₃² = 9 + 4 + 1 = **14**

```python
# Example verification
Phi = np.random.randn(5, 3)
U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
K = Phi @ Phi.T

print(f"trace(K): {np.trace(K):.4f}")
print(f"sum(σ²): {np.sum(S**2):.4f}")  # Should match
```

**Key insight**: The eigenvalues of the kernel matrix K are exactly the squared singular values of the feature matrix Φ.
</details>

---

## CATEGORY D: Different Data / Edge Cases

### Variation D1: Identical Points

Let x = [2, 2, 2] (all same) and κᵢⱼ = 1 + xᵢxⱼ.

**Part 5.1**: Compute K.
**Part 5.2**: What is rank(K)?
**Part 5.3**: What are the eigenvalues?

<details>
<summary>Solution D1</summary>

**5.1**: Since all xᵢ = 2:
κᵢⱼ = 1 + 2·2 = 5 for all i,j

K = **5 · 11ᵀ** = [[5,5,5], [5,5,5], [5,5,5]]

**5.2**: K = 5 · (11ᵀ) where 1 = [1,1,1]ᵀ
rank(11ᵀ) = 1, so **rank(K) = 1**

**5.3**: K = 5 · 11ᵀ
Eigenvalues of 11ᵀ: λ = 3 (eigenvector [1,1,1]), and λ = 0 (multiplicity 2)
Eigenvalues of K: **{15, 0, 0}**

Check: trace(K) = 15 = 15 + 0 + 0 ✓

**Key insight**: When all data points are identical, the kernel matrix has rank 1 regardless of the kernel's feature dimension.
</details>

### Variation D2: Orthogonal Features

Let x = [0, 1] and κᵢⱼ = xᵢxⱼ.

**Part 5.1**: Compute K.
**Part 5.2**: What is special about this K?
**Part 5.3**: What happens with κᵢⱼ = 1 + xᵢxⱼ?

<details>
<summary>Solution D2</summary>

**5.1**: K = [[0·0, 0·1], [1·0, 1·1]] = **[[0, 0], [0, 1]]**

**5.2**: K is diagonal! This happens because x₀ = 0 makes all off-diagonal entries zero.
- K is already an eigendecomposition
- Eigenvalues: {0, 1}
- One data point contributes nothing to the feature space

**5.3**: κᵢⱼ = 1 + xᵢxⱼ
K = [[1, 1], [1, 2]]

Now off-diagonal is nonzero.
Eigenvalues: (3 ± √5)/2 ≈ {0.38, 2.62}

**Key insight**: Zero values in data can create sparse kernel matrices. The constant term in polynomial kernels prevents this.
</details>

### Variation D3: Negative Values

Let x = [-1, 0, 1] and κᵢⱼ = (xᵢxⱼ)².

**Part 5.1**: Compute K.
**Part 5.2**: Find φ(x).
**Part 5.3**: Why is κ(-1, 1) = 1 even though -1 and 1 are "opposites"?

<details>
<summary>Solution D3</summary>

**5.1**: κᵢⱼ = (xᵢxⱼ)²
```
K[0,0] = ((-1)(-1))² = 1
K[0,1] = ((-1)(0))² = 0
K[0,2] = ((-1)(1))² = 1
K[1,1] = (0·0)² = 0
K[1,2] = (0·1)² = 0
K[2,2] = (1·1)² = 1
```

K = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]

**5.2**: κᵢⱼ = xᵢ²·xⱼ², so **φ(x) = x²**

φ(-1) = 1, φ(0) = 0, φ(1) = 1

**5.3**: The kernel uses squared terms, which loses sign information!
In the feature space: φ(-1) = φ(1) = 1

This means -1 and 1 are **indistinguishable** in this feature space.

rank(K) = 1 because φ(-1) = φ(1), so effectively only 2 distinct points, and one is at origin.

**Key insight**: Even-degree polynomial kernels are symmetric around 0, treating positive and negative values identically.
</details>

### Variation D4: Large Scale

Let N = 10,000 and κᵢⱼ = 1 + xᵢxⱼ + (xᵢxⱼ)² + (xᵢxⱼ)³.

**Part 5.1**: What is rank(K)?
**Part 5.2**: What is the memory required to store K as float64?
**Part 5.3**: How can we avoid storing K explicitly?

<details>
<summary>Solution D4</summary>

**5.1**: φ(x) = [1, x, x², x³]ᵀ ∈ ℝ⁴
rank(K) ≤ min(10000, 4) = **4**

**5.2**: K is 10000 × 10000 = 10⁸ entries
Memory = 10⁸ × 8 bytes = 800 MB = **0.8 GB**

**5.3**: The kernel trick! Instead of computing K:
- Store only x (10000 floats = 80 KB)
- Compute κᵢⱼ on-demand when needed
- For operations like Kv, compute row-by-row

```python
def kernel_vector_product(x, v):
    """Compute Kv without storing K"""
    N = len(x)
    result = np.zeros(N)
    for i in range(N):
        # Compute row i of K times v
        ki = 1 + x[i]*x + (x[i]*x)**2 + (x[i]*x)**3
        result[i] = ki @ v
    return result

# Even better: vectorize the loop
def kernel_vector_product_fast(x, v):
    # Compute all rows at once using broadcasting
    outer = x[:, np.newaxis] * x[np.newaxis, :]  # (N, N) but computed lazily
    K = 1 + outer + outer**2 + outer**3
    return K @ v
```

**Key insight**: The kernel trick's main benefit is avoiding explicit computation of potentially huge kernel matrices—but for direct kernel matrix operations, we still need clever algorithms.
</details>

---

## CATEGORY E: Proofs and Theory

### Variation E1: Kernel Composition

**Part 5.1**: Prove: If κ₁ and κ₂ are valid kernels, then κ = κ₁ + κ₂ is a valid kernel.
**Part 5.2**: Prove: If κ₁ and κ₂ are valid kernels, then κ = κ₁ · κ₂ is a valid kernel.
**Part 5.3**: Is κ = κ₁ - κ₂ always a valid kernel?

<details>
<summary>Solution E1</summary>

**5.1 (Sum)**:
Let K₁ = Φ₁Φ₁ᵀ and K₂ = Φ₂Φ₂ᵀ.

For any v: vᵀ(K₁ + K₂)v = vᵀK₁v + vᵀK₂v ≥ 0 + 0 = 0

Both terms are non-negative (PSD), so the sum is PSD. ✓

Alternatively: K₁ + K₂ corresponds to φ(x) = [φ₁(x); φ₂(x)] (concatenation).

**5.2 (Product)**:
The product of PSD matrices isn't necessarily PSD, but the **Hadamard (element-wise) product** is.

For kernels: (κ₁ · κ₂)ᵢⱼ = κ₁(xᵢ, xⱼ) · κ₂(xᵢ, xⱼ)

By Schur product theorem: If K₁, K₂ are PSD, then K₁ ⊙ K₂ (Hadamard) is PSD. ✓

Feature map: φ(x) = φ₁(x) ⊗ φ₂(x) (tensor product)

**5.3 (Difference)**:
**No**, κ₁ - κ₂ is NOT always valid.

Counterexample: κ₁(x,y) = 1 (constant), κ₂(x,y) = 2 (larger constant)
κ(x,y) = -1 gives K = -11ᵀ, which has eigenvalue -N < 0.

Not positive semi-definite! ✗

**Key insight**: Kernels are closed under addition and multiplication, but NOT subtraction. This is why kernel design focuses on building up from basic kernels.
</details>

### Variation E2: Mercer's Theorem Connection

**Part 5.1**: State Mercer's condition for a function to be a valid kernel.
**Part 5.2**: Verify that κ(x,y) = xy satisfies Mercer's condition.
**Part 5.3**: Show that κ(x,y) = x - y does NOT satisfy Mercer's condition.

<details>
<summary>Solution E2</summary>

**5.1 (Mercer's Condition)**:
A symmetric function κ(x,y) is a valid kernel if and only if for any finite set {x₁, ..., xₙ} and any coefficients c₁, ..., cₙ:

**Σᵢ Σⱼ cᵢcⱼκ(xᵢ,xⱼ) ≥ 0**

Equivalently: the kernel matrix K = [κ(xᵢ,xⱼ)] is positive semi-definite.

**5.2 (κ = xy is valid)**:
K = xxᵀ where x = [x₁, ..., xₙ]ᵀ

For any c: cᵀKc = cᵀ(xxᵀ)c = (xᵀc)² ≥ 0 ✓

**5.3 (κ = x - y is invalid)**:
First, note κ is not symmetric: κ(x,y) = x - y ≠ y - x = κ(y,x)

But even if we meant κ(x,y) = |x - y|:

Let x₁ = 0, x₂ = 1, x₃ = 2
K = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]

Eigenvalues: {-√2, √2+1, -√2+1} ≈ {-1.41, 2.41, -0.41}

Has negative eigenvalues → NOT PSD ✗

**Key insight**: Mercer's condition is the mathematical criterion for valid kernels. Distance functions like |x-y| are NOT valid kernels (but exp(-|x-y|²) is).
</details>

### Variation E3: Rank Upper Bound Proof

**Part 5.1**: Prove: rank(ΦΦᵀ) = rank(Φ).
**Part 5.2**: Prove: rank(ΦΦᵀ) ≤ min(N, d) where Φ is N×d.
**Part 5.3**: When is rank(K) = d exactly?

<details>
<summary>Solution E3</summary>

**5.1**: We'll show rank(ΦΦᵀ) = rank(Φ).

**Claim**: null(ΦΦᵀ) = null(Φᵀ)

Proof:
- If Φᵀv = 0, then ΦΦᵀv = Φ·0 = 0, so v ∈ null(ΦΦᵀ)
- If ΦΦᵀv = 0, then vᵀΦΦᵀv = ||Φᵀv||² = 0, so Φᵀv = 0

Thus null(ΦΦᵀ) = null(Φᵀ), and by rank-nullity:
rank(ΦΦᵀ) = N - dim(null(ΦΦᵀ)) = N - dim(null(Φᵀ)) = rank(Φᵀ) = rank(Φ) ✓

**5.2**: For Φ ∈ ℝ^(N×d):
rank(Φ) ≤ min(N, d)

By part 5.1: rank(K) = rank(Φ) ≤ **min(N, d)** ✓

**5.3**: rank(K) = d exactly when:
1. d ≤ N (more samples than features)
2. Φ has full column rank (columns are linearly independent)

This requires: the N feature vectors φ(x₁), ..., φ(xₙ) span all of ℝᵈ.

Sufficient condition: N ≥ d and samples are "in general position" (no degeneracy).

**Key insight**: The kernel matrix rank tells you the effective dimensionality of your data in feature space—it can be much smaller than N.
</details>

---

## CATEGORY F: Coding Challenges (No Loops)

### Variation F1: Polynomial Kernel Implementation

Implement κᵢⱼ = (c + x^(i)ᵀx^(j))^p for X ∈ ℝ^(N×d), without loops.

<details>
<summary>Solution F1</summary>

```python
import numpy as np

def polynomial_kernel(X, c=1.0, p=2):
    """
    Compute polynomial kernel matrix.

    Args:
        X: array of shape (N, d)
        c: constant term (default 1)
        p: polynomial degree (default 2)

    Returns:
        K: kernel matrix of shape (N, N)
    """
    # Linear kernel: Xᵢᵀ Xⱼ for all pairs
    linear = X @ X.T  # (N, N)

    # Polynomial kernel
    K = (c + linear) ** p

    return K

# Test
np.random.seed(42)
X = np.random.randn(100, 5)
K = polynomial_kernel(X, c=1, p=3)
print(f"Shape: {K.shape}")  # (100, 100)
print(f"Symmetric: {np.allclose(K, K.T)}")  # True
print(f"PSD: {np.all(np.linalg.eigvalsh(K) >= -1e-10)}")  # True
```

**Key insight**: The matrix multiplication `X @ X.T` computes all N² dot products simultaneously—this is the key to vectorization.
</details>

### Variation F2: RBF Kernel Implementation

Implement κᵢⱼ = exp(-γ||x^(i) - x^(j)||²), without loops.

<details>
<summary>Solution F2</summary>

```python
import numpy as np

def rbf_kernel(X, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel matrix.

    Args:
        X: array of shape (N, d)
        gamma: kernel bandwidth parameter

    Returns:
        K: kernel matrix of shape (N, N)
    """
    # ||xᵢ - xⱼ||² = ||xᵢ||² + ||xⱼ||² - 2xᵢᵀxⱼ

    # Compute squared norms: ||xᵢ||² for each i
    sq_norms = np.sum(X**2, axis=1)  # Shape: (N,)

    # Compute pairwise squared distances
    # ||xᵢ - xⱼ||² = ||xᵢ||² + ||xⱼ||² - 2xᵢᵀxⱼ
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T

    # RBF kernel
    K = np.exp(-gamma * sq_dists)

    return K

# Test
np.random.seed(42)
X = np.random.randn(100, 5)
K = rbf_kernel(X, gamma=0.5)
print(f"Shape: {K.shape}")  # (100, 100)
print(f"Diagonal (should be 1): {K[0,0]:.6f}")  # 1.0
print(f"Symmetric: {np.allclose(K, K.T)}")  # True
print(f"PSD: {np.all(np.linalg.eigvalsh(K) >= -1e-10)}")  # True
```

**Key insight**: The squared distance formula ||x-y||² = ||x||² + ||y||² - 2xᵀy allows vectorized computation. This is a common pattern in ML.
</details>

### Variation F3: Custom Kernel from Problem 5

Implement κᵢⱼ = (1 + xᵢxⱼ + 2(xᵢxⱼ)²)² for 1D input.

<details>
<summary>Solution F3</summary>

```python
import numpy as np

def custom_kernel(x):
    """
    Compute kernel matrix for κᵢⱼ = (1 + xᵢxⱼ + 2(xᵢxⱼ)²)²

    Args:
        x: 1D array of shape (N,)

    Returns:
        K: kernel matrix of shape (N, N)
    """
    # Compute outer product: xᵢxⱼ
    outer = np.outer(x, x)  # (N, N)

    # Inner part: 1 + xy + 2(xy)²
    inner = 1 + outer + 2 * outer**2

    # Square the whole thing
    K = inner ** 2

    return K

# Test
np.random.seed(2026)
x = np.random.randn(100)
K = custom_kernel(x)

print(f"Shape: {K.shape}")  # (100, 100)
print(f"Symmetric: {np.allclose(K, K.T)}")  # True

# Verify rank
rank = np.linalg.matrix_rank(K)
print(f"Rank: {rank}")  # Should be 5 (feature dim)

# Verify against explicit feature map
def phi(x):
    """Feature map: [1, √2·x, √5·x², 2x³, 2x⁴]"""
    return np.column_stack([
        np.ones_like(x),
        np.sqrt(2) * x,
        np.sqrt(5) * x**2,
        2 * x**3,
        2 * x**4
    ])

Phi = phi(x)
K_explicit = Phi @ Phi.T
print(f"Match explicit: {np.allclose(K, K_explicit)}")  # True
```

**Key insight**: Always verify your kernel implementation against the explicit feature map when possible—it's a great sanity check.
</details>

### Variation F4: Kernel Matrix Properties

Implement functions to compute rank, trace, and top eigenvalues of K without using np.linalg except for eigenvalues.

<details>
<summary>Solution F4</summary>

```python
import numpy as np

def kernel_properties(K, tol=1e-10):
    """
    Compute properties of kernel matrix.

    Args:
        K: kernel matrix of shape (N, N)
        tol: tolerance for rank computation

    Returns:
        dict with rank, trace, top eigenvalues
    """
    # Trace: sum of diagonal elements
    trace = np.trace(K)  # Or: np.sum(np.diag(K))

    # Eigenvalues (need np.linalg for this)
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Rank: count non-zero eigenvalues
    rank = np.sum(np.abs(eigenvalues) > tol)

    # Frobenius norm: sqrt(sum of eigenvalues²) = sqrt(trace(KᵀK))
    frobenius = np.sqrt(np.sum(K**2))

    return {
        'trace': trace,
        'rank': rank,
        'top_5_eigenvalues': eigenvalues[:5],
        'frobenius_norm': frobenius,
        'trace_via_eig': np.sum(eigenvalues)  # Should match trace
    }

# Test
np.random.seed(42)
x = np.random.randn(1000)
outer = np.outer(x, x)
K = (1 + outer + 2*outer**2)**2

props = kernel_properties(K)
print(f"Trace: {props['trace']:.2f}")
print(f"Trace via eigenvalues: {props['trace_via_eig']:.2f}")
print(f"Rank: {props['rank']}")  # 5
print(f"Top 5 eigenvalues: {props['top_5_eigenvalues']}")
```

**Key insight**: trace(K) = sum of eigenvalues, which can be used to verify computations. For kernel matrices, only a few eigenvalues are typically large.
</details>

### Variation F5: Efficient Kernel-Vector Product

Implement Kv for κᵢⱼ = (1 + xᵢxⱼ)² without explicitly forming K.

<details>
<summary>Solution F5</summary>

```python
import numpy as np

def kernel_vector_product_explicit(x, v):
    """Naive: form K then multiply."""
    K = (1 + np.outer(x, x))**2
    return K @ v

def kernel_vector_product_efficient(x, v):
    """
    Compute Kv without storing full K.

    κᵢⱼ = (1 + xᵢxⱼ)² = 1 + 2xᵢxⱼ + xᵢ²xⱼ²

    Kv[i] = Σⱼ κᵢⱼ vⱼ
          = Σⱼ (1 + 2xᵢxⱼ + xᵢ²xⱼ²) vⱼ
          = Σⱼ vⱼ + 2xᵢ Σⱼ xⱼvⱼ + xᵢ² Σⱼ xⱼ²vⱼ
    """
    # Precompute sums
    sum_v = np.sum(v)           # Σⱼ vⱼ
    sum_xv = np.sum(x * v)      # Σⱼ xⱼvⱼ
    sum_x2v = np.sum(x**2 * v)  # Σⱼ xⱼ²vⱼ

    # Compute result
    result = sum_v + 2 * x * sum_xv + x**2 * sum_x2v

    return result

# Test
np.random.seed(42)
N = 10000
x = np.random.randn(N)
v = np.random.randn(N)

# Verify correctness (on smaller problem)
x_small = x[:100]
v_small = v[:100]
result_explicit = kernel_vector_product_explicit(x_small, v_small)
result_efficient = kernel_vector_product_efficient(x_small, v_small)
print(f"Match: {np.allclose(result_explicit, result_efficient)}")  # True

# Time comparison
import time

# Efficient method
start = time.time()
for _ in range(100):
    result_eff = kernel_vector_product_efficient(x, v)
time_eff = time.time() - start

# Would take too long / too much memory for explicit with N=10000
print(f"Efficient method: {time_eff:.3f}s for 100 iterations")
print(f"Memory: efficient uses O(N), explicit would use O(N²)")
```

**Key insight**: For polynomial kernels, Kv can be computed in O(N) time and space instead of O(N²) by exploiting the structure. This is crucial for kernel methods at scale.
</details>

---

## CATEGORY G: Application Problems

### Variation G1: Kernel PCA

Using the kernel κᵢⱼ = (1 + xᵢxⱼ)², project data onto the top principal component in feature space.

<details>
<summary>Solution G1</summary>

```python
import numpy as np

def kernel_pca(x, n_components=1):
    """
    Kernel PCA for 1D input with polynomial kernel.

    Args:
        x: 1D array of shape (N,)
        n_components: number of principal components

    Returns:
        projections: array of shape (N, n_components)
    """
    N = len(x)

    # Compute kernel matrix
    outer = np.outer(x, x)
    K = (1 + outer)**2

    # Center the kernel matrix
    # K_centered = K - 1ₙK - K1ₙ + 1ₙK1ₙ
    # where 1ₙ = (1/N) * ones(N,N)
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project: α_k = eigenvector_k / sqrt(λ_k)
    # Projection of point i onto component k: Σⱼ α_k[j] * K_centered[i,j]
    projections = np.zeros((N, n_components))
    for k in range(n_components):
        if eigenvalues[k] > 1e-10:
            alpha = eigenvectors[:, k] / np.sqrt(eigenvalues[k])
            projections[:, k] = K_centered @ alpha

    return projections, eigenvalues[:n_components]

# Test
np.random.seed(42)
x = np.random.randn(100)
proj, eigs = kernel_pca(x, n_components=3)
print(f"Projections shape: {proj.shape}")
print(f"Top 3 eigenvalues: {eigs}")
```

**Key insight**: Kernel PCA performs PCA in the feature space φ(x) without ever explicitly computing φ(x)—all operations use the kernel matrix.
</details>

### Variation G2: Kernel Ridge Regression

Solve the kernel ridge regression problem: min_α ||Kα - y||² + λ||α||²

<details>
<summary>Solution G2</summary>

```python
import numpy as np

def kernel_ridge_regression(x_train, y_train, x_test, lambd=1.0):
    """
    Kernel ridge regression with polynomial kernel (1 + xy)².

    The solution is: α = (K + λI)⁻¹ y
    Predictions: y_pred = K_test @ α

    Args:
        x_train: training inputs, shape (N,)
        y_train: training targets, shape (N,)
        x_test: test inputs, shape (M,)
        lambd: regularization parameter

    Returns:
        y_pred: predictions on test set
    """
    N = len(x_train)

    # Training kernel matrix
    outer_train = np.outer(x_train, x_train)
    K_train = (1 + outer_train)**2  # (N, N)

    # Solve for α: (K + λI)α = y
    alpha = np.linalg.solve(K_train + lambd * np.eye(N), y_train)

    # Test kernel matrix: κ(x_test[i], x_train[j])
    outer_test = np.outer(x_test, x_train)  # (M, N)
    K_test = (1 + outer_test)**2

    # Predictions
    y_pred = K_test @ alpha

    return y_pred, alpha

# Test: fit a nonlinear function
np.random.seed(42)
x_train = np.linspace(-3, 3, 50)
y_train = np.sin(x_train) + 0.1 * np.random.randn(50)

x_test = np.linspace(-3, 3, 100)
y_pred, alpha = kernel_ridge_regression(x_train, y_train, x_test, lambd=0.1)

# True function for comparison
y_true = np.sin(x_test)
mse = np.mean((y_pred - y_true)**2)
print(f"MSE: {mse:.4f}")
```

**Key insight**: Kernel ridge regression is the "kernelized" version of ridge regression. The solution lives in the span of training kernel vectors.
</details>

### Variation G3: Kernel SVM Dual Problem

The dual SVM problem involves: Σᵢ Σⱼ αᵢαⱼyᵢyⱼκ(xᵢ,xⱼ)

Compute this for given α, y, and kernel.

<details>
<summary>Solution G3</summary>

```python
import numpy as np

def svm_dual_objective(alpha, y, K):
    """
    Compute the dual SVM objective (to be maximized):

    L(α) = Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼKᵢⱼ

    Args:
        alpha: dual variables, shape (N,)
        y: labels (+1 or -1), shape (N,)
        K: kernel matrix, shape (N, N)

    Returns:
        objective value
    """
    # First term: Σᵢ αᵢ
    term1 = np.sum(alpha)

    # Second term: (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼKᵢⱼ
    # = (1/2) αᵀ (y yᵀ ⊙ K) α
    # = (1/2) (α ⊙ y)ᵀ K (α ⊙ y)
    alpha_y = alpha * y
    term2 = 0.5 * alpha_y @ K @ alpha_y

    return term1 - term2

def compute_kernel_alignment(K1, K2):
    """
    Kernel alignment: measures similarity between two kernels.

    A(K1, K2) = <K1, K2>_F / (||K1||_F ||K2||_F)
    """
    numerator = np.sum(K1 * K2)
    denominator = np.sqrt(np.sum(K1**2)) * np.sqrt(np.sum(K2**2))
    return numerator / denominator

# Test
np.random.seed(42)
N = 100
x = np.random.randn(N)
y = np.sign(x + 0.5 * np.random.randn(N))  # Noisy labels

# Kernel
outer = np.outer(x, x)
K = (1 + outer)**2

# Random alpha (in practice, solved via QP)
alpha = np.abs(np.random.randn(N)) * 0.1

obj = svm_dual_objective(alpha, y, K)
print(f"Dual objective: {obj:.4f}")

# Compare polynomial kernel to RBF
K_rbf = np.exp(-0.5 * (x[:, None] - x[None, :])**2)
alignment = compute_kernel_alignment(K, K_rbf)
print(f"Kernel alignment (poly vs RBF): {alignment:.4f}")
```

**Key insight**: The SVM dual problem only involves data through the kernel matrix K. Different kernels lead to different decision boundaries without changing the optimization algorithm.
</details>

---

## SUMMARY: Key Kernel Concepts

| Concept | Formula | Key Property |
|---------|---------|--------------|
| Kernel function | κ(x,y) = φ(x)ᵀφ(y) | Inner product in feature space |
| Kernel matrix | Kᵢⱼ = κ(xᵢ,xⱼ) | Symmetric, PSD |
| Feature map | φ: ℝᵈ → ℝᵖ | Maps to (possibly infinite) feature space |
| Linear kernel | κ(x,y) = xᵀy | φ(x) = x, rank ≤ d |
| Polynomial kernel | κ(x,y) = (c + xᵀy)ᵖ | Finite feature space |
| RBF kernel | κ(x,y) = exp(-γ\|\|x-y\|\|²) | Infinite feature space |
| Kernel trick | K = ΦΦᵀ | Avoid explicit φ computation |
| Rank(K) | ≤ min(N, dim(φ)) | Effective dimensionality |
| trace(K) | Σᵢ σᵢ² | Sum of squared singular values |
| PSD property | vᵀKv ≥ 0 ∀v | All eigenvalues ≥ 0 |

---

## CATEGORY H: High-Probability Exam Variations

> These variations target skills frequently tested on USAAIO but not covered above.

### Variation H1: Kernel Centering (Exam-Style)

**Setup**: For kernel PCA, we need the **centered kernel matrix** K̃ where the data has zero mean in feature space.

Given: K = ΦΦᵀ where Φ is N×d (rows are φ(xᵢ)).

The centering matrix is **H = I - (1/N)11ᵀ** where 1 is the N-vector of ones.

**Part 5.1**: Show that HΦ centers the rows of Φ (subtracts the mean feature vector).

**Part 5.2**: Express the centered kernel matrix K̃ in terms of K and H.

**Part 5.3**: For κᵢⱼ = xᵢxⱼ and x = [1, 2, 3], compute K and K̃.

**Part 5.4**: Implement centered kernel matrix computation without loops.

<details>
<summary>Solution H1</summary>

**5.1**: Let μ = (1/N)Σᵢ φ(xᵢ) be the mean feature vector (as a row).

The i-th row of HΦ:
(HΦ)ᵢ = Σⱼ Hᵢⱼ φ(xⱼ)ᵀ = φ(xᵢ)ᵀ - (1/N)Σⱼ φ(xⱼ)ᵀ = φ(xᵢ)ᵀ - μ

So HΦ has rows (φ(xᵢ) - μ)ᵀ — the centered features. ✓

**5.2**: K̃ = (HΦ)(HΦ)ᵀ = HΦΦᵀHᵀ = HKHᵀ = **HKH** (since H is symmetric)

Expanding: **K̃ = K - (1/N)1·1ᵀK - (1/N)K·1·1ᵀ + (1/N²)1·1ᵀK·1·1ᵀ**

Or more compactly: K̃ᵢⱼ = Kᵢⱼ - (1/N)Σₖ Kᵢₖ - (1/N)Σₖ Kₖⱼ + (1/N²)Σₖₗ Kₖₗ

**5.3**: For x = [1, 2, 3]:
```
K = [[1, 2, 3],
     [2, 4, 6],
     [3, 6, 9]]
```

Row means: [2, 4, 6]
Column means: [2, 4, 6]
Grand mean: 4

K̃ᵢⱼ = Kᵢⱼ - row_meanᵢ - col_meanⱼ + grand_mean

```
K̃ = [[1-2-2+4, 2-2-4+4, 3-2-6+4],
     [2-4-2+4, 4-4-4+4, 6-4-6+4],
     [3-6-2+4, 6-6-4+4, 9-6-6+4]]
   = [[1, 0, -1],
      [0, 0, 0],
      [-1, 0, 1]]
```

Verify: trace(K̃) = 2 (sum of centered variances), K̃·1 = 0 (centered). ✓

**5.4**:
```python
def centered_kernel_matrix(K):
    """
    Center kernel matrix: K̃ = HKH where H = I - (1/N)11ᵀ

    Args:
        K: kernel matrix of shape (N, N)

    Returns:
        K_centered: centered kernel matrix
    """
    N = K.shape[0]

    # Row means, column means, grand mean
    row_means = K.mean(axis=1, keepdims=True)  # (N, 1)
    col_means = K.mean(axis=0, keepdims=True)  # (1, N)
    grand_mean = K.mean()

    # K̃ᵢⱼ = Kᵢⱼ - row_meanᵢ - col_meanⱼ + grand_mean
    K_centered = K - row_means - col_means + grand_mean

    return K_centered

# Test
x = np.array([1, 2, 3])
K = np.outer(x, x)
K_tilde = centered_kernel_matrix(K)
print(K_tilde)
# [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

# Verify centering: K̃ @ ones = 0
print(K_tilde @ np.ones(3))  # [0, 0, 0]
```

**Key insight**: Kernel centering is essential for kernel PCA because PCA assumes zero-mean data. The centering formula K̃ = HKH operates entirely in kernel space—we never need explicit features.
</details>

### Variation H2: Low-Rank Kernel Approximation (Exam-Style)

**Setup**: For large N, storing the N×N kernel matrix is expensive. When rank(K) = r << N, we can use a low-rank factorization.

Given: κᵢⱼ = (1 + xᵢxⱼ)² and N = 1000 samples.

**Part 5.1**: What is the exact rank of K? Express K as K = AAᵀ where A is N×r.

**Part 5.2**: How much memory does storing A save compared to storing K? (Assume float64)

**Part 5.3**: Given A, compute Kv (kernel-vector product) efficiently. What is the time complexity?

**Part 5.4**: Implement the low-rank factorization and efficient Kv computation.

<details>
<summary>Solution H2</summary>

**5.1**: The kernel (1 + xy)² has feature map φ(x) = [1, √2·x, x²]ᵀ ∈ ℝ³.

So **rank(K) = 3** (assuming samples are in general position).

The feature matrix Φ has shape (N, 3):
```
Φ = [[1, √2·x₀, x₀²],
     [1, √2·x₁, x₁²],
     ...
     [1, √2·xₙ₋₁, xₙ₋₁²]]
```

Then **K = ΦΦᵀ**, so **A = Φ** with shape (1000, 3).

**5.2**: Memory comparison:
- Full K: 1000 × 1000 × 8 bytes = **8 MB**
- Low-rank A: 1000 × 3 × 8 bytes = **24 KB**

Savings: **8MB / 24KB ≈ 333× less memory**

**5.3**: To compute Kv:
- Naive: Kv requires O(N²) operations
- Low-rank: Kv = A(Aᵀv)
  1. Compute Aᵀv: O(Nr) operations → gives r-vector
  2. Compute A(Aᵀv): O(Nr) operations → gives N-vector

Total: **O(Nr) = O(3N) = O(N)** instead of O(N²)

**5.4**:
```python
import numpy as np

def build_feature_matrix(x):
    """
    Build explicit feature matrix for κ(x,y) = (1 + xy)²
    φ(x) = [1, √2·x, x²]

    Args:
        x: 1D array of shape (N,)

    Returns:
        A: feature matrix of shape (N, 3)
    """
    N = len(x)
    A = np.column_stack([
        np.ones(N),
        np.sqrt(2) * x,
        x**2
    ])
    return A

def kernel_vector_product_lowrank(A, v):
    """
    Compute Kv = A @ Aᵀ @ v efficiently.

    Args:
        A: feature matrix of shape (N, r)
        v: vector of shape (N,)

    Returns:
        Kv: result of shape (N,)
    """
    # Step 1: Aᵀv (r-dimensional)
    Atv = A.T @ v  # O(Nr)

    # Step 2: A(Aᵀv) (N-dimensional)
    Kv = A @ Atv   # O(Nr)

    return Kv

# Test
np.random.seed(2026)
N = 1000
x = np.random.randn(N)
v = np.random.randn(N)

# Build low-rank factorization
A = build_feature_matrix(x)
print(f"A shape: {A.shape}")  # (1000, 3)
print(f"A memory: {A.nbytes / 1024:.1f} KB")  # 24 KB

# Efficient Kv
Kv_efficient = kernel_vector_product_lowrank(A, v)

# Verify against explicit K
outer = np.outer(x, x)
K = (1 + outer)**2
Kv_explicit = K @ v
print(f"K memory: {K.nbytes / 1024**2:.1f} MB")  # 8 MB

print(f"Match: {np.allclose(Kv_efficient, Kv_explicit)}")  # True

# Timing comparison
import time

start = time.time()
for _ in range(100):
    _ = kernel_vector_product_lowrank(A, v)
time_efficient = time.time() - start

start = time.time()
for _ in range(100):
    _ = K @ v
time_explicit = time.time() - start

print(f"Low-rank: {time_efficient:.4f}s")
print(f"Explicit: {time_explicit:.4f}s")
print(f"Speedup: {time_explicit/time_efficient:.1f}x")
```

**Key insight**: When rank(K) = r << N, the low-rank factorization K = AAᵀ reduces both memory (from O(N²) to O(Nr)) and matrix-vector products (from O(N²) to O(Nr)). This is why polynomial kernels are computationally attractive—they have finite, small rank.
</details>

---

## ATOMIC SKILLS TESTED

1. **Feature Map Derivation**: Expand kernel to find explicit φ(x)
2. **Rank Analysis**: rank(K) = rank(Φ) ≤ min(N, d)
3. **SVD Connection**: K = U(ΣΣᵀ)Uᵀ, eigenvalues are σᵢ²
4. **Trace/Determinant**: trace(K) = Σσᵢ², det via eigenvalues
5. **Vectorized Implementation**: np.outer for all pairs, no loops
6. **Kernel Properties**: Symmetry, PSD, composition rules
7. **Matrix Algebra**: Outer products, eigendecomposition, rank-nullity
8. **Kernel Centering**: K̃ = HKH for zero-mean features in kernel space
9. **Low-Rank Exploitation**: K = AAᵀ for O(Nr) memory and O(Nr) matvecs

---

## COMMON MISCONCEPTIONS

1. **"Feature dimension equals kernel matrix rank"** — No, rank(K) ≤ min(N, d), and can be less if data is degenerate
2. **"RBF kernel has finite feature space"** — No, it's infinite-dimensional (Taylor series never terminates)
3. **"det(K) gives useful information"** — Usually det(K) = 0 since N >> d; trace is more informative
4. **"Kernel matrices are always full rank"** — No, rank is bounded by feature dimension
5. **"Subtracting kernels gives a kernel"** — No, the difference of PSD matrices may not be PSD

---

*Generated for USAAIO Problem 5 practice. Total variations: 27+*

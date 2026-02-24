# Problem 2 Variations: Permutation Matrices & Linear Transformations (EXHAUSTIVE)

> Original: Matrix representation of permutations, outer product decomposition
> Core Skills: Recognizing transformations, constructing matrices, decomposition

---

## CATEGORY A: Different Permutations (4×4)

### Variation A1: Reverse
For any vector **x** = [x₀, x₁, x₂, x₃]^T, we have **Ax** = [x₃, x₂, x₁, x₀]^T

**Part 2.1**: What operation does A perform?
**Part 2.2**: Write A in matrix form.
**Part 2.3**: Compute A². What does this tell you about A?
**Part 2.4**: Find the eigenvalues of A.

<details>
<summary>Solution A1</summary>

**2.1**: Reversal (reflection)

**2.2**: 
```
A = | 0 0 0 1 |
    | 0 0 1 0 |
    | 0 1 0 0 |
    | 1 0 0 0 |
```

**2.3**: A² = I (reversing twice gives identity). A is an involution.

**2.4**: Since A² = I, eigenvalues satisfy λ² = 1, so λ ∈ {1, -1}.
For 4×4 reversal: λ = 1 (multiplicity 2), λ = -1 (multiplicity 2)
Eigenvectors for λ=1: [1,0,0,1]^T, [0,1,1,0]^T (symmetric vectors)
Eigenvectors for λ=-1: [1,0,0,-1]^T, [0,1,-1,0]^T (antisymmetric vectors)
</details>

### Variation A2: Left Cyclic Shift
**Ax** = [x₁, x₂, x₃, x₀]^T

**Part 2.1**: Write A in matrix form.
**Part 2.2**: Compute f(0), f(1), f(2), f(3) for the decomposition A = Σᵢ ê^(f(i)) · ê^(i)ᵀ
**Part 2.3**: What is the period of A (smallest n such that Aⁿ = I)?
**Part 2.4**: Find all eigenvalues.

<details>
<summary>Solution A2</summary>

**2.1**: 
```
A = | 0 1 0 0 |
    | 0 0 1 0 |
    | 0 0 0 1 |
    | 1 0 0 0 |
```

**2.2**: 
- Position 0 receives x₁, so f(1) = 0
- Position 1 receives x₂, so f(2) = 1
- Position 2 receives x₃, so f(3) = 2
- Position 3 receives x₀, so f(0) = 3

f(0)=3, f(1)=0, f(2)=1, f(3)=2

**2.3**: A⁴ = I, period = 4

**2.4**: λ⁴ = 1, so λ ∈ {1, -1, i, -i}
</details>

### Variation A3: Swap First and Last
**Ax** = [x₃, x₁, x₂, x₀]^T

**Part 2.1**: Write A.
**Part 2.2**: Compute A².
**Part 2.3**: Is A symmetric?

<details>
<summary>Solution A3</summary>

**2.1**: 
```
A = | 0 0 0 1 |
    | 0 1 0 0 |
    | 0 0 1 0 |
    | 1 0 0 0 |
```

**2.2**: A² = I (swapping twice returns to original)

**2.3**: Yes, A = Aᵀ (symmetric)
</details>

### Variation A4: Three-Cycle
**Ax** = [x₁, x₂, x₀, x₃]^T (cycles first three elements)

**Part 2.1**: Write A.
**Part 2.2**: What is the period?
**Part 2.3**: Compute A².

<details>
<summary>Solution A4</summary>

**2.1**: 
```
A = | 0 1 0 0 |
    | 0 0 1 0 |
    | 1 0 0 0 |
    | 0 0 0 1 |
```

**2.2**: Period = 3 (A³ = I for the cyclic part, but x₃ is fixed)

**2.3**: A²x = [x₂, x₀, x₁, x₃]^T
```
A² = | 0 0 1 0 |
     | 1 0 0 0 |
     | 0 1 0 0 |
     | 0 0 0 1 |
```
</details>

### Variation A5: Adjacent Transposition
**Ax** = [x₁, x₀, x₂, x₃]^T (swap x₀ and x₁ only)

**Part 2.1**: Write A.
**Part 2.2**: Verify A² = I.
**Part 2.3**: What is det(A)?

<details>
<summary>Solution A5</summary>

**2.1**: 
```
A = | 0 1 0 0 |
    | 1 0 0 0 |
    | 0 0 1 0 |
    | 0 0 0 1 |
```

**2.2**: A² = I ✓ (transpositions are involutions)

**2.3**: det(A) = -1 (odd permutation - single swap)
</details>

---

## CATEGORY B: Different Dimensions

### Variation B1 (3×3 Cyclic)
**Ax** = [x₂, x₀, x₁]^T for x ∈ ℝ³

**Part 2.1**: Write A.
**Part 2.2**: Compute A², A³.
**Part 2.3**: Find eigenvalues.

<details>
<summary>Solution B1</summary>

**2.1**: 
```
A = | 0 0 1 |
    | 1 0 0 |
    | 0 1 0 |
```

**2.2**: 
A²x = [x₁, x₂, x₀]^T
A³x = [x₀, x₁, x₂]^T = x, so A³ = I

**2.3**: λ³ = 1, so λ ∈ {1, e^(2πi/3), e^(4πi/3)} = {1, -1/2 + i√3/2, -1/2 - i√3/2}
</details>

### Variation B2 (5×5 Reversal)
**Ax** = [x₄, x₃, x₂, x₁, x₀]^T for x ∈ ℝ⁵

**Part 2.1**: Write A.
**Part 2.2**: What is A²?
**Part 2.3**: How many eigenvalues equal 1? Equal -1?

<details>
<summary>Solution B2</summary>

**2.1**: 
```
A = | 0 0 0 0 1 |
    | 0 0 0 1 0 |
    | 0 0 1 0 0 |
    | 0 1 0 0 0 |
    | 1 0 0 0 0 |
```

**2.2**: A² = I

**2.3**: For odd n, reversal has:
- λ=1: (n+1)/2 = 3 times (symmetric vectors, including middle element)
- λ=-1: (n-1)/2 = 2 times (antisymmetric vectors)
</details>

### Variation B3 (2×2)
**Ax** = [x₁, x₀]^T for x ∈ ℝ²

**Part 2.1**: Write A.
**Part 2.2**: Find eigenvalues and eigenvectors.
**Part 2.3**: Diagonalize A.

<details>
<summary>Solution B3</summary>

**2.1**: 
```
A = | 0 1 |
    | 1 0 |
```

**2.2**: 
det(A - λI) = λ² - 1 = 0, so λ = ±1
λ=1: eigenvector [1,1]^T
λ=-1: eigenvector [1,-1]^T

**2.3**: A = PDP⁻¹ where
P = [1  1; 1 -1], D = [1 0; 0 -1]
P⁻¹ = (1/2)[1 1; 1 -1]
</details>

### Variation B4 (6×6 Cyclic)
**Ax** = [x₅, x₀, x₁, x₂, x₃, x₄]^T

**Part 2.1**: What is the period?
**Part 2.2**: List all eigenvalues.

<details>
<summary>Solution B4</summary>

**2.1**: Period = 6 (A⁶ = I)

**2.2**: λ⁶ = 1, the 6th roots of unity:
λ ∈ {1, e^(πi/3), e^(2πi/3), -1, e^(4πi/3), e^(5πi/3)}
= {1, (1+i√3)/2, (-1+i√3)/2, -1, (-1-i√3)/2, (1-i√3)/2}
</details>

---

## CATEGORY C: Outer Product Decomposition

### Variation C1: Explicit Decomposition
For the permutation Ax = [x₂, x₀, x₁]^T:

**Part 2.1**: Write A = Σᵢ ê^(f(i)) · ê^(i)ᵀ explicitly.
**Part 2.2**: Verify by computing each outer product.
**Part 2.3**: Verify by matrix multiplication.

<details>
<summary>Solution C1</summary>

**2.1**: 
- x₀ goes to position 1: f(0) = 1
- x₁ goes to position 2: f(1) = 2
- x₂ goes to position 0: f(2) = 0

A = ê⁽¹⁾ê⁽⁰⁾ᵀ + ê⁽²⁾ê⁽¹⁾ᵀ + ê⁽⁰⁾ê⁽²⁾ᵀ

**2.2**: 
ê⁽¹⁾ê⁽⁰⁾ᵀ = [0;1;0][1,0,0] = [[0,0,0],[1,0,0],[0,0,0]]
ê⁽²⁾ê⁽¹⁾ᵀ = [0;0;1][0,1,0] = [[0,0,0],[0,0,0],[0,1,0]]
ê⁽⁰⁾ê⁽²⁾ᵀ = [1;0;0][0,0,1] = [[0,0,1],[0,0,0],[0,0,0]]

Sum = [[0,0,1],[1,0,0],[0,1,0]] ✓

**2.3**: 
A[x₀;x₁;x₂] = [[0,0,1],[1,0,0],[0,1,0]][x₀;x₁;x₂] = [x₂;x₀;x₁] ✓
</details>

### Variation C2: Inverse from Decomposition
Given A = Σᵢ ê^(f(i)) · ê^(i)ᵀ, find A⁻¹.

**Part 2.1**: What is the relationship between A and A⁻¹ for permutation matrices?
**Part 2.2**: Express A⁻¹ using outer products.
**Part 2.3**: Verify for the swap matrix A where Ax = [x₁, x₀]^T.

<details>
<summary>Solution C2</summary>

**2.1**: For permutation matrices, A⁻¹ = Aᵀ (they are orthogonal)

**2.2**: If A = Σᵢ ê^(f(i)) · ê^(i)ᵀ, then
A⁻¹ = Aᵀ = Σᵢ ê^(i) · ê^(f(i))ᵀ = Σᵢ ê^(f⁻¹(i)) · ê^(i)ᵀ

**2.3**: A = [[0,1],[1,0]]
Aᵀ = [[0,1],[1,0]] = A
So A⁻¹ = A, which makes sense since A² = I.
</details>

### Variation C3: Composition of Permutations
Let A correspond to [x₁, x₀, x₂]^T and B correspond to [x₀, x₂, x₁]^T.

**Part 2.1**: Compute AB (apply B first, then A).
**Part 2.2**: Compute BA.
**Part 2.3**: Is matrix multiplication commutative for permutation matrices?

<details>
<summary>Solution C3</summary>

**2.1**: 
B: [x₀,x₁,x₂] → [x₀,x₂,x₁]
A: [x₀,x₂,x₁] → [x₂,x₀,x₁]
AB: [x₀,x₁,x₂] → [x₂,x₀,x₁]

**2.2**: 
A: [x₀,x₁,x₂] → [x₁,x₀,x₂]
B: [x₁,x₀,x₂] → [x₁,x₂,x₀]
BA: [x₀,x₁,x₂] → [x₁,x₂,x₀]

**2.3**: AB ≠ BA in general. Permutation matrix multiplication is NOT commutative.
</details>

---

## CATEGORY D: Properties of Permutation Matrices

### Variation D1: Determinant
**Part 2.1**: What is det(A) for a transposition (single swap)?
**Part 2.2**: What is det(A) for a 3-cycle?
**Part 2.3**: General rule for det of permutation matrix?

<details>
<summary>Solution D1</summary>

**2.1**: det = -1 (odd permutation)

**2.2**: A 3-cycle can be written as 2 transpositions, so det = (-1)² = 1 (even permutation)

**2.3**: det(A) = sign(σ) = (-1)^(number of transpositions) = ±1
- Even permutation: det = 1
- Odd permutation: det = -1
</details>

### Variation D2: Trace
**Part 2.1**: What is trace(A) for the identity permutation?
**Part 2.2**: What is trace(A) for a cyclic permutation with no fixed points?
**Part 2.3**: What does trace(A) count?

<details>
<summary>Solution D2</summary>

**2.1**: trace(I) = n (all diagonal elements are 1)

**2.2**: trace = 0 (no element stays in place, so no 1s on diagonal)

**2.3**: trace(A) = number of fixed points (elements that don't move)
</details>

### Variation D3: Orthogonality
**Part 2.1**: Prove that permutation matrices are orthogonal (AAᵀ = I).
**Part 2.2**: What are the singular values of a permutation matrix?
**Part 2.3**: What is ||A|| (operator norm)?

<details>
<summary>Solution D3</summary>

**2.1**: Each row and column has exactly one 1. 
(AAᵀ)ᵢⱼ = Σₖ Aᵢₖ Aⱼₖ = 1 if i=j (same row), 0 otherwise.
So AAᵀ = I.

**2.2**: Since A is orthogonal, AᵀA = I, so singular values are all 1.

**2.3**: ||A|| = largest singular value = 1
</details>

---

## CATEGORY E: Non-Permutation Linear Transformations

### Variation E1: Scaling
**Ax** = [2x₀, 3x₁, x₂]^T

**Part 2.1**: Write A.
**Part 2.2**: Is A a permutation matrix?
**Part 2.3**: Find eigenvalues.

<details>
<summary>Solution E1</summary>

**2.1**: 
```
A = | 2 0 0 |
    | 0 3 0 |
    | 0 0 1 |
```

**2.2**: No, it's a diagonal scaling matrix.

**2.3**: Eigenvalues are the diagonal entries: λ = 2, 3, 1
</details>

### Variation E2: Rotation (2D)
**Ax** = [x₀ cos θ - x₁ sin θ, x₀ sin θ + x₁ cos θ]^T

**Part 2.1**: Write A for θ = 90°.
**Part 2.2**: Write A for θ = 45°.
**Part 2.3**: Verify A is orthogonal.

<details>
<summary>Solution E2</summary>

**2.1**: θ = 90°: cos 90° = 0, sin 90° = 1
```
A = | 0 -1 |
    | 1  0 |
```

**2.2**: θ = 45°: cos 45° = sin 45° = 1/√2
```
A = | 1/√2  -1/√2 |
    | 1/√2   1/√2 |
```

**2.3**: AAᵀ = [[cos²θ+sin²θ, 0], [0, cos²θ+sin²θ]] = I ✓
</details>

### Variation E3: Projection
**Ax** = [x₀, 0, 0]^T (project onto x-axis)

**Part 2.1**: Write A.
**Part 2.2**: Compute A².
**Part 2.3**: What are the eigenvalues?

<details>
<summary>Solution E3</summary>

**2.1**: 
```
A = | 1 0 0 |
    | 0 0 0 |
    | 0 0 0 |
```

**2.2**: A² = A (projection is idempotent)

**2.3**: λ = 1 (multiplicity 1), λ = 0 (multiplicity 2)
</details>

### Variation E4: Reflection
**Ax** = [x₀, -x₁]^T (reflect across x-axis)

**Part 2.1**: Write A.
**Part 2.2**: Verify A² = I.
**Part 2.3**: Find eigenvalues and eigenvectors.

<details>
<summary>Solution E4</summary>

**2.1**: 
```
A = | 1  0 |
    | 0 -1 |
```

**2.2**: A² = [[1,0],[0,1]] = I ✓

**2.3**: 
λ = 1: eigenvector [1,0]^T (points on x-axis are fixed)
λ = -1: eigenvector [0,1]^T (points on y-axis are flipped)
</details>

### Variation E5: Shear
**Ax** = [x₀ + kx₁, x₁]^T

**Part 2.1**: Write A for k = 2.
**Part 2.2**: Find eigenvalues.
**Part 2.3**: Is A diagonalizable?

<details>
<summary>Solution E5</summary>

**2.1**: 
```
A = | 1 2 |
    | 0 1 |
```

**2.2**: det(A - λI) = (1-λ)² = 0, so λ = 1 (multiplicity 2)

**2.3**: No! The eigenspace for λ=1 is only 1-dimensional (span{[1,0]^T}).
A is defective (not diagonalizable).
</details>

---

## CATEGORY F: Decomposition Variations

### Variation F1: SVD of Permutation
For the swap matrix A = [[0,1],[1,0]]:

**Part 2.1**: Find the SVD: A = UΣVᵀ.
**Part 2.2**: What are the singular values?

<details>
<summary>Solution F1</summary>

**2.1**: Since A is symmetric and orthogonal:
A = Aᵀ, so A = UΣVᵀ = UΣUᵀ (eigendecomposition = SVD)
U = (1/√2)[[1,1],[1,-1]], Σ = [[1,0],[0,1]], V = U

**2.2**: Singular values are both 1 (permutation matrices preserve length).
</details>

### Variation F2: Eigendecomposition
For the 3-cycle A where Ax = [x₂, x₀, x₁]^T:

**Part 2.1**: Find eigenvalues.
**Part 2.2**: Find eigenvectors.
**Part 2.3**: Write A = PDP⁻¹.

<details>
<summary>Solution F2</summary>

**2.1**: A³ = I, so λ³ = 1
λ₁ = 1, λ₂ = ω = e^(2πi/3), λ₃ = ω² = e^(4πi/3)

**2.2**: 
For λ=1: Av = v means [v₂,v₀,v₁] = [v₀,v₁,v₂], so v₀=v₁=v₂. v₁ = [1,1,1]^T
For λ=ω: v₂ = [1, ω², ω]^T
For λ=ω²: v₃ = [1, ω, ω²]^T

**2.3**: P = [[1,1,1],[1,ω²,ω],[1,ω,ω²]], D = diag(1,ω,ω²)
</details>

---

## CATEGORY G: Word Problems

### Variation G1: Card Shuffling
A deck has 4 cards in positions [0,1,2,3]. A "riffle shuffle" moves them to [2,0,3,1].

**Part 2.1**: Write the permutation matrix.
**Part 2.2**: After how many shuffles do the cards return to original order?

<details>
<summary>Solution G1</summary>

**2.1**: 
Position 0 gets card from position 2: A₀₂ = 1
Position 1 gets card from position 0: A₁₀ = 1
Position 2 gets card from position 3: A₂₃ = 1
Position 3 gets card from position 1: A₃₁ = 1
```
A = | 0 0 1 0 |
    | 1 0 0 0 |
    | 0 0 0 1 |
    | 0 1 0 0 |
```

**2.2**: Compute powers until Aⁿ = I.
This permutation is (0 2 3 1), a 4-cycle, so period = 4.
</details>

### Variation G2: Rubik's Cube Face
A face rotation permutes corner positions [0,1,2,3] → [3,0,1,2].

**Part 2.1**: Write the permutation matrix.
**Part 2.2**: What is the period?
**Part 2.3**: Express as a product of transpositions.

<details>
<summary>Solution G2</summary>

**2.1**: This is a right cyclic shift.
```
A = | 0 0 0 1 |
    | 1 0 0 0 |
    | 0 1 0 0 |
    | 0 0 1 0 |
```

**2.2**: Period = 4 (A⁴ = I)

**2.3**: (0 3 2 1) = (0 1)(0 2)(0 3) - three transpositions (odd permutation)
</details>

---

## CATEGORY H: Coding Implementations

### Variation H1: Build Permutation Matrix from Mapping
Write a NumPy function that takes a permutation mapping and returns the permutation matrix.

**Part 2.1**: Implement `build_permutation_matrix(mapping)` where `mapping[i]` is the new position of element `i`.
**Part 2.2**: Test with the cyclic shift [1, 2, 3, 0].
**Part 2.3**: Verify your matrix is orthogonal.

<details>
<summary>Solution H1</summary>

**2.1**:
```python
import numpy as np

def build_permutation_matrix(mapping):
    """
    Build permutation matrix from mapping.
    mapping[i] = j means element at position i goes to position j.
    So row j gets a 1 in column i.
    """
    n = len(mapping)
    P = np.zeros((n, n))
    for i, j in enumerate(mapping):
        P[j, i] = 1
    return P
```

**2.2**:
```python
mapping = [1, 2, 3, 0]  # x_0→pos 1, x_1→pos 2, x_2→pos 3, x_3→pos 0
P = build_permutation_matrix(mapping)
print(P)
# [[0. 0. 0. 1.]
#  [1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]]

# Verify: P @ [a,b,c,d] should give [d,a,b,c]
x = np.array([1, 2, 3, 4])
print(P @ x)  # [4. 1. 2. 3.]  ✓
```

**2.3**:
```python
print(np.allclose(P @ P.T, np.eye(4)))  # True
print(np.allclose(P.T @ P, np.eye(4)))  # True
```
</details>

### Variation H2: Apply Permutation Without Building Matrix
**Part 2.1**: Implement `apply_permutation(x, mapping)` that permutes vector `x` according to `mapping` without building the full matrix.
**Part 2.2**: What is the time complexity? Space complexity?
**Part 2.3**: When is this better than matrix multiplication?

<details>
<summary>Solution H2</summary>

**2.1**:
```python
def apply_permutation(x, mapping):
    """
    Apply permutation without building matrix.
    mapping[i] = j means x[i] goes to position j.
    """
    n = len(x)
    result = np.zeros_like(x)
    for i, j in enumerate(mapping):
        result[j] = x[i]
    return result

# Or more elegantly using fancy indexing:
def apply_permutation_fancy(x, mapping):
    result = np.zeros_like(x)
    result[mapping] = x
    return result
```

**2.2**:
- Time: O(n) - single pass through mapping
- Space: O(n) - for result array

**2.3**: Better when:
- Permuting once (matrix mult is O(n²))
- Memory constrained (matrix is O(n²) space)
- n is large and you only need the permuted vector, not the matrix
</details>

### Variation H3: Compute Period Numerically
**Part 2.1**: Implement `compute_period(P)` that finds the smallest k where P^k = I.
**Part 2.2**: Test on a 4-cycle and a transposition.
**Part 2.3**: What's the maximum possible period for an n×n permutation matrix?

<details>
<summary>Solution H3</summary>

**2.1**:
```python
def compute_period(P):
    """Find smallest k where P^k = I."""
    n = P.shape[0]
    I = np.eye(n)
    current = P.copy()
    for k in range(1, n + 1):  # Period can't exceed n! but n is a practical bound for cycles
        if np.allclose(current, I):
            return k
        current = current @ P
    # For very long periods (products of cycles), extend search
    for k in range(n + 1, 1000):
        if np.allclose(current, I):
            return k
        current = current @ P
    return None  # Period not found

# More efficient: use cycle decomposition
def compute_period_efficient(mapping):
    """Compute period from mapping using LCM of cycle lengths."""
    from math import gcd
    n = len(mapping)
    visited = [False] * n
    cycle_lengths = []

    for start in range(n):
        if visited[start]:
            continue
        length = 0
        current = start
        while not visited[current]:
            visited[current] = True
            current = mapping[current]
            length += 1
        cycle_lengths.append(length)

    # Period = LCM of all cycle lengths
    def lcm(a, b):
        return a * b // gcd(a, b)

    period = 1
    for length in cycle_lengths:
        period = lcm(period, length)
    return period
```

**2.2**:
```python
# 4-cycle: [1,2,3,0]
P_4cycle = build_permutation_matrix([1, 2, 3, 0])
print(compute_period(P_4cycle))  # 4

# Transposition: [1,0,2,3]
P_trans = build_permutation_matrix([1, 0, 2, 3])
print(compute_period(P_trans))  # 2
```

**2.3**: Maximum period for n×n is the Landau function g(n), the maximum LCM of cycle lengths summing to n.
- g(4) = 4 (from a single 4-cycle)
- g(5) = 6 (from 2-cycle + 3-cycle: lcm(2,3) = 6)
- g(6) = 6 (from 3-cycle + 3-cycle or 2-cycle + 4-cycle won't beat it)
- g(7) = 12 (from 3-cycle + 4-cycle)

</details>

### Variation H4: Eigenvalue Computation
**Part 2.1**: Use NumPy to compute eigenvalues of a 5-cycle permutation matrix.
**Part 2.2**: Verify they are 5th roots of unity.
**Part 2.3**: Why might numerical eigenvalues not be exactly on the unit circle?

<details>
<summary>Solution H4</summary>

**2.1**:
```python
# 5-cycle: [1,2,3,4,0]
mapping = [1, 2, 3, 4, 0]
P = build_permutation_matrix(mapping)
eigenvalues = np.linalg.eigvals(P)
print(eigenvalues)
```

**2.2**:
```python
# 5th roots of unity: e^(2πik/5) for k = 0,1,2,3,4
roots_of_unity = np.exp(2j * np.pi * np.arange(5) / 5)
print("Expected:", roots_of_unity)
print("Magnitudes:", np.abs(eigenvalues))  # Should all be 1.0
print("On unit circle:", np.allclose(np.abs(eigenvalues), 1))  # True
```

**2.3**:
- Floating point precision: eigenvalue algorithms iterate numerically
- Small imaginary parts might appear in "real" eigenvalues
- Magnitudes might be 0.9999999 or 1.0000001 instead of exactly 1
- Use `np.allclose` with tolerance, not exact equality
</details>

---

## CATEGORY I: Proof & Theory Questions

### Variation I1: Prove Permutation Matrices Form a Group
**Part 2.1**: What are the group axioms?
**Part 2.2**: Prove the set of n×n permutation matrices forms a group under matrix multiplication.
**Part 2.3**: What group is this isomorphic to?

<details>
<summary>Solution I1</summary>

**2.1**: Group axioms under operation ∘:
1. **Closure**: a ∘ b is in the set
2. **Associativity**: (a ∘ b) ∘ c = a ∘ (b ∘ c)
3. **Identity**: ∃e such that e ∘ a = a ∘ e = a
4. **Inverse**: ∀a, ∃a⁻¹ such that a ∘ a⁻¹ = e

**2.2**:
1. **Closure**: Product of two permutation matrices is a permutation matrix (composing two permutations gives a permutation)
2. **Associativity**: Matrix multiplication is associative
3. **Identity**: I is a permutation matrix (identity permutation)
4. **Inverse**: P⁻¹ = Pᵀ is also a permutation matrix (inverse permutation)

**2.3**: Isomorphic to Sₙ, the symmetric group on n elements. Each permutation matrix corresponds to exactly one permutation σ ∈ Sₙ.
</details>

### Variation I2: Prove Eigenvalue Property
**Part 2.1**: Prove that if P is a permutation matrix with period k, then all eigenvalues satisfy λᵏ = 1.
**Part 2.2**: Does the converse hold?
**Part 2.3**: When does a permutation matrix have real eigenvalues only?

<details>
<summary>Solution I2</summary>

**2.1**:
If Pᵏ = I and Pv = λv for eigenvector v ≠ 0, then:
Pᵏv = λᵏv
But Pᵏv = Iv = v
So λᵏv = v, meaning λᵏ = 1.

**2.2**: No. The converse doesn't hold.
Example: All eigenvalues of a 2-cycle satisfy λ² = 1, but this doesn't mean the period is 2.
Actually, for this example it does (the period IS 2).
Better example: If you know λ⁴ = 1 for all eigenvalues, the period could be 1, 2, or 4.

**2.3**: A permutation matrix has only real eigenvalues when:
- It's the identity (all eigenvalues = 1)
- It's a product of disjoint transpositions (eigenvalues ∈ {1, -1})
- More generally: when the permutation has only 1-cycles and 2-cycles (no cycles of length ≥ 3)
</details>

### Variation I3: Prove det = sign of Permutation
**Part 2.1**: What is the sign/parity of a permutation?
**Part 2.2**: Prove det(P) = sign(σ) for permutation matrix P corresponding to σ.
**Part 2.3**: Use this to prove det(AB) = det(A)det(B) for permutation matrices.

<details>
<summary>Solution I3</summary>

**2.1**: The sign of a permutation σ is:
- sign(σ) = (-1)^(number of transpositions in any decomposition)
- Equivalently: (-1)^(number of inversions)
- An inversion is a pair (i,j) where i < j but σ(i) > σ(j)

**2.2**:
The determinant can be computed using the Leibniz formula:
det(P) = Σ_{σ∈Sₙ} sign(σ) ∏ᵢ Pᵢ,σ(i)

For a permutation matrix P corresponding to permutation τ:
- Pᵢⱼ = 1 iff j = τ⁻¹(i), i.e., iff τ(j) = i
- So ∏ᵢ Pᵢ,σ(i) = 1 iff σ(i) = τ⁻¹(i) for all i, i.e., σ = τ⁻¹
- For all other σ, the product is 0

Therefore: det(P) = sign(τ⁻¹) = sign(τ) (since sign is a homomorphism)

**2.3**:
For permutation matrices P_σ and P_τ:
- P_σ P_τ = P_{σ∘τ} (composition of permutations)
- det(P_σ P_τ) = det(P_{σ∘τ}) = sign(σ∘τ) = sign(σ)·sign(τ) = det(P_σ)·det(P_τ)

This follows because sign: Sₙ → {±1} is a group homomorphism.
</details>

---

## CATEGORY J: Edge Cases & Special Permutations

### Variation J1: Identity Permutation
**Ax** = [x₀, x₁, x₂, x₃]^T (no change)

**Part 2.1**: Write A.
**Part 2.2**: What are the eigenvalues?
**Part 2.3**: What is the period?

<details>
<summary>Solution J1</summary>

**2.1**: A = I (identity matrix)
```
A = | 1 0 0 0 |
    | 0 1 0 0 |
    | 0 0 1 0 |
    | 0 0 0 1 |
```

**2.2**: All eigenvalues are 1 (with multiplicity 4).

**2.3**: Period = 1 (A¹ = I trivially)
</details>

### Variation J2: Derangement (No Fixed Points)
Find a 4×4 permutation matrix with trace = 0 (no element stays in place).

**Part 2.1**: Give an example.
**Part 2.2**: How many such permutations exist for n=4?
**Part 2.3**: What is the general formula for the number of derangements D(n)?

<details>
<summary>Solution J2</summary>

**2.1**: The cyclic permutation [x₁, x₂, x₃, x₀] works:
```
A = | 0 1 0 0 |
    | 0 0 1 0 |
    | 0 0 0 1 |
    | 1 0 0 0 |
```

Other examples: [x₃, x₂, x₁, x₀] (reversal), [x₁, x₀, x₃, x₂] (two disjoint transpositions)

**2.2**: D(4) = 9 derangements of 4 elements.
They are: all 4-cycles (6) + all products of two disjoint 2-cycles (3) = 9

**2.3**: D(n) = n! · Σₖ₌₀ⁿ (-1)ᵏ/k! ≈ n!/e

Recurrence: D(n) = (n-1)(D(n-1) + D(n-2))
Base: D(1) = 0, D(2) = 1
</details>

### Variation J3: Maximum Period
For n = 5, find a permutation with the maximum possible period.

**Part 2.1**: What is the maximum period for n=5?
**Part 2.2**: Give the permutation.
**Part 2.3**: Verify by computing the cycle structure.

<details>
<summary>Solution J3</summary>

**2.1**: Maximum period = lcm of cycle lengths summing to 5.
- 5-cycle: period 5
- 3-cycle + 2-cycle: period lcm(3,2) = 6 ✓ (maximum!)
- 4-cycle + 1-cycle: period 4
- 2-cycle + 2-cycle + 1-cycle: period 2

Maximum = 6

**2.2**: [x₁, x₂, x₀, x₄, x₃]
- Positions 0,1,2 form a 3-cycle: 0→1→2→0
- Positions 3,4 form a 2-cycle: 3↔4

**2.3**:
```
A = | 0 0 1 0 0 |
    | 1 0 0 0 0 |
    | 0 1 0 0 0 |
    | 0 0 0 0 1 |
    | 0 0 0 1 0 |
```

A³ fixes positions 0,1,2 but swaps 3,4.
A⁶ = I ✓
</details>

### Variation J4: Self-Inverse Permutations
Find all 3×3 permutation matrices A where A = A⁻¹.

**Part 2.1**: What condition must A satisfy?
**Part 2.2**: List all such 3×3 matrices.
**Part 2.3**: How many n×n self-inverse permutation matrices exist?

<details>
<summary>Solution J4</summary>

**2.1**: A = A⁻¹ means A² = I. These are called **involutions**.
A permutation is an involution iff it consists only of fixed points and 2-cycles (transpositions).

**2.2**: For n=3:
1. Identity: I (0 transpositions)
2. (0 1): swap first two, [x₁, x₀, x₂]
3. (0 2): swap first and last, [x₂, x₁, x₀]
4. (1 2): swap last two, [x₀, x₂, x₁]

Total: 4 involutions in S₃

**2.3**: The number of involutions in Sₙ is given by:
a(n) = a(n-1) + (n-1)·a(n-2)

Where a(0) = a(1) = 1.

- a(2) = 2 (identity, swap)
- a(3) = 4
- a(4) = 10
- a(5) = 26

This is OEIS sequence A000085.
</details>

---

## CATEGORY K: Advanced Composition

### Variation K1: Commuting Permutations
**Part 2.1**: When do two permutation matrices A and B commute (AB = BA)?
**Part 2.2**: Do all cyclic permutations commute with each other?
**Part 2.3**: Find two non-identity 3×3 permutation matrices that commute.

<details>
<summary>Solution K1</summary>

**2.1**: A and B commute when their corresponding permutations σ and τ commute in Sₙ.
This happens when:
- They have the same cycle structure AND
- They are powers of the same permutation, OR
- Their cycle supports are disjoint

**2.2**: Yes! All cyclic permutations of the same length commute.
If C is the basic cycle [x₁, x₂, ..., xₙ₋₁, x₀], then Cⁱ and Cʲ commute because
CⁱCʲ = Cⁱ⁺ʲ = Cʲ⁺ⁱ = CʲCⁱ

**2.3**: For 3×3:
A = 3-cycle [x₁, x₂, x₀], B = A² = [x₂, x₀, x₁]
AB = A³ = I = BA ✓

In fact, {I, A, A²} forms a commutative group (cyclic group Z₃).
</details>

### Variation K2: Conjugacy
**Part 2.1**: What does it mean for permutations to be conjugate?
**Part 2.2**: Prove conjugate permutations have the same cycle structure.
**Part 2.3**: Are [x₁, x₂, x₀, x₃] and [x₀, x₃, x₂, x₁] conjugate?

<details>
<summary>Solution K2</summary>

**2.1**: Permutations σ and τ are conjugate if ∃ρ such that τ = ρσρ⁻¹.
In matrix terms: B = PAP⁻¹ for some permutation matrix P.

**2.2**: If τ = ρσρ⁻¹ and σ has a k-cycle (a₁ a₂ ... aₖ), then τ has the k-cycle (ρ(a₁) ρ(a₂) ... ρ(aₖ)).
The conjugation just "relabels" the elements, preserving cycle lengths.

**2.3**:
- [x₁, x₂, x₀, x₃]: cycle structure is (0 1 2)(3) = one 3-cycle + one fixed point
- [x₀, x₃, x₂, x₁]: This is (1 3) = one 2-cycle + two fixed points

Different cycle structures ⟹ NOT conjugate!
</details>

### Variation K3: Order of Product
Given A is a 3-cycle and B is a disjoint 2-cycle:

**Part 2.1**: What is the order (period) of AB?
**Part 2.2**: What if A and B share one element?
**Part 2.3**: Give a specific example for each case.

<details>
<summary>Solution K3</summary>

**2.1**: If A and B are disjoint (act on different elements):
- Order(AB) = lcm(Order(A), Order(B)) = lcm(3, 2) = 6

**2.2**: If they share one element, the product structure depends on the specific overlap.
Example: If A = (0 1 2) and B = (0 3), then
AB = (0 1 2)(0 3) = (0 3 1 2) - a 4-cycle!
Order = 4

**2.3**:
Disjoint case (n=5):
A: [x₁, x₂, x₀, x₃, x₄] (3-cycle on 0,1,2)
B: [x₀, x₁, x₂, x₄, x₃] (2-cycle on 3,4)
AB: [x₁, x₂, x₀, x₄, x₃], Order = 6

Overlapping case (n=4):
A: [x₁, x₂, x₀, x₃] (3-cycle on 0,1,2)
B: [x₃, x₁, x₂, x₀] (2-cycle on 0,3)
AB: [x₃, x₂, x₁, x₀] = reversal, Order = 2 (not obvious without computation!)

Actually let me recalculate:
A maps: 0→1, 1→2, 2→0, 3→3
B maps: 0→3, 1→1, 2→2, 3→0
AB (apply A first): 0→1→1, 1→2→2, 2→0→3, 3→3→0
So AB: [x₁, x₂, x₃, x₀], a 4-cycle! Order = 4 ✓
</details>

---

## KEY FORMULAS SUMMARY

| Concept | Formula/Property |
|---------|------------------|
| Permutation matrix | Exactly one 1 per row and column |
| Outer product decomposition | A = Σᵢ ê^(f(i)) · ê^(i)ᵀ |
| Orthogonality | AAᵀ = AᵀA = I |
| Inverse | A⁻¹ = Aᵀ |
| Determinant | det(A) = ±1 (sign of permutation) |
| Trace | Number of fixed points |
| Eigenvalues of n-cycle | nth roots of unity |
| Period | Smallest n where Aⁿ = I |
| Involution | A² = I (transpositions, reversals) |

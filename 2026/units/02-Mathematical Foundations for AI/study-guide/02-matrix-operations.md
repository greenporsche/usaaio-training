# 02 — Matrix Operations

**Prerequisites**: `01-vector-spaces.md` (basis, linear independence, span)
**USAAIO Relevance**: Matrix operations are the computational engine of all ML algorithms. USAAIO tests matrix multiplication interpretations, inverse conditions, determinant properties, and rank.

---

## Discovery

It's 1858, and you're Arthur Cayley in Cambridge. You've been studying systems of linear equations and geometric transformations. You notice that composing two transformations — say, first rotating then scaling — can be represented by a single operation on arrays of numbers. You call these "matrices."

**Motivating challenge**: Consider the transformation that rotates a 2D point by 90 degrees counterclockwise:

$$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} -y \\ x \end{bmatrix}$$

Now apply this transformation twice. What do you get? Does the result make geometric sense?

**Socratic questions**:
1. When you multiply a matrix by a vector, what is the result? (A new vector — a *transformed* version of the original)
2. When you multiply two matrices $AB$, what does the product represent? (The composition: first apply $B$, then apply $A$)
3. If multiplying by $A$ represents a transformation, what does "undoing" that transformation look like? (That's the inverse $A^{-1}$)

**Misconception trap**: Matrix multiplication is NOT element-wise! Many students confuse $AB$ (matrix multiplication) with $A \odot B$ (Hadamard/element-wise product). They are fundamentally different operations. Also, $AB \neq BA$ in general.

---

## Intuition

What you just discovered is that matrices *are* linear transformations. Every matrix $A \in \mathbb{R}^{m \times n}$ defines a function $f(\mathbf{x}) = A\mathbf{x}$ that maps $\mathbb{R}^n \to \mathbb{R}^m$.

### Three Ways to Think About $A\mathbf{x}$

**View 1: Column combination** (most important for ML)

$$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

The result is a *linear combination of the columns of $A$*, weighted by entries of $\mathbf{x}$.

**View 2: Row dot products**

$$A\mathbf{x} = \begin{bmatrix} \mathbf{r}_1^\top \mathbf{x} \\ \mathbf{r}_2^\top \mathbf{x} \\ \vdots \\ \mathbf{r}_m^\top \mathbf{x} \end{bmatrix}$$

Each entry of the result is the dot product of a row of $A$ with $\mathbf{x}$.

**View 3: Geometric transformation**

```
Before A:           After A:
   |                   |  /
   | *              *  | /  *
   |   *        ──>  * |/
---+------         ----+------
   |                  /|
   |                 / |
```

$A$ stretches, rotates, shears, or projects space. The columns of $A$ tell you where the standard basis vectors land.

### Three Ways to Think About $AB$

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$:

**View 1: Column-wise** — Each column of $AB$ is $A$ times the corresponding column of $B$

**View 2: Row-column dot products** — $(AB)_{ij} = \mathbf{a}_i^\top \mathbf{b}_j$ (row $i$ of $A$ dotted with column $j$ of $B$)

**View 3: Sum of outer products** — $AB = \sum_{k=1}^{n} \mathbf{a}_{\cdot k} \mathbf{b}_{k \cdot}^\top$ (column $k$ of $A$ times row $k$ of $B$)

### What Goes Wrong Without Matrix Understanding?

If you don't understand matrix shapes and operations:
- Neural network layers won't make sense (they are just matrix multiplications + nonlinearities)
- You can't debug shape mismatch errors
- Attention mechanisms, convolutions, and embeddings all reduce to matrix operations

---

## Math

### Matrix Multiplication

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB \in \mathbb{R}^{m \times p}$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Shapes must be compatible**: inner dimensions must match. $(m \times \underline{n}) \cdot (\underline{n} \times p) = (m \times p)$.

**Properties**:
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$
- **NOT commutative**: $AB \neq BA$ in general
- $(AB)^\top = B^\top A^\top$ ("reverse the order and transpose each")

### Transpose

$$(A^\top)_{ij} = A_{ji}$$

- $(A + B)^\top = A^\top + B^\top$
- $(AB)^\top = B^\top A^\top$
- $(A^\top)^\top = A$
- A matrix is **symmetric** if $A = A^\top$

### Trace

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

**Properties** (all require square matrices or compatible products):
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(cA) = c \cdot \text{tr}(A)$
- $\text{tr}(A^\top) = \text{tr}(A)$
- **Cyclic property**: $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$
  - Caution: $\text{tr}(ABC) \neq \text{tr}(BAC)$ in general
- $\text{tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$ (Frobenius inner product)

*Reasoning required*: The cyclic property is frequently used in ML derivations (e.g., deriving PCA).

### Determinant

For $A \in \mathbb{R}^{n \times n}$:

**2x2**: $\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$

**nxn** (cofactor expansion along row 1):

$$\det(A) = \sum_{j=1}^{n} (-1)^{1+j} A_{1j} \det(M_{1j})$$

where $M_{1j}$ is the $(n-1) \times (n-1)$ matrix obtained by deleting row 1 and column $j$.

**Properties**:
- $\det(AB) = \det(A)\det(B)$
- $\det(A^{-1}) = 1/\det(A)$
- $\det(A^\top) = \det(A)$
- $\det(cA) = c^n \det(A)$ for $A \in \mathbb{R}^{n \times n}$
- $A$ is invertible $\iff$ $\det(A) \neq 0$
- **Geometric meaning**: $|\det(A)|$ = volume scaling factor of the transformation

### Matrix Inverse

$A^{-1}$ exists iff $\det(A) \neq 0$ (equivalently: $A$ has full rank, columns are linearly independent, all eigenvalues are nonzero).

**Properties**:
- $AA^{-1} = A^{-1}A = I$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^{-1})^\top = (A^\top)^{-1}$
- $(A^{-1})^{-1} = A$

**2x2 inverse**: $\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

*Reasoning not required*: Computing large inverses by hand; use numpy in practice.

### Special Matrices

| Type | Definition | Properties |
|------|-----------|------------|
| **Symmetric** | $A = A^\top$ | Real eigenvalues, orthogonal eigenvectors |
| **Orthogonal** | $Q^\top Q = QQ^\top = I$ | Preserves lengths and angles; $Q^{-1} = Q^\top$ |
| **Diagonal** | $A_{ij} = 0$ for $i \neq j$ | Easy powers: $(D^k)_{ii} = d_i^k$ |
| **Positive definite** | $\mathbf{x}^\top A \mathbf{x} > 0, \forall \mathbf{x} \neq \mathbf{0}$ | All eigenvalues $> 0$; unique Cholesky decomposition |
| **Positive semi-definite** | $\mathbf{x}^\top A \mathbf{x} \geq 0, \forall \mathbf{x}$ | All eigenvalues $\geq 0$; covariance matrices are PSD |

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

# --- Matrix Multiplication: Three Views ---

A = np.array([[1, 2], [3, 4]], dtype=float)  # (2, 2)
B = np.array([[5, 6], [7, 8]], dtype=float)  # (2, 2)

# View 1: Standard matmul
C1 = A @ B  # (2, 2)
print(f"A @ B = \n{C1}")

# View 2: Row-column dot products (from scratch)
m, n = A.shape  # 2, 2
_, p = B.shape  # 2
C2 = np.zeros((m, p))  # (2, 2)
for i in range(m):
    for j in range(p):
        C2[i, j] = np.dot(A[i, :], B[:, j])  # scalar = (n,) . (n,)
assert np.allclose(C1, C2)

# View 3: Sum of outer products (from scratch)
C3 = np.zeros((m, p))  # (2, 2)
for k in range(n):
    C3 += np.outer(A[:, k], B[k, :])  # (m, p) += (m,) outer (p,)
assert np.allclose(C1, C3)

# --- Trace ---
print(f"\ntr(A) = {np.trace(A)}")  # scalar: 1 + 4 = 5

# Verify cyclic property: tr(AB) = tr(BA)
print(f"tr(AB) = {np.trace(A @ B)}")  # scalar
print(f"tr(BA) = {np.trace(B @ A)}")  # scalar — same!

# --- Determinant ---
print(f"\ndet(A) = {np.linalg.det(A)}")  # scalar: 1*4 - 2*3 = -2

# 2x2 from scratch
def det_2x2(M: np.ndarray) -> float:
    """Compute determinant of 2x2 matrix from scratch."""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]  # scalar

print(f"det_2x2(A) = {det_2x2(A)}")  # scalar: -2.0

# --- Inverse ---
A_inv = np.linalg.inv(A)  # (2, 2)
print(f"\nA^-1 = \n{A_inv}")
print(f"A @ A^-1 = \n{np.round(A @ A_inv, 10)}")  # (2, 2) — should be I

# 2x2 inverse from scratch
def inv_2x2(M: np.ndarray) -> np.ndarray:
    """Compute inverse of 2x2 matrix from scratch."""
    det = det_2x2(M)  # scalar
    return np.array([[M[1, 1], -M[0, 1]],
                     [-M[1, 0], M[0, 0]]]) / det  # (2, 2)

assert np.allclose(A_inv, inv_2x2(A))

# --- Verify: det(AB) = det(A) * det(B) ---
print(f"\ndet(AB) = {np.linalg.det(A @ B):.4f}")  # scalar
print(f"det(A)*det(B) = {np.linalg.det(A) * np.linalg.det(B):.4f}")  # scalar
```

### PyTorch Equivalent

```python
import torch

A = torch.tensor([[1., 2.], [3., 4.]])  # (2, 2)
B = torch.tensor([[5., 6.], [7., 8.]])  # (2, 2)

# Matrix multiply
C = A @ B  # (2, 2) — same operator as NumPy

# Trace
tr = torch.trace(A)  # scalar

# Determinant
det = torch.linalg.det(A)  # scalar

# Inverse
A_inv = torch.linalg.inv(A)  # (2, 2)

# Batch operations — PyTorch excels here
batch_A = torch.randn(32, 4, 4)  # (B, n, n) — 32 matrices
batch_inv = torch.linalg.inv(batch_A)  # (B, n, n) — inverts all 32 at once
batch_det = torch.linalg.det(batch_A)  # (B,) — 32 determinants
```

---

## Resources

- [3Blue1Brown: Linear transformations and matrices](https://www.3blue1brown.com/lessons/linear-transformations) — transformations as matrix columns
- [3Blue1Brown: The determinant](https://www.3blue1brown.com/lessons/determinant) — geometric meaning
- MML Book, Chapter 2.2-2.6 — formal treatment of matrix properties

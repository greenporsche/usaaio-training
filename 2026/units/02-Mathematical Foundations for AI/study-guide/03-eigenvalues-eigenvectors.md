# 03 — Eigenvalues and Eigenvectors

**Prerequisites**: `01-vector-spaces.md` (basis, linear independence), `02-matrix-operations.md` (matrix multiply, inverse, determinant)
**USAAIO Relevance**: **CRITICAL topic.** Eigendecomposition is the backbone of PCA, spectral methods, Markov chains, and stability analysis. USAAIO Round 1 frequently tests eigenvalue computation, proofs about eigendecomposition, and the power method. Round 2 may require from-scratch implementation.

---

## Discovery

It's 1904, and you're David Hilbert in Gottingen, extending the spectral theory of integral operators. But the core idea goes back further — to Euler studying rigid body rotation in 1751. Imagine you're spinning a top. As it rotates, most points move. But the axis of rotation *stays fixed in direction*. That axis is special — the rotation doesn't change its direction, only possibly its length.

**Motivating challenge**: Consider the matrix

$$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$

Apply $A$ to the vector $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$: you get $\begin{bmatrix} 3 \\ 0 \end{bmatrix} = 3\mathbf{v}_1$. The direction didn't change — it just scaled by 3!

Now try $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$: you get $\begin{bmatrix} 4 \\ 2 \end{bmatrix}$. Is this a scalar multiple of $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$? No! This direction *does* change.

Can you find another vector whose direction is preserved by $A$?

**Socratic questions**:
1. If $A\mathbf{x} = \lambda\mathbf{x}$, what does $A^2\mathbf{x}$ equal? ($\lambda^2 \mathbf{x}$ — eigenvalues compose under matrix powers!)
2. If $A$ has $n$ linearly independent eigenvectors, can you express *any* vector as a combination of eigenvectors? (Yes — they form a basis)
3. Why would decomposing $A$ in terms of its eigenvectors be useful? (It turns matrix operations into scalar operations on eigenvalues)

**Misconception trap**: Not every matrix is diagonalizable. A matrix might not have $n$ linearly independent eigenvectors (defective matrices). However, for USAAIO purposes, you'll mostly work with symmetric matrices, which are *always* diagonalizable with real eigenvalues.

---

## Intuition

What you discovered — directions that are merely scaled, not rotated — is the essence of **eigenvectors** (German: "eigen" = "own/self"). The scaling factor is the **eigenvalue**.

### Geometric Picture

```
  Before A:              After A:
                          e1 stretched by λ1
  e2 ↑                       ↑
     |  *              e2 ↑  |  *
     | *                  |  | *
     |*                   | *|*
  ---+---> e1          ---+-----> e1
     |                    |
```

$A$ acts on eigenvectors by pure scaling. On other vectors, it both stretches AND rotates. In the eigenbasis, $A$ becomes a diagonal matrix — the simplest possible form.

### The Power of Eigendecomposition

If $A = Q\Lambda Q^{-1}$:
- $A^k = Q\Lambda^k Q^{-1}$ — matrix powers become trivial
- $e^{At} = Q e^{\Lambda t} Q^{-1}$ — matrix exponentials for differential equations
- Eigenvalues reveal stability: $|\lambda| < 1$ means that component decays

### What Goes Wrong Without Eigendecomposition?

- **PCA** is literally eigendecomposition of the covariance matrix
- **Google's PageRank** is the dominant eigenvector of the web graph
- **Markov chain convergence** depends on the spectral gap
- **Neural network training** stability depends on eigenvalues of the Hessian

---

## Math

### Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ and nonzero vector $\mathbf{x}$ satisfying

$$A\mathbf{x} = \lambda\mathbf{x}$$

are called an **eigenvalue** and **eigenvector** of $A$, respectively.

### Finding Eigenvalues: The Characteristic Equation

$$A\mathbf{x} = \lambda\mathbf{x} \iff (A - \lambda I)\mathbf{x} = \mathbf{0}$$

For a nonzero solution $\mathbf{x}$ to exist, $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0 \quad \text{(characteristic equation)}$$

This yields a degree-$n$ polynomial in $\lambda$ (the **characteristic polynomial**), which has exactly $n$ roots (counting multiplicity, possibly complex).

*Reasoning required*: USAAIO expects you to compute eigenvalues by solving the characteristic equation for small matrices ($2 \times 2$, $3 \times 3$, upper triangular).

**Example**: For $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$:

$$\det\begin{bmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{bmatrix} = (3-\lambda)(2-\lambda) = 0$$

So $\lambda_1 = 3$, $\lambda_2 = 2$.

**Special case — triangular matrices**: Eigenvalues are the diagonal entries (since $\det(A - \lambda I)$ = product of diagonal terms).

### Finding Eigenvectors

For each eigenvalue $\lambda_i$, solve $(A - \lambda_i I)\mathbf{x} = \mathbf{0}$:

For $\lambda_1 = 3$: $(A - 3I)\mathbf{x} = \begin{bmatrix} 0 & 1 \\ 0 & -1 \end{bmatrix}\mathbf{x} = \mathbf{0} \implies \mathbf{x}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

For $\lambda_2 = 2$: $(A - 2I)\mathbf{x} = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}\mathbf{x} = \mathbf{0} \implies \mathbf{x}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$

### Eigendecomposition

**Theorem**: If $A \in \mathbb{R}^{n \times n}$ has $n$ linearly independent eigenvectors $\mathbf{q}_0, \ldots, \mathbf{q}_{n-1}$ with eigenvalues $\lambda_0, \ldots, \lambda_{n-1}$, then:

$$A = Q\Lambda Q^{-1}$$

where $Q = [\mathbf{q}_0 \cdots \mathbf{q}_{n-1}]$ and $\Lambda = \text{diag}(\lambda_0, \ldots, \lambda_{n-1})$.

**Proof**:

$$AQ = A[\mathbf{q}_0 \cdots \mathbf{q}_{n-1}] = [\lambda_0\mathbf{q}_0 \cdots \lambda_{n-1}\mathbf{q}_{n-1}] = Q\Lambda$$

Since the eigenvectors are linearly independent, $Q$ is invertible, so:

$$A = Q\Lambda Q^{-1}$$

$\blacksquare$

### Matrix Powers via Eigendecomposition

$$A^k = (Q\Lambda Q^{-1})^k = Q\Lambda Q^{-1} Q\Lambda Q^{-1} \cdots = Q\Lambda^k Q^{-1}$$

Since $\Lambda^k = \text{diag}(\lambda_0^k, \ldots, \lambda_{n-1}^k)$, this is extremely efficient.

### Outer Product Form

Define the left eigenvectors $P = Q^{-\top}$, i.e., $P = [\mathbf{p}_0 \cdots \mathbf{p}_{n-1}]$ where $\mathbf{p}_i$ are columns of $Q^{-\top}$.

Then: $A = \sum_{i=0}^{n-1} \lambda_i \mathbf{q}_i \mathbf{p}_i^\top$

**Proof**: Since $Q^{-1} = P^\top$:

$$A = Q\Lambda Q^{-1} = Q\Lambda P^\top = \sum_{i=0}^{n-1} \lambda_i \mathbf{q}_i \mathbf{p}_i^\top$$

$\blacksquare$

*Reasoning required*: This outer product form is crucial for understanding rank-$k$ approximations.

### Spectral Theorem (Symmetric Matrices)

**Theorem**: If $A = A^\top$ (real symmetric), then:
1. All eigenvalues are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. $A = Q\Lambda Q^\top$ where $Q$ is orthogonal ($Q^{-1} = Q^\top$)

**Proof sketch** (eigenvalues are real):
Let $A\mathbf{x} = \lambda\mathbf{x}$ where $\mathbf{x} \neq \mathbf{0}$. Take the conjugate transpose:
$\bar{\mathbf{x}}^\top A^\top = \bar{\lambda} \bar{\mathbf{x}}^\top$

Since $A = A^\top$: $\bar{\mathbf{x}}^\top A = \bar{\lambda} \bar{\mathbf{x}}^\top$

Multiply on the right by $\mathbf{x}$: $\bar{\mathbf{x}}^\top A \mathbf{x} = \bar{\lambda} \bar{\mathbf{x}}^\top \mathbf{x}$

But also $\bar{\mathbf{x}}^\top A \mathbf{x} = \bar{\mathbf{x}}^\top (\lambda \mathbf{x}) = \lambda \bar{\mathbf{x}}^\top \mathbf{x}$

Since $\bar{\mathbf{x}}^\top \mathbf{x} = \|\mathbf{x}\|^2 > 0$: $\lambda = \bar{\lambda}$, so $\lambda$ is real. $\blacksquare$

*Reasoning required*: This proof appears in USAAIO-style problem sets.

### Key Properties of Eigenvalues

| Property | Formula |
|----------|---------|
| Sum of eigenvalues | $\sum \lambda_i = \text{tr}(A)$ |
| Product of eigenvalues | $\prod \lambda_i = \det(A)$ |
| Eigenvalues of $A^k$ | $\lambda_i^k$ |
| Eigenvalues of $A^{-1}$ | $1/\lambda_i$ |
| Eigenvalues of $A + cI$ | $\lambda_i + c$ |
| Eigenvalues of $A^\top$ | same as $A$ |

### The Power Method

An iterative algorithm to find the **dominant eigenvalue** (largest in absolute value):

1. Start with random $\mathbf{b}_0$
2. Repeat: $\mathbf{b}_{k+1} = \frac{A\mathbf{b}_k}{\|A\mathbf{b}_k\|}$
3. Converges to the eigenvector for the dominant eigenvalue
4. Eigenvalue: $\lambda = \mathbf{b}_k^\top A \mathbf{b}_k$ (Rayleigh quotient)

**Why it works**: Express $\mathbf{b}_0 = \sum c_i \mathbf{q}_i$. After $k$ iterations:

$$A^k \mathbf{b}_0 = \sum c_i \lambda_i^k \mathbf{q}_i = \lambda_1^k \left( c_1 \mathbf{q}_1 + \sum_{i>1} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^k \mathbf{q}_i \right)$$

Since $|\lambda_i/\lambda_1| < 1$, the non-dominant terms vanish, leaving only $\mathbf{q}_1$.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def eigendecompose(A: np.ndarray) -> tuple:
    """Eigendecomposition using np.linalg.eig.

    Args:
        A: shape (n, n)
    Returns:
        eigenvalues: shape (n,)
        Q: shape (n, n) — right eigenvectors as columns
        Q_inv: shape (n, n)
    """
    eigenvalues, Q = np.linalg.eig(A)  # (n,), (n, n)
    Q_inv = np.linalg.inv(Q)  # (n, n)
    return eigenvalues, Q, Q_inv

def power_method(A: np.ndarray, num_iter: int = 100, tol: float = 1e-10) -> tuple:
    """Power method to find the dominant eigenvalue/eigenvector.

    Args:
        A: shape (n, n) — must have a unique dominant eigenvalue
        num_iter: maximum iterations
        tol: convergence tolerance
    Returns:
        eigenvalue: scalar
        eigenvector: shape (n,)
    """
    n = A.shape[0]
    b = np.random.randn(n)  # (n,) — random starting vector
    b = b / np.linalg.norm(b)  # (n,) — normalize

    for _ in range(num_iter):
        b_new = A @ b  # (n,) = (n, n) @ (n,)
        b_new = b_new / np.linalg.norm(b_new)  # (n,) — normalize

        # Check convergence
        if np.abs(np.abs(np.dot(b_new, b)) - 1.0) < tol:
            b = b_new
            break
        b = b_new  # (n,)

    eigenvalue = b @ A @ b  # scalar — Rayleigh quotient
    return eigenvalue, b

def deflate_and_find_all(A: np.ndarray, k: int) -> tuple:
    """Find top-k eigenvalues/vectors using power method + deflation.

    Args:
        A: shape (n, n)
        k: number of top eigenvalues to find
    Returns:
        eigenvalues: shape (k,)
        eigenvectors: shape (k, n)
    """
    n = A.shape[0]
    eigenvalues = np.zeros(k)  # (k,)
    eigenvectors = np.zeros((k, n))  # (k, n)
    A_deflated = A.copy()  # (n, n)

    for i in range(k):
        lam, v = power_method(A_deflated)  # scalar, (n,)
        eigenvalues[i] = lam  # store
        eigenvectors[i] = v  # store

        # Deflate: remove this eigenvalue's contribution
        A_deflated = A_deflated - lam * np.outer(v, v)  # (n, n)

    return eigenvalues, eigenvectors

# --- Demo ---
A = np.array([[3, 1],
              [0, 2]], dtype=float)  # (2, 2)

vals, Q, Q_inv = eigendecompose(A)
print(f"Eigenvalues: {vals}")  # [3., 2.]
print(f"Eigenvectors:\n{Q}")  # columns are eigenvectors

# Verify: A = Q Lambda Q^-1
Lambda = np.diag(vals)  # (2, 2)
A_reconstructed = Q @ Lambda @ Q_inv  # (2, 2)
print(f"\nReconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Power method
lam_dom, v_dom = power_method(A)
print(f"\nDominant eigenvalue (power method): {lam_dom:.4f}")
print(f"Dominant eigenvector: {v_dom}")

# Matrix power via eigendecomposition
k = 10
Ak_direct = np.linalg.matrix_power(A, k)  # (2, 2)
Ak_eigen = Q @ np.diag(vals**k) @ Q_inv  # (2, 2)
print(f"\nA^{k} error: {np.linalg.norm(Ak_direct - Ak_eigen):.2e}")

# Symmetric matrix — spectral theorem
S = np.array([[4, 2], [2, 3]], dtype=float)  # (2, 2) symmetric
vals_s, Q_s = np.linalg.eigh(S)  # (2,), (2, 2) — use eigh for symmetric!
print(f"\nSymmetric eigenvalues: {vals_s}")
print(f"Q^T Q (should be I):\n{np.round(Q_s.T @ Q_s, 10)}")
```

### PyTorch Equivalent

```python
import torch

A = torch.tensor([[4., 2.], [2., 3.]])  # (2, 2) symmetric

# For symmetric matrices — preferred for ML (covariance matrices, etc.)
eigenvalues, eigenvectors = torch.linalg.eigh(A)  # (n,), (n, n)

# For general matrices
eigenvalues_gen, eigenvectors_gen = torch.linalg.eig(A)  # complex tensors

# Batch eigendecomposition — common in ML
batch_A = torch.randn(32, 5, 5)  # (B, n, n)
batch_A = batch_A + batch_A.transpose(-2, -1)  # (B, n, n) — make symmetric
batch_vals, batch_vecs = torch.linalg.eigh(batch_A)  # (B, n), (B, n, n)
```

---

## Resources

- [3Blue1Brown: Eigenvectors and eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues) — essential visual explanation
- MML Book, Chapter 4: Matrix Decompositions
- Strang, *Linear Algebra and Its Applications*, Chapter 6
- BeaverEdge Assignment: `../assignments/AI 200, linalg, eigendecomposition, assignment.md`

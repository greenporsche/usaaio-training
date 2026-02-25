# 01 — Vector Spaces

**Prerequisites**: High school algebra (Algebra 2 level)
**USAAIO Relevance**: Foundation for every other topic. Vector spaces underpin eigendecomposition, SVD, PCA. Round 1 may test linear independence, basis, and rank directly.

---

## Discovery

It's 1844, and you're Hermann Grassmann, a German schoolteacher who has grown dissatisfied with the rigid coordinate-based geometry of your time. You want a way to talk about directions, magnitudes, and linear combinations without tying everything to specific coordinate axes. You're looking for the *structure* behind algebra.

Consider this problem: You have three sensors measuring temperature at three locations. Each sensor reading is a number. A collection of three readings is a "vector" in some sense — but what makes it a *vector*?

**Motivating challenge**: Suppose you have three measurement vectors:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 3 \\ 0 \end{bmatrix}$$

Can every possible 3D measurement be expressed as a combination of these three vectors? Why or why not?

**Socratic questions**:
1. What operation do you need to "combine" vectors? (Addition and scaling — that's it!)
2. If $\mathbf{v}_3 = 2\mathbf{v}_1 + 3\mathbf{v}_2$, does $\mathbf{v}_3$ give you any "new direction" that $\mathbf{v}_1$ and $\mathbf{v}_2$ don't already cover?
3. How many truly independent directions do you need to reach any point in 3D space?

**Misconception trap**: Students often think "3 vectors in 3D = we can reach everything." But $\mathbf{v}_3$ lies in the plane spanned by $\mathbf{v}_1$ and $\mathbf{v}_2$ — you can never reach $\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$. The *number* of vectors doesn't matter — what matters is whether they point in genuinely different directions.

---

## Intuition

What you just discovered is the core idea behind **linear independence** and **span** — concepts Grassmann formalized in his 1844 *Ausdehnungslehre* (Theory of Extension), which was so ahead of its time that it was largely ignored for decades.

### Geometric Picture

Think of vectors as arrows from the origin:

```
        z
        |    * can't reach here!
        |   /
        |  /
        | /
  ------+-------> y
       /|
      / |
     /  |
    x   |

  v1, v2, v3 all live in the x-y plane
```

- **Span** of $\{\mathbf{v}_1, \mathbf{v}_2\}$ = the entire $x$-$y$ plane
- Adding $\mathbf{v}_3 = 2\mathbf{v}_1 + 3\mathbf{v}_2$ doesn't expand the span — it's *redundant*
- To span all of $\mathbb{R}^3$, you need a vector with a nonzero $z$-component

### What Goes Wrong Without This Concept?

In machine learning, your data lives in high-dimensional space. If your features are linearly dependent (e.g., "temperature in Celsius" and "temperature in Fahrenheit"), you have redundancy that:
- Wastes computation
- Makes matrix inversions fail (singular matrices!)
- Causes numerical instability in optimization

Understanding vector spaces tells you the *true dimensionality* of your data.

---

## Math

### Definition: Vector Space

A **vector space** $V$ over a field $\mathbb{F}$ (typically $\mathbb{R}$) is a set equipped with two operations:
- **Vector addition**: $+: V \times V \to V$
- **Scalar multiplication**: $\cdot: \mathbb{F} \times V \to V$

satisfying the following **8 axioms** for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $a, b \in \mathbb{F}$:

| # | Axiom | Expression |
|---|-------|-----------|
| 1 | Closure (addition) | $\mathbf{u} + \mathbf{v} \in V$ |
| 2 | Closure (scalar mult.) | $a\mathbf{v} \in V$ |
| 3 | Commutativity | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 4 | Associativity (addition) | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| 5 | Additive identity | $\exists \mathbf{0}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 6 | Additive inverse | $\exists (-\mathbf{v}): \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| 7 | Compatibility | $a(b\mathbf{v}) = (ab)\mathbf{v}$ |
| 8 | Scalar identity | $1\mathbf{v} = \mathbf{v}$ |

Plus distributive laws: $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ and $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.

*Reasoning required*: You should be able to verify whether a given set with given operations forms a vector space.

### Definition: Subspace

A **subspace** $W \subseteq V$ is a subset that is itself a vector space under the same operations. Shortcut test:
1. $\mathbf{0} \in W$
2. $\mathbf{u}, \mathbf{v} \in W \implies \mathbf{u} + \mathbf{v} \in W$
3. $a \in \mathbb{F}, \mathbf{v} \in W \implies a\mathbf{v} \in W$

### Definition: Linear Independence

Vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are **linearly independent** if:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

Otherwise they are **linearly dependent** — at least one vector can be written as a combination of the others.

### Definition: Span

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \left\{ \sum_{i=1}^{k} c_i \mathbf{v}_i \mid c_i \in \mathbb{F} \right\}$$

The span is always a subspace of $V$.

### Definition: Basis and Dimension

A **basis** of $V$ is a linearly independent set that spans $V$. The **dimension** $\dim(V)$ is the number of vectors in any basis (this is well-defined — all bases have the same size).

**Theorem**: If $\dim(V) = n$, then:
- Any $n$ linearly independent vectors in $V$ form a basis
- Any $n$ vectors that span $V$ form a basis
- Any set of more than $n$ vectors in $V$ must be linearly dependent

*Reasoning required* for USAAIO.

### Rank

For matrix $A \in \mathbb{R}^{m \times n}$:

$$\text{rank}(A) = \dim(\text{column space of } A) = \dim(\text{row space of } A)$$

**Key fact**: $\text{rank}(A) \leq \min(m, n)$. If $\text{rank}(A) = \min(m, n)$, $A$ has **full rank**.

### Orthogonality

Vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** if $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = 0$.

A set $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ is **orthonormal** if:

$$\mathbf{e}_i^\top \mathbf{e}_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases} = \delta_{ij}$$

**Why orthonormal bases are powerful**: Projection onto basis vector $\mathbf{e}_i$ is simply $\mathbf{e}_i^\top \mathbf{x}$ — no matrix inversion needed.

### Gram-Schmidt Process

Given linearly independent $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$, produce orthonormal $\{\mathbf{e}_1, \ldots, \mathbf{e}_k\}$:

**Step 1**: $\mathbf{u}_1 = \mathbf{v}_1$, $\mathbf{e}_1 = \mathbf{u}_1 / \|\mathbf{u}_1\|$

**Step $j$** (for $j = 2, \ldots, k$):

$$\mathbf{u}_j = \mathbf{v}_j - \sum_{i=1}^{j-1} \langle \mathbf{v}_j, \mathbf{e}_i \rangle \mathbf{e}_i$$

$$\mathbf{e}_j = \frac{\mathbf{u}_j}{\|\mathbf{u}_j\|}$$

**Derivation**: Each step subtracts the components of $\mathbf{v}_j$ along the already-computed orthonormal directions, leaving only the component orthogonal to the existing set. This is exactly a projection-and-subtract operation.

*Reasoning required*: USAAIO may ask you to apply Gram-Schmidt by hand or prove properties of the resulting basis.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def is_linearly_independent(vectors: np.ndarray) -> bool:
    """Check if column vectors are linearly independent.

    Args:
        vectors: shape (n, k) — k column vectors in R^n
    Returns:
        True if the vectors are linearly independent
    """
    n, k = vectors.shape  # (n, k)
    if k > n:
        return False  # More vectors than dimensions → must be dependent
    rank = np.linalg.matrix_rank(vectors)  # scalar
    return rank == k

def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization from scratch.

    Args:
        V: shape (n, k) — k linearly independent column vectors in R^n
    Returns:
        Q: shape (n, k) — k orthonormal column vectors
    """
    n, k = V.shape  # (n, k)
    Q = np.zeros((n, k))  # (n, k)

    for j in range(k):
        u = V[:, j].copy()  # (n,)

        # Subtract projections onto all previous orthonormal vectors
        for i in range(j):
            # Project v_j onto e_i: <v_j, e_i> * e_i
            proj = np.dot(V[:, j], Q[:, i])  # scalar
            u = u - proj * Q[:, i]  # (n,)

        # Normalize
        norm = np.linalg.norm(u)  # scalar
        Q[:, j] = u / norm  # (n,)

    return Q  # (n, k)

# --- Verify ---
V = np.array([[1, 0, 2],
              [0, 1, 3],
              [0, 0, 1]], dtype=float)  # (3, 3)

Q = gram_schmidt(V)  # (3, 3)
print("Orthonormal basis Q:")
print(Q)

# Verify orthonormality: Q^T Q should be identity
print("\nQ^T Q (should be I):")
print(np.round(Q.T @ Q, 10))  # (3, 3) — should be I_3

# Check linear independence of original vectors
print(f"\nLinearly independent: {is_linearly_independent(V)}")
```

### PyTorch Equivalent

```python
import torch

def gram_schmidt_torch(V: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt using PyTorch (for GPU acceleration).

    Args:
        V: shape (n, k)
    Returns:
        Q: shape (n, k)
    """
    n, k = V.shape  # (n, k)
    Q = torch.zeros(n, k, dtype=V.dtype, device=V.device)  # (n, k)

    for j in range(k):
        u = V[:, j].clone()  # (n,)
        if j > 0:
            # Vectorized projection subtraction
            projections = Q[:, :j].T @ V[:, j]  # (j,) = (j, n) @ (n,)
            u = u - Q[:, :j] @ projections  # (n,) = (n, j) @ (j,)
        Q[:, j] = u / torch.norm(u)  # (n,)

    return Q  # (n, k)

# Note: PyTorch provides torch.linalg.qr() which does this internally
# Q, R = torch.linalg.qr(V)  # Q: (n, k), R: (k, k)
```

---

## Resources

- [3Blue1Brown: Linear combinations, span, and basis vectors](https://www.3blue1brown.com/lessons/span) — best visual introduction
- MML Book, Chapter 2: Linear Algebra — rigorous treatment
- [MIT 18.06 Lecture 1-4](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Gilbert Strang's legendary lectures

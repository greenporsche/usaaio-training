# NumPy Vectorization & Broadcasting

**Prerequisites**: NumPy basics (array creation, indexing, shapes)
**USAAIO Relevance**: **This is the most critical topic.** USAAIO problems frequently include "no loops allowed" constraints. You MUST solve array problems using broadcasting and vectorized operations. This is not optional -- it is the core skill being tested.

---

## Discovery

### "No Loops Allowed" -- The USAAIO Mindset

In a typical USAAIO coding problem, you might see:

> *Given a matrix of N points in D dimensions, compute the pairwise Euclidean distance matrix. **Constraint: No explicit loops.***

A beginner writes nested loops. An AI programmer writes one line of vectorized NumPy. The difference is not just speed -- it is a fundamentally different way of thinking about computation.

**Array thinking** means: instead of processing one element at a time, you describe the operation on the entire array at once. The computer figures out how to parallelize it.

This mirrors how AI algorithms actually work:
- A neural network forward pass is matrix multiplication, not a loop over neurons
- Computing loss is a vectorized operation over all samples, not a loop over each one
- Gradient descent updates all weights simultaneously

> **Socratic question**: Consider computing the mean of a dataset. The loop version processes one number at a time. The vectorized version (`np.mean(data)`) processes all numbers at once. But the CPU still has to touch every number -- so why is the vectorized version faster?

### Broadcasting: NumPy's Secret Weapon

Broadcasting is the mechanism that allows NumPy to perform operations on arrays of **different shapes**. It is the key to eliminating loops.

Without broadcasting, to add a bias vector to every row of a matrix, you would need a loop. With broadcasting, you just write `matrix + bias` and NumPy figures it out.

---

## Intuition

### Broadcasting Rules -- Visual Guide

When NumPy operates on two arrays, it compares their shapes **element-wise from the trailing (rightmost) dimensions**:

**Rule 1**: If the arrays have different numbers of dimensions, the shape of the smaller array is **padded with 1s on the left**.

**Rule 2**: Arrays with size 1 along a dimension are **stretched** to match the other array.

**Rule 3**: If sizes disagree and neither is 1, raise an error.

```
Example 1: Matrix + Vector (most common pattern)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A shape:  (3, 4)
  B shape:      (4,)

  Step 1: Pad B with 1 on left    -> B becomes (1, 4)
  Step 2: Stretch B along axis 0  -> B becomes (3, 4)
  Result:  (3, 4)

  Visually:
  A = [[1, 2, 3, 4],      B = [10, 20, 30, 40]
       [5, 6, 7, 8],
       [9, 0, 1, 2]]
                           B "stretched" to:
                           [[10, 20, 30, 40],
                            [10, 20, 30, 40],
                            [10, 20, 30, 40]]

  A + B = [[11, 22, 33, 44],
           [15, 26, 37, 48],
           [19, 20, 31, 42]]


Example 2: Column Vector + Row Vector (outer operation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A shape: (3, 1)
  B shape: (1, 4)

  Stretch A along axis 1 -> (3, 4)
  Stretch B along axis 0 -> (3, 4)
  Result:  (3, 4)

  Visually:
  A = [[1],    B = [[10, 20, 30, 40]]
       [2],
       [3]]

  A stretched:          B stretched:
  [[1, 1, 1, 1],       [[10, 20, 30, 40],
   [2, 2, 2, 2],        [10, 20, 30, 40],
   [3, 3, 3, 3]]        [10, 20, 30, 40]]

  A + B = [[11, 21, 31, 41],
           [12, 22, 32, 42],
           [13, 23, 33, 43]]


Example 3: Incompatible shapes (ERROR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A shape: (3, 4)
  B shape: (3,)

  Trailing dimensions: 4 vs 3  -> neither is 1 -> ERROR!

  Fix: reshape B to (3, 1) so it broadcasts along columns
  A shape: (3, 4)
  B shape: (3, 1)  -> stretches to (3, 4)  ✓


Example 4: 3D Broadcasting
━━━━━━━━━━━━━━━━━━━━━━━━━━

  A shape: (2, 3, 4)
  B shape:    (3, 1)

  Step 1: Pad B -> (1, 3, 1)
  Step 2: Stretch -> (2, 3, 4)
  Result: (2, 3, 4)
```

### The Reshape Trick for Broadcasting

The most powerful pattern: **reshape to create compatible dimensions for broadcasting**.

```
Goal: compute outer product of vectors a (N,) and b (M,)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  a.shape = (N,)  ->  a.reshape(N, 1)  =  (N, 1)
  b.shape = (M,)  ->  b.reshape(1, M)  =  (1, M)

  a.reshape(N, 1) * b.reshape(1, M)  ->  (N, M)

  Each element (i, j) = a[i] * b[j]

Goal: pairwise differences between vectors a (N,) and b (M,)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  a.reshape(N, 1) - b.reshape(1, M)  ->  (N, M)

  Each element (i, j) = a[i] - b[j]
```

### Common Failure Modes

```
MISTAKE 1: Forgetting to reshape for broadcasting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  a = np.array([1, 2, 3])        # (3,)
  b = np.array([10, 20, 30, 40]) # (4,)
  a + b  # ERROR: shapes (3,) and (4,) not compatible

  Fix: a.reshape(3, 1) + b.reshape(1, 4)  -> (3, 4)

MISTAKE 2: Confusing (N,) with (N, 1) or (1, N)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v = np.array([1, 2, 3])  # shape (3,) -- neither row nor column!
  v.T  # still (3,)! Transpose has no effect on 1D arrays!

  To get column vector: v.reshape(-1, 1)  or  v[:, np.newaxis]  -> (3, 1)
  To get row vector:    v.reshape(1, -1)  or  v[np.newaxis, :]  -> (1, 3)

MISTAKE 3: Accumulating in a loop when you could broadcast
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # BAD: loop to normalize each row
  for i in range(len(data)):
      data[i] = data[i] / data[i].sum()

  # GOOD: vectorized with keepdims
  data = data / data.sum(axis=1, keepdims=True)
```

---

## Math

### Broadcasting Formalization

Given arrays A with shape $(d_1^A, d_2^A, \ldots, d_n^A)$ and B with shape $(d_1^B, d_2^B, \ldots, d_m^B)$:

1. **Align right**: pad the shorter shape with 1s on the left until both have the same number of dimensions.

2. **Check compatibility**: for each dimension $k$, the result dimension is:

$$d_k^R = \begin{cases} d_k^A & \text{if } d_k^A = d_k^B \\ \max(d_k^A, d_k^B) & \text{if } d_k^A = 1 \text{ or } d_k^B = 1 \\ \text{error} & \text{otherwise} \end{cases}$$

### Pairwise Distance Matrix

Given $N$ points in $D$ dimensions, stored as matrix $X$ with shape $(N, D)$:

$$\text{dist}(i, j) = \sqrt{\sum_{k=1}^{D} (X_{i,k} - X_{j,k})^2}$$

Vectorized using broadcasting:

$$\text{diff} = X_{(N,1,D)} - X_{(1,N,D)} \quad \rightarrow \quad \text{shape: } (N, N, D)$$
$$\text{dist} = \sqrt{\sum_{\text{axis}=2} \text{diff}^2} \quad \rightarrow \quad \text{shape: } (N, N)$$

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Vectorized for a batch of vectors, shape $(B, C)$:

```
exp_x = np.exp(x - x.max(axis=1, keepdims=True))  # (B, C) -- numerical stability
softmax = exp_x / exp_x.sum(axis=1, keepdims=True)  # (B, C)
```

---

## Code

### Element-wise Operations (Replacing Simple Loops)

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# Arithmetic (all element-wise)
a + 10        # [11, 12, 13, 14, 15]
a * 2         # [2, 4, 6, 8, 10]
a ** 2        # [1, 4, 9, 16, 25]
1 / a         # [1.0, 0.5, 0.333, 0.25, 0.2]

# Math functions
np.sqrt(a)    # [1.0, 1.414, 1.732, 2.0, 2.236]
np.exp(a)     # [2.718, 7.389, 20.086, 54.598, 148.413]
np.log(a)     # [0.0, 0.693, 1.099, 1.386, 1.609]
np.abs(a - 3) # [2, 1, 0, 1, 2]

# Comparison (returns boolean array)
a > 3         # [False, False, False, True, True]
a == 3        # [False, False, True, False, False]
```

### Broadcasting in Practice

```python
# Pattern 1: Matrix + Vector (add bias to each row)
X = np.random.randn(100, 5)   # (100, 5) -- 100 samples, 5 features
bias = np.array([1, 2, 3, 4, 5])  # (5,)
result = X + bias               # (100, 5) -- bias added to every row

# Pattern 2: Normalize each column (subtract mean, divide by std)
X = np.random.randn(100, 5)    # (100, 5)
mean = X.mean(axis=0)           # (5,)
std = X.std(axis=0)             # (5,)
X_normalized = (X - mean) / std # (100, 5)

# Pattern 3: Normalize each ROW (need keepdims!)
X = np.random.randn(100, 5)           # (100, 5)
row_sum = X.sum(axis=1, keepdims=True) # (100, 1) -- keepdims is critical!
X_row_normalized = X / row_sum          # (100, 5)
# Without keepdims: X.sum(axis=1) -> (100,) which can't broadcast with (100, 5)

# Pattern 4: Outer product
a = np.array([1, 2, 3])        # (3,)
b = np.array([10, 20, 30, 40]) # (4,)
outer = a[:, np.newaxis] * b[np.newaxis, :]  # (3, 1) * (1, 4) -> (3, 4)
# [[10, 20, 30, 40],
#  [20, 40, 60, 80],
#  [30, 60, 90, 120]]

# Equivalent: np.outer(a, b)

# Pattern 5: Pairwise differences
a = np.array([1, 3, 5])        # (3,)
diff = a[:, np.newaxis] - a[np.newaxis, :]  # (3, 1) - (1, 3) -> (3, 3)
# [[ 0, -2, -4],
#  [ 2,  0, -2],
#  [ 4,  2,  0]]
```

### Replacing Loops with Vectorized Operations

```python
# ============================================================
# EXAMPLE 1: Compute squared Euclidean distance matrix
# ============================================================

# LOOP VERSION (FORBIDDEN)
def distances_loop(X):
    """X shape: (N, D). Returns (N, N) distance matrix."""
    N = X.shape[0]
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.sum((X[i] - X[j]) ** 2)
    return dist

# VECTORIZED VERSION
def distances_vectorized(X):
    """X shape: (N, D). Returns (N, N) distance matrix."""
    # X[:, np.newaxis, :]  shape: (N, 1, D)
    # X[np.newaxis, :, :]  shape: (1, N, D)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, D)
    return np.sum(diff ** 2, axis=-1)                   # (N, N)

# Even faster using the expansion trick:
# ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
def distances_fast(X):
    """X shape: (N, D). Returns (N, N) distance matrix."""
    sq_norms = np.sum(X ** 2, axis=1)  # (N,)
    # sq_norms[:, np.newaxis]  shape: (N, 1)
    # sq_norms[np.newaxis, :]  shape: (1, N)
    return sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T

# ============================================================
# EXAMPLE 2: Softmax (with numerical stability)
# ============================================================

# LOOP VERSION (FORBIDDEN)
def softmax_loop(X):
    """X shape: (B, C). Returns (B, C) softmax."""
    B, C = X.shape
    result = np.zeros_like(X)
    for i in range(B):
        max_val = np.max(X[i])
        exp_vals = np.exp(X[i] - max_val)
        result[i] = exp_vals / np.sum(exp_vals)
    return result

# VECTORIZED VERSION
def softmax_vectorized(X):
    """X shape: (B, C). Returns (B, C) softmax."""
    exp_x = np.exp(X - X.max(axis=1, keepdims=True))  # (B, C)
    return exp_x / exp_x.sum(axis=1, keepdims=True)    # (B, C)

# ============================================================
# EXAMPLE 3: One-hot encoding
# ============================================================

# LOOP VERSION (FORBIDDEN)
def onehot_loop(labels, num_classes):
    """labels shape: (N,) of ints. Returns (N, C) one-hot."""
    N = len(labels)
    result = np.zeros((N, num_classes))
    for i in range(N):
        result[i, labels[i]] = 1.0
    return result

# VECTORIZED VERSION
def onehot_vectorized(labels, num_classes):
    """labels shape: (N,) of ints. Returns (N, C) one-hot."""
    return (labels[:, np.newaxis] == np.arange(num_classes)[np.newaxis, :]).astype(float)
    # labels[:, np.newaxis]         shape: (N, 1)
    # np.arange(num_classes)        shape: (C,) -> broadcast to (1, C)
    # Comparison produces            shape: (N, C) boolean

# Alternative using fancy indexing:
def onehot_fancy(labels, num_classes):
    result = np.zeros((len(labels), num_classes))
    result[np.arange(len(labels)), labels] = 1.0
    return result

# ============================================================
# EXAMPLE 4: Conditional operations with np.where
# ============================================================

# LOOP VERSION (FORBIDDEN)
def relu_loop(x):
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            result[i] = x[i]
        else:
            result[i] = 0
    return result

# VECTORIZED VERSION
def relu_vectorized(x):
    return np.where(x > 0, x, 0)
    # Or equivalently: np.maximum(x, 0)

# ============================================================
# EXAMPLE 5: Weighted sum / dot product
# ============================================================

# LOOP VERSION (FORBIDDEN)
def weighted_sum_loop(X, w):
    """X shape: (N, D), w shape: (D,). Returns (N,)."""
    N = X.shape[0]
    result = np.zeros(N)
    for i in range(N):
        for j in range(X.shape[1]):
            result[i] += X[i, j] * w[j]
    return result

# VECTORIZED VERSION
def weighted_sum_vectorized(X, w):
    return X @ w  # matrix-vector product: (N, D) @ (D,) -> (N,)
    # Or equivalently: np.dot(X, w) or np.einsum('ij,j->i', X, w)
```

### np.where -- Vectorized Conditionals

```python
a = np.array([-2, -1, 0, 1, 2])

# Basic: np.where(condition, value_if_true, value_if_false)
np.where(a > 0, a, 0)          # [0, 0, 0, 1, 2]  (ReLU)
np.where(a > 0, 1, -1)         # [-1, -1, -1, 1, 1]  (sign function)
np.where(a != 0, 1/a, 0)       # safe division

# Finding indices where condition is true
indices = np.where(a > 0)       # (array([3, 4]),)
a[np.where(a > 0)]             # [1, 2]

# 2D conditions
m = np.array([[1, 2], [3, 4]])
np.where(m > 2, m, -1)         # [[-1, -1], [3, 4]]
```

### einsum -- The Swiss Army Knife

`np.einsum` uses Einstein summation notation. It can express almost any array operation:

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# Matrix multiply: C_ik = sum_j A_ij * B_jk
C = np.einsum('ij,jk->ik', A, B)     # same as A @ B, shape (3, 5)

# Dot product: sum_i a_i * b_i
a = np.random.randn(5)
b = np.random.randn(5)
np.einsum('i,i->', a, b)              # same as np.dot(a, b), scalar

# Outer product: C_ij = a_i * b_j
np.einsum('i,j->ij', a[:3], b[:4])   # same as np.outer(a[:3], b[:4])

# Matrix-vector product: y_i = sum_j A_ij * v_j
np.einsum('ij,j->i', A, v)           # same as A @ v, shape (3,)

# Trace: sum of diagonal elements
S = np.random.randn(4, 4)
np.einsum('ii->', S)                  # same as np.trace(S)

# Transpose
np.einsum('ij->ji', A)               # same as A.T

# Sum over axis
np.einsum('ij->j', A)                # column sums, same as A.sum(axis=0)
np.einsum('ij->i', A)                # row sums, same as A.sum(axis=1)

# Batch matrix multiply: D_bij = sum_k A_bik * B_bkj
A_batch = np.random.randn(10, 3, 4)
B_batch = np.random.randn(10, 4, 5)
np.einsum('bij,bjk->bik', A_batch, B_batch)  # shape (10, 3, 5)

# Element-wise multiply then sum (Frobenius inner product)
np.einsum('ij,ij->', A, A)           # same as np.sum(A * A)
```

**einsum reading guide**: Each letter is a dimension index. Repeated indices on the left that do not appear on the right are summed over.

### Complete Vectorization Patterns Reference

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │              VECTORIZATION PATTERN COOKBOOK                      │
# └─────────────────────────────────────────────────────────────────┘

import numpy as np

# --- Reductions ---
X = np.random.randn(100, 5)  # (N, D)

np.sum(X)                    # sum of all elements -> scalar
np.sum(X, axis=0)            # column sums -> (D,)
np.sum(X, axis=1)            # row sums -> (N,)
np.mean(X, axis=0)           # column means -> (D,)
np.std(X, axis=0)            # column stds -> (D,)
np.min(X, axis=1)            # row minimums -> (N,)
np.argmax(X, axis=1)         # index of max in each row -> (N,)
np.cumsum(X, axis=0)         # cumulative sum down rows -> (N, D)

# --- Boolean reductions ---
mask = X > 0
np.any(mask, axis=1)         # True if any element in row > 0 -> (N,)
np.all(mask, axis=0)         # True if all elements in col > 0 -> (D,)
np.count_nonzero(mask, axis=1)  # count > 0 per row -> (N,)

# --- Sorting ---
np.sort(X, axis=0)           # sort each column -> (N, D)
np.argsort(X, axis=1)        # indices that would sort each row -> (N, D)
idx = np.argsort(X[:, 0])    # sort rows by first column
X_sorted = X[idx]            # apply sort -> (N, D)

# --- Clipping and thresholding ---
np.clip(X, -1, 1)            # clamp to [-1, 1]
np.maximum(X, 0)             # ReLU (element-wise max with 0)
np.minimum(X, 1)             # cap at 1

# --- Normalization patterns ---
# Min-max normalization (per column)
X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Z-score normalization (per column)
X_zscore = (X - X.mean(axis=0)) / X.std(axis=0)

# L2 normalization (per row)
norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))  # (N, 1)
X_l2 = X / norms                                       # (N, D)
```

---

## Resources

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy einsum explained](https://ajcr.net/Basic-guide-to-einsum/)
- [From Python to NumPy (book)](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)

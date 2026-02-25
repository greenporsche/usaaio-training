# NumPy Vectorization Exercises

**Time target**: 2-5 minutes each | **Total**: 6 exercises

---

## Exercise 1: Broadcasting Shape Prediction

For each pair of shapes, predict the result shape or state "ERROR":

```
(a) (5, 3) + (3,)
(b) (5, 3) + (5,)
(c) (4, 1) * (1, 3)
(d) (2, 3, 4) + (3, 4)
(e) (2, 3, 4) + (2, 1, 4)
(f) (3, 1, 5) * (1, 4, 5)
```

<details>
<summary>Solution</summary>

```
(a) (5, 3) + (3,)      -> (5, 3)       # (3,) pads to (1,3), stretches to (5,3)
(b) (5, 3) + (5,)      -> ERROR         # trailing dims: 3 vs 5, neither is 1
(c) (4, 1) * (1, 3)    -> (4, 3)       # both stretch: 4x1->4x3, 1x3->4x3
(d) (2, 3, 4) + (3, 4) -> (2, 3, 4)   # (3,4) pads to (1,3,4), stretches
(e) (2, 3, 4) + (2, 1, 4) -> (2, 3, 4) # middle dim: 3 vs 1, stretch 1->3
(f) (3, 1, 5) * (1, 4, 5) -> (3, 4, 5) # dim 0: 3 vs 1, dim 1: 1 vs 4
```

**Process**: Align shapes from the right. For each dimension: if equal, keep. If one is 1, stretch to match. Otherwise, error.
</details>

---

## Exercise 2: Vectorize This Loop (Normalize Rows)

Convert this loop to a single vectorized expression. **No loops allowed.**

```python
import numpy as np

X = np.random.randn(100, 5)  # (100, 5)

# Loop version
result = np.zeros_like(X)
for i in range(X.shape[0]):
    row_min = X[i].min()
    row_max = X[i].max()
    result[i] = (X[i] - row_min) / (row_max - row_min)
```

<details>
<summary>Solution</summary>

```python
row_min = X.min(axis=1, keepdims=True)  # (100, 1)
row_max = X.max(axis=1, keepdims=True)  # (100, 1)
result = (X - row_min) / (row_max - row_min)  # (100, 5)
```

**Critical detail**: `keepdims=True` is essential. Without it, `X.min(axis=1)` returns shape `(100,)`, which cannot broadcast with shape `(100, 5)` for row-wise subtraction (trailing dims: 5 vs 100 -- error!). With `keepdims=True`, the result is `(100, 1)` which broadcasts correctly.
</details>

---

## Exercise 3: Fix the Broadcasting Error

This code produces a ValueError. Fix it without using any loops.

```python
import numpy as np

# Goal: subtract each column's mean from that column
X = np.random.randn(50, 4)           # (50, 4)
col_means = np.mean(X, axis=0)       # (4,)
row_means = np.mean(X, axis=1)       # (50,)

# This works:
X_centered_cols = X - col_means

# This FAILS:
X_centered_rows = X - row_means      # ValueError!
```

<details>
<summary>Solution</summary>

```python
# Fix: reshape row_means to (50, 1) so broadcasting works
X_centered_rows = X - row_means[:, np.newaxis]   # (50, 4) - (50, 1) -> (50, 4)
# Or equivalently:
X_centered_rows = X - row_means.reshape(-1, 1)
```

**Why col_means works but row_means doesn't**:
- `col_means` shape is `(4,)`. Broadcasting: `(50, 4) - (4,)` pads to `(50, 4) - (1, 4)` -> `(50, 4)`. Trailing dim matches.
- `row_means` shape is `(50,)`. Broadcasting: `(50, 4) - (50,)` -> trailing dims: 4 vs 50 -- ERROR!

The fix is to add a dimension so the shape becomes `(50, 1)`, which broadcasts to `(50, 4)`.
</details>

---

## Exercise 4: Vectorize This Loop (Pairwise Operations)

Convert this nested loop to a vectorized expression. **No loops allowed.**

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0, 4.0])  # (4,)
b = np.array([10.0, 20.0, 30.0])     # (3,)

# Goal: create a matrix M where M[i,j] = a[i]^2 + b[j]^2
# Loop version:
M = np.zeros((len(a), len(b)))
for i in range(len(a)):
    for j in range(len(b)):
        M[i, j] = a[i]**2 + b[j]**2
```

<details>
<summary>Solution</summary>

```python
M = a[:, np.newaxis]**2 + b[np.newaxis, :]**2
# a[:, np.newaxis]**2   shape: (4, 1)
# b[np.newaxis, :]**2   shape: (1, 3)
# Result                 shape: (4, 3)
```

Result:
```
[[  101.,  401.,  901.],
 [  104.,  404.,  904.],
 [  109.,  409.,  909.],
 [  116.,  416.,  916.]]
```

**Pattern**: Whenever you have a nested loop computing `f(a[i], b[j])`, reshape `a` to a column `(N, 1)` and `b` to a row `(1, M)`, then apply `f` element-wise. Broadcasting creates the outer product.
</details>

---

## Exercise 5: What Shape Results?

Predict the output shape. Then verify mentally or in Python.

```python
import numpy as np

X = np.random.randn(10, 3)     # (10, 3) -- 10 points in 3D

# Pairwise distance computation
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
sq_diff = diff ** 2
sum_sq = np.sum(sq_diff, axis=-1)
dist = np.sqrt(sum_sq)
```

What is the shape of `diff`, `sq_diff`, `sum_sq`, and `dist`?

<details>
<summary>Solution</summary>

```python
X[:, np.newaxis, :]    # shape: (10, 1, 3)
X[np.newaxis, :, :]    # shape: (1, 10, 3)

diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
# (10, 1, 3) - (1, 10, 3) -> (10, 10, 3)
# diff[i, j, k] = X[i, k] - X[j, k]

sq_diff = diff ** 2
# (10, 10, 3) -- element-wise square

sum_sq = np.sum(sq_diff, axis=-1)
# (10, 10) -- sum along last axis (the 3 coordinate dimensions)
# sum_sq[i, j] = sum of squared differences between point i and point j

dist = np.sqrt(sum_sq)
# (10, 10) -- Euclidean distance matrix
# dist[i, j] = distance between point i and point j
```

**This is the most important vectorization pattern for USAAIO.** Memorize it: to compute pairwise operations between N items, reshape to `(N, 1, ...)` and `(1, N, ...)`, then broadcast to `(N, N, ...)`.
</details>

---

## Exercise 6: einsum Challenge

Rewrite each NumPy operation using `np.einsum`:

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# Rewrite these using einsum:
r1 = A @ B                    # matrix multiply
r2 = A @ v                    # matrix-vector multiply
r3 = np.sum(A, axis=0)        # column sums
r4 = np.trace(A @ A.T)        # trace of A * A^T (Frobenius norm squared)
r5 = np.sum(A * A)            # sum of all squared elements
```

<details>
<summary>Solution</summary>

```python
r1 = np.einsum('ij,jk->ik', A, B)       # C_ik = sum_j A_ij * B_jk
r2 = np.einsum('ij,j->i', A, v)         # y_i = sum_j A_ij * v_j
r3 = np.einsum('ij->j', A)              # s_j = sum_i A_ij
r4 = np.einsum('ij,ij->', A, A)         # sum_ij A_ij * A_ij
r5 = np.einsum('ij,ij->', A, A)         # same as r4!
```

**einsum reading guide**:
- Left of `->`: input indices. Repeated indices across inputs = element-wise multiply
- Right of `->`: output indices. Any index NOT on the right is summed over
- `'ij,jk->ik'`: `j` appears in both inputs and NOT in output, so sum over `j` -> matrix multiply
- `'ij->'`: both `i` and `j` missing from output -> sum over everything -> scalar

Note: `r4` and `r5` are the same because `trace(A @ A^T) = sum(A * A)` (the Frobenius norm squared). This is a useful identity.
</details>

# NumPy Basics Exercises

**Time target**: 2-5 minutes each | **Total**: 6 exercises

---

## Exercise 1: Shape Prediction

What is the shape of each result?

```python
import numpy as np

a = np.arange(24).reshape(2, 3, 4)

s1 = a[0]
s2 = a[:, 1]
s3 = a[:, :, 2]
s4 = a[0, 1, 2]
s5 = a[:, 1:3, ::2]
s6 = a.reshape(-1, 6)
```

<details>
<summary>Solution</summary>

```python
a.shape    # (2, 3, 4)

s1.shape   # (3, 4)     -- selected first element along axis 0
s2.shape   # (2, 4)     -- selected index 1 along axis 1
s3.shape   # (2, 3)     -- selected index 2 along axis 2
s4         # scalar (8)  -- single element, no dimensions
s5.shape   # (2, 2, 2)  -- axis 1: indices 1,2; axis 2: indices 0,2
s6.shape   # (4, 6)     -- 24 elements reshaped: 24/6 = 4 rows
```

**Key insight**: indexing with a single integer removes that dimension. Slicing preserves it. `a[0]` removes axis 0 (shape goes from (2,3,4) to (3,4)), but `a[0:1]` preserves it (shape becomes (1,3,4)).
</details>

---

## Exercise 2: View or Copy?

For each operation, does `b` share memory with `a`? (Is it a view or a copy?)

```python
a = np.arange(10)

b1 = a[2:5]
b2 = a[[2, 3, 4]]
b3 = a[a > 5]
b4 = a.reshape(2, 5)
b5 = a.flatten()
b6 = a[::2]
```

<details>
<summary>Solution</summary>

```python
b1 = a[2:5]           # VIEW -- basic slicing
b2 = a[[2, 3, 4]]     # COPY -- fancy indexing (list of indices)
b3 = a[a > 5]          # COPY -- boolean indexing
b4 = a.reshape(2, 5)   # VIEW -- reshape returns view when possible
b5 = a.flatten()        # COPY -- flatten always copies (use ravel for view)
b6 = a[::2]             # VIEW -- stride slicing
```

**Quick test**: `b.base is a` returns `True` if `b` is a view of `a`.

**Rule**: Basic slicing (with `:`) and reshape return views. Fancy indexing (integer array or boolean array) and `flatten()` return copies.
</details>

---

## Exercise 3: Boolean Indexing

Given:

```python
scores = np.array([78, 92, 65, 88, 95, 71, 83, 90])
names = np.array(['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Hank'])
```

Write **one-line** NumPy expressions (no loops) to find:

1. Names of students who scored above 85
2. The highest score among students who scored below 90
3. How many students scored between 70 and 90 (inclusive)

<details>
<summary>Solution</summary>

```python
# 1. Names above 85
names[scores > 85]
# array(['Bob', 'Dave', 'Eve', 'Hank'])

# 2. Highest score below 90
scores[scores < 90].max()
# 88

# 3. Count between 70 and 90 inclusive
np.count_nonzero((scores >= 70) & (scores <= 90))
# 5 (78, 88, 71, 83, 90)
# Alternative: ((scores >= 70) & (scores <= 90)).sum()
```

**Key pattern**: Boolean indexing lets you use one array's condition to filter another array of the same length. This replaces loops with conditional logic.
</details>

---

## Exercise 4: Axis Parameter

What does each expression return?

```python
a = np.array([[10, 20, 30],
              [40, 50, 60]])  # shape: (2, 3)

r1 = np.sum(a, axis=0)
r2 = np.sum(a, axis=1)
r3 = np.argmax(a, axis=0)
r4 = np.argmax(a, axis=1)
r5 = np.sum(a, axis=0, keepdims=True)
r6 = np.mean(a)
```

<details>
<summary>Solution</summary>

```python
r1 = np.sum(a, axis=0)    # [50, 70, 90]     shape: (3,)   -- sum down each column
r2 = np.sum(a, axis=1)    # [60, 150]         shape: (2,)   -- sum across each row
r3 = np.argmax(a, axis=0) # [1, 1, 1]         shape: (3,)   -- index of max in each column
r4 = np.argmax(a, axis=1) # [2, 2]            shape: (2,)   -- index of max in each row
r5 = np.sum(a, axis=0, keepdims=True)  # [[50, 70, 90]]  shape: (1, 3)
r6 = np.mean(a)           # 35.0              scalar         -- mean of all elements
```

**Memory aid**: `axis=N` means "collapse dimension N". The result loses that dimension (unless `keepdims=True`).
- `axis=0` on shape (2, 3): collapse dim 0 (rows) -> result shape (3,)
- `axis=1` on shape (2, 3): collapse dim 1 (cols) -> result shape (2,)
</details>

---

## Exercise 5: Reshape Puzzle

Which of these reshapes are valid? For valid ones, what's the result?

```python
a = np.arange(12)  # [0, 1, 2, ..., 11], shape (12,)

r1 = a.reshape(3, 4)
r2 = a.reshape(4, 4)
r3 = a.reshape(2, -1)
r4 = a.reshape(-1, -1)
r5 = a.reshape(2, 3, 2)
r6 = a.reshape(12, 1)
```

<details>
<summary>Solution</summary>

```python
r1 = a.reshape(3, 4)     # VALID: 3*4=12. Shape (3, 4)
r2 = a.reshape(4, 4)     # INVALID: 4*4=16 != 12. ValueError!
r3 = a.reshape(2, -1)    # VALID: -1 inferred as 6. Shape (2, 6)
r4 = a.reshape(-1, -1)   # INVALID: only ONE dimension can be -1. ValueError!
r5 = a.reshape(2, 3, 2)  # VALID: 2*3*2=12. Shape (2, 3, 2)
r6 = a.reshape(12, 1)    # VALID: 12*1=12. Shape (12, 1) -- column vector
```

**Rules**:
1. Total elements must be preserved: product of new shape = product of old shape
2. At most one dimension can be `-1` (inferred from the others)
3. Reshape fills in **row-major order** (C order) by default: elements fill left-to-right, then top-to-bottom
</details>

---

## Exercise 6: Stacking Challenge

Given:

```python
a = np.array([[1, 2], [3, 4]])  # shape (2, 2)
b = np.array([[5, 6], [7, 8]])  # shape (2, 2)
```

What is the shape and content of each?

```python
r1 = np.concatenate([a, b], axis=0)
r2 = np.concatenate([a, b], axis=1)
r3 = np.stack([a, b], axis=0)
r4 = np.stack([a, b], axis=2)
```

<details>
<summary>Solution</summary>

```python
r1 = np.concatenate([a, b], axis=0)
# Shape: (4, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

r2 = np.concatenate([a, b], axis=1)
# Shape: (2, 4)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]

r3 = np.stack([a, b], axis=0)
# Shape: (2, 2, 2)
# [[[1, 2],
#   [3, 4]],
#  [[5, 6],
#   [7, 8]]]

r4 = np.stack([a, b], axis=2)
# Shape: (2, 2, 2)
# [[[1, 5],
#   [2, 6]],
#  [[3, 7],
#   [4, 8]]]
```

**Key difference**:
- `concatenate` joins arrays along an **existing** axis (no new dimension)
- `stack` creates a **new** axis and stacks arrays along it

`np.stack([a, b], axis=0)` is like wrapping `[a, b]` in a new dimension at position 0.
`np.stack([a, b], axis=2)` inserts a new dimension at position 2, interleaving elements.
</details>

# NumPy Basics

**Prerequisites**: Python fundamentals, list comprehensions
**USAAIO Relevance**: NumPy is the foundation of every AI library. Array creation, indexing, and shape manipulation appear in nearly every USAAIO coding problem. You must be fluent in these operations.

---

## Discovery

### Why Arrays Are Faster Than Lists

Consider adding two lists of 1 million numbers:

```python
# Pure Python: ~100ms
result = [a[i] + b[i] for i in range(1_000_000)]

# NumPy: ~1ms (100x faster)
result = a + b
```

Why is NumPy 100x faster? Three reasons:

**1. Contiguous memory layout**
```
Python list:     [ptr] -> [ptr] -> [ptr] -> ...
                   |        |        |
                   v        v        v
                 [1.0]    [2.0]    [3.0]    (scattered in memory)

NumPy array:     [1.0][2.0][3.0][4.0]...   (contiguous block)
```
Python lists store **pointers** to objects scattered across memory. NumPy arrays store raw numbers in a single contiguous block. The CPU cache can load chunks of a NumPy array at once, but has to chase pointers for lists.

**2. SIMD (Single Instruction, Multiple Data)**
Modern CPUs can add 4 or 8 numbers simultaneously with a single instruction. NumPy uses these SIMD instructions; Python loops cannot.

**3. No type checking per element**
Python checks the type of every object at every operation. NumPy arrays have a single dtype -- all elements are the same type, so no per-element type checking is needed.

> **Socratic question**: If NumPy arrays must have a single dtype, what happens when you put an integer and a string in the same array? Try `np.array([1, 'hello'])` and see what dtype results.

### The ndarray: NumPy's Core Object

Every NumPy array (`ndarray`) has:
- **data**: pointer to a contiguous block of memory
- **dtype**: the type of each element (float64, int32, etc.)
- **shape**: tuple of dimension sizes, e.g. `(3, 4)` for 3 rows, 4 columns
- **strides**: bytes to skip to get to the next element along each dimension

```
Array with shape (3, 4), dtype float64 (8 bytes each):
strides = (32, 8)  -- skip 32 bytes for next row, 8 bytes for next column

Memory layout:
[0.0][1.0][2.0][3.0][4.0][5.0][6.0][7.0][8.0][9.0][10.][11.]
|--- row 0 --------|--- row 1 ---------|--- row 2 ---------|
```

---

## Intuition

### Shape: The Most Important Property

Shape tells you **what kind of mathematical object** you have:

```
shape ()        -> scalar      (just a number)
shape (N,)      -> vector      (1D: N elements)
shape (M, N)    -> matrix      (2D: M rows, N columns)
shape (B, M, N) -> batch       (3D: B matrices, each M x N)
```

**In USAAIO, always think in shapes.** Before writing any operation, write down the shapes of your inputs and what shape you expect as output.

### Indexing Mental Model

```
2D array a with shape (3, 4):

         col 0  col 1  col 2  col 3
row 0  [  0      1      2      3  ]
row 1  [  4      5      6      7  ]
row 2  [  8      9     10     11  ]

a[1, 2] = 6        (single element)
a[1] = [4,5,6,7]   (entire row -- axis 0 indexed)
a[:, 2] = [2,6,10] (entire column -- axis 1 indexed)
a[0:2, 1:3]:        (subarray -- rows 0-1, cols 1-2)
  [[1, 2],
   [5, 6]]
```

### Views vs Copies

This is the most common source of bugs for beginners:

```
SLICING = VIEW (shared memory)
┌────────────┐
│ a = [1,2,3,4,5]  │
│             ▲     │
│ b = a[1:4]  │     │  b is a VIEW into a
│   [2, 3, 4]─┘     │  modifying b modifies a!
└────────────────────┘

FANCY/BOOLEAN INDEXING = COPY (independent)
┌────────────────────┐
│ a = [1,2,3,4,5]    │
│                     │
│ c = a[[1,2,3]]     │  c is a COPY
│   [2, 3, 4]        │  modifying c does NOT affect a
└────────────────────┘
```

**Rule of thumb**: if you use `:` (slice notation), you get a view. If you use a list of indices or a boolean array, you get a copy.

---

## Math

### Dtype Hierarchy

NumPy promotes types automatically (upcasting):

```
bool -> int8 -> int16 -> int32 -> int64
                                     \
                          float16 -> float32 -> float64
```

When you combine arrays of different dtypes, NumPy promotes to the "larger" type:
- `int32 + float64 = float64`
- `bool + int64 = int64`

### Shape Arithmetic

Understanding how shapes transform is critical:

| Operation | Input Shape | Output Shape |
|-----------|------------|-------------|
| `a[i]` | `(M, N)` | `(N,)` |
| `a[:, j]` | `(M, N)` | `(M,)` |
| `a[i:j]` | `(M, N)` | `(j-i, N)` |
| `a.reshape(P, Q)` | `(M, N)` | `(P, Q)` where P*Q = M*N |
| `a.T` | `(M, N)` | `(N, M)` |
| `a.ravel()` | `(M, N)` | `(M*N,)` |

---

## Code

### Array Creation

```python
import numpy as np

# From Python data
a = np.array([1, 2, 3])                    # shape: (3,)
b = np.array([[1, 2], [3, 4]])              # shape: (2, 2)

# Filled arrays
zeros = np.zeros((3, 4))                    # shape: (3, 4), all 0.0
ones = np.ones((2, 3), dtype=np.int32)      # shape: (2, 3), all 1
full = np.full((2, 2), 3.14)               # shape: (2, 2), all 3.14
eye = np.eye(3)                             # shape: (3, 3), identity matrix

# Ranges
r1 = np.arange(0, 10, 2)                   # [0, 2, 4, 6, 8]
r2 = np.linspace(0, 1, 5)                  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random
np.random.seed(42)                          # reproducibility
rand_uniform = np.random.rand(3, 4)         # shape: (3, 4), uniform [0, 1)
rand_normal = np.random.randn(3, 4)         # shape: (3, 4), standard normal
rand_int = np.random.randint(0, 10, (3, 4)) # shape: (3, 4), ints in [0, 10)

# zeros_like / ones_like (match shape and dtype)
template = np.array([[1.0, 2.0], [3.0, 4.0]])
z = np.zeros_like(template)                 # shape: (2, 2), dtype float64
```

### Dtypes

```python
a = np.array([1, 2, 3])             # default: int64 (on 64-bit systems)
b = np.array([1.0, 2.0, 3.0])       # default: float64
c = np.array([True, False, True])    # default: bool

# Explicit dtype
d = np.array([1, 2, 3], dtype=np.float32)  # force float32

# Check and convert
print(a.dtype)           # int64
e = a.astype(np.float64) # convert to float64 (creates copy)

# Common dtypes in AI
# np.float32  -- standard for neural networks (saves memory)
# np.float64  -- standard for scientific computing (more precision)
# np.int64    -- default for integers
# np.bool_    -- boolean masks
```

### Basic Indexing and Slicing

```python
a = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Single element
a[0, 0]       # 0
a[2, 3]       # 11
a[-1, -1]     # 11 (negative indexing from end)

# Row selection
a[0]          # [0, 1, 2, 3]       shape: (4,)
a[0, :]       # [0, 1, 2, 3]       same thing, explicit

# Column selection
a[:, 1]       # [1, 5, 9]          shape: (3,)

# Slicing: start:stop:step (stop is exclusive)
a[0:2]        # rows 0-1            shape: (2, 4)
a[:, 1:3]     # cols 1-2            shape: (3, 2)
a[::2]        # every other row     shape: (2, 4)
a[:, ::-1]    # reverse columns     shape: (3, 4)

# Combined
a[0:2, 1:3]   # [[1, 2], [5, 6]]   shape: (2, 2)
a[::2, ::2]    # [[0, 2], [8, 10]]  shape: (2, 2)
```

### Fancy Indexing

```python
a = np.arange(12).reshape(3, 4)

# Integer array indexing -- select specific elements
a[[0, 2], [1, 3]]    # [1, 11]   elements at (0,1) and (2,3)

# Select specific rows
a[[0, 2]]            # [[0,1,2,3], [8,9,10,11]]   shape: (2, 4)

# Select specific columns
a[:, [0, 3]]         # [[0,3], [4,7], [8,11]]     shape: (3, 2)

# Combine: select rows 0,2 and columns 1,3
a[np.ix_([0, 2], [1, 3])]  # [[1,3], [9,11]]      shape: (2, 2)
# np.ix_ creates an open mesh for cross-indexing
```

### Boolean Indexing

```python
a = np.arange(12).reshape(3, 4)

# Create boolean mask
mask = a > 5                    # shape: (3, 4), dtype bool
# [[False, False, False, False],
#  [False, False,  True,  True],
#  [ True,  True,  True,  True]]

# Apply mask -- returns 1D array of matching elements
a[mask]                         # [6, 7, 8, 9, 10, 11]

# Direct condition
a[a % 2 == 0]                  # [0, 2, 4, 6, 8, 10]

# Compound conditions (use & for AND, | for OR, ~ for NOT)
a[(a > 3) & (a < 8)]           # [4, 5, 6, 7]
a[(a < 2) | (a > 9)]           # [0, 1, 10, 11]
a[~(a > 5)]                    # [0, 1, 2, 3, 4, 5]

# Boolean indexing for assignment
b = a.copy()
b[b < 5] = 0                   # set all elements < 5 to 0
```

### Views vs Copies -- The Critical Distinction

```python
a = np.arange(6)  # [0, 1, 2, 3, 4, 5]

# Slicing creates a VIEW
b = a[1:4]        # [1, 2, 3] -- shares memory with a
b[0] = 99         # modifies BOTH b and a!
print(a)          # [0, 99, 2, 3, 4, 5]

# Fancy indexing creates a COPY
a = np.arange(6)
c = a[[1, 2, 3]]  # [1, 2, 3] -- independent copy
c[0] = 99         # modifies only c
print(a)          # [0, 1, 2, 3, 4, 5] (unchanged)

# Boolean indexing creates a COPY
a = np.arange(6)
d = a[a > 2]       # [3, 4, 5] -- independent copy

# To explicitly copy a slice:
e = a[1:4].copy()  # now e is independent

# Check if array is a view
print(b.base is a)  # True if b is a view of a
```

### Shape Manipulation

```python
a = np.arange(12)  # shape: (12,)

# reshape: change shape without changing data
b = a.reshape(3, 4)     # shape: (3, 4)
c = a.reshape(2, 2, 3)  # shape: (2, 2, 3)
d = a.reshape(-1, 4)    # shape: (3, 4)  -- infer first dim
e = a.reshape(4, -1)    # shape: (4, 3)  -- infer second dim

# Total elements must match: 12 = 3*4 = 2*2*3 = 4*3

# transpose
m = np.arange(6).reshape(2, 3)  # shape: (2, 3)
m.T                              # shape: (3, 2)

# ravel (flatten to 1D, returns view if possible)
m.ravel()    # shape: (6,)

# flatten (always returns copy)
m.flatten()  # shape: (6,)

# Adding / removing dimensions
a = np.array([1, 2, 3])              # shape: (3,)
a[np.newaxis, :]                      # shape: (1, 3) -- row vector
a[:, np.newaxis]                      # shape: (3, 1) -- column vector
np.expand_dims(a, axis=0)            # shape: (1, 3)
np.expand_dims(a, axis=1)            # shape: (3, 1)

b = np.zeros((1, 3, 1))              # shape: (1, 3, 1)
np.squeeze(b)                         # shape: (3,) -- remove all size-1 dims
np.squeeze(b, axis=0)                # shape: (3, 1) -- remove only axis 0
```

### Stacking and Concatenation

```python
a = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
b = np.array([[5, 6], [7, 8]])  # shape: (2, 2)

# concatenate: join along EXISTING axis
np.concatenate([a, b], axis=0)  # shape: (4, 2) -- stack vertically
# [[1,2],[3,4],[5,6],[7,8]]

np.concatenate([a, b], axis=1)  # shape: (2, 4) -- stack horizontally
# [[1,2,5,6],[3,4,7,8]]

# stack: join along NEW axis
np.stack([a, b], axis=0)        # shape: (2, 2, 2)
np.stack([a, b], axis=1)        # shape: (2, 2, 2) -- different arrangement

# Shortcuts
np.vstack([a, b])  # same as concatenate axis=0: (4, 2)
np.hstack([a, b])  # same as concatenate axis=1: (2, 4)

# split
c = np.arange(12).reshape(3, 4)
np.split(c, 3, axis=0)          # 3 arrays of shape (1, 4)
np.split(c, 2, axis=1)          # 2 arrays of shape (3, 2)
```

### The Axis Parameter

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape: (2, 3)

# axis=0: operation along rows (collapse rows -> result has no row dim)
np.sum(a, axis=0)     # [5, 7, 9]      shape: (3,)
# Think: "sum DOWN each column"

# axis=1: operation along columns (collapse columns -> result has no col dim)
np.sum(a, axis=1)     # [6, 15]         shape: (2,)
# Think: "sum ACROSS each row"

# axis=None: operation on all elements (flatten first)
np.sum(a)             # 21              scalar

# keepdims: preserve the collapsed dimension as size 1
np.sum(a, axis=0, keepdims=True)  # [[5, 7, 9]]  shape: (1, 3)
np.sum(a, axis=1, keepdims=True)  # [[6], [15]]   shape: (2, 1)
# keepdims is crucial for broadcasting (covered in next section)
```

---

## Resources

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Visual NumPy Introduction (Jay Alammar)](https://jalammar.github.io/visual-numpy/)

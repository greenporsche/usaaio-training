# Tensors

**Prerequisites**: NumPy arrays, basic linear algebra (shapes, matrix multiplication)
**USAAIO Relevance**: Every PyTorch operation starts with tensors. Shape errors are the #1 debugging issue in competition. Mastering tensor creation, manipulation, and broadcasting is non-negotiable.

---

## Discovery

### Why Tensors Instead of NumPy Arrays?

If you already know NumPy, you might wonder: why learn a new array library? NumPy arrays and PyTorch tensors look almost identical:

```python
import numpy as np
import torch

np_arr = np.array([1.0, 2.0, 3.0])
pt_tensor = torch.tensor([1.0, 2.0, 3.0])
```

Three reasons make tensors essential:

1. **GPU acceleration**: NumPy runs on CPU only. PyTorch tensors can live on GPU, making matrix operations 10–100x faster for large models.

2. **Automatic differentiation**: Tensors track operations in a computation graph. Call `.backward()` and PyTorch computes all gradients for you — this is what makes training neural networks possible.

3. **Deep learning ecosystem**: Every PyTorch layer (`nn.Linear`, `nn.Conv2d`, etc.) expects tensors. The entire training pipeline — data loading, forward pass, loss computation, backward pass — operates on tensors.

> **Socratic question**: If PyTorch tensors are so similar to NumPy arrays, why did the PyTorch team build a new library instead of adding GPU/autograd support to NumPy?
>
> *Think about it*: NumPy's C internals were not designed for computation graph tracking. Retrofitting autograd into NumPy would require rewriting the core. It was cleaner to build a new library with gradient tracking baked in from day one.

### What IS a Tensor?

Mathematically, a tensor is a multi-dimensional array. The number of dimensions is its **rank** (or **order**):

| Rank | Name | Example |
|---|---|---|
| 0 | Scalar | `torch.tensor(3.14)` — a single number |
| 1 | Vector | `torch.tensor([1, 2, 3])` — shape `(3,)` |
| 2 | Matrix | `torch.randn(3, 4)` — shape `(3, 4)` |
| 3 | 3D tensor | `torch.randn(2, 3, 4)` — e.g., batch of matrices |
| 4 | 4D tensor | `torch.randn(B, C, H, W)` — batch of images |

---

## Intuition

### Memory Layout: Contiguous vs Non-Contiguous

A tensor's data is stored as a flat 1D block of numbers. The **stride** tells PyTorch how to navigate the flat memory to find a specific element:

```
Matrix (2x3):
[[1, 2, 3],        Memory: [1, 2, 3, 4, 5, 6]
 [4, 5, 6]]        Stride: (3, 1) — skip 3 for next row, 1 for next col
```

When you transpose, PyTorch does NOT copy data. It just swaps strides:

```
Transposed (3x2):
[[1, 4],            Memory: [1, 2, 3, 4, 5, 6]  (same!)
 [2, 5],            Stride: (1, 3) — skip 1 for next row, 3 for next col
 [3, 6]]
```

This is why `view()` fails after `transpose()` — the data is no longer contiguous in memory. Use `.contiguous()` first, or just use `reshape()` which handles it automatically.

### Broadcasting: Same Rules as NumPy

Broadcasting lets you operate on tensors of different shapes without explicit copying:

```
Shape alignment (right to left):

  (4, 3, 2)
+       (2,)    ← left-pad with 1s → (1, 1, 2)
= (4, 3, 2)    ← broadcast: (4, 3, 2) + (1, 1, 2) → (4, 3, 2)

  (3, 1)
+ (1, 4)
= (3, 4)       ← both dimensions broadcast
```

Rules:
1. Align shapes from the right
2. Missing dims on the left are treated as size 1
3. Size-1 dims are stretched to match the other tensor
4. If both dims are > 1 and different → **error**

### view vs clone vs reshape

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   view()    │     │   reshape()  │     │   clone()    │
├─────────────┤     ├──────────────┤     ├──────────────┤
│ Same memory │     │ Same memory  │     │ NEW memory   │
│ Must be     │     │ if possible, │     │ Independent  │
│ contiguous  │     │ else copies  │     │ copy         │
│ Fast, no    │     │ Always works │     │ Always works │
│ copy ever   │     │              │     │ Gradient     │
│             │     │              │     │ flows through│
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Math

### Tensor Operations as Linear Algebra

Matrix multiplication in PyTorch:

$$C = AB \quad \Leftrightarrow \quad \texttt{C = A @ B}$$

For batched matrix multiplication with shapes `(B, M, K)` and `(B, K, N)`:

$$C_{b,i,j} = \sum_k A_{b,i,k} \cdot B_{b,k,j} \quad \Leftrightarrow \quad \texttt{C = torch.bmm(A, B)}$$

Result shape: `(B, M, N)`.

For general Einstein summation:

$$C_{ij} = \sum_k A_{ik} B_{kj} \quad \Leftrightarrow \quad \texttt{C = torch.einsum('ik,kj->ij', A, B)}$$

### NumPy-PyTorch Conversion

The conversion is zero-copy when possible:

$$\texttt{torch.from\_numpy(arr)} \quad \text{shares memory (no copy)}$$
$$\texttt{tensor.numpy()} \quad \text{shares memory (CPU only, no grad)}$$

---

## Code

### Tensor Creation

```python
import torch

# From Python data
x = torch.tensor([1, 2, 3])                    # shape: (3,), dtype: int64
x = torch.tensor([1.0, 2.0, 3.0])              # shape: (3,), dtype: float32
x = torch.tensor([[1, 2], [3, 4]])              # shape: (2, 2)

# Factory functions
z = torch.zeros(3, 4)                           # shape: (3, 4), all zeros
o = torch.ones(2, 3, dtype=torch.float64)       # shape: (2, 3), all ones, float64
r = torch.randn(5, 5)                           # shape: (5, 5), N(0, 1)
u = torch.rand(3, 3)                            # shape: (3, 3), Uniform(0, 1)
e = torch.eye(4)                                # shape: (4, 4), identity
f = torch.full((2, 3), fill_value=7.0)          # shape: (2, 3), all 7.0

# Sequences
a = torch.arange(0, 10, 2)                      # tensor([0, 2, 4, 6, 8])
l = torch.linspace(0, 1, 5)                     # tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# Like-functions (match shape/dtype/device of existing tensor)
x = torch.randn(3, 4)
z = torch.zeros_like(x)                         # same shape, dtype, device as x
o = torch.ones_like(x)
r = torch.randn_like(x)
```

### dtype and Device

```python
# dtype
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = x.to(torch.float64)     # cast
x = x.float()               # shorthand for float32
x = x.long()                # shorthand for int64

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 4, device=device)     # create on device directly
x = x.to(device)                          # move existing tensor
x = x.cuda()                              # explicit GPU
x = x.cpu()                               # explicit CPU
print(x.device)                            # e.g., cuda:0 or cpu
```

### Shape Manipulation

```python
x = torch.randn(2, 3, 4)                  # shape: (2, 3, 4)

# Reshape
x.reshape(2, 12)                           # shape: (2, 12)
x.reshape(2, -1)                           # shape: (2, 12) — infer last dim
x.view(6, 4)                               # shape: (6, 4) — must be contiguous

# Transpose / Permute
x.transpose(1, 2)                          # shape: (2, 4, 3) — swap dims 1 and 2
x.permute(2, 0, 1)                         # shape: (4, 2, 3) — arbitrary reorder

# Squeeze / Unsqueeze
y = torch.randn(1, 3, 1, 4)
y.squeeze()                                # shape: (3, 4) — remove ALL size-1 dims
y.squeeze(0)                               # shape: (3, 1, 4) — remove dim 0 only
z = torch.randn(3, 4)
z.unsqueeze(0)                             # shape: (1, 3, 4) — add dim at position 0
z.unsqueeze(-1)                            # shape: (3, 4, 1) — add dim at end
z[None, :, :]                              # shape: (1, 3, 4) — equivalent to unsqueeze(0)
z[:, :, None]                              # shape: (3, 4, 1) — equivalent to unsqueeze(-1)

# Flatten
x = torch.randn(2, 3, 4)
x.flatten()                                # shape: (24,) — flatten all dims
x.flatten(1)                               # shape: (2, 12) — flatten from dim 1 onward
```

### Indexing and Masking

```python
x = torch.randn(4, 5)

# Basic indexing
x[0]                                       # shape: (5,) — first row
x[:, 0]                                    # shape: (4,) — first column
x[1:3, 2:5]                               # shape: (2, 3) — slice

# Boolean masking
mask = x > 0                               # shape: (4, 5), dtype: bool
x[mask]                                    # shape: (N,) — flat tensor of positive elements

# Fancy (advanced) indexing
indices = torch.tensor([0, 2, 3])
x[indices]                                 # shape: (3, 5) — rows 0, 2, 3

# gather and scatter (for competition use)
# gather: select elements along a dim using indices
src = torch.tensor([[1, 2], [3, 4]])       # shape: (2, 2)
idx = torch.tensor([[0, 0], [1, 0]])       # shape: (2, 2)
torch.gather(src, dim=1, index=idx)        # tensor([[1, 1], [4, 3]])
```

### NumPy Interop

```python
import numpy as np

# NumPy → PyTorch (shared memory!)
np_arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_arr)          # shares memory
np_arr[0] = 99.0
print(tensor[0])                           # tensor(99.) — linked!

# PyTorch → NumPy (shared memory, CPU only)
tensor = torch.randn(3)
np_arr = tensor.numpy()                    # shares memory

# To break the link, use .clone() or .copy()
tensor = torch.from_numpy(np_arr.copy())   # independent copy
```

### Matrix Operations

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Matrix multiply
C = A @ B                                  # shape: (3, 5)
C = torch.matmul(A, B)                     # same as above
C = torch.mm(A, B)                         # 2D only

# Batched matrix multiply
A = torch.randn(8, 3, 4)                  # batch of 8 matrices
B = torch.randn(8, 4, 5)
C = torch.bmm(A, B)                        # shape: (8, 3, 5)
C = A @ B                                  # also works with batches

# Einstein summation
C = torch.einsum('bik,bkj->bij', A, B)    # batched matmul

# Element-wise operations
x = torch.randn(3, 4)
x * 2                                      # scalar multiply
x + x                                      # element-wise add
x ** 2                                     # element-wise square
torch.exp(x)                               # element-wise exp
torch.clamp(x, min=0)                      # ReLU equivalent

# Reductions
x.sum()                                    # scalar sum of all elements
x.sum(dim=0)                               # shape: (4,) — sum along rows
x.sum(dim=1)                               # shape: (3,) — sum along columns
x.mean(dim=-1, keepdim=True)               # shape: (3, 1) — keep dim for broadcasting
x.max(dim=1)                               # returns (values, indices)
x.argmax(dim=1)                            # shape: (3,) — index of max per row
```

### Reproducibility

```python
torch.manual_seed(42)                      # CPU seed
torch.cuda.manual_seed_all(42)             # GPU seed (all devices)

# For full reproducibility:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

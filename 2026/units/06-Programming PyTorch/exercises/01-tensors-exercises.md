# Tensor Exercises

**Topic**: Tensors — creation, dtypes, operations, broadcasting, shape manipulation
**Difficulty**: Foundational → Intermediate

---

## Exercise 1: Predict the Shape

For each operation, predict the output shape **before** running the code. Then verify.

```python
import torch

a = torch.randn(3, 4, 5)

# What is the shape of each?
b = a.reshape(3, 20)
c = a.permute(2, 0, 1)
d = a.transpose(1, 2)
e = a[:, :2, :]
f = a.unsqueeze(0)
g = a.flatten(1)
h = a.sum(dim=1)
i = a.mean(dim=-1, keepdim=True)
j = a[0]
```

<details>
<summary>Solution</summary>

```python
b = a.reshape(3, 20)                  # (3, 20) — flatten last two dims
c = a.permute(2, 0, 1)                # (5, 3, 4) — reorder dims: 2→0, 0→1, 1→2
d = a.transpose(1, 2)                 # (3, 5, 4) — swap dims 1 and 2
e = a[:, :2, :]                       # (3, 2, 5) — slice dim 1 to first 2
f = a.unsqueeze(0)                    # (1, 3, 4, 5) — add dim at position 0
g = a.flatten(1)                      # (3, 20) — flatten from dim 1 onward
h = a.sum(dim=1)                      # (3, 5) — reduce dim 1
i = a.mean(dim=-1, keepdim=True)      # (3, 4, 1) — reduce last dim, keep it
j = a[0]                              # (4, 5) — select first element along dim 0
```

**Key insight**: `keepdim=True` preserves the reduced dimension as size 1, which is essential for broadcasting the result back to the original shape.
</details>

---

## Exercise 2: Broadcasting Compatibility

For each pair of shapes, determine whether broadcasting is valid. If valid, state the output shape. If invalid, explain why.

```
1. (3, 4) + (4,)
2. (2, 3, 4) + (3, 4)
3. (2, 3, 4) + (2, 4)
4. (5, 1) + (1, 3)
5. (2, 3) + (3, 2)
6. (1, 5, 1) + (3, 1, 4)
7. (8, 1, 6, 1) + (7, 1, 5)
```

<details>
<summary>Solution</summary>

```
1. (3, 4) + (4,)        → (3, 4)       ✓  (4,) becomes (1, 4), then broadcasts
2. (2, 3, 4) + (3, 4)   → (2, 3, 4)    ✓  (3, 4) becomes (1, 3, 4)
3. (2, 3, 4) + (2, 4)   → INVALID       ✗  dim 1: 3 vs no match (2,4 left-pads to 1,2,4 — dim 1: 3 vs 2, neither is 1)
   Actually: (2, 4) → (1, 2, 4). Compare: (2, 3, 4) vs (1, 2, 4).
   dim 0: 2 vs 1 → OK. dim 1: 3 vs 2 → FAIL (neither is 1).
4. (5, 1) + (1, 3)      → (5, 3)       ✓  both dims broadcast
5. (2, 3) + (3, 2)      → INVALID       ✗  dim 0: 2 vs 3 (neither is 1), dim 1: 3 vs 2 (neither is 1)
6. (1, 5, 1) + (3, 1, 4)→ (3, 5, 4)    ✓  each dim: max(1,3)=3, max(5,1)=5, max(1,4)=4
7. (8, 1, 6, 1) + (7, 1, 5) → INVALID  ✗  (7,1,5) becomes (1,7,1,5). dim 1: 1 vs 7 OK, dim 2: 6 vs 1 OK, dim 3: 1 vs 5 OK, dim 0: 8 vs 1 OK → (8, 7, 6, 5) ✓

Correction for #7: (7, 1, 5) left-pads to (1, 7, 1, 5).
Compare with (8, 1, 6, 1):
  dim 0: 8 vs 1 → 8 ✓
  dim 1: 1 vs 7 → 7 ✓
  dim 2: 6 vs 1 → 6 ✓
  dim 3: 1 vs 5 → 5 ✓
Result: (8, 7, 6, 5) ✓ — VALID
```

**Key insight**: Always align dimensions from the right, then check each pair. Both must be equal, or one must be 1.
</details>

---

## Exercise 3: NumPy Interop Bug

What is wrong with this code? Fix it.

```python
import numpy as np
import torch

# Load data
data = np.load('features.npy')       # shape: (1000, 784), dtype: float64
labels = np.load('labels.npy')       # shape: (1000,), dtype: int32

# Convert to tensors
X = torch.from_numpy(data)
y = torch.from_numpy(labels)

# Pass through model
model = torch.nn.Linear(784, 10)
output = model(X)
loss = torch.nn.functional.cross_entropy(output, y)
```

<details>
<summary>Solution</summary>

Two bugs:

1. **dtype mismatch on X**: `nn.Linear` weights are `float32`, but `X` is `float64` (from NumPy's default). PyTorch will raise a RuntimeError about dtype mismatch.

2. **dtype mismatch on y**: `cross_entropy` expects `torch.long` (int64) labels, but `labels` is int32.

Fixed code:

```python
X = torch.from_numpy(data).float()          # float64 → float32
y = torch.from_numpy(labels).long()          # int32 → int64

output = model(X)
loss = torch.nn.functional.cross_entropy(output, y)
```

**Key insight**: Always cast tensors to the correct dtype. Models use `float32` by default; classification labels must be `long` (int64).
</details>

---

## Exercise 4: Implement Softmax with Tensors

Implement softmax from scratch using only tensor operations (no `torch.softmax` or `torch.nn.functional.softmax`). Handle numerical stability.

```python
def my_softmax(logits, dim=-1):
    """
    Compute softmax along the given dimension.
    Args:
        logits: tensor of any shape
        dim: dimension along which to compute softmax
    Returns:
        Tensor of same shape with softmax applied along dim
    """
    # YOUR CODE HERE
    pass

# Test
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 3.0]])
result = my_softmax(logits, dim=-1)
expected = torch.softmax(logits, dim=-1)
assert torch.allclose(result, expected, atol=1e-6)
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
def my_softmax(logits, dim=-1):
    # Numerical stability: subtract max to prevent overflow in exp
    max_vals = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_vals                  # max is now 0, all others negative
    exp_vals = torch.exp(shifted)                # no overflow possible
    return exp_vals / exp_vals.sum(dim=dim, keepdim=True)
```

**Key insight**: Without the max subtraction, `exp(1000)` overflows to infinity. Subtracting the max does not change the result (it cancels in numerator and denominator) but prevents overflow. This is the "log-sum-exp trick" applied to softmax.

**Why `keepdim=True`?** The max and sum reduce one dimension. With `keepdim=True`, the result shape retains that dimension as size 1, enabling broadcasting against the original tensor.
</details>

---

## Exercise 5: Tensor Puzzle — Matrix Multiplication Without @

Compute the matrix product $C = AB$ where $A$ is `(M, K)` and $B$ is `(K, N)`, using only element-wise operations, `sum`, `unsqueeze`, and broadcasting. No `@`, `torch.mm`, `torch.matmul`, or `torch.einsum`.

```python
def matmul_broadcast(A, B):
    """
    Matrix multiplication using broadcasting.
    A: (M, K)
    B: (K, N)
    Returns: (M, N)
    """
    # YOUR CODE HERE
    pass

# Test
A = torch.randn(3, 4)
B = torch.randn(4, 5)
result = matmul_broadcast(A, B)
expected = A @ B
assert torch.allclose(result, expected, atol=1e-5)
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
def matmul_broadcast(A, B):
    # A: (M, K) → (M, K, 1)
    # B: (K, N) → (1, K, N)
    # Element-wise multiply broadcasts to (M, K, N)
    # Sum over K dimension → (M, N)
    return (A.unsqueeze(2) * B.unsqueeze(0)).sum(dim=1)
```

Step by step:
- `A.unsqueeze(2)`: shape `(M, K)` → `(M, K, 1)`
- `B.unsqueeze(0)`: shape `(K, N)` → `(1, K, N)`
- Multiply: `(M, K, 1) * (1, K, N)` → `(M, K, N)` via broadcasting
- Sum over dim 1 (K): `(M, K, N)` → `(M, N)`

This is exactly $C_{ij} = \sum_k A_{ik} B_{kj}$.

**Key insight**: Broadcasting + sum can express any tensor contraction (Einstein summation). Understanding this pattern is powerful for implementing custom operations without loops.
</details>

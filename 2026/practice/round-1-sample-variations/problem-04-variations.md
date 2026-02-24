# Problem 4 Variations: NumPy Array Manipulation (EXHAUSTIVE)

> **Original Problem**: Generate random array, squeeze, expand_dims, swapaxes, boolean indexing, flatten
> **Core Skills**: Array creation, shape manipulation, axis operations, conditional assignment, reshaping
> **Unit**: 03 (AI Programming in Python - NumPy)

---

## ORIGINAL PROBLEM (Reference)

```python
import numpy as np

# 4.1: Generate shape (5, 8, 3, 1, 2) with seed 2026
np.random.seed(2026)
arr = np.random.randn(5, 8, 3, 1, 2)

# 4.2: Remove dimension of length 1
arr = np.squeeze(arr)  # Shape: (5, 8, 3, 2)

# 4.3: Insert new dimension at axis 2
arr = np.expand_dims(arr, axis=2)  # Shape: (5, 8, 1, 3, 2)

# 4.4: Swap axes 0 and 1
arr = np.swapaxes(arr, 0, 1)  # Shape: (8, 5, 1, 3, 2)

# 4.5: Set values > 1 to 100
arr[arr > 1] = 100

# 4.6: Flatten
arr = arr.flatten()
```

---

## CATEGORY A: Different Values (Same Structure)

### Variation A1: Different Shape and Seed

Generate a NumPy array with shape (3, 4, 2, 1, 5). Each entry is a standard normal. Use random seed 42.

**Part 4.1**: Generate the array.
**Part 4.2**: Remove all dimensions of length 1.
**Part 4.3**: Insert a new dimension at axis 0.
**Part 4.4**: Swap axes 1 and 2.
**Part 4.5**: For entries below -1, reset values to -100.
**Part 4.6**: Flatten the array.

<details>
<summary>Solution A1</summary>

```python
import numpy as np

# 4.1
np.random.seed(42)
arr = np.random.randn(3, 4, 2, 1, 5)
print(f"4.1 Shape: {arr.shape}")  # (3, 4, 2, 1, 5)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (3, 4, 2, 5)

# 4.3
arr = np.expand_dims(arr, axis=0)
print(f"4.3 Shape: {arr.shape}")  # (1, 3, 4, 2, 5)

# 4.4
arr = np.swapaxes(arr, 1, 2)
print(f"4.4 Shape: {arr.shape}")  # (1, 4, 3, 2, 5)

# 4.5
arr[arr < -1] = -100

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (120,)
```

**Key insight**: `np.squeeze()` removes ALL dimensions of length 1, not just one specific dimension.
</details>

### Variation A2: Uniform Distribution

Generate a NumPy array with shape (4, 1, 6, 3, 1). Each entry is uniformly distributed between 0 and 10. Use random seed 2025.

**Part 4.1**: Generate the array using `np.random.uniform`.
**Part 4.2**: Remove dimensions of length 1.
**Part 4.3**: Insert a new dimension at axis -1 (last position).
**Part 4.4**: Swap axes 0 and 2.
**Part 4.5**: For entries between 3 and 7 (inclusive), set to 5.
**Part 4.6**: Flatten using row-major order.

<details>
<summary>Solution A2</summary>

```python
import numpy as np

# 4.1
np.random.seed(2025)
arr = np.random.uniform(0, 10, size=(4, 1, 6, 3, 1))
print(f"4.1 Shape: {arr.shape}")  # (4, 1, 6, 3, 1)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (4, 6, 3)

# 4.3
arr = np.expand_dims(arr, axis=-1)
print(f"4.3 Shape: {arr.shape}")  # (4, 6, 3, 1)

# 4.4
arr = np.swapaxes(arr, 0, 2)
print(f"4.4 Shape: {arr.shape}")  # (3, 6, 4, 1)

# 4.5
arr[(arr >= 3) & (arr <= 7)] = 5

# 4.6
arr = arr.flatten(order='C')  # 'C' is row-major (default)
print(f"4.6 Shape: {arr.shape}")  # (72,)
```

**Key insight**: Boolean conditions can be combined with `&` (and) and `|` (or). Remember parentheses!
</details>

### Variation A3: Integer Array

Generate a NumPy array with shape (2, 3, 1, 4, 2). Each entry is a random integer from 0 to 99. Use random seed 123.

**Part 4.1**: Generate the array using `np.random.randint`.
**Part 4.2**: Squeeze the array.
**Part 4.3**: Insert a new dimension at axis 1.
**Part 4.4**: Swap axes 2 and 3.
**Part 4.5**: Set all even numbers to 0.
**Part 4.6**: Flatten the array.

<details>
<summary>Solution A3</summary>

```python
import numpy as np

# 4.1
np.random.seed(123)
arr = np.random.randint(0, 100, size=(2, 3, 1, 4, 2))
print(f"4.1 Shape: {arr.shape}")  # (2, 3, 1, 4, 2)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (2, 3, 4, 2)

# 4.3
arr = np.expand_dims(arr, axis=1)
print(f"4.3 Shape: {arr.shape}")  # (2, 1, 3, 4, 2)

# 4.4
arr = np.swapaxes(arr, 2, 3)
print(f"4.4 Shape: {arr.shape}")  # (2, 1, 4, 3, 2)

# 4.5
arr[arr % 2 == 0] = 0

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (48,)
```

**Key insight**: `arr % 2 == 0` creates a boolean mask for even numbers. This works element-wise on arrays.
</details>

### Variation A4: Exponential Distribution

Generate a NumPy array with shape (1, 5, 4, 2, 1). Each entry is drawn from exponential distribution with scale=2. Use random seed 999.

**Part 4.1**: Generate the array.
**Part 4.2**: Squeeze all unit dimensions.
**Part 4.3**: Insert dimension at axis 3.
**Part 4.4**: Swap axes 0 and 1.
**Part 4.5**: Cap all values at 5 (values > 5 become 5).
**Part 4.6**: Flatten the array.

<details>
<summary>Solution A4</summary>

```python
import numpy as np

# 4.1
np.random.seed(999)
arr = np.random.exponential(scale=2, size=(1, 5, 4, 2, 1))
print(f"4.1 Shape: {arr.shape}")  # (1, 5, 4, 2, 1)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (5, 4, 2)

# 4.3
arr = np.expand_dims(arr, axis=3)
print(f"4.3 Shape: {arr.shape}")  # (5, 4, 2, 1)

# 4.4
arr = np.swapaxes(arr, 0, 1)
print(f"4.4 Shape: {arr.shape}")  # (4, 5, 2, 1)

# 4.5
arr[arr > 5] = 5
# Alternative: arr = np.minimum(arr, 5)  # More elegant "capping"

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (40,)
```

**Key insight**: `np.minimum(arr, value)` is a cleaner way to cap values. Similarly, `np.maximum` for flooring.
</details>

### Variation A5: Binomial Distribution

Generate a NumPy array with shape (6, 1, 3, 2, 1). Each entry is the number of successes in 10 trials with p=0.5. Use random seed 2024.

**Part 4.1**: Generate using `np.random.binomial(n=10, p=0.5, size=...)`.
**Part 4.2**: Remove unit dimensions.
**Part 4.3**: Insert dimension at the end.
**Part 4.4**: Swap axes 0 and 2.
**Part 4.5**: Replace values equal to 5 with -1.
**Part 4.6**: Flatten.

<details>
<summary>Solution A5</summary>

```python
import numpy as np

# 4.1
np.random.seed(2024)
arr = np.random.binomial(n=10, p=0.5, size=(6, 1, 3, 2, 1))
print(f"4.1 Shape: {arr.shape}")  # (6, 1, 3, 2, 1)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (6, 3, 2)

# 4.3
arr = np.expand_dims(arr, axis=-1)
print(f"4.3 Shape: {arr.shape}")  # (6, 3, 2, 1)

# 4.4
arr = np.swapaxes(arr, 0, 2)
print(f"4.4 Shape: {arr.shape}")  # (2, 3, 6, 1)

# 4.5
arr[arr == 5] = -1

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (36,)
```

**Key insight**: `arr == value` creates a boolean mask for exact equality matching.
</details>

---

## CATEGORY B: Different Dimensions and Complexities

### Variation B1: 6D Array

Generate a 6D array with shape (2, 3, 1, 4, 1, 2). Use seed 100.

**Part 4.1**: Generate standard normal array.
**Part 4.2**: Remove ALL unit dimensions (should remove two).
**Part 4.3**: Insert two new dimensions: one at axis 0 and one at axis 2.
**Part 4.4**: Swap axes 1 and 3.
**Part 4.5**: Set values in range (-0.5, 0.5) to 0.
**Part 4.6**: Report final shape, then flatten.

<details>
<summary>Solution B1</summary>

```python
import numpy as np

# 4.1
np.random.seed(100)
arr = np.random.randn(2, 3, 1, 4, 1, 2)
print(f"4.1 Shape: {arr.shape}")  # (2, 3, 1, 4, 1, 2)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (2, 3, 4, 2)

# 4.3
arr = np.expand_dims(arr, axis=0)
arr = np.expand_dims(arr, axis=2)
print(f"4.3 Shape: {arr.shape}")  # (1, 2, 1, 3, 4, 2)

# 4.4
arr = np.swapaxes(arr, 1, 3)
print(f"4.4 Shape: {arr.shape}")  # (1, 3, 1, 2, 4, 2)

# 4.5
arr[(arr > -0.5) & (arr < 0.5)] = 0

# 4.6
print(f"Final shape before flatten: {arr.shape}")  # (1, 3, 1, 2, 4, 2)
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (48,)
```

**Key insight**: Multiple `expand_dims` calls can add multiple dimensions. Order matters—each call shifts subsequent axis indices.
</details>

### Variation B2: 3D Array (Simpler)

Generate a 3D array with shape (10, 1, 5). Use seed 50.

**Part 4.1**: Generate uniform [0, 1) using `np.random.rand`.
**Part 4.2**: Squeeze.
**Part 4.3**: Add dimension at axis 1.
**Part 4.4**: Swap axes 0 and 2.
**Part 4.5**: Multiply all values > 0.8 by 10.
**Part 4.6**: Flatten.

<details>
<summary>Solution B2</summary>

```python
import numpy as np

# 4.1
np.random.seed(50)
arr = np.random.rand(10, 1, 5)
print(f"4.1 Shape: {arr.shape}")  # (10, 1, 5)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (10, 5)

# 4.3
arr = np.expand_dims(arr, axis=1)
print(f"4.3 Shape: {arr.shape}")  # (10, 1, 5)

# 4.4
arr = np.swapaxes(arr, 0, 2)
print(f"4.4 Shape: {arr.shape}")  # (5, 1, 10)

# 4.5
arr[arr > 0.8] *= 10
# This modifies in-place using the boolean mask

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (50,)
```

**Key insight**: `arr[mask] *= 10` is valid! You can use compound assignment operators with boolean indexing.
</details>

### Variation B3: 2D Array (Minimal)

Generate a 2D array with shape (1, 8). Use seed 7.

**Part 4.1**: Generate standard normal.
**Part 4.2**: Squeeze.
**Part 4.3**: Add dimension at axis 0 and axis 1 (making it 3D again).
**Part 4.4**: Swap axes 0 and 1.
**Part 4.5**: Negate all positive values.
**Part 4.6**: Flatten.

<details>
<summary>Solution B3</summary>

```python
import numpy as np

# 4.1
np.random.seed(7)
arr = np.random.randn(1, 8)
print(f"4.1 Shape: {arr.shape}")  # (1, 8)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (8,) — 1D!

# 4.3
arr = np.expand_dims(arr, axis=0)  # (1, 8)
arr = np.expand_dims(arr, axis=1)  # (1, 1, 8)
print(f"4.3 Shape: {arr.shape}")  # (1, 1, 8)

# 4.4
arr = np.swapaxes(arr, 0, 1)
print(f"4.4 Shape: {arr.shape}")  # (1, 1, 8) — no change since both dims are 1

# 4.5
arr[arr > 0] = -arr[arr > 0]
# Or equivalently: arr[arr > 0] *= -1

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (8,)
```

**Key insight**: Squeezing a (1, 8) array gives a 1D array of shape (8,), not (8,) vs (8, 1).
</details>

### Variation B4: Large Array

Generate an array with shape (100, 50, 1, 20). Use seed 2026.

**Part 4.1**: Generate standard normal.
**Part 4.2**: Squeeze.
**Part 4.3**: Insert dimension at axis 2.
**Part 4.4**: Transpose (reverse all axes).
**Part 4.5**: Set the top 1% of values (> 99th percentile) to the mean.
**Part 4.6**: Flatten and report the total number of elements.

<details>
<summary>Solution B4</summary>

```python
import numpy as np

# 4.1
np.random.seed(2026)
arr = np.random.randn(100, 50, 1, 20)
print(f"4.1 Shape: {arr.shape}")  # (100, 50, 1, 20)

# 4.2
arr = np.squeeze(arr)
print(f"4.2 Shape: {arr.shape}")  # (100, 50, 20)

# 4.3
arr = np.expand_dims(arr, axis=2)
print(f"4.3 Shape: {arr.shape}")  # (100, 50, 1, 20)

# 4.4
arr = arr.transpose()  # or np.transpose(arr)
# Equivalent to reversing all axes: arr.transpose(3, 2, 1, 0)
print(f"4.4 Shape: {arr.shape}")  # (20, 1, 50, 100)

# 4.5
percentile_99 = np.percentile(arr, 99)
mean_val = np.mean(arr)
arr[arr > percentile_99] = mean_val

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (100000,)
print(f"Total elements: {arr.size}")  # 100000
```

**Key insight**: `arr.transpose()` without arguments reverses all axes. With arguments, you can specify any permutation.
</details>

---

## CATEGORY C: Alternative Methods (Same Results)

### Variation C1: Using reshape instead of squeeze/expand_dims

Redo the original problem using only `reshape` and `transpose`.

**Part 4.1**: Generate (5, 8, 3, 1, 2) with seed 2026.
**Part 4.2**: Use reshape to remove the size-1 dimension.
**Part 4.3**: Use reshape to add a dimension.
**Part 4.4**: Use transpose instead of swapaxes.
**Part 4.5**: Boolean indexing (same as original).
**Part 4.6**: Reshape to 1D.

<details>
<summary>Solution C1</summary>

```python
import numpy as np

# 4.1
np.random.seed(2026)
arr = np.random.randn(5, 8, 3, 1, 2)

# 4.2: squeeze equivalent via reshape
arr = arr.reshape(5, 8, 3, 2)  # Must know which dim to remove

# 4.3: expand_dims equivalent via reshape
arr = arr.reshape(5, 8, 1, 3, 2)

# 4.4: swapaxes equivalent via transpose
arr = arr.transpose(1, 0, 2, 3, 4)  # Swap positions 0 and 1

# 4.5
arr[arr > 1] = 100

# 4.6: flatten equivalent via reshape
arr = arr.reshape(-1)  # -1 means "infer this dimension"
```

**Key insight**: `reshape(-1)` is equivalent to `flatten()`. The `-1` tells NumPy to compute the size automatically.
</details>

### Variation C2: Using slicing for dimension manipulation

**Part 4.1**: Generate (5, 8, 3, 1, 2) with seed 2026.
**Part 4.2**: Use `arr[:, :, :, 0, :]` to "squeeze" axis 3.
**Part 4.3**: Use `arr[:, :, np.newaxis, :, :]` to add axis.
**Part 4.4**: Use transpose.
**Part 4.5**: Use np.where for conditional assignment.
**Part 4.6**: Use ravel instead of flatten.

<details>
<summary>Solution C2</summary>

```python
import numpy as np

# 4.1
np.random.seed(2026)
arr = np.random.randn(5, 8, 3, 1, 2)

# 4.2: Index into the size-1 dimension
arr = arr[:, :, :, 0, :]  # Shape: (5, 8, 3, 2)

# 4.3: Use np.newaxis (alias for None)
arr = arr[:, :, np.newaxis, :, :]  # Shape: (5, 8, 1, 3, 2)

# 4.4
arr = arr.transpose(1, 0, 2, 3, 4)

# 4.5: np.where version
arr = np.where(arr > 1, 100, arr)

# 4.6: ravel returns a view when possible (more efficient)
arr = arr.ravel()
```

**Key insight**:
- `np.newaxis` (or `None`) adds a dimension: `arr[:, np.newaxis]` turns (n,) into (n, 1)
- `ravel()` returns a view when possible; `flatten()` always returns a copy
</details>

### Variation C3: Functional style with np.squeeze axis parameter

**Part 4.1**: Generate (5, 1, 3, 1, 2) with seed 2026.
**Part 4.2**: Squeeze only axis 1 (not axis 3).
**Part 4.3**: Expand at axis 0.
**Part 4.4**: Swap axes using negative indices.
**Part 4.5**: Use np.clip for value capping.
**Part 4.6**: Flatten.

<details>
<summary>Solution C3</summary>

```python
import numpy as np

# 4.1
np.random.seed(2026)
arr = np.random.randn(5, 1, 3, 1, 2)
print(f"4.1 Shape: {arr.shape}")  # (5, 1, 3, 1, 2)

# 4.2: Squeeze ONLY axis 1
arr = np.squeeze(arr, axis=1)
print(f"4.2 Shape: {arr.shape}")  # (5, 3, 1, 2) — axis 3 still has size 1!

# 4.3
arr = np.expand_dims(arr, axis=0)
print(f"4.3 Shape: {arr.shape}")  # (1, 5, 3, 1, 2)

# 4.4: Negative indices work!
arr = np.swapaxes(arr, -1, -2)  # Swap last two axes
print(f"4.4 Shape: {arr.shape}")  # (1, 5, 3, 2, 1)

# 4.5: Clip values to [-1, 1]
arr = np.clip(arr, -1, 1)

# 4.6
arr = arr.flatten()
print(f"4.6 Shape: {arr.shape}")  # (30,)
```

**Key insight**: `np.squeeze(arr, axis=k)` only removes dimension at axis k, and raises an error if that dimension isn't size 1.
</details>

---

## CATEGORY D: Boolean Indexing Variations

### Variation D1: Multiple Conditions

Generate (4, 5, 3) standard normal with seed 11.

**Part 4.1**: Set values > 2 to 100.
**Part 4.2**: Set values < -2 to -100.
**Part 4.3**: Set values between -0.5 and 0.5 to 0.
**Part 4.4**: Count how many values are now exactly 0, 100, and -100.
**Part 4.5**: What percentage of values were "extreme" (|x| > 2)?

<details>
<summary>Solution D1</summary>

```python
import numpy as np

np.random.seed(11)
arr = np.random.randn(4, 5, 3)

# 4.1
arr[arr > 2] = 100

# 4.2
arr[arr < -2] = -100

# 4.3
arr[(arr > -0.5) & (arr < 0.5)] = 0

# 4.4
count_zero = np.sum(arr == 0)
count_100 = np.sum(arr == 100)
count_neg100 = np.sum(arr == -100)
print(f"Zeros: {count_zero}, 100s: {count_100}, -100s: {count_neg100}")

# 4.5
total = arr.size  # 60
extreme = count_100 + count_neg100
percentage = 100 * extreme / total
print(f"Extreme values: {percentage:.2f}%")
```

**Key insight**: For standard normal, ~2.3% of values exceed |2| (by the 68-95-99.7 rule, about 5% exceed |2|, split between tails).
</details>

### Variation D2: Indexing with computed masks

Generate (3, 4, 5) uniform [0, 10] with seed 22.

**Part 4.1**: Create mask for values above the mean.
**Part 4.2**: Create mask for values above the median.
**Part 4.3**: Set values that are above mean BUT below median to 5.
**Part 4.4**: Explain why this might be empty or non-empty.

<details>
<summary>Solution D2</summary>

```python
import numpy as np

np.random.seed(22)
arr = np.random.uniform(0, 10, size=(3, 4, 5))

# 4.1
mean_val = np.mean(arr)
mask_above_mean = arr > mean_val
print(f"Mean: {mean_val:.3f}")
print(f"Above mean: {np.sum(mask_above_mean)} values")

# 4.2
median_val = np.median(arr)
mask_above_median = arr > median_val
print(f"Median: {median_val:.3f}")
print(f"Above median: {np.sum(mask_above_median)} values")

# 4.3
mask_between = (arr > mean_val) & (arr < median_val)
arr[mask_between] = 5
print(f"Values above mean but below median: {np.sum(mask_between)}")

# 4.4: Explanation
# For symmetric distributions (like uniform), mean ≈ median
# So there are few values strictly between them
# If mean < median: some values satisfy the condition
# If mean > median: the condition is impossible (empty set)
```

**Key insight**: For symmetric distributions, mean ≈ median. For skewed data, they differ, and this mask becomes interesting.
</details>

### Variation D3: NaN handling

Generate (5, 5) standard normal with seed 33. Insert some NaN values.

**Part 4.1**: Set 5 random positions to NaN.
**Part 4.2**: Replace NaN values with 0.
**Part 4.3**: Alternative: Replace NaN with the mean of non-NaN values.

<details>
<summary>Solution D3</summary>

```python
import numpy as np

np.random.seed(33)
arr = np.random.randn(5, 5)

# 4.1: Insert NaN at random positions
nan_indices = np.random.choice(25, size=5, replace=False)
arr.flat[nan_indices] = np.nan
print(f"NaN count: {np.sum(np.isnan(arr))}")

# 4.2: Replace NaN with 0
arr_copy1 = arr.copy()
arr_copy1[np.isnan(arr_copy1)] = 0
print(f"After NaN→0: NaN count = {np.sum(np.isnan(arr_copy1))}")

# 4.3: Replace NaN with mean of non-NaN values
arr_copy2 = arr.copy()
mean_non_nan = np.nanmean(arr_copy2)  # Ignores NaN
arr_copy2[np.isnan(arr_copy2)] = mean_non_nan
print(f"Mean of non-NaN: {mean_non_nan:.3f}")
```

**Key insight**:
- `np.isnan(arr)` returns boolean mask for NaN values
- `np.nanmean()`, `np.nanstd()`, etc. ignore NaN values in computation
</details>

### Variation D4: Using np.where for conditional selection

Generate (4, 4) integers 0-9 with seed 44.

**Part 4.1**: Use np.where to get indices of values > 5.
**Part 4.2**: Use np.where to create new array: if > 5, keep value; else set to 0.
**Part 4.3**: Use np.where to create new array: if > 5, set to "high"; else "low" (string array).

<details>
<summary>Solution D4</summary>

```python
import numpy as np

np.random.seed(44)
arr = np.random.randint(0, 10, size=(4, 4))
print("Original:\n", arr)

# 4.1: Get indices (returns tuple of arrays)
indices = np.where(arr > 5)
print(f"Indices where > 5: rows={indices[0]}, cols={indices[1]}")
# These can be used to index: arr[indices] gives the values

# 4.2: Conditional assignment (returns new array)
new_arr = np.where(arr > 5, arr, 0)
print("High values kept, others zeroed:\n", new_arr)

# 4.3: String labels (creates object array)
labels = np.where(arr > 5, "high", "low")
print("Labels:\n", labels)
```

**Key insight**: `np.where(condition)` returns indices; `np.where(condition, x, y)` returns array with x where True, y where False.
</details>

### Variation D5: Modifying specific axes

Generate (3, 4, 5) standard normal with seed 55.

**Part 4.1**: For each row (axis 0), find the maximum value.
**Part 4.2**: Set all maximum values in each row to 999.
**Part 4.3**: Set all values in column 0 (axis 1) to -1.

<details>
<summary>Solution D5</summary>

```python
import numpy as np

np.random.seed(55)
arr = np.random.randn(3, 4, 5)

# 4.1: Max along rows (for each of the 3 "rows")
row_maxes = arr.max(axis=(1, 2))  # Max over the 4×5 slice for each row
print(f"Max per row: {row_maxes}")

# 4.2: This is tricky—need to find WHERE each max occurs
for i in range(3):
    slice_i = arr[i]
    max_val = slice_i.max()
    arr[i][slice_i == max_val] = 999

# 4.3: Set all values in first column to -1
arr[:, 0, :] = -1  # All rows, column 0, all depths
```

**Key insight**: Axis operations in NumPy require understanding which dimensions you're collapsing vs. preserving.
</details>

---

## CATEGORY E: Transpose and Axis Operations

### Variation E1: Full transpose permutation

Generate (2, 3, 4, 5) with seed 66.

**Part 4.1**: Transpose to (5, 4, 3, 2).
**Part 4.2**: Transpose to (3, 5, 2, 4).
**Part 4.3**: Use moveaxis to move axis 0 to position 3.
**Part 4.4**: Use rollaxis to roll axis 2 to position 0.

<details>
<summary>Solution E1</summary>

```python
import numpy as np

np.random.seed(66)
arr = np.random.randn(2, 3, 4, 5)
print(f"Original: {arr.shape}")

# 4.1: Reverse all axes
arr1 = arr.transpose(3, 2, 1, 0)
print(f"4.1: {arr1.shape}")  # (5, 4, 3, 2)

# 4.2: Custom permutation
arr2 = arr.transpose(1, 3, 0, 2)
print(f"4.2: {arr2.shape}")  # (3, 5, 2, 4)

# 4.3: moveaxis
arr3 = np.moveaxis(arr, 0, 3)  # Move axis 0 to position 3
print(f"4.3: {arr3.shape}")  # (3, 4, 5, 2)

# 4.4: rollaxis (deprecated but still works)
arr4 = np.rollaxis(arr, 2, 0)  # Roll axis 2 to position 0
print(f"4.4: {arr4.shape}")  # (4, 2, 3, 5)
```

**Key insight**:
- `transpose(perm)`: axis i of output comes from axis perm[i] of input
- `moveaxis(a, src, dst)`: moves axis from src to dst, shifting others
- `rollaxis` is older; prefer `moveaxis`
</details>

### Variation E2: Swapping multiple axes

Generate (2, 3, 4, 5, 6) with seed 77.

**Part 4.1**: Swap axes 0 and 4.
**Part 4.2**: Swap axes 1 and 3.
**Part 4.3**: Verify that swapping twice returns original shape.

<details>
<summary>Solution E2</summary>

```python
import numpy as np

np.random.seed(77)
arr = np.random.randn(2, 3, 4, 5, 6)
print(f"Original: {arr.shape}")

# 4.1
arr1 = np.swapaxes(arr, 0, 4)
print(f"Swap 0↔4: {arr1.shape}")  # (6, 3, 4, 5, 2)

# 4.2
arr2 = np.swapaxes(arr, 1, 3)
print(f"Swap 1↔3: {arr2.shape}")  # (2, 5, 4, 3, 6)

# 4.3: Double swap returns original
arr3 = np.swapaxes(arr, 0, 4)
arr3 = np.swapaxes(arr3, 0, 4)
print(f"Double swap: {arr3.shape}")  # (2, 3, 4, 5, 6)
assert arr3.shape == arr.shape
```

**Key insight**: `swapaxes` is its own inverse—swapping twice restores the original.
</details>

### Variation E3: Using einsum for transposition

Generate (3, 4, 5) with seed 88.

**Part 4.1**: Use `np.einsum` to transpose to (5, 4, 3).
**Part 4.2**: Use `np.einsum` to transpose to (4, 3, 5).
**Part 4.3**: Verify against `transpose`.

<details>
<summary>Solution E3</summary>

```python
import numpy as np

np.random.seed(88)
arr = np.random.randn(3, 4, 5)

# 4.1: 'ijk->kji' reverses axes
arr1 = np.einsum('ijk->kji', arr)
print(f"4.1 einsum: {arr1.shape}")  # (5, 4, 3)

# 4.2: 'ijk->jik' swaps first two
arr2 = np.einsum('ijk->jik', arr)
print(f"4.2 einsum: {arr2.shape}")  # (4, 3, 5)

# 4.3: Verify
assert np.allclose(arr1, arr.transpose(2, 1, 0))
assert np.allclose(arr2, arr.transpose(1, 0, 2))
print("Verified!")
```

**Key insight**: `einsum` is extremely powerful—it can do transposition, summation, matrix multiplication, and more in a single call.
</details>

---

## CATEGORY F: Flatten vs Ravel vs Reshape(-1)

### Variation F1: Memory behavior

**Part 4.1**: Create array and check if ravel shares memory.
**Part 4.2**: Check if flatten shares memory.
**Part 4.3**: When does ravel NOT share memory?

<details>
<summary>Solution F1</summary>

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 4.1: ravel usually returns a view
r = arr.ravel()
print(f"ravel shares memory: {np.shares_memory(arr, r)}")  # True

r[0] = 999
print(f"Original changed: {arr[0, 0]}")  # 999 — proves it's a view

# 4.2: flatten always returns a copy
arr = np.array([[1, 2, 3], [4, 5, 6]])
f = arr.flatten()
print(f"flatten shares memory: {np.shares_memory(arr, f)}")  # False

f[0] = 999
print(f"Original unchanged: {arr[0, 0]}")  # 1

# 4.3: ravel doesn't share memory when array is non-contiguous
arr = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr.T
r = transposed.ravel()
print(f"Transposed ravel shares memory: {np.shares_memory(transposed, r)}")
# May be False because transposed array isn't contiguous
```

**Key insight**:
- `ravel()` returns a view when possible (memory efficient)
- `flatten()` always returns a copy (safe for modification)
- Non-contiguous arrays force `ravel()` to copy
</details>

### Variation F2: Order parameter

Generate (2, 3, 4) with values 0-23 using `np.arange(24).reshape(2, 3, 4)`.

**Part 4.1**: Flatten with 'C' order (row-major, default).
**Part 4.2**: Flatten with 'F' order (column-major, Fortran).
**Part 4.3**: Explain the difference.

<details>
<summary>Solution F2</summary>

```python
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)
print("Original shape:", arr.shape)
print("arr[0]:\n", arr[0])
print("arr[1]:\n", arr[1])

# 4.1: C order (row-major) — last axis varies fastest
c_flat = arr.flatten(order='C')
print(f"C order: {c_flat[:12]}")  # [0 1 2 3 4 5 6 7 8 9 10 11]

# 4.2: F order (column-major) — first axis varies fastest
f_flat = arr.flatten(order='F')
print(f"F order: {f_flat[:12]}")  # [0 12 4 16 8 20 1 13 5 17 9 21]

# 4.3: Explanation
# C order: arr[0,0,0], arr[0,0,1], arr[0,0,2], arr[0,0,3], arr[0,1,0], ...
# F order: arr[0,0,0], arr[1,0,0], arr[0,1,0], arr[1,1,0], arr[0,2,0], ...
```

**Key insight**:
- **C order** (row-major): rightmost index changes fastest — how data is stored in memory by default
- **F order** (column-major): leftmost index changes fastest — Fortran convention
</details>

---

## CATEGORY G: Edge Cases and Tricky Situations

### Variation G1: Empty arrays

**Part 4.1**: Create empty array with shape (0, 5).
**Part 4.2**: Squeeze it.
**Part 4.3**: What happens when you flatten?

<details>
<summary>Solution G1</summary>

```python
import numpy as np

# 4.1
arr = np.empty((0, 5))
print(f"Original shape: {arr.shape}")  # (0, 5)
print(f"Size: {arr.size}")  # 0

# 4.2
squeezed = np.squeeze(arr)
print(f"Squeezed shape: {squeezed.shape}")  # (0, 5) — no change! No size-1 dims

# 4.3
flat = arr.flatten()
print(f"Flattened shape: {flat.shape}")  # (0,)
print(f"Flat array: {flat}")  # []
```

**Key insight**: Empty arrays (with a 0 dimension) are valid and propagate through operations. They're useful for handling edge cases without special code.
</details>

### Variation G2: Scalar extraction

**Part 4.1**: Create (1, 1, 1) array with single value 42.
**Part 4.2**: Squeeze to scalar.
**Part 4.3**: What's the type difference between 0-d array and Python scalar?

<details>
<summary>Solution G2</summary>

```python
import numpy as np

# 4.1
arr = np.array([[[42]]])
print(f"Shape: {arr.shape}")  # (1, 1, 1)

# 4.2
scalar = np.squeeze(arr)
print(f"Squeezed shape: {scalar.shape}")  # () — 0-dimensional!
print(f"Value: {scalar}")  # 42

# 4.3: Type difference
print(f"Type of scalar: {type(scalar)}")  # numpy.int64 (0-d array)
print(f"Type of scalar.item(): {type(scalar.item())}")  # int (Python scalar)

# To get Python scalar, use .item()
python_scalar = scalar.item()
print(f"Python int: {python_scalar}")  # 42
```

**Key insight**:
- `np.squeeze` on all-ones shape gives a 0-dimensional array (not a Python scalar)
- Use `.item()` to extract the Python scalar from a 0-d NumPy array
</details>

### Variation G3: Views vs copies in boolean indexing

**Part 4.1**: Create array [1, 2, 3, 4, 5].
**Part 4.2**: Select elements > 2 using boolean indexing.
**Part 4.3**: Does this return a view or copy?

<details>
<summary>Solution G3</summary>

```python
import numpy as np

# 4.1
arr = np.array([1, 2, 3, 4, 5])

# 4.2
selected = arr[arr > 2]
print(f"Selected: {selected}")  # [3 4 5]

# 4.3: Check if it's a view
print(f"Shares memory: {np.shares_memory(arr, selected)}")  # False!

# Boolean indexing ALWAYS returns a copy
selected[0] = 999
print(f"Original unchanged: {arr}")  # [1 2 3 4 5]

# But assignment through boolean indexing modifies in place!
arr[arr > 2] = 0
print(f"After assignment: {arr}")  # [1 2 0 0 0]
```

**Key insight**:
- Boolean indexing for **reading** returns a **copy**
- Boolean indexing for **writing** (`arr[mask] = val`) modifies **in place**
</details>

### Variation G4: Broadcasting with expand_dims

**Part 4.1**: Create a = [1, 2, 3] and b = [10, 20].
**Part 4.2**: Use expand_dims to enable broadcasting for outer product.
**Part 4.3**: Compute outer product: a[:, np.newaxis] * b[np.newaxis, :].

<details>
<summary>Solution G4</summary>

```python
import numpy as np

# 4.1
a = np.array([1, 2, 3])  # Shape: (3,)
b = np.array([10, 20])   # Shape: (2,)

# 4.2: Reshape for broadcasting
a_col = np.expand_dims(a, axis=1)  # Shape: (3, 1)
b_row = np.expand_dims(b, axis=0)  # Shape: (1, 2)
print(f"a_col shape: {a_col.shape}")
print(f"b_row shape: {b_row.shape}")

# 4.3: Outer product via broadcasting
outer = a_col * b_row  # (3, 1) * (1, 2) → (3, 2)
print("Outer product:\n", outer)
# [[10 20]
#  [20 40]
#  [30 60]]

# Alternative using np.newaxis
outer2 = a[:, np.newaxis] * b[np.newaxis, :]
assert np.array_equal(outer, outer2)

# Cleanest: np.outer
outer3 = np.outer(a, b)
assert np.array_equal(outer, outer3)
```

**Key insight**: `expand_dims` and `np.newaxis` enable broadcasting between arrays of different shapes. This is the foundation of efficient NumPy code.
</details>

### Variation G5: Squeeze with specific axis error

**Part 4.1**: Create (3, 4, 5) array.
**Part 4.2**: Try to squeeze axis 1 (which has size 4, not 1).
**Part 4.3**: What error do you get?

<details>
<summary>Solution G5</summary>

```python
import numpy as np

# 4.1
arr = np.random.randn(3, 4, 5)

# 4.2 & 4.3
try:
    squeezed = np.squeeze(arr, axis=1)
except ValueError as e:
    print(f"Error: {e}")
    # "cannot select an axis to squeeze out which has size not equal to one"

# Note: np.squeeze without axis argument silently does nothing if no size-1 dims
no_change = np.squeeze(arr)
print(f"No-op squeeze: {no_change.shape}")  # (3, 4, 5) — unchanged!
```

**Key insight**: `np.squeeze(arr, axis=k)` raises `ValueError` if `arr.shape[k] != 1`. Without the axis argument, squeeze silently does nothing if there are no size-1 dimensions.
</details>

---

## CATEGORY H: Coding Challenges (USAAIO-Style)

### Variation H1: Shape Detective

Given the following code, predict the final shape WITHOUT running it:

```python
import numpy as np
np.random.seed(0)
arr = np.random.randn(2, 1, 4, 3, 1)
arr = np.squeeze(arr)
arr = np.expand_dims(arr, axis=0)
arr = np.expand_dims(arr, axis=-1)
arr = np.swapaxes(arr, 1, 2)
arr = arr.transpose(0, 2, 1, 3)
# What is arr.shape?
```

<details>
<summary>Solution H1</summary>

```
Step by step:
(2, 1, 4, 3, 1) — original
→ squeeze: (2, 4, 3) — removes axes 1 and 4 (both size 1)
→ expand_dims axis=0: (1, 2, 4, 3)
→ expand_dims axis=-1: (1, 2, 4, 3, 1)
→ swapaxes(1, 2): (1, 4, 2, 3, 1)
→ transpose(0, 2, 1, 3): (1, 2, 4, 3, 1)

Wait, transpose(0, 2, 1, 3) only has 4 indices but array is 5D!
This will raise an error: "axes don't match array"

If we meant transpose(0, 2, 1, 3, 4), then:
(1, 4, 2, 3, 1) → (1, 2, 4, 3, 1)

Final shape: ERROR (or (1, 2, 4, 3, 1) if transpose had 5 indices)
```

**Key insight**: Always count dimensions carefully. A transpose permutation must have exactly as many indices as the array has dimensions.
</details>

### Variation H2: Implement squeeze from scratch

Without using np.squeeze, implement a function that removes all size-1 dimensions.

<details>
<summary>Solution H2</summary>

```python
import numpy as np

def my_squeeze(arr):
    """Remove all dimensions of size 1."""
    new_shape = tuple(dim for dim in arr.shape if dim != 1)
    return arr.reshape(new_shape) if new_shape else arr.reshape(())

# Test
arr = np.random.randn(2, 1, 3, 1, 4)
squeezed = my_squeeze(arr)
print(f"Original: {arr.shape}")  # (2, 1, 3, 1, 4)
print(f"Squeezed: {squeezed.shape}")  # (2, 3, 4)

# Verify against np.squeeze
assert squeezed.shape == np.squeeze(arr).shape
```

**Key insight**: `squeeze` is essentially filtering the shape tuple to remove 1s, then reshaping.
</details>

### Variation H3: Implement expand_dims from scratch

Without using np.expand_dims, implement a function that inserts a new axis.

<details>
<summary>Solution H3</summary>

```python
import numpy as np

def my_expand_dims(arr, axis):
    """Insert a new axis at the given position."""
    shape = list(arr.shape)
    # Handle negative axis
    if axis < 0:
        axis = len(shape) + axis + 1
    shape.insert(axis, 1)
    return arr.reshape(shape)

# Test
arr = np.random.randn(3, 4, 5)
expanded = my_expand_dims(arr, axis=1)
print(f"Original: {arr.shape}")  # (3, 4, 5)
print(f"Expanded: {expanded.shape}")  # (3, 1, 4, 5)

# Test negative axis
expanded_neg = my_expand_dims(arr, axis=-1)
print(f"Expanded axis=-1: {expanded_neg.shape}")  # (3, 4, 5, 1)

# Verify
assert expanded.shape == np.expand_dims(arr, axis=1).shape
assert expanded_neg.shape == np.expand_dims(arr, axis=-1).shape
```

**Key insight**: `expand_dims` is inserting a 1 into the shape tuple at the specified position.
</details>

### Variation H4: Count operations

Generate (10, 20, 30) standard normal with seed 2026.

**Part 4.1**: How many values are positive?
**Part 4.2**: How many values are in each "sign category" (negative, zero, positive)?
**Part 4.3**: What's the ratio of positive to negative values?

<details>
<summary>Solution H4</summary>

```python
import numpy as np

np.random.seed(2026)
arr = np.random.randn(10, 20, 30)

# 4.1: Count positive
n_positive = np.sum(arr > 0)
print(f"Positive values: {n_positive}")  # ~3000 (half of 6000)

# 4.2: Count by category
n_negative = np.sum(arr < 0)
n_zero = np.sum(arr == 0)  # Very unlikely for continuous distribution
print(f"Negative: {n_negative}, Zero: {n_zero}, Positive: {n_positive}")

# 4.3: Ratio
ratio = n_positive / n_negative
print(f"Positive/Negative ratio: {ratio:.4f}")  # Should be ~1.0

# Alternative using np.sign
signs = np.sign(arr)
unique, counts = np.unique(signs, return_counts=True)
print(f"Sign distribution: {dict(zip(unique, counts))}")
```

**Key insight**: For standard normal, approximately 50% of values are positive, 50% negative. `np.sum(condition)` counts True values since True=1.
</details>

### Variation H5: Complex reshaping chain

Given arr of shape (24,), reshape through these intermediate shapes WITHOUT using multiple reshapes: (2, 3, 4) → (3, 2, 4) → (3, 8)

<details>
<summary>Solution H5</summary>

```python
import numpy as np

arr = np.arange(24)
print(f"Original: {arr.shape}")  # (24,)

# Method 1: Multiple operations (what NOT to do in USAAIO)
arr1 = arr.reshape(2, 3, 4)
arr1 = arr1.swapaxes(0, 1)  # → (3, 2, 4)
arr1 = arr1.reshape(3, 8)   # → (3, 8)
print(f"Method 1: {arr1.shape}")

# Method 2: Single clever reshape (ideal for USAAIO)
# Need to understand the element ordering
# Original (24,) → (2, 3, 4): elements fill in C-order
# After swap (3, 2, 4): changes physical layout
# Final (3, 8): combine last two dimensions

# We can't do this in one reshape because swapaxes changes element order!
# This is a TRICK QUESTION

# Method 3: Using transpose + reshape
arr2 = arr.reshape(2, 3, 4).transpose(1, 0, 2).reshape(3, 8)
print(f"Method 2: {arr2.shape}")

# Verify they're the same
assert np.array_equal(arr1, arr2)
```

**Key insight**: You cannot combine `swapaxes` into a single `reshape` because swapaxes reorders elements. Reshape only changes how elements are grouped, not their order.
</details>

---

## CATEGORY I: Broadcasting After Shape Manipulation

### Variation I1: Basic Broadcasting Setup

Generate arrays and manipulate shapes to enable broadcasting.

**Part 4.1**: Create `a = np.arange(12).reshape(3, 4)` and `b = np.arange(4)`.
**Part 4.2**: Use `expand_dims` on `a` to make it (3, 1, 4).
**Part 4.3**: What shape results from `a_expanded + b`? Compute it.
**Part 4.4**: Now expand `b` to (1, 4, 1) and compute `a_expanded + b_expanded`. What's the shape?

<details>
<summary>Solution I1</summary>

```python
import numpy as np

# 4.1
a = np.arange(12).reshape(3, 4)  # Shape: (3, 4)
b = np.arange(4)                  # Shape: (4,)
print(f"a shape: {a.shape}, b shape: {b.shape}")

# 4.2
a_expanded = np.expand_dims(a, axis=1)  # Shape: (3, 1, 4)
print(f"a_expanded shape: {a_expanded.shape}")

# 4.3
# Broadcasting: (3, 1, 4) + (4,) → (3, 1, 4) + (1, 1, 4) → (3, 1, 4)
result1 = a_expanded + b
print(f"a_expanded + b shape: {result1.shape}")  # (3, 1, 4)

# 4.4
b_expanded = b.reshape(1, 4, 1)  # Shape: (1, 4, 1)
# Broadcasting: (3, 1, 4) + (1, 4, 1) → (3, 4, 4)
result2 = a_expanded + b_expanded
print(f"a_expanded + b_expanded shape: {result2.shape}")  # (3, 4, 4)
```

**Key insight**: Broadcasting aligns shapes from the right. A dimension of 1 can broadcast to any size. Both arrays expand along their size-1 dimensions.
</details>

### Variation I2: Outer Product via Broadcasting

Create outer product without using `np.outer`.

**Part 4.1**: Create `x = np.array([1, 2, 3])` and `y = np.array([10, 20, 30, 40])`.
**Part 4.2**: Use `expand_dims` or `np.newaxis` to compute outer product `x @ y.T` via broadcasting.
**Part 4.3**: Verify your result matches `np.outer(x, y)`.

<details>
<summary>Solution I2</summary>

```python
import numpy as np

# 4.1
x = np.array([1, 2, 3])       # Shape: (3,)
y = np.array([10, 20, 30, 40]) # Shape: (4,)

# 4.2: Reshape for broadcasting
x_col = x[:, np.newaxis]  # Shape: (3, 1)
y_row = y[np.newaxis, :]  # Shape: (1, 4)

outer = x_col * y_row  # Broadcasting: (3, 1) * (1, 4) → (3, 4)
print(f"Outer product shape: {outer.shape}")
print(outer)
# [[10 20 30 40]
#  [20 40 60 80]
#  [30 60 90 120]]

# 4.3: Verify
assert np.array_equal(outer, np.outer(x, y))
print("Verified!")
```

**Key insight**: Outer product is just element-wise multiplication with the right shape manipulation. `x[:, None] * y[None, :]` is a common idiom.
</details>

### Variation I3: Broadcasting with Transposed Arrays

**Part 4.1**: Create `A = np.arange(24).reshape(2, 3, 4)`.
**Part 4.2**: Create `B = np.arange(3).reshape(3, 1)`.
**Part 4.3**: Can you compute `A + B` directly? What about `A + B.T`?
**Part 4.4**: Use `swapaxes` on A to make the addition work along axis 1.

<details>
<summary>Solution I3</summary>

```python
import numpy as np

# 4.1
A = np.arange(24).reshape(2, 3, 4)  # Shape: (2, 3, 4)

# 4.2
B = np.arange(3).reshape(3, 1)  # Shape: (3, 1)

# 4.3
# A + B: (2, 3, 4) + (3, 1)
# Aligning from right: (2, 3, 4) + (3, 1) → works! Broadcasts to (2, 3, 4)
result1 = A + B
print(f"A + B shape: {result1.shape}")  # (2, 3, 4)

# A + B.T: (2, 3, 4) + (1, 3)
# This also works! (2, 3, 4) + (1, 3) → (2, 3, 4)
result2 = A + B.T
print(f"A + B.T shape: {result2.shape}")  # (2, 3, 4)

# But they're different! B broadcasts along axis 2, B.T along axis 1
print(f"Same result? {np.array_equal(result1, result2)}")  # False

# 4.4: If we want B (3,) to broadcast along axis 1 specifically
A_swapped = np.swapaxes(A, 1, 2)  # Shape: (2, 4, 3)
result3 = A_swapped + B.flatten()  # (2, 4, 3) + (3,) → (2, 4, 3)
result3 = np.swapaxes(result3, 1, 2)  # Back to (2, 3, 4)
print(f"Controlled broadcast shape: {result3.shape}")
```

**Key insight**: Broadcasting alignment happens from the rightmost dimension. To control which axis broadcasts, you may need to transpose/swapaxes before and after.
</details>

### Variation I4: Batch Matrix Operations

**Part 4.1**: Create `batch = np.random.randn(5, 3, 4)` (5 matrices of 3×4).
**Part 4.2**: Create `vec = np.random.randn(4)` (a single vector).
**Part 4.3**: Compute batch matrix-vector product for all 5 matrices using broadcasting and `@` or `np.einsum`.
**Part 4.4**: What shape manipulation is needed if `vec` was shape (4, 1)?

<details>
<summary>Solution I4</summary>

```python
import numpy as np

np.random.seed(42)

# 4.1
batch = np.random.randn(5, 3, 4)  # 5 matrices, each 3×4

# 4.2
vec = np.random.randn(4)  # Shape: (4,)

# 4.3: Batch matrix-vector multiplication
# Method 1: Using @ (matmul broadcasts over batch dimensions)
result1 = batch @ vec  # (5, 3, 4) @ (4,) → (5, 3)
print(f"batch @ vec shape: {result1.shape}")  # (5, 3)

# Method 2: Using einsum
result2 = np.einsum('ijk,k->ij', batch, vec)
print(f"einsum shape: {result2.shape}")  # (5, 3)

assert np.allclose(result1, result2)

# 4.4: If vec was (4, 1)
vec_col = vec.reshape(4, 1)
# batch @ vec_col: (5, 3, 4) @ (4, 1) → (5, 3, 1)
result3 = batch @ vec_col
print(f"batch @ vec_col shape: {result3.shape}")  # (5, 3, 1)

# To get (5, 3), squeeze the last dimension
result3_squeezed = np.squeeze(result3, axis=-1)
print(f"Squeezed shape: {result3_squeezed.shape}")  # (5, 3)
```

**Key insight**: NumPy's `@` operator broadcasts over leading batch dimensions. The last two axes are treated as the matrix dimensions for multiplication.
</details>

### Variation I5: Broadcasting Failure Analysis

**Part 4.1**: Create arrays of shapes (3, 4) and (5,).
**Part 4.2**: Why can't these broadcast together?
**Part 4.3**: What `expand_dims` operations would make them compatible?
**Part 4.4**: After fixing, what's the resulting shape?

<details>
<summary>Solution I5</summary>

```python
import numpy as np

# 4.1
a = np.random.randn(3, 4)  # Shape: (3, 4)
b = np.random.randn(5)      # Shape: (5,)

# 4.2: Why can't they broadcast?
# Aligning from right: (3, 4) vs (5,)
#                       4 vs 5  ← INCOMPATIBLE! Neither is 1.
try:
    result = a + b
except ValueError as e:
    print(f"Error: {e}")
    # "operands could not be broadcast together with shapes (3,4) (5,)"

# 4.3: Make them compatible
# Option A: Make a into (3, 4, 1) and b into (5,) → result (3, 4, 5)
a_expanded = np.expand_dims(a, axis=2)  # (3, 4, 1)
result_a = a_expanded + b  # (3, 4, 1) + (5,) → (3, 4, 5)
print(f"Option A shape: {result_a.shape}")  # (3, 4, 5)

# Option B: Make a into (3, 1, 4) and b into (5, 1) → result (3, 5, 4)
a_expanded2 = np.expand_dims(a, axis=1)  # (3, 1, 4)
b_expanded = np.expand_dims(b, axis=1)   # (5, 1)
result_b = a_expanded2 + b_expanded  # (3, 1, 4) + (5, 1) → (3, 5, 4)
print(f"Option B shape: {result_b.shape}")  # (3, 5, 4)

# Option C: Make a into (3, 4, 1, 1) and b into (1, 5) → result (3, 4, 1, 5)
# Many possibilities depending on desired output structure!
```

**Key insight**: Broadcasting fails when non-singleton dimensions don't match. The fix depends on what output shape you want—there's no single "right" answer.
</details>

---

## CATEGORY J: Memory Layout and Strides

### Variation J1: Understanding Strides

**Part 4.1**: Create `arr = np.arange(24).reshape(2, 3, 4)` and print its strides.
**Part 4.2**: What do the stride values mean in bytes?
**Part 4.3**: What are the strides after `arr.T`?
**Part 4.4**: Is `arr.T` contiguous? How do you check?

<details>
<summary>Solution J1</summary>

```python
import numpy as np

# 4.1
arr = np.arange(24).reshape(2, 3, 4)
print(f"Shape: {arr.shape}")  # (2, 3, 4)
print(f"Strides: {arr.strides}")  # (96, 32, 8) for int64

# 4.2: Stride interpretation (assuming int64 = 8 bytes)
# - Stride 0 (96): Move 96 bytes = 12 elements to go from arr[0] to arr[1]
# - Stride 1 (32): Move 32 bytes = 4 elements to go from arr[i,0] to arr[i,1]
# - Stride 2 (8): Move 8 bytes = 1 element to go from arr[i,j,0] to arr[i,j,1]

# Verify: arr is C-contiguous (row-major)
print(f"C-contiguous: {arr.flags['C_CONTIGUOUS']}")  # True

# 4.3: Strides after transpose
arr_T = arr.T
print(f"arr.T shape: {arr_T.shape}")  # (4, 3, 2)
print(f"arr.T strides: {arr_T.strides}")  # (8, 32, 96) — reversed!

# 4.4: Is arr.T contiguous?
print(f"arr.T C-contiguous: {arr_T.flags['C_CONTIGUOUS']}")  # False
print(f"arr.T F-contiguous: {arr_T.flags['F_CONTIGUOUS']}")  # True

# This is why ravel() on a transposed array returns a copy!
```

**Key insight**: Strides tell NumPy how many bytes to skip to move along each axis. Transpose just reverses strides without copying data—that's why it's O(1) but makes the array non-contiguous.
</details>

### Variation J2: When Does Reshape Copy?

**Part 4.1**: Create `arr = np.arange(12).reshape(3, 4)`.
**Part 4.2**: Reshape to (4, 3). Does it share memory with original?
**Part 4.3**: Transpose and then reshape to (12,). Does it share memory?
**Part 4.4**: Explain the rule for when reshape creates a copy.

<details>
<summary>Solution J2</summary>

```python
import numpy as np

# 4.1
arr = np.arange(12).reshape(3, 4)
print(f"Original contiguous: {arr.flags['C_CONTIGUOUS']}")  # True

# 4.2: Reshape contiguous array
reshaped = arr.reshape(4, 3)
print(f"Shares memory: {np.shares_memory(arr, reshaped)}")  # True!
# Reshape of contiguous array returns a view

# 4.3: Transpose then reshape
transposed = arr.T  # Shape (4, 3), not C-contiguous
print(f"Transposed contiguous: {transposed.flags['C_CONTIGUOUS']}")  # False

flattened = transposed.reshape(12)
print(f"Shares memory after transpose+reshape: {np.shares_memory(transposed, flattened)}")
# False! Had to copy because elements aren't contiguous in memory

# 4.4: The rule
# reshape() returns a view if the new shape is compatible with the memory layout
# (i.e., elements can be accessed with regular strides)
# reshape() returns a copy if it must reorder elements in memory

# Practical test: if the original is C-contiguous, reshape to any shape is a view
# If non-contiguous (like after transpose), reshape usually copies
```

**Key insight**: `reshape` only copies when it must. For C-contiguous arrays, reshape is always a view. After transpose, the array is F-contiguous, so C-order reshape requires copying.
</details>

### Variation J3: Fortran vs C Order

**Part 4.1**: Create `arr = np.arange(12).reshape(3, 4)` and `arr_f = np.arange(12).reshape(3, 4, order='F')`.
**Part 4.2**: Print both arrays. Are they the same?
**Part 4.3**: Flatten both with 'C' and 'F' order. What do you observe?
**Part 4.4**: When would you use Fortran order?

<details>
<summary>Solution J3</summary>

```python
import numpy as np

# 4.1
arr_c = np.arange(12).reshape(3, 4)  # C order (default)
arr_f = np.arange(12).reshape(3, 4, order='F')  # Fortran order

# 4.2: Print both
print("C order:")
print(arr_c)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print("\nF order:")
print(arr_f)
# [[ 0  3  6  9]
#  [ 1  4  7 10]
#  [ 2  5  8 11]]

# They're DIFFERENT! Same elements, but filled in different order

# 4.3: Flatten with different orders
print(f"\narr_c.flatten('C'): {arr_c.flatten('C')}")  # [0 1 2 3 4 5 6 7 8 9 10 11]
print(f"arr_c.flatten('F'): {arr_c.flatten('F')}")  # [0 4 8 1 5 9 2 6 10 3 7 11]

print(f"\narr_f.flatten('C'): {arr_f.flatten('C')}")  # [0 3 6 9 1 4 7 10 2 5 8 11]
print(f"arr_f.flatten('F'): {arr_f.flatten('F')}")  # [0 1 2 3 4 5 6 7 8 9 10 11]

# 4.4: When to use F order?
# - Interfacing with Fortran code (LAPACK, BLAS)
# - Column-major access patterns (iterate over columns frequently)
# - Some scientific computing libraries expect F-order
```

**Key insight**: C-order fills rows first (row-major), F-order fills columns first (column-major). The order affects how elements are laid out in memory, which impacts performance for different access patterns.
</details>

### Variation J4: Non-Contiguous Slices

**Part 4.1**: Create `arr = np.arange(20).reshape(4, 5)`.
**Part 4.2**: Take slice `s = arr[::2, ::2]` (every other row and column). Is it contiguous?
**Part 4.3**: What are the strides of `s`?
**Part 4.4**: Can you reshape `s` to (4,) without copying?

<details>
<summary>Solution J4</summary>

```python
import numpy as np

# 4.1
arr = np.arange(20).reshape(4, 5)
print(f"Original strides: {arr.strides}")  # (40, 8) for int64

# 4.2
s = arr[::2, ::2]  # Every other element in both dimensions
print(f"Slice shape: {s.shape}")  # (2, 3)
print(f"Slice values:\n{s}")
# [[ 0  2  4]
#  [10 12 14]]

print(f"Slice contiguous: {s.flags['C_CONTIGUOUS']}")  # False!

# 4.3
print(f"Slice strides: {s.strides}")  # (80, 16)
# Stride doubled because we skip every other element!
# - Row stride: 80 = 2 rows × 5 elements × 8 bytes
# - Col stride: 16 = 2 elements × 8 bytes

# 4.4
try:
    # Try to reshape (may or may not work depending on NumPy version)
    s_flat = s.reshape(6)
    print(f"Shares memory: {np.shares_memory(s, s_flat)}")  # False - had to copy
except:
    pass

# Force a view to fail:
# s.shape = (6,)  # This would raise an error

# Correct approach: explicitly copy if you need contiguous data
s_contiguous = np.ascontiguousarray(s)
print(f"Now contiguous: {s_contiguous.flags['C_CONTIGUOUS']}")  # True
```

**Key insight**: Slicing with step > 1 creates non-contiguous views with larger strides. These can't be reshaped without copying because the elements aren't sequential in memory.
</details>

---

## CATEGORY K: np.atleast_Xd and Dimension Guarantees

### Variation K1: Ensuring Minimum Dimensions

**Part 4.1**: What does `np.atleast_1d(5)` return? Its shape?
**Part 4.2**: What does `np.atleast_2d(np.array([1, 2, 3]))` return? Its shape?
**Part 4.3**: What does `np.atleast_3d(np.array([[1, 2], [3, 4]]))` return? Its shape?
**Part 4.4**: When would you use these functions?

<details>
<summary>Solution K1</summary>

```python
import numpy as np

# 4.1: atleast_1d on scalar
result1 = np.atleast_1d(5)
print(f"atleast_1d(5): {result1}, shape: {result1.shape}")  # [5], shape: (1,)

# 4.2: atleast_2d on 1D array
arr1d = np.array([1, 2, 3])
result2 = np.atleast_2d(arr1d)
print(f"atleast_2d([1,2,3]): {result2}, shape: {result2.shape}")
# [[1 2 3]], shape: (1, 3) — adds axis at position 0

# 4.3: atleast_3d on 2D array
arr2d = np.array([[1, 2], [3, 4]])
result3 = np.atleast_3d(arr2d)
print(f"atleast_3d shape: {result3.shape}")  # (2, 2, 1) — adds axis at end!
print(f"atleast_3d:\n{result3}")

# 4.4: Use cases
# - Writing functions that accept scalars, 1D, or 2D arrays uniformly
# - Ensuring broadcasting compatibility
# - Preparing data for APIs that expect specific dimensions

def safe_batch_process(data):
    """Process data that might be scalar, 1D, or 2D."""
    data = np.atleast_2d(data)  # Now guaranteed to be 2D
    # ... process assuming 2D ...
    return data.sum(axis=1)  # Sum along rows

print(safe_batch_process(5))         # Works with scalar
print(safe_batch_process([1,2,3]))   # Works with 1D
print(safe_batch_process([[1,2],[3,4]]))  # Works with 2D
```

**Key insight**: `atleast_Xd` functions guarantee minimum dimensionality without checking. They're defensive programming tools for functions that need consistent input shapes.
</details>

### Variation K2: atleast_Xd vs expand_dims

**Part 4.1**: Compare `np.atleast_2d(arr)` vs `np.expand_dims(arr, axis=0)` for 1D array.
**Part 4.2**: Are they always equivalent? Test with 0D (scalar) and 2D inputs.
**Part 4.3**: Which is more explicit about where the new axis goes?

<details>
<summary>Solution K2</summary>

```python
import numpy as np

# 4.1: Compare for 1D array
arr1d = np.array([1, 2, 3])

atleast = np.atleast_2d(arr1d)
expand = np.expand_dims(arr1d, axis=0)

print(f"atleast_2d shape: {atleast.shape}")  # (1, 3)
print(f"expand_dims(axis=0) shape: {expand.shape}")  # (1, 3)
print(f"Same? {np.array_equal(atleast, expand)}")  # True for this case

# 4.2: Test with scalar (0D)
scalar = np.array(5)  # 0D array
print(f"\nScalar input (shape {scalar.shape}):")
print(f"atleast_2d shape: {np.atleast_2d(scalar).shape}")  # (1, 1)
print(f"expand_dims(axis=0) shape: {np.expand_dims(scalar, axis=0).shape}")  # (1,)
# Different! atleast_2d adds TWO dimensions to reach 2D

# Test with 2D
arr2d = np.array([[1, 2], [3, 4]])
print(f"\n2D input (shape {arr2d.shape}):")
print(f"atleast_2d shape: {np.atleast_2d(arr2d).shape}")  # (2, 2) — unchanged!
print(f"expand_dims(axis=0) shape: {np.expand_dims(arr2d, axis=0).shape}")  # (1, 2, 2)
# Different! expand_dims ALWAYS adds a dimension

# 4.3: Explicitness
# expand_dims is MORE explicit — you specify exactly where the axis goes
# atleast_Xd is LESS explicit — it adds axes according to internal rules
# Use expand_dims when you need precise control
# Use atleast_Xd when you just need "at least N dimensions, don't care how"
```

**Key insight**: `expand_dims` always adds exactly one axis at the specified position. `atleast_Xd` adds as many axes as needed to reach X dimensions, and does nothing if already X+ dimensions.
</details>

### Variation K3: Multiple Arrays at Once

**Part 4.1**: Use `np.atleast_2d` with multiple arrays of different shapes.
**Part 4.2**: How can this help with broadcasting setup?

<details>
<summary>Solution K3</summary>

```python
import numpy as np

# 4.1: Multiple arrays at once
scalar = 5
arr1d = np.array([1, 2, 3])
arr2d = np.array([[1, 2], [3, 4]])

# atleast_2d accepts multiple arrays and returns a list
results = np.atleast_2d(scalar, arr1d, arr2d)

for i, r in enumerate(results):
    print(f"Input {i}: shape {r.shape}")
# Input 0: shape (1, 1)
# Input 1: shape (1, 3)
# Input 2: shape (2, 2)

# 4.2: Broadcasting setup
def pairwise_distances(a, b):
    """Compute pairwise L2 distances between points in a and b."""
    a, b = np.atleast_2d(a, b)  # Ensure 2D
    # a: (n, d), b: (m, d)
    # Use broadcasting: (n, 1, d) - (1, m, d) → (n, m, d)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

# Works with any input shape
print(pairwise_distances([0, 0], [[1, 0], [0, 1], [1, 1]]))
# [[1.         1.         1.41421356]]
```

**Key insight**: `np.atleast_Xd(*arrays)` can normalize multiple arrays at once, returning a list. This is useful for functions that accept flexible input shapes.
</details>

---

## CATEGORY L: Concatenation and Stacking with Shape Manipulation

### Variation L1: Stack vs Concatenate

**Part 4.1**: Create two (3, 4) arrays.
**Part 4.2**: Use `np.stack` along axis=0. What's the shape?
**Part 4.3**: Use `np.concatenate` along axis=0. What's the shape?
**Part 4.4**: Explain the fundamental difference.

<details>
<summary>Solution L1</summary>

```python
import numpy as np

# 4.1
a = np.random.randn(3, 4)
b = np.random.randn(3, 4)
print(f"a shape: {a.shape}, b shape: {b.shape}")

# 4.2: Stack creates a NEW axis
stacked = np.stack([a, b], axis=0)
print(f"Stacked shape: {stacked.shape}")  # (2, 3, 4)

# 4.3: Concatenate along EXISTING axis
concatenated = np.concatenate([a, b], axis=0)
print(f"Concatenated shape: {concatenated.shape}")  # (6, 4)

# 4.4: The difference
# - stack: Creates new dimension, arrays become "slices" along new axis
#   Think: "stack papers into a pile" — adds thickness
# - concatenate: Joins along existing dimension, extends that axis
#   Think: "tape papers end-to-end" — makes longer

# Stack is equivalent to:
stacked_manual = np.expand_dims(a, 0)  # (1, 3, 4)
stacked_manual = np.concatenate([np.expand_dims(a, 0), np.expand_dims(b, 0)], axis=0)
print(f"Manual stack shape: {stacked_manual.shape}")  # (2, 3, 4)
```

**Key insight**: `stack` creates a NEW axis (like `expand_dims` + `concatenate`). `concatenate` extends an EXISTING axis. Choose based on whether you want a new dimension or not.
</details>

### Variation L2: vstack, hstack, dstack

**Part 4.1**: Create two (2, 3) arrays.
**Part 4.2**: Apply `vstack`, `hstack`, and `dstack`. What are the shapes?
**Part 4.3**: Which are equivalent to concatenate with axis=0, 1, or stack?

<details>
<summary>Solution L2</summary>

```python
import numpy as np

# 4.1
a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([[7, 8, 9], [10, 11, 12]])  # (2, 3)

# 4.2
v = np.vstack([a, b])
h = np.hstack([a, b])
d = np.dstack([a, b])

print(f"vstack shape: {v.shape}")  # (4, 3) — vertical = along axis 0
print(f"hstack shape: {h.shape}")  # (2, 6) — horizontal = along axis 1
print(f"dstack shape: {d.shape}")  # (2, 3, 2) — depth = NEW axis 2

print(f"\nvstack:\n{v}")
print(f"\nhstack:\n{h}")
print(f"\ndstack:\n{d}")

# 4.3: Equivalences
assert np.array_equal(v, np.concatenate([a, b], axis=0))  # vstack = concat axis 0
assert np.array_equal(h, np.concatenate([a, b], axis=1))  # hstack = concat axis 1
assert np.array_equal(d, np.stack([a, b], axis=2))        # dstack = stack axis 2

print("\nEquivalences verified!")
```

**Key insight**:
- `vstack` = `concatenate(..., axis=0)` — stacks vertically (rows)
- `hstack` = `concatenate(..., axis=1)` — stacks horizontally (columns)
- `dstack` = `stack(..., axis=2)` — stacks in depth (new third dimension)
</details>

### Variation L3: Shape Manipulation Before Concatenation

**Part 4.1**: You have arrays of shape (5,) and (3, 5). How do you concatenate them?
**Part 4.2**: You have arrays of shape (2, 3) and (2, 4). Can you concatenate? Which axis?
**Part 4.3**: You have (2, 3, 4) and (3, 4). How do you stack them?

<details>
<summary>Solution L3</summary>

```python
import numpy as np

# 4.1: (5,) and (3, 5) — need to add dimension to first
a = np.arange(5)  # (5,)
b = np.random.randn(3, 5)  # (3, 5)

a_2d = np.expand_dims(a, axis=0)  # (1, 5)
result1 = np.concatenate([a_2d, b], axis=0)
print(f"4.1 Result shape: {result1.shape}")  # (4, 5)

# Alternative: vstack handles this automatically for 1D arrays!
result1_alt = np.vstack([a, b])
print(f"4.1 vstack shape: {result1_alt.shape}")  # (4, 5)

# 4.2: (2, 3) and (2, 4) — different along axis 1
c = np.random.randn(2, 3)  # (2, 3)
d = np.random.randn(2, 4)  # (2, 4)

# Can concatenate along axis 1 (where they differ)
result2 = np.concatenate([c, d], axis=1)
print(f"4.2 Result shape: {result2.shape}")  # (2, 7)

# Cannot concatenate along axis 0 — would need matching axis 1
try:
    np.concatenate([c, d], axis=0)
except ValueError as e:
    print(f"4.2 axis=0 error: dimension mismatch")

# 4.3: (2, 3, 4) and (3, 4) — need to add batch dimension
e = np.random.randn(2, 3, 4)  # (2, 3, 4)
f = np.random.randn(3, 4)     # (3, 4)

f_3d = np.expand_dims(f, axis=0)  # (1, 3, 4)
result3 = np.concatenate([e, f_3d], axis=0)
print(f"4.3 Result shape: {result3.shape}")  # (3, 3, 4)
```

**Key insight**: Before concatenation, arrays must match on all axes EXCEPT the concatenation axis. Use `expand_dims` or `atleast_Xd` to add missing dimensions.
</details>

### Variation L4: Split After Concatenate

**Part 4.1**: Concatenate 3 arrays of shapes (2, 4), (3, 4), (5, 4) along axis 0.
**Part 4.2**: Split the result back into the original 3 arrays.
**Part 4.3**: Use `np.split` vs `np.array_split`. What's the difference?

<details>
<summary>Solution L4</summary>

```python
import numpy as np

# 4.1: Concatenate arrays of different sizes
a = np.ones((2, 4)) * 1
b = np.ones((3, 4)) * 2
c = np.ones((5, 4)) * 3

combined = np.concatenate([a, b, c], axis=0)
print(f"Combined shape: {combined.shape}")  # (10, 4)

# 4.2: Split back — need to specify WHERE to split
# Cumulative sizes: 2, 2+3=5 → split at indices [2, 5]
split_result = np.split(combined, [2, 5], axis=0)

for i, arr in enumerate(split_result):
    print(f"Split {i}: shape {arr.shape}, values all = {arr[0,0]}")
# Split 0: shape (2, 4), values all = 1.0
# Split 1: shape (3, 4), values all = 2.0
# Split 2: shape (5, 4), values all = 3.0

# 4.3: split vs array_split
# np.split: Splits into EQUAL parts, or at specified indices
# np.array_split: Allows unequal splits

# split into 2 equal parts (only works if divisible)
try:
    np.split(combined, 2, axis=0)  # Works: 10/2 = 5 each
    np.split(combined, 3, axis=0)  # Error: 10 not divisible by 3
except ValueError:
    print("np.split failed with 3 sections")

# array_split handles unequal divisions
unequal = np.array_split(combined, 3, axis=0)
for i, arr in enumerate(unequal):
    print(f"array_split {i}: shape {arr.shape}")
# array_split 0: shape (4, 4)  — gets extra
# array_split 1: shape (3, 4)
# array_split 2: shape (3, 4)
```

**Key insight**: `np.split(arr, indices)` splits at specific indices. `np.split(arr, n)` requires exact division. `np.array_split(arr, n)` handles uneven splits by giving extras to earlier chunks.
</details>

---

## CATEGORY M: Complex Boolean Indexing Chains

### Variation M1: Multi-Step Conditional Modification

Generate (4, 5, 6) standard normal with seed 123.

**Part 4.1**: Set all values in the first "page" (index 0 on axis 0) to 0, but only if they're negative.
**Part 4.2**: In the second page, double all positive values.
**Part 4.3**: In the remaining pages, clip values to [-1, 1].

<details>
<summary>Solution M1</summary>

```python
import numpy as np

np.random.seed(123)
arr = np.random.randn(4, 5, 6)

# 4.1: First page, negative values → 0
page0 = arr[0]  # View of first page
page0[page0 < 0] = 0
# Or in one line: arr[0][arr[0] < 0] = 0

# 4.2: Second page, positive values → double
arr[1][arr[1] > 0] *= 2

# 4.3: Remaining pages, clip to [-1, 1]
arr[2:] = np.clip(arr[2:], -1, 1)

# Verify
print(f"Page 0 min: {arr[0].min()}")  # Should be >= 0
print(f"Page 1 has doubled values: {(arr[1] > 2).any()}")  # Likely True
print(f"Pages 2+ range: [{arr[2:].min():.2f}, {arr[2:].max():.2f}]")  # [-1, 1]
```

**Key insight**: You can chain indexing: `arr[0][arr[0] < 0]` first selects page 0, then applies boolean mask. The modification happens in-place because we're assigning to the indexed view.
</details>

### Variation M2: Boolean Operations on Flattened Views

Generate (3, 4, 5) with values 0-59 using `np.arange(60).reshape(3, 4, 5)`.

**Part 4.1**: Set the first 20 elements (in flattened order) that are even to -1.
**Part 4.2**: Count how many -1 values are now in each "page".
**Part 4.3**: Why is this tricky? What's the gotcha with `.flat`?

<details>
<summary>Solution M2</summary>

```python
import numpy as np

arr = np.arange(60).reshape(3, 4, 5)
print(f"Original:\n{arr[0]}")

# 4.1: First 20 elements, if even → -1
# GOTCHA: arr.flat returns an iterator, not a view for boolean indexing!

# Method 1: Work with reshaped view
flat_view = arr.ravel()  # 1D view (if contiguous)
even_mask = flat_view[:20] % 2 == 0
flat_view[:20][even_mask] = -1
# But wait — this modifies arr in place!

# Reset and try again
arr = np.arange(60).reshape(3, 4, 5)

# Method 2: Use indices
flat_view = arr.reshape(-1)  # Equivalent to ravel() for contiguous
indices = np.where(flat_view[:20] % 2 == 0)[0]
flat_view[indices] = -1

# 4.2: Count -1 per page
for i in range(3):
    count = np.sum(arr[i] == -1)
    print(f"Page {i}: {count} values of -1")
# Page 0: 10 values (0,2,4,6,8,10,12,14,16,18 all in first page)
# Page 1: 0 values
# Page 2: 0 values

# 4.3: The gotcha
# arr.flat[mask] DOES NOT modify arr!
arr2 = np.arange(10).reshape(2, 5)
arr2.flat[arr2.flat > 5] = 0  # This does NOT work!
print(f"arr2 unchanged: {arr2}")  # Still has values > 5

# You must use arr.ravel() or arr.reshape(-1) for boolean assignment
```

**Key insight**: `arr.flat` returns an iterator that doesn't support boolean indexing for assignment. Use `arr.ravel()` or `arr.reshape(-1)` instead, which return actual arrays (views when possible).
</details>

### Variation M3: Conditional Assignment with Multiple Arrays

**Part 4.1**: Create `data = np.random.randn(5, 5)` and `mask = np.random.rand(5, 5) > 0.5` with seed 42.
**Part 4.2**: Replace values where mask is True with values from another array `replacement = np.ones((5, 5)) * 999`.
**Part 4.3**: Do this without a loop using `np.where` or boolean indexing.

<details>
<summary>Solution M3</summary>

```python
import numpy as np

np.random.seed(42)

# 4.1
data = np.random.randn(5, 5)
mask = np.random.rand(5, 5) > 0.5
replacement = np.ones((5, 5)) * 999

print(f"Original data[0]: {data[0]}")
print(f"Mask[0]: {mask[0]}")

# 4.2 & 4.3: Method 1 — Boolean indexing
data_copy1 = data.copy()
data_copy1[mask] = replacement[mask]
print(f"\nBoolean indexing result[0]: {data_copy1[0]}")

# Method 2 — np.where (more elegant)
data_copy2 = np.where(mask, replacement, data)
print(f"np.where result[0]: {data_copy2[0]}")

# Method 3 — In-place with boolean indexing
data_copy3 = data.copy()
data_copy3[mask] = 999  # Direct scalar also works
print(f"Direct assignment result[0]: {data_copy3[0]}")

# Verify all methods give same result
assert np.array_equal(data_copy1, data_copy2)
assert np.array_equal(data_copy1, data_copy3)
```

**Key insight**: `np.where(condition, x, y)` is powerful: it selects from `x` where condition is True, from `y` where False. Both `x` and `y` can be arrays of the same shape, enabling element-wise conditional selection.
</details>

### Variation M4: Argwhere vs Where vs Nonzero

**Part 4.1**: Create `arr = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])`.
**Part 4.2**: Use `np.where(arr > 0)` — what's the output format?
**Part 4.3**: Use `np.argwhere(arr > 0)` — what's the output format?
**Part 4.4**: Use `np.nonzero(arr)` — how does it relate to `where`?

<details>
<summary>Solution M4</summary>

```python
import numpy as np

# 4.1
arr = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
print("Array:")
print(arr)

# 4.2: np.where returns tuple of arrays (one per dimension)
where_result = np.where(arr > 0)
print(f"\nnp.where(arr > 0):")
print(f"  Rows: {where_result[0]}")  # [0 0 1 2 2]
print(f"  Cols: {where_result[1]}")  # [0 2 1 0 2]
# Use as: arr[where_result] gives [1 2 3 4 5]

# 4.3: np.argwhere returns (N, ndim) array of coordinates
argwhere_result = np.argwhere(arr > 0)
print(f"\nnp.argwhere(arr > 0):")
print(argwhere_result)
# [[0 0]
#  [0 2]
#  [1 1]
#  [2 0]
#  [2 2]]
# Each row is a coordinate pair (row, col)

# 4.4: np.nonzero is equivalent to np.where with just the condition
nonzero_result = np.nonzero(arr)  # Same as np.where(arr) — finds non-zero elements
print(f"\nnp.nonzero(arr) == np.where(arr != 0): ", end="")
print(np.array_equal(nonzero_result[0], np.where(arr != 0)[0]))  # True

# Conversion between formats:
# where → argwhere: np.column_stack(where_result)
# argwhere → where: tuple(argwhere_result.T)
print(f"\nConversion check:")
print(f"column_stack(where):\n{np.column_stack(where_result)}")
print(f"argwhere.T as tuple: {tuple(argwhere_result.T)}")
```

**Key insight**:
- `np.where(cond)` returns tuple of arrays — good for indexing: `arr[np.where(cond)]`
- `np.argwhere(cond)` returns array of coordinates — good for iterating: `for (r, c) in np.argwhere(cond)`
- `np.nonzero(arr)` is just `np.where(arr != 0)` (legacy name)
</details>

### Variation M5: Fancy Indexing After Boolean Selection

Generate (100,) array with values from standard normal, seed 0.

**Part 4.1**: Get indices of all positive values.
**Part 4.2**: From those indices, select every 3rd one.
**Part 4.3**: Set those selected values to 0 in the original array.

<details>
<summary>Solution M5</summary>

```python
import numpy as np

np.random.seed(0)
arr = np.random.randn(100)

print(f"Original: {arr[:10]}")  # First 10 values
print(f"Positive count: {np.sum(arr > 0)}")

# 4.1: Get indices of positive values
positive_indices = np.where(arr > 0)[0]
print(f"Positive indices: {positive_indices[:10]}...")  # First 10 indices

# 4.2: Every 3rd positive index
selected_indices = positive_indices[::3]
print(f"Every 3rd: {selected_indices[:5]}...")  # First 5 selected

# 4.3: Set those to 0
arr[selected_indices] = 0

# Verify
print(f"\nAfter modification:")
print(f"Values at selected indices: {arr[selected_indices]}")  # All 0
print(f"Positive count now: {np.sum(arr > 0)}")  # Reduced

# Alternative one-liner (less readable):
# arr[np.where(arr > 0)[0][::3]] = 0
```

**Key insight**: You can chain indexing operations: first get indices with `where`, then slice those indices, then use them for assignment. This is "fancy indexing" — using an array of indices to access/modify elements.
</details>

---

## SUMMARY: Key NumPy Array Manipulation Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `np.random.seed(n)` | Set random state | `np.random.seed(2026)` |
| `np.random.randn(*shape)` | Standard normal | `np.random.randn(3, 4)` |
| `np.squeeze(arr)` | Remove size-1 dims | `(2,1,3)` → `(2,3)` |
| `np.squeeze(arr, axis=k)` | Remove specific axis | Must be size 1 |
| `np.expand_dims(arr, axis=k)` | Insert new axis | `(2,3)` → `(2,1,3)` |
| `arr[:, np.newaxis]` | Insert axis via indexing | Same as expand_dims |
| `np.swapaxes(arr, a, b)` | Swap two axes | `(2,3,4)` → `(4,3,2)` |
| `arr.transpose(perm)` | Permute all axes | Flexible axis reordering |
| `arr[arr > k] = v` | Boolean assignment | Conditional modification |
| `np.where(cond, x, y)` | Conditional selection | Returns array |
| `np.where(cond)` | Get indices | Returns tuple of arrays |
| `np.argwhere(cond)` | Get coordinates | Returns (N, ndim) array |
| `arr.flatten()` | 1D copy | Always copies |
| `arr.ravel()` | 1D view (if possible) | May be view or copy |
| `arr.reshape(-1)` | 1D via reshape | Equivalent to ravel |
| `arr.strides` | Bytes per axis step | Memory layout info |
| `np.atleast_1d/2d/3d` | Ensure min dimensions | Defensive programming |
| `np.stack(arrays, axis)` | Create new axis | Combines into new dim |
| `np.concatenate(arrays, axis)` | Extend existing axis | Joins along axis |
| `np.vstack/hstack/dstack` | Stack shortcuts | Vertical/horizontal/depth |
| `np.split(arr, indices)` | Split array | Inverse of concatenate |
| `np.shares_memory(a, b)` | Check for view | Debug memory sharing |

---

## ATOMIC SKILLS TESTED

1. **Array Creation**: `np.random.seed`, `randn`, `rand`, `randint`, `uniform`, `exponential`, `binomial`
2. **Dimension Reduction**: `squeeze`, slicing with index
3. **Dimension Expansion**: `expand_dims`, `np.newaxis`, `reshape`, `atleast_Xd`
4. **Axis Manipulation**: `swapaxes`, `transpose`, `moveaxis`, `rollaxis`
5. **Boolean Indexing**: masks, compound conditions (`&`, `|`), `np.where`, `np.argwhere`
6. **Flattening**: `flatten`, `ravel`, `reshape(-1)`, order parameter
7. **Shape Reasoning**: tracking dimensions through operations
8. **Broadcasting**: shape alignment, `expand_dims` for compatibility
9. **Memory Layout**: strides, contiguity, views vs copies, C vs F order
10. **Concatenation/Stacking**: `stack`, `concatenate`, `vstack`, `hstack`, `dstack`, `split`

---

## COMMON MISCONCEPTIONS

1. **Squeeze removes one dimension** — No, it removes ALL size-1 dimensions (unless axis specified)
2. **Boolean indexing returns a view** — No, it returns a COPY for reading (but modifies in-place for writing)
3. **Transpose reverses axes** — Only without arguments; with arguments it's a permutation
4. **Flatten and ravel are identical** — No, flatten always copies; ravel may return a view
5. **reshape(-1) creates a copy** — No, it returns a view when possible (like ravel)
6. **arr.flat supports boolean assignment** — No, use `arr.ravel()` or `arr.reshape(-1)` instead
7. **Stack and concatenate are interchangeable** — No, stack creates a NEW axis; concatenate extends an EXISTING one
8. **Broadcasting always works** — No, non-singleton dimensions must match exactly
9. **Reshape always returns a view** — No, non-contiguous arrays (e.g., after transpose) require copying
10. **atleast_2d is the same as expand_dims** — No, atleast_Xd adds axes only IF needed; expand_dims ALWAYS adds

---

*Generated for USAAIO Problem 4 practice. Total variations: 45+*

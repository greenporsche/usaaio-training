# AI Programming in Python -- Cheat Sheet

> Quick reference for USAAIO 2026. Keep this open during practice.

---

## NumPy Array Creation

```python
import numpy as np

np.zeros((3, 4))            # 3x4 of zeros
np.ones((2, 3))             # 2x3 of ones
np.full((2, 2), 7)          # 2x2 filled with 7
np.eye(3)                   # 3x3 identity
np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)        # [0, 0.25, 0.5, 0.75, 1.0]
np.random.randn(3, 4)       # 3x4 standard normal
np.random.rand(3, 4)        # 3x4 uniform [0, 1)
np.random.randint(0, 10, (3, 4))  # 3x4 integers in [0, 10)
```

## Indexing & Slicing

```python
a = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

a[1, 2]           # 6           (scalar)
a[1]              # [4, 5, 6, 7] (row)
a[:, 2]           # [2, 6, 10]   (column)
a[0:2, 1:3]       # [[1,2],[5,6]] (subarray)
a[::2, ::2]       # [[0,2],[8,10]] (stride)

# Fancy indexing (returns copy)
a[[0, 2], [1, 3]]           # [1, 11]
a[np.array([0, 2])]         # rows 0 and 2

# Boolean indexing (returns copy)
a[a > 5]                    # [6, 7, 8, 9, 10, 11]
a[(a > 3) & (a < 8)]        # [4, 5, 6, 7]
```

**View vs Copy**: slicing = view (shared memory), fancy/boolean indexing = copy.

## Shape Manipulation

```python
a.reshape(4, 3)       # new shape, same data (view if possible)
a.reshape(-1, 6)      # infer first dim -> (2, 6)
a.T                   # transpose
a.ravel()             # flatten to 1D (view)
a.flatten()           # flatten to 1D (copy)

np.squeeze(a)         # remove dims of size 1
np.expand_dims(a, 0)  # add dim: (3,4) -> (1,3,4)
a[np.newaxis, :]      # same as expand_dims axis=0
a[:, np.newaxis]      # (3,4) -> (3,1,4)

np.concatenate([a, b], axis=0)  # stack vertically
np.concatenate([a, b], axis=1)  # stack horizontally
np.stack([a, b], axis=0)        # new axis: (2, 3, 4)
np.vstack([a, b])               # vertical stack
np.hstack([a, b])               # horizontal stack
```

## Broadcasting Rules

Two arrays are compatible when, **for each trailing dimension**:
1. Dimensions are equal, OR
2. One of them is 1

```
(3, 4) + (4,)     -> (3, 4)     # rule 2: (1,4) pads to match
(3, 1) + (1, 4)   -> (3, 4)     # both stretch
(3, 4) + (3,)     -> ERROR       # 4 != 3, neither is 1
(2, 3, 4) + (3, 1) -> (2, 3, 4) # pad left -> (1,3,1), broadcast
```

**Pattern**: to broadcast `(N,)` with `(M,)` for outer operation, reshape to `(N, 1)` and `(1, M)`.

## Vectorization Patterns

```python
# LOOP (FORBIDDEN in USAAIO)        # VECTORIZED
total = 0                            total = np.sum(a)
for x in a: total += x

result = []                          result = a ** 2
for x in a: result.append(x**2)

# Pairwise distances (N points in D dims)
# points shape: (N, D)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, D)
distances = np.sqrt(np.sum(diff**2, axis=-1))                # (N, N)

# Conditional
result = np.where(a > 0, a, 0)      # ReLU

# Outer product
outer = a[:, np.newaxis] * b[np.newaxis, :]  # (N, M)
```

## Key NumPy Functions

```python
np.sum(a, axis=0)       # sum along axis 0 (collapse rows)
np.mean(a, axis=1)      # mean along axis 1 (collapse cols)
np.std(a, axis=0)       # standard deviation
np.argmax(a, axis=1)    # index of max per row
np.argsort(a, axis=0)   # indices that would sort
np.where(cond, x, y)    # element-wise ternary
np.clip(a, lo, hi)      # clamp values
np.unique(a)             # sorted unique values
np.einsum('ij,jk->ik', A, B)  # matrix multiply
np.einsum('ij->j', A)         # sum over rows (column sums)
np.einsum('ii->', A)           # trace
```

**Axis convention**: `axis=0` collapses rows (operates down), `axis=1` collapses columns (operates across).

## Pandas Essentials

```python
import pandas as pd

# Creation
df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
s = pd.Series([1, 2, 3], name='x')

# Selection
df['a']                  # Series (column)
df[['a', 'b']]           # DataFrame (multiple cols)
df.loc[0:2, 'a':'b']     # label-based (inclusive)
df.iloc[0:2, 0:2]        # integer-based (exclusive end)

# Filtering
df[df['a'] > 1]
df.query('a > 1 and b < 6')

# GroupBy
df.groupby('a')['b'].mean()
df.groupby('a').agg({'b': ['mean', 'sum']})

# Merge
pd.merge(df1, df2, on='key', how='inner')  # inner/left/right/outer

# Apply
df['a'].apply(lambda x: x * 2)
df.apply(lambda row: row['a'] + row['b'], axis=1)

# Missing data
df.isna().sum()          # count NaN per column
df.fillna(0)             # fill NaN
df.dropna()              # drop rows with NaN

# Useful
df.describe()            # summary stats
df.value_counts('a')     # frequency table
df.sort_values('a')      # sort by column
df.pivot_table(values='b', index='a', aggfunc='mean')
```

## Matplotlib & Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(x, y, 'r--', label='line')
ax[0].set_title('Title'); ax[0].legend()
ax[1].scatter(x, y, c=colors, s=sizes)
plt.tight_layout(); plt.show()

# Histogram / Bar
ax.hist(data, bins=20, alpha=0.7)
ax.bar(labels, values)

# Seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
sns.boxplot(x='cat', y='val', data=df)
sns.histplot(df['col'], kde=True)
```

## Common Gotchas

| Gotcha | Fix |
|--------|-----|
| `a[0:3]` is a **view** -- modifying it modifies `a` | Use `.copy()` if you need independence |
| `axis=0` means "along rows" (collapses rows) | Think "which axis disappears" |
| `(3,) + (3, 4)` fails broadcasting | Reshape to `(3, 1)` first |
| `a == np.nan` is always `False` | Use `np.isnan(a)` |
| Integer division in Python 3: `7 / 2 = 3.5` | Use `//` for floor division |
| `df['col'] = df['col'].apply(...)` is slow | Use vectorized `df['col'] * 2` |
| `np.array([1, 2, 3]).shape` is `(3,)` not `(3, 1)` | Use `.reshape(-1, 1)` for column vector |

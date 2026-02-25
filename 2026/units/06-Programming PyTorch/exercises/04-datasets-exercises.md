# Datasets & DataLoaders Exercises

**Topic**: Dataset protocol, DataLoader, custom collate, data pipelines
**Difficulty**: Intermediate

---

## Exercise 1: Predict the Output

What does this code print? Predict the shapes and values.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TinyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(20).reshape(10, 2).float()
        self.labels = torch.arange(10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = TinyDataset()
loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=False)

for i, (x, y) in enumerate(loader):
    print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
```

<details>
<summary>Solution</summary>

```
Batch 0: x.shape=torch.Size([3, 2]), y.shape=torch.Size([3])
Batch 1: x.shape=torch.Size([3, 2]), y.shape=torch.Size([3])
Batch 2: x.shape=torch.Size([3, 2]), y.shape=torch.Size([3])
Batch 3: x.shape=torch.Size([1, 2]), y.shape=torch.Size([1])
```

There are 10 samples with batch_size=3: $\lceil 10/3 \rceil = 4$ batches. The last batch has only 1 sample (10 - 3*3 = 1). With `drop_last=False`, this incomplete batch is included.

If `drop_last=True`, batch 3 would be dropped (only 3 batches, 9 samples used).

**Key insight**: The last batch of an epoch often has fewer samples than `batch_size`. This can cause issues with `BatchNorm` (which needs at least 2 samples). Use `drop_last=True` during training if you use BatchNorm.
</details>

---

## Exercise 2: Implement a Pairs Dataset

Create a dataset that returns pairs of samples for contrastive learning. Given a base dataset, it should return `(sample_i, sample_j, is_same_class)`.

```python
class PairsDataset(Dataset):
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: a Dataset returning (feature, label) tuples
        """
        # YOUR CODE HERE
        pass

    def __len__(self):
        # Return number of possible pairs (use N^2 for simplicity)
        pass

    def __getitem__(self, idx):
        """
        Return (sample_i, sample_j, is_same_class)
        where is_same_class is 1.0 if same class, 0.0 otherwise
        """
        # YOUR CODE HERE
        pass

# Test
base = TinyDataset()
pairs = PairsDataset(base)
x1, x2, same = pairs[0]
print(f"x1.shape={x1.shape}, x2.shape={x2.shape}, same={same}")
```

<details>
<summary>Solution</summary>

```python
class PairsDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.n = len(base_dataset)

    def __len__(self):
        return self.n * self.n

    def __getitem__(self, idx):
        i = idx // self.n
        j = idx % self.n

        x_i, y_i = self.base[i]
        x_j, y_j = self.base[j]

        is_same = torch.tensor(1.0 if y_i == y_j else 0.0)
        return x_i, x_j, is_same
```

In practice, $N^2$ pairs is too many. Real contrastive learning uses random sampling or mining strategies. But this illustrates the pattern.

**Key insight**: Datasets compose — you can wrap one dataset inside another to create new views of the same data. This is a powerful pattern for data augmentation, pairing, and curriculum learning.
</details>

---

## Exercise 3: Custom Collate for Variable-Length Sequences

You have a dataset where each sample is a 1D sequence of different length. Write a custom collate function that pads sequences to the maximum length in the batch and returns a padding mask.

```python
class VariableLengthDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        sequences: list of 1D tensors with different lengths
        labels: list of integer labels
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def pad_collate(batch):
    """
    Custom collate function.
    Returns:
        padded: (B, max_len) tensor, zero-padded
        mask: (B, max_len) bool tensor, True where padded
        labels: (B,) tensor
    """
    # YOUR CODE HERE
    pass

# Test
sequences = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0]),
    torch.tensor([6.0, 7.0, 8.0, 9.0]),
]
labels = [0, 1, 0]
dataset = VariableLengthDataset(sequences, labels)
loader = DataLoader(dataset, batch_size=3, collate_fn=pad_collate)

for padded, mask, labels in loader:
    print(f"padded: {padded}")
    print(f"mask:   {mask}")
    print(f"labels: {labels}")
```

Expected output:
```
padded: tensor([[1., 2., 3., 0.],
                [4., 5., 0., 0.],
                [6., 7., 8., 9.]])
mask:   tensor([[False, False, False,  True],
                [False, False,  True,  True],
                [False, False, False, False]])
labels: tensor([0, 1, 0])
```

<details>
<summary>Solution</summary>

```python
def pad_collate(batch):
    sequences, labels = zip(*batch)

    # Find max length in this batch
    max_len = max(seq.size(0) for seq in sequences)

    # Create padded tensor and mask
    B = len(sequences)
    padded = torch.zeros(B, max_len)
    mask = torch.ones(B, max_len, dtype=torch.bool)     # True = padded

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded[i, :length] = seq
        mask[i, :length] = False                          # False = real data

    labels = torch.tensor(labels)
    return padded, mask, labels
```

**Key insight**: Padding masks are essential for variable-length data. Transformers use these masks to ignore padded positions in attention computation (`scores.masked_fill(mask, -inf)`). The convention varies: some codebases use `True` for real data and `False` for padding — always check the convention.
</details>

---

## Exercise 4: Data Pipeline Performance

Which of these DataLoader configurations will be SLOWEST and why?

```python
# Config A
loader_a = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Config B
loader_b = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Config C
loader_c = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Config D
loader_d = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                       pin_memory=True)
```

<details>
<summary>Solution</summary>

**Config B is slowest** for two reasons:

1. **batch_size=1**: No batching means no parallelism on GPU. Matrix operations on a single sample cannot utilize GPU cores efficiently. Also, many more Python loop iterations (N vs N/32).

2. **num_workers=0**: Data loading happens in the main process. The GPU waits while the CPU loads each sample sequentially.

**Performance ranking (fastest to slowest on GPU):**
1. **Config D** — 4 workers + pin_memory (data is pre-loaded in page-locked memory for faster GPU transfer)
2. **Config A** — 4 workers, batch_size=32
3. **Config C** — Single worker, batch_size=32 (GPU may starve while waiting for data)
4. **Config B** — Single sample, single worker (worst of all worlds)

**Key insight**: `num_workers > 0` uses multiprocessing to load data in parallel with training. `pin_memory=True` speeds up CPU-to-GPU transfer. These are free performance wins on GPU training. On CPU-only or for USAAIO (small datasets), `num_workers=0` and `pin_memory=False` are fine.
</details>

---

## Exercise 5: Implement a Lazy Dataset

Create a dataset that generates data on-the-fly (no preloading) for a synthetic regression task: $y = \sin(x) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, 0.1)$.

The dataset should:
- Accept a size parameter `n`
- Generate a random `x` in `[0, 2*pi]` and compute `y = sin(x) + noise` in `__getitem__`
- Return different random data each time `__getitem__` is called with the same index

```python
class SyntheticSinDataset(Dataset):
    def __init__(self, n):
        # YOUR CODE HERE
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        # Generate random sample on-the-fly
        pass

# Test
dataset = SyntheticSinDataset(1000)
x1, y1 = dataset[0]
x2, y2 = dataset[0]
print(f"Same index, different data: x1={x1.item():.4f}, x2={x2.item():.4f}")
assert x1 != x2, "Should be different each time!"
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
import math

class SyntheticSinDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.rand(1) * 2 * math.pi               # Uniform in [0, 2*pi]
        noise = torch.randn(1) * 0.1                    # N(0, 0.1)
        y = torch.sin(x) + noise
        return x, y                                     # (1,), (1,)
```

**Key insight**: Lazy datasets are useful for:
- Synthetic data (infinite effective dataset)
- Very large datasets that do not fit in memory
- Online data augmentation (each access produces a different augmented version)

The downside: no reproducibility. Each epoch sees different data. If reproducibility matters, set a seed based on `idx`: `torch.manual_seed(idx)`. But for training with augmentation, randomness is usually desired.
</details>

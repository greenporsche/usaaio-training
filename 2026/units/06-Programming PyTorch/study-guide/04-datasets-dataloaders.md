# Datasets & DataLoaders

**Prerequisites**: Python classes (`__len__`, `__getitem__`), tensors
**USAAIO Relevance**: Round 2 problems provide data in various formats. You must know how to wrap any data source into a `Dataset` and feed it through a `DataLoader` for batched training.

---

## Discovery

### Feeding Data to Neural Networks Efficiently

Training a neural network requires iterating over data in batches. Why batches?

1. **Memory**: A dataset with 60,000 images does not fit in GPU memory at once
2. **Stochasticity**: Random mini-batches provide noisy gradient estimates that help escape local minima (this is why SGD works)
3. **Parallelism**: GPU hardware is optimized for batched matrix operations

PyTorch separates concerns into two abstractions:

- **`Dataset`**: Defines *what* the data is (how to load one sample)
- **`DataLoader`**: Defines *how* to iterate (batching, shuffling, parallelism)

This separation is clean and powerful — you write the data logic once in `Dataset`, and `DataLoader` handles the rest.

> **Socratic question**: Why shuffle training data? Imagine training on a sorted dataset — all cats first, then all dogs. The model sees only cats for the first half of training, then only dogs. It cannot learn both simultaneously. Shuffling ensures each batch is representative of the full distribution.

---

## Intuition

### The Dataset Protocol

A `Dataset` is any Python object that supports:

```
┌──────────────────────────────────┐
│         Dataset Protocol         │
├──────────────────────────────────┤
│  __len__(self) → int             │  How many samples?
│  __getitem__(self, idx) → tuple  │  Return sample at index
└──────────────────────────────────┘
```

That is it. Two methods. `DataLoader` calls `__len__` to know the dataset size and `__getitem__` to fetch individual samples (or batches of indices).

### DataLoader Pipeline

```
Dataset                    DataLoader                    Training Loop
┌─────────┐    indices    ┌──────────────┐    batches   ┌─────────────┐
│ __len__ │←─────────────│  Sampler      │              │ for batch   │
│ = 60000 │              │  (shuffle)    │              │ in loader:  │
│         │    samples   │       ↓       │──────────────│   x, y =    │
│__getitem│←─────────────│  Collate      │              │   batch     │
│ (idx)   │              │  (stack into  │              │   ...       │
│→ (x, y) │              │   tensors)    │              └─────────────┘
└─────────┘              └──────────────┘
```

The flow:
1. **Sampler** generates indices (sequential or shuffled)
2. DataLoader groups indices into batches of size `batch_size`
3. `__getitem__` is called for each index
4. **Collate function** stacks individual samples into a batch tensor

### Collate Function

The default collate function stacks tensors along a new batch dimension:

```
Sample 0: (tensor(shape=(28,28)), tensor(3))
Sample 1: (tensor(shape=(28,28)), tensor(7))
Sample 2: (tensor(shape=(28,28)), tensor(1))
                    ↓ collate
Batch: (tensor(shape=(3,28,28)), tensor([3,7,1]))
```

For variable-length data (text, graphs), you need a custom collate function to pad sequences.

---

## Math

### Batch Statistics

With batch size $B$ and dataset size $N$:

- Batches per epoch: $\lceil N / B \rceil$
- Gradient variance: $\text{Var}[\nabla_B] \approx \frac{\sigma^2}{B}$ — larger batches have lower gradient noise
- Learning rate scaling: common practice is to scale LR linearly with batch size

---

## Code

### Minimal Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X: numpy array or tensor of features, shape (N, ...)
            y: numpy array or tensor of labels, shape (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Usage
import numpy as np
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, size=1000)

dataset = SimpleDataset(X_train, y_train)
print(len(dataset))           # 1000
print(dataset[0][0].shape)    # torch.Size([784])
print(dataset[0][1])          # tensor(3) — example label
```

### DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,             # Samples per batch
    shuffle=True,              # Randomize order each epoch (training only!)
    num_workers=0,             # Subprocesses for data loading (0 = main process)
    drop_last=False,           # If True, drop incomplete last batch
)

# Iterate
for batch_x, batch_y in loader:
    print(batch_x.shape)      # torch.Size([32, 784])
    print(batch_y.shape)      # torch.Size([32])
    break                      # Just show first batch
```

### Image Dataset

```python
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels                       # dict: filename → class_id
        self.filenames = list(labels.keys())
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')

        if self.transform:
            image = self.transform(image)          # e.g., ToTensor, Normalize

        label = self.labels[fname]
        return image, label                        # (C, H, W), int
```

### CSV/Tabular Dataset

```python
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path, target_col):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(
            df.drop(columns=[target_col]).values, dtype=torch.float32
        )
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### Custom Collate Function

```python
def pad_collate(batch):
    """
    Custom collate for variable-length sequences.
    Each sample is (sequence_tensor, label).
    """
    sequences, labels = zip(*batch)

    # Find max length in this batch
    max_len = max(seq.size(0) for seq in sequences)

    # Pad all sequences to max_len
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(1))
    lengths = torch.tensor([seq.size(0) for seq in sequences])

    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0), :] = seq

    labels = torch.stack(labels)
    return padded, labels, lengths             # (B, max_L, D), (B,), (B,)

# Usage
loader = DataLoader(dataset, batch_size=16, collate_fn=pad_collate)
```

### Using torchvision Datasets (MNIST example)

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                         # PIL Image → (C, H, W) tensor in [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)),    # MNIST mean and std
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Check shapes
for images, labels in train_loader:
    print(images.shape)    # torch.Size([64, 1, 28, 28])
    print(labels.shape)    # torch.Size([64])
    break
```

### Train/Validation Split

```python
from torch.utils.data import random_split

full_dataset = SimpleDataset(X, y)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)    # Reproducible split
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Subset and Sampler

```python
from torch.utils.data import Subset, WeightedRandomSampler

# Subset: use only specific indices
subset = Subset(full_dataset, indices=range(0, 100))

# WeightedRandomSampler: oversample minority classes
class_counts = [800, 200]                          # Imbalanced: 800 of class 0, 200 of class 1
weights = [1.0/c for c in class_counts]
sample_weights = [weights[label] for _, label in full_dataset]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(full_dataset))
loader = DataLoader(full_dataset, batch_size=32, sampler=sampler)
# Note: cannot use shuffle=True with a sampler
```

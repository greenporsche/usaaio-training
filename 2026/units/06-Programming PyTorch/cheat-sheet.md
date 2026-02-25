# Programming PyTorch — Cheat Sheet

> Quick reference for USAAIO 2026 | AI 310

---

## Tensor Creation

| Function | Example | Result Shape |
|---|---|---|
| `torch.zeros(m, n)` | `torch.zeros(2, 3)` | `(2, 3)` filled with 0.0 |
| `torch.ones(m, n)` | `torch.ones(3, 4)` | `(3, 4)` filled with 1.0 |
| `torch.randn(m, n)` | `torch.randn(2, 5)` | `(2, 5)` from $\mathcal{N}(0, 1)$ |
| `torch.rand(m, n)` | `torch.rand(2, 5)` | `(2, 5)` from $\text{Uniform}(0, 1)$ |
| `torch.arange(start, end, step)` | `torch.arange(0, 10, 2)` | `(5,)` → `[0, 2, 4, 6, 8]` |
| `torch.linspace(start, end, steps)` | `torch.linspace(0, 1, 5)` | `(5,)` → `[0, 0.25, 0.5, 0.75, 1.0]` |
| `torch.eye(n)` | `torch.eye(3)` | `(3, 3)` identity |
| `torch.full((m, n), val)` | `torch.full((2, 3), 7.0)` | `(2, 3)` filled with 7.0 |
| `torch.tensor(data)` | `torch.tensor([1, 2, 3])` | From Python list/NumPy |
| `torch.from_numpy(arr)` | `torch.from_numpy(np_arr)` | Shares memory with NumPy |

---

## Common dtypes

| dtype | Description |
|---|---|
| `torch.float32` / `torch.float` | Default float (32-bit) |
| `torch.float64` / `torch.double` | 64-bit float |
| `torch.float16` / `torch.half` | 16-bit float (GPU training) |
| `torch.int64` / `torch.long` | Default int (class labels) |
| `torch.bool` | Boolean masks |

Cast with `x.to(torch.float32)` or `x.float()`.

---

## Tensor Shape Operations

| Operation | Code | Effect |
|---|---|---|
| Reshape | `x.reshape(B, -1)` or `x.view(B, -1)` | Rearrange without copying (view requires contiguous) |
| Transpose | `x.T` or `x.transpose(0, 1)` | Swap two dims |
| Permute | `x.permute(0, 2, 1)` | Arbitrary dim reorder |
| Squeeze | `x.squeeze(dim)` | Remove dim of size 1 |
| Unsqueeze | `x.unsqueeze(dim)` | Add dim of size 1 |
| Flatten | `x.flatten(start_dim)` | Collapse dims |
| Expand | `x.expand(B, C, H, W)` | Broadcast without copy |
| Contiguous | `x.contiguous()` | Force contiguous memory layout |

**`view` vs `reshape`**: `view` requires contiguous memory (fails after `transpose`). `reshape` always works (may copy). Use `reshape` unless you need to guarantee no copy.

---

## Indexing & Slicing

```python
x[0]              # First element / row
x[:, 1]           # Second column
x[x > 0]          # Boolean masking
x[indices]         # Fancy indexing
x[..., -1]        # Last element along last dim (ellipsis)
x[:, None, :]     # Insert dim (same as unsqueeze(1))
```

---

## Broadcasting Rules

Same as NumPy. Dimensions are compared **right to left**:
1. If dims differ in size, the size-1 dim is broadcast
2. If a tensor has fewer dims, it is left-padded with 1s
3. Sizes must match or one must be 1

```
(3, 1) + (1, 4) → (3, 4)
(2, 3, 4) + (4,) → (2, 3, 4)    # (4,) becomes (1, 1, 4)
(2, 3, 4) + (3, 1) → (2, 3, 4)  # (3, 1) becomes (1, 3, 1)
```

---

## Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x                    # y = x^2 + 3x
y.backward()                            # dy/dx = 2x + 3
print(x.grad)                           # tensor([7.])
```

| Pattern | Code |
|---|---|
| Enable gradient tracking | `x = torch.tensor(..., requires_grad=True)` |
| Compute gradients | `y.backward()` |
| Access gradient | `x.grad` |
| Detach from graph | `x.detach()` |
| No-grad context | `with torch.no_grad(): ...` |
| Zero gradients | `x.grad.zero_()` or `optimizer.zero_grad()` |
| Higher-order derivatives | `torch.autograd.grad(y, x, create_graph=True)` |
| Functional gradient | `grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))` |

**Critical for USAAIO**: `torch.autograd.grad` with `create_graph=True` allows computing second derivatives (e.g., for PINNs):

```python
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                           create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                            create_graph=True)[0]
```

---

## nn.Module Pattern

```python
class MyModel(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()                      # MUST call super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):                       # Defines computation
        x = self.relu(self.fc1(x))              # (B, in) → (B, hidden)
        x = self.fc2(x)                         # (B, hidden) → (B, out)
        return x
```

| Method | Purpose |
|---|---|
| `model.parameters()` | Iterator over all learnable parameters |
| `model.named_parameters()` | Iterator yielding `(name, param)` tuples |
| `model.state_dict()` | Ordered dict of parameter tensors |
| `model.load_state_dict(d)` | Load parameters from dict |
| `model.train()` | Set training mode (enables dropout, batchnorm) |
| `model.eval()` | Set evaluation mode (disables dropout, batchnorm) |
| `model.to(device)` | Move all parameters to device |

---

## Common Layers

| Layer | Signature | Shape Transform |
|---|---|---|
| `nn.Linear(in, out)` | Fully connected | `(B, in) → (B, out)` |
| `nn.Conv2d(C_in, C_out, k)` | 2D convolution | `(B, C_in, H, W) → (B, C_out, H', W')` |
| `nn.BatchNorm1d(features)` | Batch normalization | Shape unchanged |
| `nn.LayerNorm(features)` | Layer normalization | Shape unchanged |
| `nn.Dropout(p)` | Dropout | Shape unchanged (zeros some elements) |
| `nn.ReLU()` | Activation | Shape unchanged |
| `nn.Embedding(num, dim)` | Lookup table | `(B, L)` → `(B, L, dim)` |

---

## nn.Sequential & nn.ModuleList

```python
# Sequential: layers applied in order
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# ModuleList: when you need indexing or loops
class Block(nn.Module):
    def __init__(self, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
```

**Warning**: Plain Python lists don't register parameters. Always use `nn.ModuleList` or `nn.ModuleDict`.

---

## Loss Functions

| Loss | Use Case | Formula | Note |
|---|---|---|---|
| `nn.MSELoss()` | Regression | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | |
| `nn.CrossEntropyLoss()` | Multi-class classification | $-\sum_c y_c \log \hat{p}_c$ | **Includes softmax** — pass raw logits |
| `nn.BCEWithLogitsLoss()` | Binary classification | $-[y\log\sigma(\hat{y}) + (1-y)\log(1-\sigma(\hat{y}))]$ | **Includes sigmoid** — pass raw logits |
| `nn.BCELoss()` | Binary classification | $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Expects probabilities (apply sigmoid first) |
| `nn.L1Loss()` | Robust regression | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | |
| `nn.NLLLoss()` | Classification | $-\log \hat{p}_{y_i}$ | Expects log-probabilities |

**`reduction` parameter**: `'mean'` (default), `'sum'`, `'none'` (per-element).

**Critical**: `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`. Do NOT apply softmax before it.

---

## Optimizers

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

| Optimizer | Key Args | When to Use |
|---|---|---|
| `SGD(params, lr, momentum)` | `momentum=0.9` | Simple, good generalization |
| `Adam(params, lr, betas, weight_decay)` | `lr=1e-3` | Default choice, adaptive LR |
| `AdamW(params, lr, weight_decay)` | `weight_decay=1e-2` | Proper L2 regularization |

### Learning Rate Schedulers

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# After each epoch:
scheduler.step()
```

| Scheduler | Behavior |
|---|---|
| `StepLR(optimizer, step_size, gamma)` | Multiply LR by $\gamma$ every `step_size` epochs |
| `CosineAnnealingLR(optimizer, T_max)` | Cosine decay: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T_{max}}\pi))$ |
| `ReduceLROnPlateau(optimizer, patience)` | Reduce LR when metric stops improving |

---

## Training Loop Template

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        logits = model(batch_x)             # (B, C)
        loss = criterion(logits, batch_y)   # scalar

        # Backward pass
        optimizer.zero_grad()               # Clear old gradients
        loss.backward()                     # Compute gradients
        optimizer.step()                    # Update parameters

    # Evaluation
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y)
```

**Order matters**: `zero_grad()` → `backward()` → `step()`. If you skip `zero_grad`, gradients accumulate.

---

## Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)           # Move model
x = x.to(device)                   # Move data
x = x.cuda()                       # Explicit GPU
x = x.cpu()                        # Explicit CPU
x.device                           # Check device
```

**Common bug**: Model on GPU + data on CPU (or vice versa) → RuntimeError. Always move both.

---

## Datasets & DataLoaders

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MyDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Shape Conventions

| Data Type | Shape | Meaning |
|---|---|---|
| Image batch | `(B, C, H, W)` | Batch, Channels, Height, Width |
| Sequence batch | `(B, L, D)` | Batch, Length, Dimension |
| Flat features | `(B, F)` | Batch, Features |
| Class labels | `(B,)` | One integer per sample |
| Logits | `(B, num_classes)` | Raw scores before softmax |

---

## Saving & Loading

```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

---

## Quick Debugging Checklist

| Problem | Check |
|---|---|
| `RuntimeError: size mismatch` | Print shapes at every layer: `print(x.shape)` |
| `RuntimeError: expected device cuda` | Move both model and data to same device |
| Loss is NaN | Learning rate too high, or division by zero |
| Loss not decreasing | Check `zero_grad()` is called, gradients exist |
| Validation loss much worse than train | Overfitting — add dropout/regularization, reduce model size |
| `requires_grad` is False | Pass `requires_grad=True` at creation or call `.requires_grad_(True)` |

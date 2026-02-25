# Training Loops

**Prerequisites**: All previous topics (tensors, autograd, nn.Module, datasets, losses, optimizers)
**USAAIO Relevance**: Round 2 requires writing complete training pipelines from scratch. There is no `model.fit()`. You must chain forward pass, loss computation, backward pass, and optimizer step correctly — and debug when things go wrong.

---

## Discovery

### Putting It All Together

The training loop is where every concept from this unit converges:

1. **Tensors** carry the data through the network
2. **nn.Module** defines the computation
3. **Loss functions** measure the error
4. **Autograd** computes gradients
5. **Optimizers** update parameters
6. **DataLoaders** feed mini-batches

The training loop is the heartbeat of deep learning. Each iteration:

```
Load batch → Forward → Compute loss → Zero grad → Backward → Step
```

> **Socratic question**: Why this specific order? Why must `zero_grad` come before `backward`? Because PyTorch accumulates gradients by default. If you skip `zero_grad`, you get the sum of gradients from all previous batches — your model learns garbage. And `step` must come after `backward` because the optimizer needs the freshly computed gradients.

Actually, the order `zero_grad → forward → loss → backward → step` and `forward → loss → zero_grad → backward → step` are both correct. What matters is that `zero_grad` happens before `backward`, and `step` happens after `backward`. The forward pass and loss computation can be before or after `zero_grad`.

### train() vs eval() — The Silent Bug

Forgetting `model.eval()` during validation is one of the most common bugs:

- **Dropout** randomly zeroes elements during `train()` but passes everything through during `eval()`
- **BatchNorm** uses batch statistics during `train()` but running statistics during `eval()`

If you evaluate with `model.train()`, your validation metrics will be noisy and potentially misleading.

---

## Intuition

### The Full Pipeline

```
                                    ┌──── EPOCH ────┐
                                    │                │
  ┌─────────────────────────────────┼────────────────┼──────────┐
  │ Training Phase                  │                │          │
  │                                 ↓                │          │
  │  model.train()           ┌───────────┐           │          │
  │       │                  │ DataLoader │           │          │
  │       ↓                  │ (shuffle)  │           │          │
  │  for batch in loader:    └─────┬─────┘           │          │
  │       │                        │                 │          │
  │       ↓                        ↓                 │          │
  │  ┌──────────┐          ┌──────────────┐          │          │
  │  │  Forward  │──────→  │  Compute     │          │          │
  │  │  Pass     │         │  Loss        │          │          │
  │  └──────────┘          └──────┬───────┘          │          │
  │                               │                  │          │
  │  ┌──────────┐          ┌──────┴───────┐          │          │
  │  │ zero_grad │←────── │  backward()  │          │          │
  │  └──────────┘          └──────┬───────┘          │          │
  │                               │                  │          │
  │                        ┌──────┴───────┐          │          │
  │                        │  step()      │──────────┘          │
  │                        └──────────────┘                     │
  │                                                             │
  │ Validation Phase                                            │
  │                                                             │
  │  model.eval()                                               │
  │  with torch.no_grad():                                      │
  │       for batch in val_loader:                              │
  │           forward pass only → compute metrics               │
  └─────────────────────────────────────────────────────────────┘
```

### Checkpointing Strategy

```
Epoch  1: val_loss = 0.45  → save checkpoint (best so far)
Epoch  2: val_loss = 0.38  → save checkpoint (new best)
Epoch  3: val_loss = 0.41  → skip (worse)
Epoch  4: val_loss = 0.35  → save checkpoint (new best)
...
Epoch 20: val_loss = 0.40  → patience exceeded → early stop
                              → load best checkpoint (epoch 4)
```

### GPU Training Checklist

```
1. device = torch.device('cuda' if available else 'cpu')
2. model = model.to(device)          ← move model
3. criterion = criterion.to(device)   ← move loss (if it has params)
4. For each batch:
   x, y = x.to(device), y.to(device) ← move data EVERY batch
```

Common mistake: moving the model but forgetting to move the data (or vice versa).

---

## Math

### Gradient Descent Convergence

For a convex function with $L$-Lipschitz gradient, SGD with learning rate $\eta = \frac{1}{L}$ converges at rate:

$$f(\theta_T) - f(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2T}$$

For non-convex functions (neural networks), SGD converges to a stationary point where $\|\nabla f\| \leq \epsilon$ in $O(1/\epsilon^2)$ steps.

### Mini-batch Gradient Estimator

The mini-batch gradient is an unbiased estimator of the full gradient:

$$\mathbb{E}[\nabla_{\theta} \mathcal{L}_B(\theta)] = \nabla_{\theta} \mathcal{L}(\theta)$$

Variance scales inversely with batch size:

$$\text{Var}[\nabla_B] = \frac{\sigma^2}{B}$$

---

## Code

### Minimal Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)              # (B, 784)
        batch_y = batch_y.to(device)              # (B,)

        # Forward
        logits = model(batch_x)                    # (B, 10)
        loss = criterion(logits, batch_y)          # scalar

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch_x.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += batch_x.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
```

### Complete Training Loop with Validation

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )
    scheduler.step()

    print(f"Epoch {epoch+1:3d} | "
          f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
```

### Checkpointing and Early Stopping

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, device
    )

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pt')
        print(f"  Saved new best model (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model for final evaluation
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")
```

### Full MNIST Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()                # (B, 1, 28, 28) → (B, 784)
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)                        # (B, 1, 28, 28) → (B, 784)
        return self.net(x)                         # (B, 784) → (B, 10)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1}: Test accuracy = {correct/total:.4f}")
```

### Training with Mixed Precision (GPU)

```python
# Mixed precision: use float16 for forward/backward, float32 for updates
# Faster and uses less memory on modern GPUs

scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Forward pass in mixed precision
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Logging Training Progress

```python
# Simple logging dictionary
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

# Plot (for Jupyter / local debugging)
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history['train_loss'], label='train')
ax1.plot(history['val_loss'], label='val')
ax1.set_title('Loss')
ax1.legend()
ax2.plot(history['train_acc'], label='train')
ax2.plot(history['val_acc'], label='val')
ax2.set_title('Accuracy')
ax2.legend()
plt.show()
```

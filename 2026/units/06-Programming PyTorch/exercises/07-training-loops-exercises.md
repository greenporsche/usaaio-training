# Training Loops Exercises

**Topic**: Full training pipeline, debugging, GPU training, evaluation
**Difficulty**: Intermediate → Advanced

---

## Exercise 1: Find the Bugs

This training loop has FIVE bugs. Find and fix all of them.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Training
    total_loss = 0
    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss

    # Validation
    correct = 0
    total = 0
    for images, labels in val_loader:
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, "
          f"val_acc={correct/total:.4f}")
```

<details>
<summary>Solution</summary>

Five bugs:

**Bug 1**: Missing `optimizer.zero_grad()` before `loss.backward()`. Gradients accumulate across batches.

**Bug 2**: Missing `model.train()` before the training loop. Dropout needs training mode to function correctly.

**Bug 3**: Missing `model.eval()` before validation. Dropout is still active during validation, giving noisy results.

**Bug 4**: Missing `torch.no_grad()` during validation. Unnecessary computation graph is built, wasting memory.

**Bug 5**: `total_loss += loss` accumulates tensor objects (with gradient graphs), causing memory leak. Should be `total_loss += loss.item()`.

Fixed code:

```python
for epoch in range(10):
    # Training
    model.train()                                          # Bug 2
    total_loss = 0
    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()                              # Bug 1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()                          # Bug 5

    # Validation
    model.eval()                                           # Bug 3
    correct = 0
    total = 0
    with torch.no_grad():                                  # Bug 4
        for images, labels in val_loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, "
          f"val_acc={correct/total:.4f}")
```

**Key insight**: These five bugs represent the most common training loop errors. Memorize the checklist: `model.train()`, `zero_grad()`, `loss.item()`, `model.eval()`, `torch.no_grad()`.
</details>

---

## Exercise 2: Training Loop from Scratch

Write a complete training pipeline for binary classification WITHOUT using any `nn` loss functions or optimizers. Implement everything manually:
- Forward pass through a 2-layer network
- Binary cross-entropy loss (with sigmoid)
- Manual gradient computation via autograd
- Manual parameter update (SGD)

```python
# Data: XOR problem
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Initialize parameters
W1 = torch.randn(2, 4, requires_grad=True)     # (in=2, hidden=4)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)     # (hidden=4, out=1)
b2 = torch.zeros(1, requires_grad=True)

lr = 0.5
for epoch in range(1000):
    # YOUR CODE: forward, loss, backward, update
    pass

# After training, should correctly classify XOR
with torch.no_grad():
    h = torch.sigmoid(X @ W1 + b1)
    pred = torch.sigmoid(h @ W2 + b2)
    print(f"Predictions: {pred.squeeze().tolist()}")
    # Should be close to [0, 1, 1, 0]
```

<details>
<summary>Solution</summary>

```python
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

lr = 0.5
for epoch in range(1000):
    # Forward pass
    h = torch.sigmoid(X @ W1 + b1)             # (4, 4)
    y_pred = torch.sigmoid(h @ W2 + b2)        # (4, 1)

    # Binary cross-entropy loss (manual)
    eps = 1e-7                                   # for numerical stability
    loss = -(y * torch.log(y_pred + eps) +
             (1 - y) * torch.log(1 - y_pred + eps)).mean()

    # Backward (autograd computes all gradients)
    loss.backward()

    # Manual SGD update
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

    # Zero gradients
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Test
with torch.no_grad():
    h = torch.sigmoid(X @ W1 + b1)
    pred = torch.sigmoid(h @ W2 + b2)
    print(f"Predictions: {[f'{p:.3f}' for p in pred.squeeze().tolist()]}")
    # Should be close to ['0.03', '0.97', '0.97', '0.03']
```

**Key insight**: The `with torch.no_grad()` block is essential during the update step. Without it, PyTorch would track the subtraction operation, creating an ever-growing computation graph. The manual update pattern shows exactly what the optimizer does under the hood.
</details>

---

## Exercise 3: Overfitting Detector

Write a function that takes training and validation loss histories and detects when the model starts overfitting. Return the epoch where validation loss first increases for 3 consecutive epochs.

```python
def detect_overfitting(train_losses, val_losses, patience=3):
    """
    Args:
        train_losses: list of per-epoch training losses
        val_losses: list of per-epoch validation losses
        patience: number of consecutive increases before declaring overfitting
    Returns:
        epoch_number where overfitting starts (or -1 if no overfitting detected)
    """
    # YOUR CODE HERE
    pass

# Test
train = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
val =   [0.9, 0.75, 0.65, 0.55, 0.52, 0.53, 0.55, 0.58, 0.62, 0.67]
#                                   ^--- best   ^--- starts increasing
epoch = detect_overfitting(train, val, patience=3)
print(f"Overfitting detected at epoch: {epoch}")  # Should be 5 (0-indexed)
```

<details>
<summary>Solution</summary>

```python
def detect_overfitting(train_losses, val_losses, patience=3):
    best_val = float('inf')
    increase_count = 0
    best_epoch = 0

    for epoch, val_loss in enumerate(val_losses):
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            increase_count = 0
        else:
            increase_count += 1

        if increase_count >= patience:
            return best_epoch

    return -1  # No overfitting detected

# Test
train = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
val =   [0.9, 0.75, 0.65, 0.55, 0.52, 0.53, 0.55, 0.58, 0.62, 0.67]

epoch = detect_overfitting(train, val, patience=3)
print(f"Overfitting detected at epoch: {epoch}")  # 4 (0-indexed best)
```

The best validation loss is at epoch 4 (0.52). After that, validation loss increases for 3+ epochs: 0.53, 0.55, 0.58. So overfitting starts at epoch 4.

**Key insight**: This is exactly the logic behind early stopping. In practice, you save the model checkpoint at `best_epoch` and stop training when patience is exceeded. The gap between training loss (decreasing) and validation loss (increasing) is the textbook overfitting signal.
</details>

---

## Exercise 4: GPU Training Conversion

Convert this CPU training loop to work on GPU. Mark every line that needs to change.

```python
model = MNISTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

<details>
<summary>Solution</summary>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # NEW
model = MNISTNet().to(device)                                           # CHANGED
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)               # unchanged
criterion = nn.CrossEntropyLoss()                                       # unchanged*

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)                                      # NEW
        labels = labels.to(device)                                      # NEW
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Changes:
1. **Define device** (line 1)
2. **Move model to device** with `.to(device)` (line 2)
3. **Move each batch to device** with `.to(device)` (inside loop, two lines)

*Note: `nn.CrossEntropyLoss()` does not need `.to(device)` because it has no parameters. Losses with learnable parameters (rare) would need it.

The optimizer does NOT need `.to(device)` — it references model parameters, which are already on GPU.

**Key insight**: Only three things need to move to GPU: the model, the input data, and the labels. Everything else (optimizer, loss function, training logic) stays the same. The optimizer's internal state (momentum buffers, etc.) automatically lives on the same device as the parameters.
</details>

---

## Exercise 5: Complete Training Pipeline

Write a complete, production-quality training function that includes ALL best practices: train/eval modes, no_grad, checkpointing, early stopping, learning rate scheduling, and logging. Fill in the blanks.

```python
def train(model, train_loader, val_loader, num_epochs, lr, device, patience=5):
    """Complete training pipeline with all best practices."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # === Training phase ===
        # YOUR CODE HERE (set mode, iterate, forward, backward, update, track metrics)

        # === Validation phase ===
        # YOUR CODE HERE (set mode, no_grad, iterate, track metrics)

        # === Scheduling ===
        # YOUR CODE HERE

        # === Checkpointing + Early stopping ===
        # YOUR CODE HERE

        # === Logging ===
        # YOUR CODE HERE

    # Load best model
    model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
    return model, history
```

<details>
<summary>Solution</summary>

```python
def train(model, train_loader, val_loader, num_epochs, lr, device, patience=5):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # === Training phase ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # === Validation phase ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # === Scheduling ===
        scheduler.step()

        # === Checkpointing + Early stopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # === Logging ===
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    return model, history
```

**Key insight**: This is the complete training template you should have memorized for USAAIO. The critical elements are: `model.train()`/`model.eval()`, `torch.no_grad()`, `loss.item()`, `zero_grad` before `backward`, gradient clipping, checkpointing on validation improvement, early stopping, and LR scheduling. Miss any one of these and you introduce a subtle bug.
</details>

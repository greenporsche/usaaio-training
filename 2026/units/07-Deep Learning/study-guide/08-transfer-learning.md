# Transfer Learning

**Prerequisites**: CNNs (Study Guide 06), architectures (Study Guide 07)
**USAAIO Relevance**: Round 2 frequently includes transfer learning pipelines. You must know how to load pretrained models, freeze layers, replace classification heads, and fine-tune with proper learning rate strategies. Data augmentation is tested alongside transfer learning.

---

## Discovery

### Why Train from Scratch?

Training a ResNet-50 from scratch on ImageNet takes days on multiple GPUs and requires 1.2 million labeled images. What if you have a dataset with only 1,000 images?

> **Socratic question**: A model trained on ImageNet (dogs, cats, cars, planes) learned to detect edges, textures, shapes, and object parts. Would these features be useful for medical image classification (X-rays, MRIs)?
>
> *Yes! Early layers learn universal features (edges, corners, textures) that transfer across domains. Later layers learn domain-specific features (dog ears, car wheels) that may need fine-tuning.*

This is the key insight of **transfer learning**: features learned on one task can be reused for another. Instead of training from random weights, start from a model pre-trained on a large dataset and adapt it to your specific task.

### Three Strategies

1. **Feature extraction**: Freeze the entire pretrained model. Use it as a fixed feature extractor. Train only a new classification head.

2. **Fine-tuning (partial)**: Freeze early layers (universal features). Unfreeze later layers + head. Train with a smaller learning rate.

3. **Full fine-tuning**: Unfreeze everything. Train with a very small learning rate to avoid destroying pretrained features.

```
Strategy spectrum:

Freeze everything ◄──────────────────────► Unfreeze everything
(Feature extraction)    (Partial fine-tune)    (Full fine-tune)

Small dataset,          Medium dataset,         Large dataset,
similar domain          somewhat different      very different
```

### When to Use Which

| Dataset Size | Domain Similarity | Strategy |
|---|---|---|
| Small (< 1K) | Similar | Feature extraction |
| Small (< 1K) | Different | Feature extraction (may fail) |
| Medium (1K–100K) | Similar | Fine-tune last few layers |
| Medium (1K–100K) | Different | Fine-tune more layers |
| Large (100K+) | Any | Full fine-tuning or train from scratch |

---

## Intuition

### What Pretrained Layers Learn

Research (Zeiler & Fergus 2014) showed that CNN layers learn increasingly abstract features:

```
Layer 1: Edges, colors       Layer 2: Corners, textures
┌────────┐                   ┌────────┐
│ ╱  ╲   │                   │ ╱╲  ╲╱ │
│ │  ─   │                   │ grid   │
│ ╲  •   │                   │ spots  │
└────────┘                   └────────┘

Layer 3: Parts, patterns     Layer 4-5: Object parts, full objects
┌────────┐                   ┌────────┐
│ wheels │                   │ 🐕 dog │
│ eyes   │                   │ 🚗 car │
│ text   │                   │ 🌸 flower│
└────────┘                   └────────┘

Universal ◄──────────────────────────────► Domain-specific
(transfer well)                             (may need fine-tuning)
```

### The Freeze/Unfreeze Pattern

```
Feature Extraction:          Fine-tuning:
┌──────────┐                 ┌──────────┐
│ New Head │ ← train          │ New Head │ ← train (lr = 1e-3)
├──────────┤                 ├──────────┤
│ Layer 5  │ ← FROZEN         │ Layer 5  │ ← train (lr = 1e-5)
│ Layer 4  │ ← FROZEN         │ Layer 4  │ ← train (lr = 1e-5)
│ Layer 3  │ ← FROZEN         │ Layer 3  │ ← FROZEN
│ Layer 2  │ ← FROZEN         │ Layer 2  │ ← FROZEN
│ Layer 1  │ ← FROZEN         │ Layer 1  │ ← FROZEN
└──────────┘                 └──────────┘
```

### Data Augmentation

When fine-tuning with limited data, augmentation is essential to prevent overfitting:

```
Original image → Random transformations:

┌─────┐     ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ 🐕  │  →  │ 🐕↔ │  │ 🐕🔄│  │🌓🐕 │  │ 🐕✂ │
│     │     │flip │  │rotate│  │color │  │crop  │
└─────┘     └─────┘  └─────┘  └─────┘  └─────┘
```

Each epoch, the model sees slightly different versions of the same image. This acts as a regularizer.

---

## Math

### Feature Extraction as Representation Learning

A pretrained model $f$ can be decomposed as:

$$f(x) = h(g(x))$$

where $g: \mathbb{R}^{C \times H \times W} \to \mathbb{R}^d$ is the feature extractor (all layers except the last) and $h: \mathbb{R}^d \to \mathbb{R}^K$ is the classification head.

In feature extraction, we fix $g$ and learn a new $h'$:

$$\hat{y} = h'(g(x)), \quad h': \mathbb{R}^d \to \mathbb{R}^{K'}$$

where $K'$ is the number of classes in our new task.

The feature $g(x) \in \mathbb{R}^d$ is a learned representation — a compressed, semantically meaningful encoding of the image.

### Learning Rate Schedules for Fine-Tuning

**Discriminative learning rates**: Use different learning rates for different parts of the network.

For layer group $i$ at depth $d_i$ (deeper = closer to output):

$$\text{lr}_i = \text{lr}_{\text{base}} \times \gamma^{L - d_i}$$

where $\gamma < 1$ (e.g., $\gamma = 0.1$). Earlier layers get exponentially smaller learning rates.

**Linear warmup**: Start with a very small LR and increase linearly:

$$\text{lr}(t) = \text{lr}_{\text{target}} \times \frac{t}{T_{\text{warmup}}}$$

for steps $t \leq T_{\text{warmup}}$. This prevents large gradient updates from destroying pretrained features in the first few steps.

### Augmentation as Regularization

Data augmentation can be viewed as expanding the training distribution. For an augmentation transform $T$ (random crop, flip, color jitter), the effective training set becomes:

$$\mathcal{D}_{\text{aug}} = \{(T(x_i), y_i) : (x_i, y_i) \in \mathcal{D}, T \sim p(T)\}$$

This is equivalent to adding a regularization term to the loss that encourages invariance to $T$.

---

## Code

### Feature Extraction Pipeline

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load pretrained ResNet-18
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False                         # No gradient computation

# Replace the classification head
num_classes = 5  # Our custom task
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Only this is trainable
# model.fc.in_features = 512 for ResNet-18

# Only optimize the new head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# Verify: only fc layer has gradients
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
# Trainable: 2,565 / 11,179,077 (0.0%)
```

### Fine-Tuning with Layer Freezing

```python
import torch
import torch.nn as nn
import torchvision

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Freeze early layers (layer1, layer2)
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name or 'conv1' in name or 'bn1' in name:
        param.requires_grad = False

# Replace head
model.fc = nn.Linear(512, 5)

# Discriminative learning rates
param_groups = [
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
optimizer = torch.optim.Adam(param_groups)
```

### Data Augmentation Pipeline

```python
import torchvision.transforms as T

# Training transforms (with augmentation)
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),     # Random crop + resize
    T.RandomHorizontalFlip(p=0.5),                   # 50% chance horizontal flip
    T.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1),           # Color perturbation
    T.RandomRotation(15),                             # ±15 degree rotation
    T.ToTensor(),                                     # PIL → tensor, scale to [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],          # ImageNet normalization
                std=[0.229, 0.224, 0.225]),
])

# Validation/Test transforms (NO augmentation)
val_transform = T.Compose([
    T.Resize(256),                                    # Resize shorter side to 256
    T.CenterCrop(224),                                # Center crop to 224×224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
```

### Complete Transfer Learning Training Pipeline

```python
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

def train_transfer_learning(
    model, train_loader, val_loader, optimizer,
    num_epochs=10, device='cpu'
):
    """Full transfer learning training loop."""
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)                     # (B, num_classes)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} | "
              f"Train Acc: {train_correct/train_total:.4f} | "
              f"Val Acc: {val_correct/val_total:.4f}")
```

### Progressive Unfreezing

```python
def unfreeze_layer(model, layer_name):
    """Unfreeze a specific layer group."""
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True

# Training schedule:
# Phase 1: Train only the head (5 epochs)
# Phase 2: Unfreeze layer4 (5 epochs)
# Phase 3: Unfreeze layer3 (5 epochs)

# Phase 1
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
# ... train for 5 epochs ...

# Phase 2
unfreeze_layer(model, 'layer4')
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
# ... train for 5 epochs ...

# Phase 3
unfreeze_layer(model, 'layer3')
optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
# ... train for 5 epochs ...
```

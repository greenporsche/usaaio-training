# nn.Module

**Prerequisites**: Python classes (OOP), tensors, autograd
**USAAIO Relevance**: Every Round 2 problem requires building custom modules from scratch. You cannot use high-level wrappers like `nn.MultiheadAttention` — you must compose from `nn.Linear`, `nn.Parameter`, and basic operations.

---

## Discovery

### The Building Blocks of Neural Networks

A neural network is just a function that transforms inputs to outputs. But a function with millions of parameters needs structure. PyTorch's answer is `nn.Module` — a base class that:

1. **Registers parameters** so the optimizer can find them
2. **Composes hierarchically** — modules contain modules, forming a tree
3. **Manages state** — training vs. evaluation mode, device placement

Every layer in PyTorch (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm1d`) is an `nn.Module`. Your custom models are also `nn.Module` subclasses. The entire model, from a single linear layer to GPT-4, is a tree of modules.

> **Socratic question**: Why not just use plain functions with global variables for parameters? Think about what happens when you have 50 layers, each with weights and biases. How do you collect all parameters for the optimizer? How do you move them all to GPU? `nn.Module` solves this with automatic parameter registration.

### The Module Contract

When you subclass `nn.Module`, you agree to:

1. Call `super().__init__()` in your `__init__`
2. Define `forward()` — this is the computation
3. Register all learnable parameters as `nn.Parameter` or as sub-modules (which have their own parameters)
4. **Never** call `forward()` directly — call the module itself: `output = model(input)`

Why not call `forward()` directly? Because `model(input)` calls `__call__`, which runs hooks (for debugging, profiling) before and after `forward()`. Always use `model(input)`.

---

## Intuition

### The Module Tree

```
MyModel (nn.Module)
├── self.encoder (nn.Sequential)
│   ├── 0: nn.Linear(784, 256)
│   │   ├── weight: Parameter (256, 784)
│   │   └── bias: Parameter (256,)
│   ├── 1: nn.ReLU()
│   └── 2: nn.Linear(256, 128)
│       ├── weight: Parameter (128, 256)
│       └── bias: Parameter (128,)
├── self.classifier (nn.Linear(128, 10))
│   ├── weight: Parameter (10, 128)
│   └── bias: Parameter (10,)
└── self.dropout (nn.Dropout(0.5))
```

- `model.parameters()` traverses this tree and yields every `Parameter`
- `model.to('cuda')` moves every parameter to GPU
- `model.state_dict()` serializes the tree for saving/loading

### nn.Parameter vs Regular Tensor

```
┌──────────────────────────────────────┐
│ nn.Parameter(tensor)                 │
│  ├── requires_grad = True (default)  │
│  ├── Registered in module            │
│  └── Found by model.parameters()     │
│                                      │
│ Regular tensor (self.x = tensor)     │
│  ├── requires_grad = depends         │
│  ├── NOT registered                  │
│  └── NOT found by parameters()       │
│      (invisible to optimizer!)       │
└──────────────────────────────────────┘
```

If you need a learnable value that is not a standard layer, wrap it in `nn.Parameter`:

```python
self.scale = nn.Parameter(torch.ones(1))   # learnable scalar — optimizer will update it
```

### train() vs eval()

```
model.train()                    model.eval()
  ├── Dropout: active              ├── Dropout: disabled
  ├── BatchNorm: uses batch        ├── BatchNorm: uses running
  │   statistics                   │   statistics
  └── Other: no effect             └── Other: no effect
```

**Always** set mode correctly before forward pass. Forgetting `model.eval()` during validation is a common bug.

---

## Math

### Linear Layer

`nn.Linear(in_features, out_features)` computes:

$$y = xW^T + b$$

where $W \in \mathbb{R}^{\text{out} \times \text{in}}$, $b \in \mathbb{R}^{\text{out}}$, and $x \in \mathbb{R}^{B \times \text{in}}$.

Note: the weight matrix is stored as `(out, in)`, not `(in, out)`. This is because PyTorch computes $xW^T$, transposing the weight.

### Kaiming Initialization

By default, `nn.Linear` uses Kaiming uniform initialization:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{1}{\text{fan\_in}}}, \sqrt{\frac{1}{\text{fan\_in}}}\right)$$

This keeps activations and gradients at approximately unit variance across layers, preventing vanishing/exploding signals.

---

## Code

### Minimal Custom Module

```python
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()                        # MUST call super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)  # Registered as sub-module
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (B, in_dim)
        x = torch.relu(self.fc1(x))               # (B, in_dim) → (B, hidden_dim)
        x = self.fc2(x)                            # (B, hidden_dim) → (B, out_dim)
        return x                                   # (B, out_dim)

model = TwoLayerNet(784, 256, 10)
print(model)
# TwoLayerNet(
#   (fc1): Linear(in_features=784, out_features=256, bias=True)
#   (fc2): Linear(in_features=256, out_features=10, bias=True)
# )
```

### Inspecting Parameters

```python
model = TwoLayerNet(784, 256, 10)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total}")
# 784*256 + 256 + 256*10 + 10 = 203,530

# Named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# fc1.weight: torch.Size([256, 784])
# fc1.bias: torch.Size([256])
# fc2.weight: torch.Size([10, 256])
# fc2.bias: torch.Size([10])

# State dict (for saving/loading)
state = model.state_dict()
print(state.keys())
# odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
```

### nn.Sequential

```python
# When layers are applied in order, use Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

x = torch.randn(32, 784)
output = model(x)                              # (32, 10)
```

### nn.ModuleList

```python
# When you need dynamic or indexed access to layers
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.fc(x)) + x      # Residual connection

class StackedResidual(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        # ModuleList registers all sub-modules
        self.blocks = nn.ModuleList([
            ResidualBlock(dim) for _ in range(n_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

model = StackedResidual(128, 4)
print(sum(p.numel() for p in model.parameters()))
# 4 * (128*128 + 128) = 66,048
```

### nn.ModuleDict

```python
class MultiTaskModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, task_dims):
        super().__init__()
        self.shared = nn.Linear(in_dim, hidden_dim)
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in task_dims.items()
        })

    def forward(self, x, task_name):
        x = torch.relu(self.shared(x))        # (B, in_dim) → (B, hidden_dim)
        return self.heads[task_name](x)        # (B, hidden_dim) → (B, task_dim)

model = MultiTaskModel(784, 256, {'classify': 10, 'regress': 1})
```

### nn.Parameter for Custom Learnable Values

```python
class ScaledLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Custom learnable scalar — optimizer will update this
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.linear(x) / self.temperature    # Learnable temperature scaling
```

### Custom Activation as nn.Module

```python
class Swish(nn.Module):
    """Swish activation: x * sigmoid(beta * x) with learnable beta."""
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
```

### Building MHA from Scratch (USAAIO Style)

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from scratch.
    USAAIO requires building this — no nn.MultiheadAttention allowed.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)   # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape                         # (B, L, d_model)

        Q = self.W_q(x)                            # (B, L, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (B, n_heads, L, d_k)
        Q = Q.reshape(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)                    # (B, H, L, L)
        context = attn @ V                                      # (B, H, L, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).reshape(B, L, D)     # (B, L, d_model)
        return self.W_o(context)                                # (B, L, d_model)
```

### Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TwoLayerNet(784, 256, 10).to(device)   # Move ALL parameters to device

# Verify
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")             # All should show cuda:0

# Input must match device
x = torch.randn(32, 784).to(device)
output = model(x)                                # Works — both on same device
```

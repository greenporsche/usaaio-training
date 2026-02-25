# 01 — Physics-Informed Neural Networks (PINNs)

> This is the **most critical file** in Unit 12. The 2025 USAAIO Round 2 Problem 1 (100 points) was entirely about PINNs for the heat equation. Master this material thoroughly.

---

## Discovery

### The Problem with Data-Driven Learning

Standard neural networks learn from data: given input-output pairs $(x_i, y_i)$, minimize $\sum \|f_\theta(x_i) - y_i\|^2$. But many physical systems are governed by known differential equations. If you have 1000 data points and a governing PDE, why throw away the PDE?

### The PINN Insight (Raissi et al., 2019)

**Key idea:** Encode the PDE directly into the loss function. Instead of (or in addition to) fitting data, penalize the neural network for violating the physics.

A neural network $u_\theta(t, x)$ approximates the solution to a PDE. The loss has three components:

$$\mathcal{L} = \lambda_{PDE}\mathcal{L}_{PDE} + \lambda_{IC}\mathcal{L}_{IC} + \lambda_{BC}\mathcal{L}_{BC}$$

Where:
- $\mathcal{L}_{PDE}$: How badly the network violates the PDE at random interior points
- $\mathcal{L}_{IC}$: How badly the network violates initial conditions
- $\mathcal{L}_{BC}$: How badly the network violates boundary conditions

No labeled solution data needed — the physics **is** the supervision signal.

### Why This Matters for USAAIO

Round 2 tested exactly this pipeline:
1. Given a PDE and its analytical solution
2. Prove the solution satisfies the PDE (math)
3. Build a neural network to approximate the solution (code)
4. Train it using physics-informed loss (code)
5. Evaluate against the analytical solution (code)

---

## Intuition

### PDE as a Constraint

Consider the 1D heat equation:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Or equivalently: $u_t - \alpha u_{xx} = 0$.

This says: "At every point $(t, x)$ in the domain, the time derivative of $u$ equals $\alpha$ times the spatial second derivative." A neural network that satisfies this equation at many random points is likely a good approximation of the true solution.

### The Three Constraint Types

**PDE residual** — sample random points $(t_i, x_i)$ in the interior of the domain. Compute $u_t - \alpha u_{xx}$ using automatic differentiation. This should be zero:

$$\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left| \frac{\partial u_\theta}{\partial t}(t_i, x_i) - \alpha \frac{\partial^2 u_\theta}{\partial x^2}(t_i, x_i) \right|^2$$

**Initial condition** — at $t = 0$, the solution must equal the prescribed initial profile $h(x)$:

$$\mathcal{L}_{IC} = \frac{1}{N_{IC}} \sum_{j=1}^{N_{IC}} |u_\theta(0, x_j) - h(x_j)|^2$$

**Boundary conditions** — at spatial boundaries (e.g., $x = 0$ and $x = L$), the solution must satisfy prescribed values:

$$\mathcal{L}_{BC} = \frac{1}{N_{BC}} \sum_{k=1}^{N_{BC}} |u_\theta(t_k, 0) - g_0(t_k)|^2 + |u_\theta(t_k, L) - g_L(t_k)|^2$$

### Visual Intuition

```
Domain: t ∈ [0, T], x ∈ [0, L]

     t=T ┌─────────────────────┐
         │   PDE residual      │
         │   (random points)   │
         │        • •  •       │
         │     •    •     •    │
         │  •     •    •    •  │
     t=0 └─────────────────────┘
         x=0    IC (t=0)      x=L
         │                     │
         BC                    BC
         (x=0)                 (x=L)
```

---

## Mastery

### The 1D Heat Equation — Complete Walkthrough

#### Problem Setup

**PDE:** $u_t - \alpha u_{xx} = 0$ on $t \in [0, 1]$, $x \in [0, 1]$

**IC:** $u(0, x) = \sin(\pi x)$

**BC (Dirichlet):** $u(t, 0) = 0$, $u(t, 1) = 0$

**Analytical solution:** $u(t, x) = e^{-\alpha \pi^2 t} \sin(\pi x)$

#### Step 1: Verify the Analytical Solution

**Prove** that $u(t, x) = e^{-\alpha \pi^2 t} \sin(\pi x)$ satisfies the PDE, IC, and BC.

**PDE verification:**

$$u_t = \frac{\partial}{\partial t}\left[e^{-\alpha\pi^2 t}\sin(\pi x)\right] = -\alpha\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)$$

$$u_x = \frac{\partial}{\partial x}\left[e^{-\alpha\pi^2 t}\sin(\pi x)\right] = \pi e^{-\alpha\pi^2 t}\cos(\pi x)$$

$$u_{xx} = \frac{\partial^2}{\partial x^2}\left[e^{-\alpha\pi^2 t}\sin(\pi x)\right] = -\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)$$

$$u_t - \alpha u_{xx} = -\alpha\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x) - \alpha\left(-\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)\right) = 0 \quad\checkmark$$

**IC verification:**

$$u(0, x) = e^{0}\sin(\pi x) = \sin(\pi x) \quad\checkmark$$

**BC verification:**

$$u(t, 0) = e^{-\alpha\pi^2 t}\sin(0) = 0 \quad\checkmark$$

$$u(t, 1) = e^{-\alpha\pi^2 t}\sin(\pi) = 0 \quad\checkmark$$

#### Step 2: Build the Neural Network

```python
import torch
import torch.nn as nn

class HeatPINN(nn.Module):
    """
    Neural network approximation of u(t, x).
    Input: (t, x) concatenated -> shape (B, 2)
    Output: u(t, x) -> shape (B, 1)
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tx):
        # tx: (B, 2) where tx[:, 0] = t, tx[:, 1] = x
        return self.net(tx)  # (B, 1)
```

**Architecture choices:**
- **Tanh activation** — smooth and differentiable everywhere. Important because we need second-order derivatives. ReLU has zero second derivative almost everywhere and would break PDE loss computation.
- **2 inputs, 1 output** — the network learns the mapping $(t, x) \mapsto u$.
- **4 hidden layers of 64** — sufficient for smooth solutions. More complex PDEs may need wider/deeper networks.

#### Step 3: Create Training Datasets

**PDE collocation points** — random points in the interior:

```python
from torch.utils.data import Dataset, DataLoader

class PDEDataset(Dataset):
    """Random (t, x) points in the domain interior."""
    def __init__(self, n_points, t_range=(0, 1), x_range=(0, 1)):
        self.t = torch.rand(n_points, 1) * (t_range[1] - t_range[0]) + t_range[0]
        self.x = torch.rand(n_points, 1) * (x_range[1] - x_range[0]) + x_range[0]
        self.tx = torch.cat([self.t, self.x], dim=1)  # (N, 2)

    def __len__(self):
        return len(self.tx)

    def __getitem__(self, idx):
        return self.tx[idx]
```

**IC dataset** — points at $t = 0$:

```python
class ICDataset(Dataset):
    """Points at t=0 with known initial condition."""
    def __init__(self, n_points, x_range=(0, 1)):
        self.x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
        self.t = torch.zeros(n_points, 1)
        self.tx = torch.cat([self.t, self.x], dim=1)  # (N, 2)
        self.u = torch.sin(torch.pi * self.x)          # IC: sin(πx)

    def __len__(self):
        return len(self.tx)

    def __getitem__(self, idx):
        return self.tx[idx], self.u[idx]
```

**BC dataset** — points at $x = 0$ and $x = 1$:

```python
class BCDataset(Dataset):
    """Points at x=0 and x=1 with known boundary values."""
    def __init__(self, n_points, t_range=(0, 1)):
        t_vals = torch.linspace(t_range[0], t_range[1], n_points).unsqueeze(1)
        # x = 0 boundary
        tx_left = torch.cat([t_vals, torch.zeros(n_points, 1)], dim=1)
        u_left = torch.zeros(n_points, 1)
        # x = 1 boundary
        tx_right = torch.cat([t_vals, torch.ones(n_points, 1)], dim=1)
        u_right = torch.zeros(n_points, 1)
        # Combine both boundaries
        self.tx = torch.cat([tx_left, tx_right], dim=0)   # (2N, 2)
        self.u = torch.cat([u_left, u_right], dim=0)      # (2N, 1)

    def __len__(self):
        return len(self.tx)

    def __getitem__(self, idx):
        return self.tx[idx], self.u[idx]
```

#### Step 4: Compute Derivatives with autograd

This is the **core technical skill** for PINNs:

```python
def compute_pde_residual(model, tx, alpha=0.01):
    """
    Compute the PDE residual: u_t - alpha * u_xx
    tx must have requires_grad=True
    """
    tx = tx.requires_grad_(True)
    u = model(tx)  # (B, 1)

    # Compute gradient of u w.r.t. (t, x)
    # grad_outputs=torch.ones_like(u) because u is (B, 1), not scalar
    grads = torch.autograd.grad(
        outputs=u,
        inputs=tx,
        grad_outputs=torch.ones_like(u),
        create_graph=True  # CRITICAL: needed for second derivatives
    )[0]  # (B, 2)

    u_t = grads[:, 0:1]  # ∂u/∂t, shape (B, 1)
    u_x = grads[:, 1:2]  # ∂u/∂x, shape (B, 1)

    # Second derivative: ∂²u/∂x²
    grads2 = torch.autograd.grad(
        outputs=u_x,
        inputs=tx,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]  # (B, 2)

    u_xx = grads2[:, 1:2]  # ∂²u/∂x², shape (B, 1)

    residual = u_t - alpha * u_xx  # Should be 0
    return residual
```

**Why `create_graph=True`?**

Without it, `autograd.grad` computes the derivative but does not build a computation graph for the derivative itself. We need `create_graph=True` for two reasons:
1. We need second derivatives ($u_{xx}$ depends on $u_x$, which is itself a derivative)
2. We need to backpropagate through the PDE loss to update model parameters

**Why `grad_outputs=torch.ones_like(u)`?**

`autograd.grad` computes vector-Jacobian products. When $u$ is a vector (batch of values), we need to specify the "vector" part. Using `ones_like(u)` means we want $\sum_i \frac{\partial u_i}{\partial \text{inputs}}$, which for independent samples gives us the per-sample gradients.

**Alternative pattern** using `torch.sum`:
```python
u = model(tx)
u_x = torch.autograd.grad(torch.sum(u), tx, create_graph=True)[0][:, 1:2]
```

This is equivalent and sometimes cleaner. `torch.sum(u)` is a scalar, so no `grad_outputs` needed.

#### Step 5: Training Loop

```python
# Setup
model = HeatPINN(hidden_dim=64, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
alpha = 0.01

# Data
pde_dataset = PDEDataset(n_points=10000)
pde_loader = DataLoader(pde_dataset, batch_size=256, shuffle=True)
ic_dataset = ICDataset(n_points=100)
bc_dataset = BCDataset(n_points=100)

# Get FULL IC and BC data (not mini-batched!)
tx_ic, u_ic = ic_dataset.tx, ic_dataset.u
tx_bc, u_bc = bc_dataset.tx, bc_dataset.u

# Train
for epoch in range(1000):
    epoch_loss = 0.0
    for tx_pde in pde_loader:
        # PDE loss
        residual = compute_pde_residual(model, tx_pde, alpha)
        loss_pde = torch.mean(residual ** 2)

        # IC loss (full data, every step)
        u_ic_pred = model(tx_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

        # BC loss (full data, every step)
        u_bc_pred = model(tx_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc) ** 2)

        # Total loss
        loss = loss_pde + loss_ic + loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")
```

**Critical design decision:** IC and BC data use **full batch** every step, not mini-batches. Why?

- PDE collocation points are soft constraints — the network should approximately satisfy the PDE at many points. Random subsets work fine.
- IC and BC are **hard constraints** — the solution must exactly satisfy them. Subsampling causes the network to "forget" boundary conditions between updates, leading to unstable training.

#### Step 6: Evaluation

```python
# Create test grid
t_test = torch.linspace(0, 1, 50)
x_test = torch.linspace(0, 1, 50)
T, X = torch.meshgrid(t_test, x_test, indexing='ij')
tx_test = torch.stack([T.flatten(), X.flatten()], dim=1)

# Predicted solution
with torch.no_grad():
    u_pred = model(tx_test).reshape(50, 50)

# Analytical solution
u_exact = torch.exp(-alpha * torch.pi**2 * T) * torch.sin(torch.pi * X)

# Error
error = torch.abs(u_pred - u_exact)
print(f"Max error: {error.max().item():.6f}")
print(f"Mean error: {error.mean().item():.6f}")
```

### Extending to Other PDEs

The PINN framework is **PDE-agnostic**. To solve a different PDE:

1. Change the residual computation (different derivatives, different equation)
2. Change the IC/BC data
3. Everything else stays the same

**Wave equation:** $u_{tt} = c^2 u_{xx}$

```python
def wave_residual(model, tx, c=1.0):
    tx = tx.requires_grad_(True)
    u = model(tx)
    grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    grads_tt = torch.autograd.grad(u_t, tx, torch.ones_like(u_t), create_graph=True)[0]
    u_tt = grads_tt[:, 0:1]
    u_x = grads[:, 1:2]
    grads_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grads_xx[:, 1:2]
    return u_tt - c**2 * u_xx
```

**Burgers' equation:** $u_t + u \cdot u_x = \nu u_{xx}$

```python
def burgers_residual(model, tx, nu=0.01):
    tx = tx.requires_grad_(True)
    u = model(tx)
    grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    grads2 = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grads2[:, 1:2]
    return u_t + u * u_x - nu * u_xx
```

---

## Connection

### PINNs Connect Multiple Units

| Concept | Source Unit | How It Appears in PINNs |
|---------|------------|------------------------|
| Neural network architecture | Unit 5 | The approximation network |
| Loss function design | Unit 5 | Compound physics-informed loss |
| Automatic differentiation | Unit 5 | `torch.autograd.grad` for PDE derivatives |
| Dataset and DataLoader | Unit 3 | PDE, IC, BC datasets |
| Training loop | Unit 5 | Mini-batch PDE + full IC/BC |
| Optimization | Unit 12.02 | Adam, L-BFGS for PINNs |
| Calculus / PDEs | Mathematics | Verifying analytical solutions |

### From PINNs to the Exam

The 2025 Round 2 Problem 1 followed this exact structure:

1. **(Non-coding)** Prove the analytical solution satisfies the PDE — **substitution and simplification**
2. **(Coding)** Build `HeatPINN` as an `nn.Module` — **standard MLP with Tanh**
3. **(Non-coding)** Analyze output shapes — **trace tensor dimensions**
4. **(Coding)** Create PDE dataset and DataLoader — **`torch.rand` for random points**
5. **(Coding)** Create IC and BC datasets — **`torch.linspace` for grid points**
6. **(Coding)** Set up optimizer — **`Adam` with learning rate**
7. **(Coding)** Use `autograd.grad` to compute derivatives — **the critical skill**
8. **(Coding)** Write training loop — **compound loss, mini-batch PDE, full IC/BC**
9. **(Non-coding)** Explain why full IC/BC data — **constraints vs. soft penalties**
10. **(Coding)** Evaluate and visualize — **compare against analytical solution**

Expect similar structure in 2026 with a different PDE.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| PINN idea | Encode PDE in loss, not just data |
| Loss components | $\mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}$ |
| Activation | Tanh (smooth, twice differentiable) |
| autograd key | `create_graph=True` for higher-order derivatives |
| Training data | Mini-batch PDE, FULL IC/BC |
| Heat equation | $u_t = \alpha u_{xx}$, solution: $e^{-\alpha\pi^2 t}\sin(\pi x)$ |
| Generalization | Change the residual function for any PDE |

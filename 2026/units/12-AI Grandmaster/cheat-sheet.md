# Unit 12: AI Grandmaster (AI 900) — Cheat Sheet

## Physics-Informed Neural Networks (PINNs)

### Core Idea
Train a neural network $u_\theta(t, x)$ so that it satisfies a PDE, initial conditions (IC), and boundary conditions (BC) **simultaneously** through a compound loss.

### PINN Loss

$$\mathcal{L} = \lambda_{PDE}\|N[u_\theta] - f\|^2 + \lambda_{BC}\|B[u_\theta] - g\|^2 + \lambda_{IC}\|u_\theta(x,0) - h(x)\|^2$$

Where $N[\cdot]$ is the differential operator, $B[\cdot]$ is the boundary operator, $f$ is the PDE source, $g$ is the BC data, $h$ is the IC data.

### Heat Equation

$$u_t = \alpha u_{xx}$$

Analytical solution (Dirichlet BC on $[0,1]$):

$$u(t, x) = e^{-\alpha \pi^2 t} \sin(\pi x)$$

Verification: $u_t = -\alpha\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)$, $u_{xx} = -\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)$, so $u_t - \alpha u_{xx} = 0$ ✓

### Wave Equation

$$u_{tt} = c^2 u_{xx}$$

D'Alembert solution: $u(t, x) = f(x - ct) + g(x + ct)$

Standing wave: $u(t, x) = \cos(c\pi t)\sin(\pi x)$

### PyTorch autograd for Derivatives

```python
# First-order derivative
u = model(tx)  # tx requires grad
u_t = torch.autograd.grad(
    outputs=u, inputs=tx,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0][:, 0:1]  # ∂u/∂t

u_x = torch.autograd.grad(
    outputs=u, inputs=tx,
    grad_outputs=torch.ones_like(u),
    create_graph=True
)[0][:, 1:2]  # ∂u/∂x
```

**For vector outputs** (batch of scalars):
```python
u_x = torch.autograd.grad(
    torch.sum(U), inputs,
    create_graph=True
)[0]
```

**Second-order derivative:**
```python
u_xx = torch.autograd.grad(
    outputs=u_x, inputs=tx,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True
)[0][:, 1:2]
```

**Key flags:**
- `create_graph=True` — needed for higher-order derivatives and backprop through the derivative
- `retain_graph=True` — keep computation graph for multiple backward passes
- `grad_outputs` — weighting tensor, use `torch.ones_like(u)` for uniform weighting

### PINN Architecture Pattern

```python
class HeatPINN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tx):
        return self.net(tx)  # (B, 2) -> (B, 1)
```

Input: `(t, x)` concatenated, shape `(B, 2)`. Output: `u(t, x)`, shape `(B, 1)`.

### PINN Training Data

| Dataset | Points | Sampling | Batch |
|---------|--------|----------|-------|
| PDE (collocation) | 10k+ random $(t, x)$ in domain | `torch.rand` | Mini-batch |
| IC | All $x$ at $t=0$ | Uniform grid | **Full** |
| BC | All $t$ at $x=0$ and $x=L$ | Uniform grid | **Full** |

**Critical:** Use mini-batch for PDE data but **FULL** IC/BC data every step. IC and BC are hard constraints — subsampling them causes instability.

### PINN Training Loop Skeleton

```python
for epoch in range(epochs):
    for pde_batch in pde_loader:
        tx_pde = pde_batch.requires_grad_(True)
        u_pde = model(tx_pde)
        # Compute derivatives via autograd
        loss_pde = mse(residual, 0)
        loss_ic = mse(model(tx_ic), u_ic_true)
        loss_bc = mse(model(tx_bc), u_bc_true)
        loss = loss_pde + loss_ic + loss_bc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Advanced Optimization

### Second-Order Methods

| Method | Update Rule | Pros | Cons |
|--------|------------|------|------|
| Newton's | $\theta \leftarrow \theta - H^{-1}\nabla L$ | Quadratic convergence | $O(n^3)$ per step |
| L-BFGS | Approximate $H^{-1}$ from recent gradients | Near-Newton, cheap | Needs full-batch or large batch |
| Natural Gradient | $\theta \leftarrow \theta - F^{-1}\nabla L$ | Invariant to parameterization | Fisher matrix expensive |

**L-BFGS in PyTorch:**
```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)
def closure():
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss
optimizer.step(closure)
```

### Meta-Learning (MAML)

Inner loop (task adaptation):
$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{task}}(\theta)$$

Outer loop (meta-update):
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\text{tasks}} \mathcal{L}_{\text{task}}(\theta')$$

Requires second-order gradients (differentiating through the inner loop).

### Curriculum Learning

Train on easy examples first, gradually increase difficulty. For PINNs: start with smooth solutions, add sharp gradients later.

---

## Paper-to-Implementation

### Reading Order
1. **Abstract** — what problem, what method, what result
2. **Figures** — architecture diagrams, result plots
3. **Method section** — the algorithm / equations
4. **Experiments** — datasets, baselines, metrics
5. **Related work** — context (read last)

### Implementation Checklist
1. Identify the **key equation or algorithm**
2. Map notation to **tensor shapes** (write shapes as comments)
3. Implement **layer by layer**, verifying shapes
4. Write a **smoke test** with random input
5. Compare output **shapes** against paper's description
6. Train on a **toy problem** first

### Shape-First Implementation
```python
# Paper says: Q = XW_Q, shape: (B, L, d_k)
Q = x @ W_Q  # (B, L, d) @ (d, d_k) -> (B, L, d_k)
```

---

## Novel Architectures

### Attention Variants Quick Reference

| Variant | Key Change | Complexity |
|---------|-----------|------------|
| Standard MHA | $\text{softmax}(QK^T/\sqrt{d_k})V$ | $O(L^2 d)$ |
| Linear Attention | $\phi(Q)(\phi(K)^T V)$ | $O(Ld^2)$ |
| Multi-Query (MQA) | Shared K, V across heads | $O(L^2 d / h)$ memory |
| Grouped Query (GQA) | Groups of heads share K, V | Between MHA and MQA |
| Multi-Latent (MLA) | Low-rank KV compression | Reduced KV cache |

### Custom Module Pattern
```python
class NovelLayer(nn.Module):
    def __init__(self, d_model, ...):
        super().__init__()
        # Define parameters matching paper notation
    def forward(self, x):
        # Step-by-step with shape comments
        return output
```

---

## Competition Strategy (Round 2)

### Time Management
- **4 hours** = 240 minutes for ~3 problems
- ~80 min per problem, ~6 min per part (average)
- **Read all problems first** (5 min) — pick the one you know best
- Budget 5 min reading + understanding per problem before coding

### Execution Rules
1. **Use stated results to continue** — if Part 3 says "the output shape is (B, L, d)", use that even if your Part 2 is wrong
2. **Partial credit** — write something for every part, even pseudocode
3. **Non-coding parts** — mathematical proofs and explanations; be precise, use $\LaTeX$
4. **Coding parts** — focus on correctness first, then efficiency
5. **Skip and return** — if stuck for > 8 min on a part, skip and come back

### LaTeX Speed Tips
- `\frac{a}{b}`, `\partial`, `\nabla`, `\sum_{i=1}^{n}`, `\mathcal{L}`
- Aligned equations: `\begin{align*} ... \end{align*}`
- Matrices: `\begin{bmatrix} a & b \\ c & d \end{bmatrix}`

### Mental Frameworks for Proofs
- **Substitution verification:** Plug the proposed solution into the PDE and show LHS = RHS
- **Dimension analysis:** Check units/shapes at every step
- **Boundary checking:** Verify IC and BC are satisfied by substitution

### Common Pitfalls
- Forgetting `create_graph=True` in autograd
- Not using `.requires_grad_(True)` on input tensors
- Mini-batching IC/BC data (should be full)
- Wrong indexing for partial derivatives from joint input
- Not detaching tensors when computing metrics (use `.item()`)

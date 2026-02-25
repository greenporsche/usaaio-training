# Exercises: Physics-Informed Neural Networks

## Exercise 1: Verify a PDE Solution

**Difficulty:** Introductory

The 1D diffusion equation with a source term is:

$$u_t = \alpha u_{xx} + S(x, t)$$

where $S(x, t) = \sin(\pi x)\cos(t)$.

A proposed solution is:

$$u(x, t) = \frac{1}{\alpha\pi^2 + 1}\sin(\pi x)\left[\sin(t) + \alpha\pi^2 \cos(t) + (1 - \alpha\pi^2)e^{-\alpha\pi^2 t}\right] \cdot \frac{1}{\alpha\pi^2 + 1}$$

Wait — that is getting complicated. Let us use a simpler form.

Consider instead:

$$u_t - u_{xx} = 0 \quad \text{on } x \in [0, \pi], \, t > 0$$

with $u(0, t) = u(\pi, t) = 0$ and $u(x, 0) = \sin(2x)$.

**Proposed solution:** $u(x, t) = e^{-4t}\sin(2x)$

**Tasks:**

(a) Compute $u_t$ and $u_{xx}$.

(b) Verify that $u_t - u_{xx} = 0$.

(c) Verify the initial condition $u(x, 0) = \sin(2x)$.

(d) Verify the boundary conditions $u(0, t) = u(\pi, t) = 0$.

---

## Exercise 2: Set Up PINN Loss for a New Equation

**Difficulty:** Intermediate

Consider the **advection equation**:

$$u_t + c \, u_x = 0, \quad x \in [0, 2\pi], \, t \in [0, 1]$$

with $c = 1$, IC: $u(x, 0) = \sin(x)$, and periodic BC: $u(0, t) = u(2\pi, t)$.

**Tasks:**

(a) Write the PDE residual function in PyTorch. Your function should take a model and a tensor `tx` of shape `(B, 2)` and return the residual $u_t + c \, u_x$ of shape `(B, 1)`.

(b) What is the analytical solution to this PDE? (Hint: the solution translates the initial condition.)

(c) How would you modify the BC dataset to handle periodic boundary conditions (where $u(0, t) = u(2\pi, t)$) instead of fixed Dirichlet conditions?

(d) Would you expect this problem to be harder or easier for a PINN than the heat equation? Explain why. (Hint: think about the smoothness of the solution over time.)

---

## Exercise 3: PINN Architecture Choices

**Difficulty:** Intermediate

You are building a PINN for the **Burgers' equation**:

$$u_t + u \, u_x = \nu u_{xx}$$

This equation develops sharp gradients (shock waves) for small $\nu$.

**Tasks:**

(a) Explain why `nn.Tanh` is preferred over `nn.ReLU` as the activation function in a PINN. What specific property of Tanh is essential?

(b) The Burgers' equation has a nonlinear term $u \cdot u_x$. Write the residual computation in PyTorch, computing $u_t + u \cdot u_x - \nu u_{xx}$ using `autograd.grad`.

(c) For small $\nu$ (e.g., $\nu = 0.001$), the solution develops near-discontinuities. Propose two modifications to the standard PINN architecture or training that might help capture sharp gradients.

(d) If you used `nn.ReLU` activations, what would happen to $u_{xx}$ computed via autograd? Would the PDE loss be meaningful?

---

## Exercise 4: Training Data Design

**Difficulty:** Advanced

You are solving a 2D Poisson equation on a circular domain:

$$\nabla^2 u = u_{xx} + u_{yy} = f(x, y), \quad (x, y) \in \Omega = \{(x,y) : x^2 + y^2 < 1\}$$

with $u = 0$ on the boundary $\partial\Omega = \{(x,y) : x^2 + y^2 = 1\}$.

**Tasks:**

(a) How would you sample random collocation points uniformly inside the unit disk? (Hint: rejection sampling or polar coordinates with the correct Jacobian.)

(b) How would you sample boundary points on the circle?

(c) This is a steady-state PDE (no time dependence). The input to the network is $(x, y)$ and the output is $u(x, y)$. There is no IC. Write the loss function with only PDE and BC terms.

(d) The domain is not rectangular. Discuss whether this affects the PINN approach compared to traditional numerical methods like finite differences (which require a grid).

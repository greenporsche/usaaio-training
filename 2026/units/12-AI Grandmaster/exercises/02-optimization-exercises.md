# Exercises: Advanced Optimization

## Exercise 1: L-BFGS Closure

**Difficulty:** Introductory

The PyTorch L-BFGS optimizer requires a `closure` function that recomputes the loss.

**Tasks:**

(a) Explain why L-BFGS needs a closure while Adam does not. What does L-BFGS do inside a single `optimizer.step()` call that requires re-evaluating the loss?

(b) Write a complete L-BFGS training loop for a simple regression problem. Given:
```python
model = nn.Linear(10, 1)
X = torch.randn(100, 10)
y = torch.randn(100, 1)
```
Write the closure function and the training loop for 50 steps.

(c) What happens if you forget to call `optimizer.zero_grad()` inside the closure?

(d) L-BFGS works best with full-batch gradients. Why is this the case? What goes wrong with mini-batches?

---

## Exercise 2: Two-Phase PINN Training

**Difficulty:** Intermediate

A common PINN training strategy uses Adam first, then switches to L-BFGS.

**Tasks:**

(a) Explain the rationale for this two-phase approach. What does each optimizer contribute?

(b) Write a two-phase training function:
```python
def train_pinn_two_phase(model, pde_loader, tx_ic, u_ic, tx_bc, u_bc,
                          adam_epochs=5000, lbfgs_epochs=100, adam_lr=1e-3):
    # Phase 1: Adam
    # Phase 2: L-BFGS
    pass
```
Sketch the implementation (pseudocode is acceptable).

(c) During the L-BFGS phase, you use full-batch PDE data instead of mini-batches. How does this affect memory usage? What is the trade-off?

(d) Some practitioners add a learning rate schedule to the Adam phase (e.g., reduce LR by 0.5 every 1000 epochs). Would this help or hurt the transition to L-BFGS? Explain.

---

## Exercise 3: MAML Inner Loop

**Difficulty:** Advanced

Consider a simple MAML setup for few-shot regression.

**Tasks:**

(a) Explain in your own words why MAML requires `create_graph=True` in the inner loop gradient computation.

(b) Given a model `model` and a single task with support set `(x_s, y_s)` and query set `(x_q, y_q)`, write the inner loop adaptation step that produces adapted parameters `theta_prime`. Use `torch.autograd.grad` explicitly (not `loss.backward()`).

(c) After the inner loop, you need to compute the outer loss on the query set using the adapted parameters. Why can't you simply do:
```python
for p, p_prime in zip(model.parameters(), theta_prime):
    p.data = p_prime
loss = criterion(model(x_q), y_q)
```
What would be wrong with this approach?

(d) How does the memory cost of MAML scale with the number of inner loop steps? If you do $K$ inner steps, how many computation graphs are maintained?

---

## Exercise 4: Learning Rate Schedules

**Difficulty:** Intermediate

You are training a model for 10,000 steps and want to experiment with learning rate schedules.

**Tasks:**

(a) Write code to create three different schedulers:
1. Cosine annealing from `lr=0.01` to `lr=0` over 10,000 steps
2. Step decay: multiply LR by 0.1 at steps 3000, 6000, and 9000
3. Warm-up + cosine: linear warm-up from 0 to 0.01 over 500 steps, then cosine decay

(b) For each scheduler, describe what the learning rate curve looks like (no plotting needed — just describe the shape).

(c) Which scheduler would you choose for:
- Fine-tuning a pretrained model? Why?
- Training a PINN from scratch? Why?
- A competition with limited training time? Why?

(d) Some practitioners use "warm restarts" where the cosine schedule resets periodically. What is the potential benefit of this approach?

# Exercises: Competition Strategy

## Exercise 1: Time Allocation

**Difficulty:** Introductory

You are taking a 4-hour Round 2 exam with three problems:

- **Problem A:** Transformer variant — 14 parts, familiar topic
- **Problem B:** PINN for a new PDE — 12 parts, practiced extensively
- **Problem C:** Generative model — 10 parts, least familiar

**Tasks:**

(a) In what order would you attempt the problems? Justify your choice.

(b) Allocate time (in minutes) to each problem. Remember: 240 minutes total, and you need 10 minutes for initial reading and 10 minutes for final review.

(c) For Problem C (your weakest), which types of parts would you prioritize? (Hint: think about which part types have the highest points-per-minute ratio.)

(d) At the 160-minute mark, you have completed Problem B (your first choice) and 9 of 14 parts of Problem A. You have not started Problem C. What do you do?

---

## Exercise 2: Using Stated Results

**Difficulty:** Intermediate

Consider this exam scenario:

> **Part 4:** Show that after the projection layer, the tensor shape is $(B, L, h \cdot d_k)$. [8 points]
>
> **Part 5:** Reshape this tensor into $(B, h, L, d_k)$ and implement the multi-head splitting. [10 points]
>
> **Part 6:** Implement scaled dot-product attention using the reshaped tensors. [12 points]

You are stuck on Part 4 — you cannot figure out the projection dimensions.

**Tasks:**

(a) What information does Part 4 give you, even if you cannot solve it?

(b) Write the code for Part 5 using the stated result from Part 4, even though you did not solve Part 4.

(c) Write the code for Part 6 using the shapes from Part 5.

(d) If you skip Part 4 entirely and write correct code for Parts 5 and 6, how many points do you expect to receive (out of 30 total for Parts 4–6)?

---

## Exercise 3: Partial Credit Maximization

**Difficulty:** Intermediate

For each of the following scenarios, write the **maximum partial-credit answer** in under 3 minutes.

**(a)** "Prove that the function $f(x) = e^{-x^2}$ is a solution to $f'(x) = -2xf(x)$." You don't remember the derivative of $e^{-x^2}$.

**(b)** "Implement a custom DataLoader that yields batches of (input, target) pairs from a dataset." You remember the Dataset class but not the exact DataLoader interface.

**(c)** "Explain why batch normalization is less effective than layer normalization in transformer models." You know vaguely but cannot articulate precisely.

**(d)** "Compute the Jacobian of $f(x) = Ax + b$ where $A \in \mathbb{R}^{m \times n}$." You are unsure about the exact definition of the Jacobian.

---

## Exercise 4: Mock Exam Sprint

**Difficulty:** Advanced (Timed)

**Set a timer for 20 minutes.** Complete as many parts as possible.

**Mini-Problem: Implement a Simple PINN**

Consider $u_t = u_{xx}$ on $x \in [0, 1]$, $t \in [0, 0.5]$, with $u(x, 0) = \sin(\pi x)$ and $u(0, t) = u(1, t) = 0$.

**Part 1** (5 pts): Write the analytical solution.

**Part 2** (10 pts): Define a `SimplePINN` network: 2 inputs, 3 hidden layers of 32 units with Tanh, 1 output.

**Part 3** (10 pts): Write a function `pde_residual(model, tx)` that computes $u_t - u_{xx}$ using `autograd.grad`.

**Part 4** (10 pts): Generate 5000 random PDE collocation points, 50 IC points, and 50 BC points (at both boundaries).

**Part 5** (15 pts): Write a training loop for 2000 epochs using Adam with lr=0.001. Use mini-batches of 256 for PDE data and full IC/BC data.

**Part 6** (10 pts): After training, evaluate the model on a 50x50 grid and compute the maximum absolute error against the analytical solution.

After the timer: count how many parts you completed. Which parts took longest? Where could you have been faster?

# 05 — Competition Strategy

> Time management, exam technique, and mental frameworks for USAAIO Round 2.

---

## Discovery

### The Round 2 Format

- **Duration:** 4 hours (240 minutes)
- **Problems:** 3 problems, each 100 points
- **Total:** 300 points
- **Format:** Jupyter notebooks with coding and non-coding parts
- **Parts per problem:** 8–14 parts
- **Average:** ~6 minutes per part (but this varies widely)

### What Determines Your Score

Your score is determined by three factors, in order of importance:

1. **Completeness** — answering every part with something reasonable
2. **Correctness** — getting the right answer
3. **Time management** — not spending 45 minutes on one part

Most students lose points not because they can't solve the problems, but because they **run out of time** or **get stuck on one part and skip the rest**.

---

## Intuition

### The First 10 Minutes

Before writing any code or proofs, spend 10 minutes on reconnaissance:

1. **Read all three problems** — just the titles, introductions, and part lists (3 min)
2. **Rank by confidence** — which problem do you know the most about? (1 min)
3. **Count the parts** — more parts = more partial credit opportunities (1 min)
4. **Plan your order** — start with your strongest problem (2 min)
5. **Set time checkpoints** — at 80 min, 160 min, 200 min, check progress (1 min)

### Time Allocation Strategy

**The 80-80-80 split:**

- Problem 1: 80 minutes (your strongest)
- Problem 2: 80 minutes (your second strongest)
- Problem 3: 80 minutes (your weakest)

This gives you equal time for each problem. If you finish one early, invest the surplus in your weakest problem.

**Within each problem:**

- 5 min: Read the full problem and understand the method
- 60 min: Work through parts sequentially
- 15 min: Review, fix bugs, clean up

**Per-part budget:**

| Part Type | Budget | Notes |
|-----------|--------|-------|
| Simple proof (IC/BC verification) | 3-5 min | Substitution + simplification |
| Shape analysis | 2-3 min | Trace dimensions |
| Implement nn.Module | 8-12 min | Shape-first method |
| Create Dataset | 5-8 min | Pattern from PINNs unit |
| Training loop | 8-12 min | Standard pattern |
| Analysis/explanation | 3-5 min | Refer to training results |

---

## Mastery

### Rule 1: Use Stated Results to Continue

**This is the single most important exam strategy.**

If Part 3 says: "Show that the output shape is $(B, L, d)$" — and you cannot solve Part 3 — **use the stated answer** $(B, L, d)$ to continue with Part 4.

The exam is designed so that later parts depend on earlier parts. But the **stated results are given to you** precisely so that a mistake in one part does not cascade. You get partial credit for every correct part, even if earlier parts are wrong.

**Example:**

> Part 5: "Using the `HeatPINN` model from Part 2, create a training loop..."

If your Part 2 implementation is buggy, you can still:
- Write a correct training loop structure
- Use the correct loss formulation
- Handle the optimizer correctly
- Get most of the points for Part 5

### Rule 2: The 8-Minute Rule

If you have been stuck on a part for **8 minutes** with no clear path forward:

1. **Write what you know** — even pseudocode or a partial answer gets partial credit
2. **Mark it** — add a comment `# TODO: revisit if time permits`
3. **Move on** — the next part may be easier and worth the same points
4. **Come back later** — fresh eyes after other problems often break the block

### Rule 3: Write Something for Every Part

Partial credit is real. Even if you cannot fully solve a part:

- **For proofs:** Write the setup, state what needs to be shown, write the first step
- **For code:** Write the function signature, shape comments, and pseudocode
- **For analysis:** State the expected behavior even if you cannot demonstrate it

A part with 3/10 points is better than a part with 0/10 points.

### Rule 4: Non-Coding Parts Are Free Points

Non-coding parts (proofs, explanations, shape analysis) typically require less time than coding parts but are worth the same points.

**Proof patterns for PDE verification:**
```
1. State the PDE: u_t - α u_xx = 0
2. Compute u_t from the proposed solution
3. Compute u_xx from the proposed solution
4. Substitute into the PDE and simplify to 0
5. Check IC: substitute t=0
6. Check BC: substitute x=0 and x=L
```

**Shape analysis pattern:**
```
1. Start with input shape: (B, L, d)
2. Apply each operation: Linear(d, k) -> (B, L, k)
3. Track through reshapes, transposes, and matmuls
4. State final output shape
```

### Rule 5: Code Patterns Are Reusable

Most coding parts reuse the same patterns. Memorize these:

**nn.Module skeleton:**
```python
class Model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # parameters
    def forward(self, x):
        # computation
        return output
```

**Dataset skeleton:**
```python
class MyDataset(Dataset):
    def __init__(self, ...):
        # generate or load data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  # or (input, target)
```

**Training loop skeleton:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for batch in dataloader:
        pred = model(batch)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**autograd derivative skeleton:**
```python
x = x.requires_grad_(True)
y = model(x)
dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
```

### LaTeX Typesetting Speed

Non-coding parts require mathematical typesetting. Speed matters.

**Essential shortcuts:**

| What | LaTeX | Rendered |
|------|-------|----------|
| Fraction | `\frac{a}{b}` | $\frac{a}{b}$ |
| Partial | `\frac{\partial u}{\partial t}` | $\frac{\partial u}{\partial t}$ |
| Sum | `\sum_{i=1}^{N}` | $\sum_{i=1}^{N}$ |
| Norm | `\|x\|^2` | $\|x\|^2$ |
| Matrix | `\begin{bmatrix} a & b \\ c & d \end{bmatrix}` | matrix |
| Set | `\mathbb{R}^{d}` | $\mathbb{R}^{d}$ |
| Loss | `\mathcal{L}` | $\mathcal{L}$ |
| Gradient | `\nabla_\theta` | $\nabla_\theta$ |
| Expectation | `\mathbb{E}` | $\mathbb{E}$ |
| Aligned eqs | `\begin{align*} a &= b \\ c &= d \end{align*}` | aligned |

**Proof template:**

```latex
$$\begin{align*}
u_t &= \frac{\partial}{\partial t}\left[e^{-\alpha\pi^2 t}\sin(\pi x)\right] \\
&= -\alpha\pi^2 e^{-\alpha\pi^2 t}\sin(\pi x)
\end{align*}$$
```

### Mental Frameworks for Proof Construction

**Substitution verification:**
1. Write the equation to verify: $N[u] = 0$
2. Compute each derivative of the proposed solution
3. Substitute into the equation
4. Simplify — terms should cancel

**Dimensional consistency:**
1. Write dimensions of each term
2. Verify they match across equations

**Induction (for discrete problems):**
1. Base case: verify for $n = 0$
2. Inductive step: assume true for $n$, prove for $n + 1$

---

## Connection

### Pre-Exam Checklist

The day before the exam:

- [ ] Review cheat sheet (this unit's `cheat-sheet.md`)
- [ ] Run through one timed mini-problem (30 min)
- [ ] Verify your environment: PyTorch, NumPy, matplotlib working
- [ ] Prepare your workspace: quiet, comfortable, hydrated
- [ ] Set 3 alarms: 80 min, 160 min, 200 min (for time checkpoints)

### During the Exam

| Time | Action |
|------|--------|
| 0:00 | Read all problems, rank, plan order |
| 0:10 | Start Problem 1 (strongest) |
| 1:20 | Time check — switch to Problem 2 |
| 1:30 | Start Problem 2 |
| 2:40 | Time check — switch to Problem 3 |
| 2:50 | Start Problem 3 |
| 3:20 | Time check — 40 min left |
| 3:40 | Stop new work — review and clean up |
| 4:00 | Submit |

### After the Exam

Win or lose, review your performance:
- Which parts took longest? Why?
- Where did you get stuck? What would have helped?
- Did you use stated results when stuck? Or did you waste time?

---

## Summary

| Rule | What to Do |
|------|-----------|
| Reconnaissance first | Read all problems before starting (10 min) |
| Use stated results | If stuck, use the answer the problem gives you |
| 8-minute rule | Stuck for 8 min → write something, move on |
| Write everything | Partial answers get partial credit |
| Non-coding = free | Proofs and explanations are fast points |
| Reuse patterns | nn.Module, Dataset, training loop, autograd |
| Time checkpoints | Check at 80, 160, 200 minutes |

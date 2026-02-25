# Batch Normalization & Dropout Exercises

**Topic**: Batch normalization computation, training vs. inference, dropout mechanics
**Difficulty**: Foundational → Advanced

---

## Exercise 1: Batch Norm by Hand

Given a batch of 4 values (single feature): $x = [2.0, 4.0, 6.0, 8.0]$, with $\gamma = 1.5$, $\beta = 0.5$, $\epsilon = 0$:

1. Compute $\mu_B$ (batch mean)
2. Compute $\sigma_B^2$ (batch variance)
3. Compute $\hat{x}_i$ for each sample (normalized)
4. Compute $y_i$ for each sample (after scale and shift)

<details>
<summary>Solution</summary>

**1.** $\mu_B = \frac{2+4+6+8}{4} = 5.0$

**2.** $\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9+1+1+9}{4} = 5.0$

**3.** Normalized:
- $\hat{x}_1 = \frac{2-5}{\sqrt{5}} = \frac{-3}{2.236} = -1.342$
- $\hat{x}_2 = \frac{4-5}{\sqrt{5}} = \frac{-1}{2.236} = -0.447$
- $\hat{x}_3 = \frac{6-5}{\sqrt{5}} = \frac{1}{2.236} = 0.447$
- $\hat{x}_4 = \frac{8-5}{\sqrt{5}} = \frac{3}{2.236} = 1.342$

Verify: mean $= 0$ (check: $-1.342 - 0.447 + 0.447 + 1.342 = 0$ ✓), variance $= 1$ ✓

**4.** Scale and shift ($\gamma = 1.5$, $\beta = 0.5$):
- $y_1 = 1.5(-1.342) + 0.5 = -2.013 + 0.5 = -1.513$
- $y_2 = 1.5(-0.447) + 0.5 = -0.671 + 0.5 = -0.171$
- $y_3 = 1.5(0.447) + 0.5 = 0.671 + 0.5 = 1.171$
- $y_4 = 1.5(1.342) + 0.5 = 2.013 + 0.5 = 2.513$

Verify output stats: mean $= 0.5 = \beta$ ✓, std $= 1.5 = \gamma$ ✓

**Key insight**: After batch norm, the output has mean $\beta$ and standard deviation $|\gamma|$. If $\gamma = \sigma_B$ and $\beta = \mu_B$, batch norm is a no-op (identity). The network can learn to undo normalization if needed.
</details>

---

## Exercise 2: Batch Norm for Multi-Feature Input

Given a batch of shape $(B=3, D=2)$:

$$X = \begin{bmatrix}1 & 10 \\ 3 & 20 \\ 5 & 30\end{bmatrix}$$

With $\gamma = [1, 1]$, $\beta = [0, 0]$, $\epsilon = 0$:

Compute batch norm **per feature** (each column is normalized independently).

<details>
<summary>Solution</summary>

**Feature 1** ($x = [1, 3, 5]$):
- $\mu_1 = 3$, $\sigma_1^2 = \frac{4+0+4}{3} = \frac{8}{3} = 2.667$, $\sigma_1 = 1.633$
- $\hat{x} = [\frac{1-3}{1.633}, \frac{3-3}{1.633}, \frac{5-3}{1.633}] = [-1.225, 0, 1.225]$

**Feature 2** ($x = [10, 20, 30]$):
- $\mu_2 = 20$, $\sigma_2^2 = \frac{100+0+100}{3} = \frac{200}{3} = 66.667$, $\sigma_2 = 8.165$
- $\hat{x} = [\frac{10-20}{8.165}, \frac{20-20}{8.165}, \frac{30-20}{8.165}] = [-1.225, 0, 1.225]$

**Result** ($\gamma = 1, \beta = 0$):
$$Y = \begin{bmatrix}-1.225 & -1.225 \\ 0 & 0 \\ 1.225 & 1.225\end{bmatrix}$$

Both features are normalized to the same statistics despite having very different scales (1–5 vs 10–30).

**Key insight**: Batch norm normalizes EACH FEATURE independently across the batch. Feature 1 (range 1–5) and feature 2 (range 10–30) end up with identical normalized values because the pattern is the same (linearly spaced). This is exactly what makes batch norm powerful — it removes the scale differences between features.
</details>

---

## Exercise 3: Training vs. Inference Mode Bug

The following code has a subtle bug related to batch normalization. Find and fix it.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Training
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
correct = 0
for X_batch, y_batch in test_loader:
    logits = model(X_batch)
    correct += (logits.argmax(1) == y_batch).sum().item()
accuracy = correct / len(test_dataset)
```

<details>
<summary>Solution</summary>

**Bug**: Missing `model.eval()` before evaluation and `model.train()` before training. Also missing `torch.no_grad()` during evaluation.

**Fixed code**:

```python
# Training
for epoch in range(10):
    model.train()                                    # ← ADDED
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()                                         # ← ADDED
correct = 0
with torch.no_grad():                               # ← ADDED
    for X_batch, y_batch in test_loader:
        logits = model(X_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
accuracy = correct / len(test_dataset)
```

**Why this matters**:
- In training mode, `BatchNorm1d` uses batch statistics ($\mu_B, \sigma_B^2$) and updates running statistics
- In eval mode, it uses running statistics ($\mu_{run}, \sigma^2_{run}$)
- Without `model.eval()`, evaluation with batch size 1 computes variance of a single value (0 or undefined), giving garbage results
- `torch.no_grad()` saves memory and computation during inference

**Key insight**: `model.eval()` affects BatchNorm AND Dropout. Always set the correct mode. This is one of the most common bugs in deep learning code.
</details>

---

## Exercise 4: Dropout Probability Analysis

For a layer with 100 neurons and dropout probability $p = 0.5$:

1. On average, how many neurons are active during one training forward pass?
2. What is the scaling factor applied to active neurons?
3. What is the expected value of the output of a neuron with value $h = 2.0$ during training?
4. If we increase $p$ to 0.8, how many neurons are active on average, and what is the scaling factor?

<details>
<summary>Solution</summary>

**1.** Average active neurons $= 100 \times (1 - p) = 100 \times 0.5 = 50$

**2.** Scaling factor $= \frac{1}{1-p} = \frac{1}{0.5} = 2.0$

**3.** Expected output with dropout:

$E[\tilde{h}] = P(\text{active}) \times h \times \text{scale} + P(\text{dropped}) \times 0$

$= 0.5 \times 2.0 \times 2.0 + 0.5 \times 0 = 2.0$

So $E[\tilde{h}] = h = 2.0$ — the expected value is unchanged (inverted dropout ensures this).

**4.** With $p = 0.8$:
- Active neurons: $100 \times 0.2 = 20$
- Scaling factor: $\frac{1}{0.2} = 5.0$

Only 20 of 100 neurons are active, but each is scaled by 5x. The expected value is still $h$.

**Key insight**: The $\frac{1}{1-p}$ scaling in inverted dropout ensures that the expected output is the same during training and inference. Without this scaling, you would need to multiply outputs by $(1-p)$ at inference time (the original dropout formulation). PyTorch uses inverted dropout by default, so no adjustment is needed at inference.
</details>

---

## Exercise 5: Batch Norm Gradient Complexity

Consider the batch norm backward pass. For a batch of size $m$ and feature dimension $d$:

1. Why does $\frac{\partial L}{\partial x_i}$ depend on ALL samples in the batch (not just sample $i$)?
2. What happens to batch norm gradients when $m = 1$?
3. Why is batch norm problematic with very small batch sizes?

<details>
<summary>Solution</summary>

**1.** The gradient $\frac{\partial L}{\partial x_i}$ depends on all samples because:
- $\mu_B = \frac{1}{m}\sum_j x_j$ — every $x_j$ contributes to the mean
- $\sigma_B^2 = \frac{1}{m}\sum_j (x_j - \mu_B)^2$ — every $x_j$ contributes to the variance
- The normalization $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ depends on $\mu_B$ and $\sigma_B^2$

So changing ANY $x_j$ changes the mean and variance, which changes the normalized output for ALL samples. The gradient must account for this coupling.

This makes batch norm different from element-wise operations like ReLU — it is a **batch-level** operation.

**2.** When $m = 1$:
- $\mu_B = x_1$ (just the input)
- $\sigma_B^2 = 0$ (variance of a single sample is 0)
- $\hat{x}_1 = \frac{x_1 - x_1}{\sqrt{0 + \epsilon}} = 0$ for all inputs!
- The output is always $\gamma \cdot 0 + \beta = \beta$ regardless of input.

The layer becomes a constant function — useless.

**3.** Small batch sizes ($m \leq 4$):
- Batch statistics are noisy estimates of population statistics
- Variance estimate is unreliable with few samples
- The noise in statistics acts as strong (and unhelpful) regularization
- Training becomes unstable

**Alternatives for small batches**:
- **Layer Normalization**: Normalizes across features (not batch). Used in transformers.
- **Group Normalization**: Normalizes across groups of channels. Used in detection/segmentation.
- **Instance Normalization**: Normalizes each sample independently. Used in style transfer.

**Key insight**: Batch normalization is powerful but has a strong assumption: the batch must be large enough to provide reliable statistics. For batch size 1 (online learning) or very small batches, use LayerNorm or GroupNorm instead.
</details>

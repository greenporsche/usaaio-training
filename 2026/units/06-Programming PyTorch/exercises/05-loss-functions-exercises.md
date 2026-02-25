# Loss Functions Exercises

**Topic**: MSE, CrossEntropy, BCE, custom losses, reduction
**Difficulty**: Intermediate → Advanced

---

## Exercise 1: Compute Cross-Entropy by Hand

Given logits `z = [2.0, 1.0, 0.1]` and true class `y = 0`, compute the cross-entropy loss step by step.

1. Compute the softmax probabilities
2. Compute the negative log probability of the true class
3. Verify your answer matches `nn.CrossEntropyLoss`

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1]])    # (1, 3)
labels = torch.tensor([0])                    # true class = 0

# Step 1: softmax
# Step 2: -log(p[true_class])
# Step 3: verify

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
print(f"PyTorch loss: {loss.item()}")
# YOUR ANSWER: ?
```

<details>
<summary>Solution</summary>

**Step 1: Softmax**

$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

- $e^{2.0} = 7.389$
- $e^{1.0} = 2.718$
- $e^{0.1} = 1.105$
- Sum = $7.389 + 2.718 + 1.105 = 11.212$

$p = [0.659, 0.242, 0.099]$

**Step 2: Negative log probability**

$\mathcal{L} = -\log(p_0) = -\log(0.659) = 0.417$

**Step 3: Verify**

```python
# Manual computation
import math
exp_z = [math.exp(2.0), math.exp(1.0), math.exp(0.1)]
sum_exp = sum(exp_z)
probs = [e / sum_exp for e in exp_z]
loss_manual = -math.log(probs[0])
print(f"Manual loss: {loss_manual:.4f}")     # 0.4170

# PyTorch
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))
print(f"PyTorch loss: {loss.item():.4f}")    # 0.4170 — matches!
```

**Key insight**: Cross-entropy loss is just $-\log(\text{softmax}(z)[\text{true class}])$. The softmax converts logits to probabilities, and we want the probability of the correct class to be high (so its negative log is low).
</details>

---

## Exercise 2: The CrossEntropyLoss Trap

What is wrong with this code? Predict the output, then fix it.

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
probs = torch.softmax(logits, dim=-1)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(probs, torch.tensor([0]))
print(f"Loss: {loss.item():.4f}")
```

<details>
<summary>Solution</summary>

**The bug**: `CrossEntropyLoss` applies softmax internally. Passing already-softmaxed probabilities means softmax is applied **twice** — this is wrong.

```
Input to CE: [0.659, 0.242, 0.099]     (already softmaxed)
CE applies softmax again: softmax([0.659, 0.242, 0.099])
  = [0.385, 0.254, 0.220, ...]         (flatter distribution)
Loss = -log(0.385) = 0.955             (WRONG — too high)
```

The correct loss should be 0.4170 (as computed in Exercise 1).

**Fix**: Pass raw logits to `CrossEntropyLoss`:

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, torch.tensor([0]))     # Pass logits, NOT probs
print(f"Loss: {loss.item():.4f}")              # 0.4170
```

**Alternative**: If you already have probabilities, use `NLLLoss` with `log()`:

```python
probs = torch.softmax(logits, dim=-1)
loss = nn.NLLLoss()(torch.log(probs), torch.tensor([0]))   # 0.4170
```

**Key insight**: `CrossEntropyLoss = LogSoftmax + NLLLoss`. Never apply softmax before it. This is the single most common PyTorch loss bug.
</details>

---

## Exercise 3: Implement Triplet Loss

Implement the triplet loss used in metric learning:

$$\mathcal{L} = \max(d(\text{anchor}, \text{positive}) - d(\text{anchor}, \text{negative}) + \text{margin}, 0)$$

where $d(a, b) = \|a - b\|_2$ is the Euclidean distance.

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: (B, D) embeddings
            positive: (B, D) same-class embeddings
            negative: (B, D) different-class embeddings
        Returns:
            scalar loss
        """
        # YOUR CODE HERE
        pass

# Test
torch.manual_seed(42)
B, D = 8, 64
anchor = torch.randn(B, D)
positive = anchor + 0.1 * torch.randn(B, D)    # Close to anchor
negative = torch.randn(B, D)                    # Random (far)

loss_fn = TripletLoss(margin=1.0)
loss = loss_fn(anchor, positive, negative)
print(f"Loss: {loss.item():.4f}")
assert loss.item() >= 0, "Loss must be non-negative"
print("Passed!")
```

<details>
<summary>Solution</summary>

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = torch.nn.functional.pairwise_distance(anchor, positive)  # (B,)
        d_neg = torch.nn.functional.pairwise_distance(anchor, negative)  # (B,)
        losses = torch.clamp(d_pos - d_neg + self.margin, min=0)         # (B,)
        return losses.mean()
```

Or without `pairwise_distance`:

```python
def forward(self, anchor, positive, negative):
    d_pos = ((anchor - positive) ** 2).sum(dim=-1).sqrt()    # (B,)
    d_neg = ((anchor - negative) ** 2).sum(dim=-1).sqrt()    # (B,)
    losses = torch.clamp(d_pos - d_neg + self.margin, min=0)
    return losses.mean()
```

**Key insight**: Triplet loss pushes the anchor closer to the positive and farther from the negative. The margin ensures a minimum gap. If the negative is already far enough (d_neg > d_pos + margin), the loss is zero for that triplet — no gradient signal.
</details>

---

## Exercise 4: Loss Function Behavior

For each scenario, identify which loss function is most appropriate and explain why.

1. Predicting house prices (continuous, unbounded output)
2. Classifying images into 1000 categories
3. Detecting whether an email is spam (binary)
4. Predicting multiple labels (an image can be "sunny" AND "beach" simultaneously)
5. Training a PINN to satisfy $u_{xx} + u_{yy} = 0$

<details>
<summary>Solution</summary>

1. **House prices**: `nn.MSELoss()` — regression with continuous targets. Alternative: `nn.L1Loss()` if outliers are a concern (more robust).

2. **1000-class classification**: `nn.CrossEntropyLoss()` — multi-class, mutually exclusive classes. Model outputs 1000 logits, CE applies softmax + negative log-likelihood.

3. **Spam detection**: `nn.BCEWithLogitsLoss()` — binary classification. Model outputs a single logit. Alternatively, `nn.CrossEntropyLoss()` with 2 output logits, but BCE is simpler.

4. **Multi-label classification**: `nn.BCEWithLogitsLoss()` — each label is an independent binary classification. Model outputs one logit per label, and sigmoid (included in BCEWithLogits) is applied independently. NOT `CrossEntropyLoss` (which assumes mutually exclusive classes).

5. **PINN residual**: Custom MSE on the PDE residual — `(u_xx + u_yy).pow(2).mean()`. This is not classification or standard regression; it is a physics constraint that should be driven to zero.

**Key insight**: The loss function encodes your assumptions about the problem. Mutually exclusive classes → CrossEntropy. Independent labels → BCE. Continuous target → MSE/L1. Physics constraint → custom residual loss.
</details>

---

## Exercise 5: Weighted Multi-Task Loss

Implement a loss function for a model that simultaneously performs classification and regression. The model predicts:
- Class logits `(B, C)` for classification
- A continuous value `(B, 1)` for regression

The total loss should be: $\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{cls}} + (1 - \alpha) \cdot \mathcal{L}_{\text{reg}}$

Additionally, implement automatic loss weighting using learnable parameters (uncertainty weighting from "Multi-Task Learning Using Uncertainty to Weigh Losses"):

$$\mathcal{L} = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{cls}} + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{reg}} + \log(\sigma_1) + \log(\sigma_2)$$

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes, use_uncertainty=False):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            # Learnable log-variance parameters
            # YOUR CODE HERE
            pass

    def forward(self, cls_logits, cls_labels, reg_pred, reg_target, alpha=0.5):
        """
        Returns: total_loss, dict of individual losses
        """
        # YOUR CODE HERE
        pass
```

<details>
<summary>Solution</summary>

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes, use_uncertainty=False):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()
        self.use_uncertainty = use_uncertainty

        if use_uncertainty:
            # log(sigma^2) — learnable, initialized to 0 (sigma=1)
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, cls_logits, cls_labels, reg_pred, reg_target, alpha=0.5):
        l_cls = self.cls_loss(cls_logits, cls_labels)
        l_reg = self.reg_loss(reg_pred, reg_target)

        if self.use_uncertainty:
            # Uncertainty weighting (Kendall et al., 2018)
            # L = (1/2*sigma_1^2) * L_cls + (1/2*sigma_2^2) * L_reg + log(sigma_1) + log(sigma_2)
            # Using log_var = log(sigma^2), so 1/sigma^2 = exp(-log_var)
            total = (torch.exp(-self.log_var_cls) * l_cls + self.log_var_cls / 2 +
                     torch.exp(-self.log_var_reg) * l_reg + self.log_var_reg / 2)
        else:
            total = alpha * l_cls + (1 - alpha) * l_reg

        return total, {'cls': l_cls.item(), 'reg': l_reg.item(), 'total': total.item()}
```

**Key insight**: The uncertainty weighting automatically balances task losses during training. If one task has high uncertainty ($\sigma$ is large), its loss contribution is down-weighted. The $\log(\sigma)$ term prevents $\sigma$ from growing to infinity (which would make the loss trivially zero). This is especially useful when tasks have very different loss scales.
</details>

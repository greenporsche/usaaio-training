# Adversarial Attacks

**Prerequisites**: Neural networks, gradient computation, image classification, backpropagation
**USAAIO Relevance**: FGSM and PGD implementation, perturbation budget analysis, and adversarial training are testable. Understanding why neural networks are vulnerable to imperceptible perturbations is conceptually important.

---

## Discovery

### The Core Question

> Neural networks achieve superhuman accuracy on image classification, yet a tiny, imperceptible change to an image — invisible to humans — can cause the network to confidently predict a completely wrong class. Why does this happen, and how?

### Historical Context

- **Szegedy et al. (2013)**: Discovered adversarial examples — small perturbations that fool neural networks. Showed that the same perturbation transfers across different models.
- **Goodfellow, Shlens, Szegedy (2014)**: Introduced FGSM (Fast Gradient Sign Method) — a single-step attack. Argued adversarial vulnerability is due to linear behavior in high dimensions, not nonlinearity.
- **Madry et al. (2018)**: Introduced PGD (Projected Gradient Descent) attack and adversarial training as a defense. Showed adversarial training with PGD is the most robust defense.
- Adversarial robustness remains an open problem in deep learning.

### Socratic Warm-Up

1. If a 224x224x3 image has 150,528 pixels, and you change each by a tiny amount $\epsilon$, what's the maximum change in the dot product with any weight vector?
2. Why does FGSM use `sign(gradient)` instead of the raw gradient?
3. If you train on adversarial examples, does the model become robust to *all* attacks or just the one you trained against?

### Misconception Traps

- **"Adversarial examples exploit model bugs."** — They exploit a fundamental property: linear models in high dimensions are sensitive to small, coordinated perturbations. This isn't a bug — it's inherent to how neural networks work.
- **"Adversarial perturbations are random noise."** — They are carefully *computed* to maximize the loss. Random noise rarely fools a network.
- **"Adversarial training fully solves the problem."** — It improves robustness against specific attack types but reduces clean accuracy and doesn't protect against all attacks.

---

## Intuition

### Why Small Perturbations Fool Networks

Consider a linear classifier $f(x) = w^T x$. A perturbation $\delta = \epsilon \cdot \text{sign}(w)$ changes the output by:

$$w^T(x + \delta) - w^Tx = w^T \delta = \epsilon \sum_i |w_i|$$

For a 150,528-dimensional input, even $\epsilon = 0.01$ (invisible change) produces a huge output change: $\epsilon \cdot \|w\|_1 \approx 0.01 \times 150{,}528 = 1{,}505$.

```
High-dimensional input space:

Clean image x:    [0.5, 0.3, 0.8, 0.1, ...]  → "cat" (99%)
                       +ε   +ε   -ε   +ε
Adversarial x':   [0.51, 0.31, 0.79, 0.11, ...] → "dog" (95%)

Each pixel changes by < 0.01 (invisible!)
But across 150K dimensions, the total effect is MASSIVE.
```

### FGSM: One-Step Attack

```
1. Forward pass:  x → model → loss L(f(x), y_true)
2. Backward pass: ∂L/∂x  (gradient of loss w.r.t. INPUT pixels)
3. Perturb:       x_adv = x + ε · sign(∂L/∂x)
4. Clamp:         x_adv = clamp(x_adv, 0, 1)

        Original                    Perturbation              Adversarial
     ┌────────┐                  ┌────────┐               ┌────────┐
     │  cat   │  +  ε ×          │ +-+-+- │   =           │  dog?  │
     │  🐱    │                  │ -+-+-+ │               │  🐱    │
     │        │                  │ +-+-+- │               │        │
     └────────┘                  └────────┘               └────────┘
     P(cat)=99%              sign(gradient)            P(dog)=95%
                          (looks like noise)        (looks identical to human)
```

### PGD: Multi-Step Attack

```
PGD = Iterative FGSM with projection

Step 0: x^(0) = x + uniform random noise within ε-ball
Step 1: x^(1) = Π[x^(0) + α·sign(∇L)]     (FGSM step + project)
Step 2: x^(2) = Π[x^(1) + α·sign(∇L)]     (FGSM step + project)
...
Step K: x^(K) = Π[x^(K-1) + α·sign(∇L)]   (final adversarial example)

Π = project back onto ε-ball around original x
α = step size (smaller than ε)
K = number of iterations (typically 10-50)
```

### The L-infinity Ball

```
Perturbation budget: ||x_adv - x||_∞ ≤ ε

Each pixel can change by at most ε:
x_i - ε ≤ x_adv_i ≤ x_i + ε

For ε = 8/255 ≈ 0.031:
┌──────────────────┐
│ ─ε ─ ─ ─ ─ ─ +ε │  ← pixel i can vary in this range
│ ←───── 2ε ─────→ │
└──────────────────┘

Typical ε values:
- ε = 4/255  (subtle, hard to detect)
- ε = 8/255  (standard benchmark)
- ε = 16/255 (visible if you look carefully)
```

---

## Math

### FGSM (Fast Gradient Sign Method)

Given input $x$, true label $y$, model $f_\theta$, and loss $\mathcal{L}$:

$$x_{adv} = x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(f_\theta(x), y)\right)$$

**Why `sign`?** The $\ell_\infty$ norm constraint $\|x_{adv} - x\|_\infty \leq \epsilon$ means each pixel can change by at most $\epsilon$. The sign function maximizes the perturbation within this budget:

$$\text{sign}(\nabla_x \mathcal{L}) = \arg\max_{\delta: \|\delta\|_\infty \leq 1} \nabla_x \mathcal{L}^T \delta$$

This follows from the dual norm: the maximum of a linear function over the $\ell_\infty$ ball is achieved by aligning with the sign of the gradient.

### PGD (Projected Gradient Descent)

PGD is iterative FGSM with projection:

$$x^{(0)} = x + \text{Uniform}(-\epsilon, \epsilon)$$
$$x^{(k+1)} = \Pi_{B_\epsilon(x)}\left[x^{(k)} + \alpha \cdot \text{sign}\left(\nabla_{x^{(k)}} \mathcal{L}(f_\theta(x^{(k)}), y)\right)\right]$$

where $\Pi_{B_\epsilon(x)}$ projects onto the $\ell_\infty$ ball of radius $\epsilon$ around $x$:

$$\Pi_{B_\epsilon(x)}[z] = \text{clamp}(z, x - \epsilon, x + \epsilon)$$

Followed by clamping to valid pixel range: $x^{(k+1)} = \text{clamp}(x^{(k+1)}, 0, 1)$.

### Adversarial Training

Train on worst-case perturbations:

$$\min_\theta \; \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[\max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y)\right]$$

In practice:
1. For each training batch, compute PGD adversarial examples
2. Train the model on these adversarial examples
3. This is a min-max optimization (inner max by PGD, outer min by SGD)

### Targeted vs Untargeted Attacks

**Untargeted** (maximize loss for true class):
$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{true}))$$

**Targeted** (minimize loss for target class):
$$x_{adv} = x - \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y_{target}))$$

Note the sign flip: we go *against* the gradient for the target class.

---

## Code

### FGSM Attack

```python
import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon):
    """
    model: trained classifier
    images: (B, C, H, W) in [0, 1]
    labels: (B,) true labels
    epsilon: perturbation budget
    Returns: adversarial images
    """
    images.requires_grad_(True)

    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    # Backward pass — gradient w.r.t. INPUT
    loss.backward()

    # FGSM perturbation
    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation

    # Clamp to valid pixel range
    adv_images = adv_images.clamp(0, 1)
    return adv_images.detach()
```

### PGD Attack

```python
def pgd_attack(model, images, labels, epsilon, alpha, num_steps):
    """
    model: trained classifier
    images: (B, C, H, W) in [0, 1]
    labels: (B,) true labels
    epsilon: perturbation budget (L-inf)
    alpha: step size per iteration
    num_steps: number of PGD iterations
    Returns: adversarial images
    """
    adv_images = images.clone().detach()

    # Random start within epsilon ball
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = adv_images.clamp(0, 1)

    for _ in range(num_steps):
        adv_images.requires_grad_(True)

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # FGSM step
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project onto epsilon ball around original images
        delta = adv_images - images
        delta = delta.clamp(-epsilon, epsilon)
        adv_images = (images + delta).clamp(0, 1)

    return adv_images.detach()
```

### Adversarial Training Loop

```python
def adversarial_train_step(model, images, labels, optimizer, epsilon=8/255, alpha=2/255, pgd_steps=7):
    model.eval()  # for generating adversarial examples
    adv_images = pgd_attack(model, images, labels, epsilon, alpha, pgd_steps)

    model.train()
    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Evaluating Robustness

```python
def evaluate_robustness(model, test_loader, epsilon, alpha=2/255, pgd_steps=20):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in test_loader:
        # Clean accuracy
        outputs = model(images)
        clean_correct += (outputs.argmax(1) == labels).sum().item()

        # Adversarial accuracy
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, pgd_steps)
        adv_outputs = model(adv_images)
        adv_correct += (adv_outputs.argmax(1) == labels).sum().item()

        total += labels.size(0)

    print(f"Clean accuracy: {100 * clean_correct / total:.1f}%")
    print(f"PGD-{pgd_steps} accuracy (eps={epsilon:.4f}): {100 * adv_correct / total:.1f}%")
```

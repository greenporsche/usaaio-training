# Adversarial Attack Exercises

**5 exercises** | Covers: FGSM computation, PGD iteration, perturbation analysis, adversarial training, targeted attacks

---

## Exercise 1: FGSM by Hand

**Target time**: 5 minutes

A simple linear classifier $f(x) = w^T x$ with weights $w = [2, -1, 3]$ classifies input $x = [0.5, 0.8, 0.3]$.

The loss gradient w.r.t. input is $\nabla_x \mathcal{L} = [0.4, -0.2, 0.6]$.

Perturbation budget: $\epsilon = 0.1$.

**Part 1**: Compute the FGSM perturbation: $\delta = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$.

**Part 2**: Compute the adversarial example: $x_{adv} = x + \delta$.

**Part 3**: Compute $f(x)$ and $f(x_{adv})$. How much did the output change?

<details>
<summary>Solution</summary>

**Part 1**:
$\text{sign}(\nabla_x \mathcal{L}) = \text{sign}([0.4, -0.2, 0.6]) = [1, -1, 1]$

$\delta = 0.1 \times [1, -1, 1] = [0.1, -0.1, 0.1]$

**Part 2**:
$x_{adv} = [0.5, 0.8, 0.3] + [0.1, -0.1, 0.1] = [0.6, 0.7, 0.4]$

**Part 3**:
$f(x) = 2(0.5) + (-1)(0.8) + 3(0.3) = 1.0 - 0.8 + 0.9 = 1.1$

$f(x_{adv}) = 2(0.6) + (-1)(0.7) + 3(0.4) = 1.2 - 0.7 + 1.2 = 1.7$

Change: $|f(x_{adv}) - f(x)| = |1.7 - 1.1| = 0.6$.

Note: The maximum possible change is $\epsilon \cdot \|w\|_1 = 0.1 \times (2+1+3) = 0.6$. FGSM achieves exactly this maximum — it's optimal for the $\ell_\infty$ budget against linear models.

</details>

---

## Exercise 2: PGD Iteration

**Target time**: 5 minutes

Starting from $x = [0.5, 0.5]$ with $\epsilon = 0.15$, step size $\alpha = 0.1$.

**Step 0**: $x^{(0)} = [0.5, 0.5]$ (no random initialization for simplicity).

The gradient at each step is:
- Step 1: $\nabla \mathcal{L}|_{x^{(0)}} = [0.3, -0.7]$
- Step 2: $\nabla \mathcal{L}|_{x^{(1)}} = [0.5, -0.2]$

**Part 1**: Compute $x^{(1)}$ after the first PGD step (FGSM + project onto $\epsilon$-ball).

**Part 2**: Compute $x^{(2)}$ after the second PGD step.

**Part 3**: Verify that $\|x^{(2)} - x\|_\infty \leq \epsilon$ at every step.

<details>
<summary>Solution</summary>

**Part 1**: FGSM step:
$x' = x^{(0)} + \alpha \cdot \text{sign}(\nabla \mathcal{L}) = [0.5, 0.5] + 0.1 \cdot [1, -1] = [0.6, 0.4]$

Project onto $\epsilon$-ball: $\delta = x' - x = [0.1, -0.1]$. $\|\delta\|_\infty = 0.1 \leq 0.15$. No clipping needed.

$x^{(1)} = [0.6, 0.4]$

**Part 2**: FGSM step:
$x' = x^{(1)} + \alpha \cdot \text{sign}(\nabla \mathcal{L}) = [0.6, 0.4] + 0.1 \cdot [1, -1] = [0.7, 0.3]$

Project onto $\epsilon$-ball: $\delta = x' - x = [0.7 - 0.5, 0.3 - 0.5] = [0.2, -0.2]$. $\|\delta\|_\infty = 0.2 > 0.15$!

Clamp: $\delta_{clipped} = [\text{clamp}(0.2, -0.15, 0.15), \text{clamp}(-0.2, -0.15, 0.15)] = [0.15, -0.15]$

$x^{(2)} = x + \delta_{clipped} = [0.5 + 0.15, 0.5 - 0.15] = [0.65, 0.35]$

**Part 3**:
- $x^{(1)}$: $\|[0.6, 0.4] - [0.5, 0.5]\|_\infty = \|[0.1, -0.1]\|_\infty = 0.1 \leq 0.15$ ✓
- $x^{(2)}$: $\|[0.65, 0.35] - [0.5, 0.5]\|_\infty = \|[0.15, -0.15]\|_\infty = 0.15 \leq 0.15$ ✓

The projection ensures we never exceed the budget.

</details>

---

## Exercise 3: Why Small Perturbations Have Large Effects

**Target time**: 3 minutes

A neural network's final linear layer has 1000 weights for a single output neuron, each with magnitude approximately $|w_i| \approx 0.1$.

**Part 1**: What is the approximate $\ell_1$ norm $\|w\|_1$?

**Part 2**: With FGSM perturbation $\delta = \epsilon \cdot \text{sign}(w)$ and $\epsilon = 8/255 \approx 0.031$, what is the change in the output $w^T \delta$?

**Part 3**: If the logit gap between the correct and second-best class is 2.0, is this perturbation likely to cause misclassification?

<details>
<summary>Solution</summary>

**Part 1**: $\|w\|_1 = \sum_{i=1}^{1000} |w_i| \approx 1000 \times 0.1 = 100$

**Part 2**: $w^T \delta = w^T (\epsilon \cdot \text{sign}(w)) = \epsilon \sum_i |w_i| = \epsilon \|w\|_1 = 0.031 \times 100 = 3.1$

**Part 3**: Yes! The perturbation changes the output by 3.1, which exceeds the logit gap of 2.0. This means FGSM can flip the classification from the correct class to the second-best class. The key insight: in high dimensions, many tiny per-dimension changes accumulate into a large total effect.

</details>

---

## Exercise 4: Targeted FGSM

**Target time**: 4 minutes

A classifier outputs logits for 3 classes: $f(x) = [3.0, 1.5, 2.0]$ (class 0 is correct).

The gradients of the loss $\mathcal{L}(f(x), y_{target})$ w.r.t. $x$ for different targets are:
- Target class 1: $\nabla_x \mathcal{L}_{target=1} = [0.3, -0.5, 0.1, 0.4]$
- Target class 2: $\nabla_x \mathcal{L}_{target=2} = [-0.2, 0.4, -0.6, 0.3]$

$x = [0.5, 0.5, 0.5, 0.5]$, $\epsilon = 0.05$.

**Part 1**: For a targeted attack toward class 1, compute $x_{adv}$. (Remember: targeted = *subtract* $\epsilon \cdot \text{sign}(\nabla)$.)

**Part 2**: For a targeted attack toward class 2, compute $x_{adv}$.

**Part 3**: Which targeted attack changes the input more (in $\ell_2$ norm)? Does it matter?

<details>
<summary>Solution</summary>

**Part 1**: Targeted attack toward class 1:
$x_{adv} = x - \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}_{target=1})$
$= [0.5, 0.5, 0.5, 0.5] - 0.05 \cdot \text{sign}([0.3, -0.5, 0.1, 0.4])$
$= [0.5, 0.5, 0.5, 0.5] - 0.05 \cdot [1, -1, 1, 1]$
$= [0.45, 0.55, 0.45, 0.45]$

**Part 2**: Targeted attack toward class 2:
$x_{adv} = [0.5, 0.5, 0.5, 0.5] - 0.05 \cdot \text{sign}([-0.2, 0.4, -0.6, 0.3])$
$= [0.5, 0.5, 0.5, 0.5] - 0.05 \cdot [-1, 1, -1, 1]$
$= [0.55, 0.45, 0.55, 0.45]$

**Part 3**: Both have the same $\ell_2$ norm of change:
$\|\delta_1\|_2 = 0.05\sqrt{4} = 0.1$
$\|\delta_2\|_2 = 0.05\sqrt{4} = 0.1$

The $\ell_2$ norm is always the same: $\epsilon\sqrt{d}$ for any FGSM perturbation in $d$ dimensions. The $\ell_\infty$ norm is exactly $\epsilon$ by construction. What differs is the *direction* of the perturbation, which determines which class the model switches to.

</details>

---

## Exercise 5: Adversarial Training Analysis

**Target time**: 3 minutes

A model achieves:
- Clean accuracy: 95% (standard training) vs. 87% (adversarial training)
- PGD-20 accuracy (ε=8/255): 0% (standard) vs. 53% (adversarial)

**Part 1**: What is the robustness gap for each model (clean accuracy - adversarial accuracy)?

**Part 2**: Why does adversarial training reduce clean accuracy?

**Part 3**: A new attack (AutoAttack) achieves 45% accuracy against the adversarially-trained model (vs. 53% for PGD-20). What does this tell us?

<details>
<summary>Solution</summary>

**Part 1**:
- Standard model: $95\% - 0\% = 95\%$ robustness gap
- Adversarial model: $87\% - 53\% = 34\%$ robustness gap

Adversarial training dramatically reduces the gap, but a 34% gap remains.

**Part 2**: Adversarial training forces the model to be correct not just at each data point $x$, but in the entire $\epsilon$-ball around $x$. This is a harder task — the decision boundary must be smoother and farther from training points. This "robustness-accuracy tradeoff" means the model sacrifices some performance on clean data to gain robustness. Intuitively, adversarially-robust features tend to be simpler and more interpretable but less discriminative.

**Part 3**: PGD-20 overestimates robustness — it's not the strongest possible attack. AutoAttack (a stronger, ensemble-based attack) finds adversarial examples that PGD-20 misses, reducing accuracy from 53% to 45%. This reveals an 8% "false robustness" where PGD-20 failed to find existing adversarial examples. When evaluating robustness, always use the strongest available attack — reported robustness is an *upper bound* on true robustness.

</details>

---

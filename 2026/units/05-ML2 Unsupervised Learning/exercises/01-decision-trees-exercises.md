# Decision Trees — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: Compute Gini Impurity

A node contains 30 samples: 18 belong to class A and 12 belong to class B.

1. Compute the Gini impurity of this node.
2. If the node were split into two children — Left (12A, 3B) and Right (6A, 9B) — compute the weighted Gini impurity after the split.
3. What is the Gini gain (reduction in impurity) from this split?

<details>
<summary>Solution</summary>

**Part 1**: Gini impurity of parent node.

$p_A = 18/30 = 0.6$, $p_B = 12/30 = 0.4$

$G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$

**Part 2**: Weighted Gini after split.

Left (15 samples): $G_L = 1 - (12/15)^2 - (3/15)^2 = 1 - 0.64 - 0.04 = 0.32$

Right (15 samples): $G_R = 1 - (6/15)^2 - (9/15)^2 = 1 - 0.16 - 0.36 = 0.48$

Weighted: $G_{\text{split}} = \frac{15}{30}(0.32) + \frac{15}{30}(0.48) = 0.16 + 0.24 = 0.40$

**Part 3**: Gini gain.

$\Delta G = 0.48 - 0.40 = 0.08$

</details>

---

## Exercise 2: Compute Entropy and Information Gain

A dataset has 14 samples: 9 positive (+) and 5 negative (-). A candidate split on feature "Outlook" produces:

| Outlook | + | - | Total |
|---------|---|---|-------|
| Sunny | 2 | 3 | 5 |
| Overcast | 4 | 0 | 4 |
| Rain | 3 | 2 | 5 |

1. Compute the entropy of the parent node.
2. Compute the entropy of each child.
3. Compute the information gain.

<details>
<summary>Solution</summary>

**Part 1**: Parent entropy.

$H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14}$

$= -0.643 \times (-0.637) - 0.357 \times (-1.485) = 0.410 + 0.530 = 0.940$ bits

**Part 2**: Child entropies.

Sunny: $H = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} = 0.971$ bits

Overcast: $H = -\frac{4}{4}\log_2\frac{4}{4} = 0$ bits (pure node!)

Rain: $H = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} = 0.971$ bits

**Part 3**: Information gain.

$IG = H(S) - \frac{5}{14}H(\text{Sunny}) - \frac{4}{14}H(\text{Overcast}) - \frac{5}{14}H(\text{Rain})$

$= 0.940 - \frac{5}{14}(0.971) - \frac{4}{14}(0) - \frac{5}{14}(0.971)$

$= 0.940 - 0.347 - 0 - 0.347 = 0.246$ bits

</details>

---

## Exercise 3: Trace a Decision Tree Split (Continuous Feature)

Given the following 1D dataset:

| x | y (class) |
|---|-----------|
| 1 | - |
| 2 | - |
| 3 | + |
| 4 | - |
| 5 | + |
| 6 | + |
| 7 | + |
| 8 | + |

Candidate thresholds are midpoints: 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5.

1. Compute the Gini impurity of the parent.
2. For threshold = 2.5 (left: x <= 2.5, right: x > 2.5), compute the Gini gain.
3. For threshold = 4.5 (left: x <= 4.5, right: x > 4.5), compute the Gini gain.
4. Which threshold is better?

<details>
<summary>Solution</summary>

**Part 1**: Parent: 5 positive, 3 negative out of 8.

$G = 1 - (5/8)^2 - (3/8)^2 = 1 - 0.3906 - 0.1406 = 0.4688$

**Part 2**: Threshold = 2.5.

Left ($x \leq 2.5$): 0+, 2- → $G_L = 1 - 0 - 1 = 0$ (pure!)
Right ($x > 2.5$): 5+, 1- → $G_R = 1 - (5/6)^2 - (1/6)^2 = 1 - 0.6944 - 0.0278 = 0.2778$

Weighted: $G = \frac{2}{8}(0) + \frac{6}{8}(0.2778) = 0 + 0.2083 = 0.2083$

Gain: $0.4688 - 0.2083 = 0.2604$

**Part 3**: Threshold = 4.5.

Left ($x \leq 4.5$): 1+, 3- → $G_L = 1 - (1/4)^2 - (3/4)^2 = 1 - 0.0625 - 0.5625 = 0.375$
Right ($x > 4.5$): 4+, 0- → $G_R = 0$ (pure!)

Weighted: $G = \frac{4}{8}(0.375) + \frac{4}{8}(0) = 0.1875$

Gain: $0.4688 - 0.1875 = 0.2813$

**Part 4**: Threshold 4.5 has higher Gini gain (0.2813 > 0.2604), so it is the better split.

</details>

---

## Exercise 4: Overfitting in Decision Trees

Consider a training set with 100 samples and a test set with 50 samples. You train decision trees with varying max depths:

| Max Depth | Train Accuracy | Test Accuracy |
|-----------|---------------|---------------|
| 1 | 72% | 70% |
| 3 | 88% | 85% |
| 5 | 95% | 83% |
| 10 | 99% | 78% |
| None (unlimited) | 100% | 72% |

1. At which depth does overfitting begin?
2. What max depth would you choose and why?
3. Name two other techniques (besides limiting depth) to prevent overfitting in decision trees.

<details>
<summary>Solution</summary>

**Part 1**: Overfitting begins at depth 5. Test accuracy peaks at depth 3 (85%) and declines after that, while training accuracy continues to increase. The gap between train and test accuracy grows from depth 5 onward.

**Part 2**: Choose max depth = 3. It has the highest test accuracy (85%) and a small train-test gap (3%), indicating good generalization.

**Part 3**: Two other techniques:

1. **Minimum samples per leaf** (`min_samples_leaf`): Require each leaf to have at least $k$ samples, preventing the tree from memorizing individual points.

2. **Cost-complexity pruning** (post-pruning): Grow a full tree, then prune subtrees that don't improve validation performance. Uses the objective $R_\alpha(T) = R(T) + \alpha|T|$ where increasing $\alpha$ favors smaller trees.

Other valid answers: minimum samples per split, maximum number of leaf nodes, minimum impurity decrease.

</details>

---

## Exercise 5: Comparing Impurity Measures

For a binary classification node with class proportion $p$ (for the positive class):

1. Write the Gini impurity as a function of $p$.
2. Write the entropy as a function of $p$.
3. At $p = 0.5$, compute both Gini and entropy.
4. At $p = 0.1$, compute both. Which measure is more "sensitive" to the impurity at this point?
5. Explain why misclassification error is not suitable as a splitting criterion, even though it's used for evaluating final predictions.

<details>
<summary>Solution</summary>

**Part 1**: $G(p) = 1 - p^2 - (1-p)^2 = 2p(1-p)$

**Part 2**: $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$

**Part 3**: At $p = 0.5$:

$G(0.5) = 2 \times 0.5 \times 0.5 = 0.5$

$H(0.5) = -0.5\log_2(0.5) - 0.5\log_2(0.5) = -0.5(-1) - 0.5(-1) = 1.0$

**Part 4**: At $p = 0.1$:

$G(0.1) = 2 \times 0.1 \times 0.9 = 0.18$

$H(0.1) = -0.1\log_2(0.1) - 0.9\log_2(0.9) = -0.1(-3.322) - 0.9(-0.152) = 0.332 + 0.137 = 0.469$

Entropy (0.469) is relatively higher than Gini (0.18) compared to their maximums (1.0 vs 0.5). Entropy $= 0.469/1.0 = 46.9\%$ of max, while Gini $= 0.18/0.5 = 36\%$ of max. Entropy is more sensitive to impurity in skewed distributions.

**Part 5**: Misclassification error $E(p) = \min(p, 1-p)$ is piecewise linear. It doesn't differentiate between a 60/40 split and an 80/20 split on the majority side — both might have the same weighted misclassification error after a split, even though 80/20 is clearly "more pure." Gini and entropy are strictly concave, so any split that moves proportions away from 50/50 is rewarded. This makes them better for choosing splits, even though misclassification error is the natural evaluation metric.

</details>

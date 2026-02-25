# Boosting — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: AdaBoost Weight Update

Given 6 training samples with labels $y_i \in \{-1, +1\}$, initial weights $w_i = 1/6$, and a weak learner $h_1$ that predicts as shown:

| $i$ | $y_i$ | $h_1(x_i)$ | Correct? |
|-----|--------|------------|----------|
| 1 | +1 | +1 | Yes |
| 2 | +1 | +1 | Yes |
| 3 | -1 | -1 | Yes |
| 4 | -1 | +1 | No |
| 5 | +1 | -1 | No |
| 6 | -1 | -1 | Yes |

1. Compute the weighted error $\epsilon_1$.
2. Compute the learner weight $\alpha_1$.
3. Compute the updated (unnormalized) weights.
4. Normalize the weights. Which samples got larger weights?

<details>
<summary>Solution</summary>

**Part 1**: $\epsilon_1 = \sum_{i: h_1(x_i) \neq y_i} w_i = w_4 + w_5 = \frac{1}{6} + \frac{1}{6} = \frac{1}{3} \approx 0.333$

**Part 2**: $\alpha_1 = \frac{1}{2}\ln\frac{1 - \epsilon_1}{\epsilon_1} = \frac{1}{2}\ln\frac{2/3}{1/3} = \frac{1}{2}\ln 2 \approx 0.347$

**Part 3**: Unnormalized weights $\tilde{w}_i = w_i \exp(-\alpha_1 y_i h_1(x_i))$:

For correct predictions ($y_i h_1(x_i) = +1$): $\tilde{w}_i = \frac{1}{6}\exp(-0.347) = \frac{1}{6}(0.707) = 0.1178$

For incorrect predictions ($y_i h_1(x_i) = -1$): $\tilde{w}_i = \frac{1}{6}\exp(+0.347) = \frac{1}{6}(1.414) = 0.2357$

| $i$ | $\tilde{w}_i$ |
|-----|---------------|
| 1 | 0.1178 |
| 2 | 0.1178 |
| 3 | 0.1178 |
| 4 | **0.2357** |
| 5 | **0.2357** |
| 6 | 0.1178 |

**Part 4**: Sum = $4(0.1178) + 2(0.2357) = 0.4714 + 0.4714 = 0.9428$

Wait — let me recompute. Actually $\exp(-0.347) = 1/\sqrt{2} \approx 0.7071$ and $\exp(0.347) = \sqrt{2} \approx 1.4142$.

Normalized: $Z = 4 \times \frac{0.7071}{6} + 2 \times \frac{1.4142}{6} = \frac{2.8284 + 2.8284}{6} = \frac{5.6569}{6} = 0.9428$

Normalized weights:
- Correct samples: $w_i = 0.1178/0.9428 = 0.125 = 1/8$
- Incorrect samples: $w_i = 0.2357/0.9428 = 0.250 = 1/4$

Samples 4 and 5 (misclassified) doubled in weight relative to the correct samples. The next weak learner will focus on getting these two right.

</details>

---

## Exercise 2: AdaBoost Learner Weight

A weak learner achieves the following weighted errors across three boosting rounds:

| Round | $\epsilon_t$ |
|-------|--------------|
| 1 | 0.3 |
| 2 | 0.45 |
| 3 | 0.1 |

1. Compute $\alpha_t$ for each round.
2. Which round's learner gets the most influence in the final vote?
3. What happens if $\epsilon_t = 0.5$? What about $\epsilon_t > 0.5$?
4. In the final prediction $H(x) = \text{sign}(\alpha_1 h_1 + \alpha_2 h_2 + \alpha_3 h_3)$, what are the relative weights?

<details>
<summary>Solution</summary>

**Part 1**:

$\alpha_1 = \frac{1}{2}\ln\frac{0.7}{0.3} = \frac{1}{2}\ln(2.333) = \frac{1}{2}(0.847) = 0.424$

$\alpha_2 = \frac{1}{2}\ln\frac{0.55}{0.45} = \frac{1}{2}\ln(1.222) = \frac{1}{2}(0.201) = 0.100$

$\alpha_3 = \frac{1}{2}\ln\frac{0.9}{0.1} = \frac{1}{2}\ln(9) = \frac{1}{2}(2.197) = 1.099$

**Part 2**: Round 3 ($\alpha_3 = 1.099$) gets the most influence because it had the lowest error (0.1).

**Part 3**:
- $\epsilon_t = 0.5$: $\alpha_t = \frac{1}{2}\ln(1) = 0$. The learner is no better than random and gets zero weight — effectively ignored.
- $\epsilon_t > 0.5$: $\alpha_t < 0$. The learner is worse than random! AdaBoost flips its predictions (negative weight) and uses the inverted learner. In practice, if $\epsilon \geq 0.5$, boosting usually stops.

**Part 4**: Relative weights are $0.424 : 0.100 : 1.099$, or approximately $4.2 : 1 : 11$. Round 3's learner has about 11x the influence of Round 2's learner.

</details>

---

## Exercise 3: Gradient Boosting Residuals

You are performing gradient boosting for regression with squared loss. The true values and current predictions after 2 rounds are:

| $i$ | $y_i$ | $F_2(x_i)$ |
|-----|--------|------------|
| 1 | 10 | 8.5 |
| 2 | 15 | 14.2 |
| 3 | 7 | 8.0 |
| 4 | 20 | 18.0 |
| 5 | 12 | 11.5 |

The learning rate is $\eta = 0.1$.

1. Compute the residuals (pseudo-residuals for squared loss).
2. Suppose a tree $h_3$ fits these residuals and produces predictions: $h_3 = [1.2, 0.6, -0.8, 1.5, 0.4]$. Compute $F_3(x_i)$.
3. Compute the new residuals after round 3.
4. Compute the MSE before and after adding $h_3$.

<details>
<summary>Solution</summary>

**Part 1**: For squared loss, pseudo-residuals = actual residuals:

$r_i = y_i - F_2(x_i)$

| $i$ | $r_i$ |
|-----|--------|
| 1 | 1.5 |
| 2 | 0.8 |
| 3 | -1.0 |
| 4 | 2.0 |
| 5 | 0.5 |

**Part 2**: $F_3(x_i) = F_2(x_i) + \eta \cdot h_3(x_i)$

| $i$ | $F_3(x_i)$ |
|-----|------------|
| 1 | $8.5 + 0.1(1.2) = 8.62$ |
| 2 | $14.2 + 0.1(0.6) = 14.26$ |
| 3 | $8.0 + 0.1(-0.8) = 7.92$ |
| 4 | $18.0 + 0.1(1.5) = 18.15$ |
| 5 | $11.5 + 0.1(0.4) = 11.54$ |

**Part 3**: New residuals:

| $i$ | $y_i - F_3(x_i)$ |
|-----|-------------------|
| 1 | $10 - 8.62 = 1.38$ |
| 2 | $15 - 14.26 = 0.74$ |
| 3 | $7 - 7.92 = -0.92$ |
| 4 | $20 - 18.15 = 1.85$ |
| 5 | $12 - 11.54 = 0.46$ |

Residuals decreased in magnitude — the model improved!

**Part 4**:

MSE before (round 2): $\frac{1}{5}(1.5^2 + 0.8^2 + 1.0^2 + 2.0^2 + 0.5^2) = \frac{1}{5}(2.25 + 0.64 + 1.0 + 4.0 + 0.25) = \frac{8.14}{5} = 1.628$

MSE after (round 3): $\frac{1}{5}(1.38^2 + 0.74^2 + 0.92^2 + 1.85^2 + 0.46^2) = \frac{1}{5}(1.904 + 0.548 + 0.846 + 3.423 + 0.212) = \frac{6.933}{5} = 1.387$

MSE decreased from 1.628 to 1.387 — a 14.8% improvement. Note how the small learning rate (0.1) makes conservative updates.

</details>

---

## Exercise 4: Boosting vs. Bagging Comparison

Answer the following conceptual questions:

1. In bagging, can trees be trained in parallel? What about boosting? Explain why.
2. If your dataset has 5% mislabeled samples (noisy labels), which method is more affected: random forests or AdaBoost? Why?
3. A random forest with 100 trees and a gradient boosting model with 100 trees both achieve 90% training accuracy. Which one is more likely to improve with 200 trees?
4. Explain the "bias-variance tradeoff" difference between bagging and boosting.

<details>
<summary>Solution</summary>

**Part 1**:
- **Bagging (Random Forests)**: Yes, trees can be trained in parallel. Each tree is built on an independent bootstrap sample, so no tree depends on any other.
- **Boosting**: No, trees must be trained sequentially. Each tree depends on the errors of all previous trees (through sample weights in AdaBoost or residuals in gradient boosting).

**Part 2**: **AdaBoost** is more affected. Mislabeled samples will be repeatedly misclassified, causing their weights to increase exponentially. AdaBoost will increasingly focus on these "hard" examples (which are actually noise), leading to overfitting. Random forests average over independent trees, so noise in one bootstrap sample is diluted.

**Part 3**: **Boosting** is more likely to improve. With 100 trees at 90% accuracy, boosting can add 100 more trees to fit the remaining 10% error (residuals). Random forests, in contrast, already have diminishing returns — adding more trees reduces the $\frac{(1-\rho)}{B}\sigma^2$ term, which is already small at $B=100$. The irreducible term $\rho\sigma^2$ doesn't change.

**Part 4**:
- **Bagging** primarily reduces **variance** while keeping bias roughly constant. Each tree has high variance but low bias; averaging many trees keeps the low bias while reducing variance.
- **Boosting** primarily reduces **bias** by sequentially correcting errors. Each round adds a learner that fits the residual, reducing the overall bias. However, too many rounds can increase variance (overfitting).

</details>

---

## Exercise 5: Learning Rate and Number of Estimators

A gradient boosting model is trained with different configurations:

| Config | Learning Rate ($\eta$) | # Trees | Train MSE | Test MSE |
|--------|----------------------|---------|-----------|----------|
| A | 1.0 | 10 | 0.05 | 0.85 |
| B | 0.1 | 100 | 0.10 | 0.30 |
| C | 0.01 | 1000 | 0.15 | 0.25 |
| D | 0.01 | 100 | 0.80 | 0.82 |
| E | 0.001 | 1000 | 0.90 | 0.91 |

1. Which configuration shows the most overfitting? How can you tell?
2. Which is the best configuration? Why?
3. Why does Config D have high training AND test error?
4. What general relationship exists between learning rate and optimal number of trees?
5. If you had unlimited computation, how would you find the optimal (learning rate, number of trees) pair?

<details>
<summary>Solution</summary>

**Part 1**: **Config A** (train MSE = 0.05, test MSE = 0.85) shows the most overfitting. The gap of 0.80 between train and test error is by far the largest. The high learning rate ($\eta = 1.0$) causes each tree to make large updates, fitting the training data very closely (including noise) without generalizing.

**Part 2**: **Config C** is the best (lowest test MSE = 0.25). The small learning rate with many trees achieves good generalization. The train-test gap is reasonable (0.10), suggesting the model generalizes well.

**Part 3**: Config D is **underfitting**. With learning rate 0.01 and only 100 trees, the total "learning capacity" is $\eta \times \text{trees} = 0.01 \times 100 = 1.0$, which may be too little. The model hasn't had enough rounds to adequately fit the training data, let alone generalize. (Compare to Config C: $0.01 \times 1000 = 10$.)

**Part 4**: Smaller learning rates require more trees to achieve the same effective model complexity. Roughly: $\eta_1 \times T_1 \approx \eta_2 \times T_2$ for similar final performance (though smaller $\eta$ with more trees usually generalizes better due to the regularization effect of small steps).

**Part 5**: Use **early stopping with cross-validation**:
1. Set a small learning rate (e.g., 0.01 or 0.001).
2. Set a large maximum number of trees (e.g., 10,000).
3. Use a validation set or k-fold CV; monitor validation error after each tree.
4. Stop when validation error hasn't improved for $k$ consecutive rounds.
5. Optionally, grid search over learning rates: {0.3, 0.1, 0.05, 0.01}.

</details>

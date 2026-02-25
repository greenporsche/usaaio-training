# Random Forests — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: Bootstrap Probability

A dataset has $n = 100$ samples. You create a bootstrap sample by drawing 100 samples with replacement.

1. What is the probability that a specific sample $x_i$ is NOT chosen in a single draw?
2. What is the probability that $x_i$ is NOT chosen in any of the 100 draws?
3. What is the probability that $x_i$ appears at least once in the bootstrap sample?
4. Approximately how many unique samples do you expect in the bootstrap sample?

<details>
<summary>Solution</summary>

**Part 1**: $P(\text{not chosen in one draw}) = 1 - \frac{1}{100} = 0.99$

**Part 2**: $P(\text{not chosen in 100 draws}) = (1 - \frac{1}{100})^{100} = 0.99^{100} \approx 0.366$

(In general, $(1 - 1/n)^n \to 1/e \approx 0.368$ as $n \to \infty$.)

**Part 3**: $P(\text{at least once}) = 1 - 0.366 = 0.634$

**Part 4**: Expected unique samples = $n \times P(\text{at least once}) = 100 \times 0.634 \approx 63.4$ unique samples.

The remaining ~36.6 samples are out-of-bag (OOB) and can be used for validation.

</details>

---

## Exercise 2: Feature Subsampling

You have a dataset with $p = 16$ features.

1. For a classification random forest, how many features should be considered at each split (standard rule of thumb)?
2. For a regression random forest, how many?
3. If one feature is extremely predictive (say, feature 7), what happens in a standard decision tree vs. a random forest?
4. Why does feature subsampling reduce the correlation between trees?

<details>
<summary>Solution</summary>

**Part 1**: Classification: $m = \lfloor\sqrt{p}\rfloor = \lfloor\sqrt{16}\rfloor = 4$ features per split.

**Part 2**: Regression: $m = \lfloor p/3 \rfloor = \lfloor 16/3 \rfloor = 5$ features per split.

**Part 3**:
- **Decision tree**: Every split will use feature 7 at the root (and possibly several subsequent levels), since it's always the best available feature.
- **Random forest**: Feature 7 is only available in $m/p = 4/16 = 25\%$ of split decisions. Many trees will be forced to find alternative features, discovering other useful patterns in the data.

**Part 4**: Without feature subsampling, if one feature dominates, all trees will make similar first splits, leading to highly correlated trees. Correlated trees don't benefit as much from averaging (recall: $\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2$). Feature subsampling forces trees to explore different features, reducing $\rho$, which reduces the irreducible first term.

</details>

---

## Exercise 3: OOB Error Calculation

A random forest has 5 trees. The following table shows which samples are in-bag (IB) or out-of-bag (OOB) for each tree, along with the tree's prediction:

| Sample | True Label | T1 | T2 | T3 | T4 | T5 |
|--------|------------|----|----|----|----|-----|
| $x_1$ | + | IB | OOB(+) | IB | OOB(+) | OOB(-) |
| $x_2$ | - | OOB(+) | IB | OOB(-) | OOB(-) | IB |
| $x_3$ | + | OOB(+) | OOB(+) | IB | IB | OOB(-) |
| $x_4$ | - | IB | IB | OOB(-) | OOB(+) | OOB(-) |

1. For each sample, compute the OOB prediction (majority vote of OOB trees only).
2. Compute the OOB error rate.

<details>
<summary>Solution</summary>

**Part 1**: OOB predictions:

- $x_1$: OOB trees are T2(+), T4(+), T5(-). Majority vote: **+** (2-1). True label: +. Correct.
- $x_2$: OOB trees are T1(+), T3(-), T4(-). Majority vote: **-** (2-1). True label: -. Correct.
- $x_3$: OOB trees are T1(+), T2(+), T5(-). Majority vote: **+** (2-1). True label: +. Correct.
- $x_4$: OOB trees are T3(-), T4(+), T5(-). Majority vote: **-** (2-1). True label: -. Correct.

**Part 2**: OOB error = 0/4 = **0%** (all samples correctly predicted OOB).

Note: In practice with small forests and few samples, OOB error can be noisy. With hundreds of trees, it closely approximates cross-validation error.

</details>

---

## Exercise 4: Variance Reduction Analysis

Consider a random forest with $B = 100$ trees. Each tree has prediction variance $\sigma^2 = 4$.

1. If the trees are independent ($\rho = 0$), what is the variance of the forest's prediction?
2. If the pairwise correlation is $\rho = 0.5$, what is the variance?
3. If $\rho = 0.9$ (highly correlated trees — e.g., no feature subsampling with a dominant feature), what is the variance?
4. For case 2, how many trees would you need to achieve the same variance as case 3?

<details>
<summary>Solution</summary>

Using $\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2$:

**Part 1**: $\rho = 0$: $\text{Var} = 0 + \frac{1}{100} \times 4 = 0.04$

**Part 2**: $\rho = 0.5$: $\text{Var} = 0.5 \times 4 + \frac{0.5}{100} \times 4 = 2.0 + 0.02 = 2.02$

**Part 3**: $\rho = 0.9$: $\text{Var} = 0.9 \times 4 + \frac{0.1}{100} \times 4 = 3.6 + 0.004 = 3.604$

**Part 4**: We want $0.5 \times 4 + \frac{0.5}{B} \times 4 = 3.604$.

$2.0 + \frac{2.0}{B} = 3.604$

$\frac{2.0}{B} = 1.604$

$B = 2.0 / 1.604 \approx 1.25$

This means even with just $B = 2$ trees at $\rho = 0.5$, you'd have lower variance than $B = 100$ trees at $\rho = 0.9$! This demonstrates that **reducing correlation ($\rho$) is far more important than adding more trees**. Feature subsampling is the real power of random forests.

</details>

---

## Exercise 5: Feature Importance Interpretation

A random forest trained on a housing dataset reports the following feature importances (Mean Decrease in Impurity):

| Feature | Importance |
|---------|-----------|
| Square footage | 0.45 |
| Number of bedrooms | 0.20 |
| Zip code | 0.15 |
| Year built | 0.10 |
| Has pool | 0.05 |
| Color of front door | 0.05 |

1. Do these importances necessarily reflect *causal* relationships? Explain.
2. Square footage and number of bedrooms are correlated. How might this affect their reported importances?
3. "Color of front door" has the same importance as "Has pool." A colleague suggests removing "Color of front door" since it's clearly irrelevant. Is the importance score evidence that it IS relevant, or could something else explain it?
4. How would you use **permutation importance** to get a more reliable ranking?

<details>
<summary>Solution</summary>

**Part 1**: No. Feature importance measures **predictive association**, not causation. "Zip code" predicts price because of location desirability, schools, etc. — it doesn't *cause* high prices. Changing a zip code wouldn't change a house's value.

**Part 2**: Correlated features *split* their importance. If bedrooms and square footage are highly correlated, the tree might split on either one at a given node. The importance gets divided between them, underestimating the true importance of each. If you removed one, the other would absorb its importance.

**Part 3**: MDI importance can be inflated for:
- **High-cardinality features**: Features with many unique values (like zip code) get more split opportunities, which can inflate MDI.
- **Noisy features**: Even random noise features can get nonzero MDI in deep trees because they'll occasionally reduce impurity by chance (overfitting to noise).

"Color of front door" with 0.05 importance is likely noise — it might have many categories giving it extra splitting opportunities. This does NOT mean it's actually predictive.

**Part 4**: Permutation importance works by:
1. Train the forest normally and compute OOB accuracy.
2. For each feature, randomly shuffle (permute) that feature's values in the OOB data.
3. Re-compute OOB accuracy with the shuffled feature.
4. Importance = drop in accuracy after shuffling.

This is more reliable because: (a) it measures actual impact on predictions, not just impurity decrease, (b) it's less biased toward high-cardinality features, and (c) a truly irrelevant feature like door color will show zero (or near-zero) drop when shuffled.

</details>

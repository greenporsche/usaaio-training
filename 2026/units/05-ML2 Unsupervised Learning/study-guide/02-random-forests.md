# Random Forests

**Prerequisites**: Decision Trees (01-decision-trees.md), basic statistics (variance, correlation)
**USAAIO Relevance**: Tested in Round 1 (concepts, OOB error, feature importance) and Round 2 (implement bagging, compare to single tree)

---

## Discovery

It's 2001, and you're Leo Breiman at UC Berkeley. You've spent decades studying decision trees (you co-invented CART in 1984), and you know their fatal flaw: **high variance**. Train the same tree on slightly different data, and you get a completely different tree. A single tree is an unreliable witness.

But you've noticed something in other fields: **committees make better decisions than individuals**. Jury trials use 12 people, not 1. Prediction markets aggregate thousands of opinions. Even noisy, biased individuals cancel each other out when aggregated.

**Your question**: Can you build a "committee of trees" that's more stable than any single tree?

You try an obvious approach: train 100 trees on the same data. But they all learn the same splits — identical trees give identical predictions. A committee of clones is no committee at all.

**The breakthrough**: What if each tree sees a *different* version of the data? And at each split, only considers a *random subset* of features? Now the trees are diverse — they make different mistakes — and their average smooths out the noise.

**Socratic questions**:
- Why would averaging reduce variance? (Hint: think of $\text{Var}(\bar{X}) = \sigma^2/n$ for independent variables.)
- Why do we subsample *features* and not just *data*? (Hint: what if one feature dominates?)
- If the trees are correlated, does averaging still help as much?

---

## Intuition

What you just discovered is Breiman's Random Forest (2001), one of the most successful ML algorithms ever — often competitive with deep learning on tabular data.

### Bootstrap Aggregation (Bagging)

The foundation is **bagging** (Bootstrap AGGregatING):

```
Original data:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Bootstrap 1:    [2, 5, 5, 3, 8, 1, 7, 7, 3, 9]    → Tree 1
Bootstrap 2:    [4, 1, 6, 6, 10, 3, 2, 8, 8, 1]   → Tree 2
Bootstrap 3:    [7, 3, 3, 9, 1, 5, 10, 2, 6, 4]   → Tree 3
...
Bootstrap B:    [...]                                → Tree B

Prediction = majority vote of all B trees
```

Each bootstrap sample draws $n$ points **with replacement** from $n$ original points. On average, each sample contains ~63.2% unique points (the rest are duplicates).

**Why 63.2%?** The probability a specific point is NOT chosen in one draw is $(1 - 1/n)$. Over $n$ draws: $(1 - 1/n)^n \to 1/e \approx 0.368$. So probability of being included at least once = $1 - 1/e \approx 0.632$.

### Feature Subsampling — The Secret Sauce

Bagging alone isn't enough. If one feature is very strong, every tree will split on it first, making the trees highly correlated. Random forests fix this:

```
At each split, instead of considering all p features:
  → Randomly select m features
  → Find best split among only those m features
  → Typical: m = sqrt(p) for classification, m = p/3 for regression
```

This forces diversity: some trees will find the "obvious" splits, others will discover subtler patterns.

### Out-of-Bag (OOB) Error — Free Cross-Validation

The ~36.8% of points NOT in a bootstrap sample are "out-of-bag" for that tree. Use them for validation — no need for a separate test set!

```
Point x_i is OOB for trees: {T_3, T_7, T_15, T_22, ...}
OOB prediction for x_i = majority vote of those trees
OOB error = fraction of incorrect OOB predictions across all points
```

OOB error closely approximates leave-one-out cross-validation error.

### Why It Works — Variance Reduction

For $B$ trees with individual variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}(\text{forest}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

- The second term $\to 0$ as $B \to \infty$ — just add more trees!
- The first term is irreducible and depends on $\rho$.
- Feature subsampling reduces $\rho$, which is why it helps.

### Failure Cases

- **Slow prediction**: Each test point traverses $B$ trees. Not a problem for batch prediction but matters for real-time applications.
- **No interpretability**: You can't draw a single tree anymore. But feature importance partially compensates.
- **Correlated features**: If many features are copies of each other, the forest doesn't benefit from feature subsampling as much.

---

## Math

### Bagging Variance Reduction

*Reasoning not required for USAAIO, but understand the result.*

Let $f_1, \ldots, f_B$ be the outputs of $B$ trees, each with variance $\sigma^2$, pairwise correlation $\rho$.

The forest prediction is $\bar{f} = \frac{1}{B}\sum_{b=1}^{B} f_b$.

$$\text{Var}(\bar{f}) = \frac{1}{B^2}\left(\sum_b \text{Var}(f_b) + \sum_{b \neq b'}\text{Cov}(f_b, f_{b'})\right)$$

$$= \frac{1}{B^2}\left(B\sigma^2 + B(B-1)\rho\sigma^2\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

### Feature Importance

**Mean Decrease in Impurity (MDI)**: For each feature $j$, sum the impurity decreases across all trees and all splits using feature $j$:

$$\text{Imp}(j) = \frac{1}{B}\sum_{b=1}^{B}\sum_{t \in T_b} \mathbb{1}[v(t) = j] \cdot \Delta G(t)$$

where $v(t)$ is the split feature at node $t$ and $\Delta G(t)$ is the impurity decrease.

**Permutation Importance**: Randomly shuffle feature $j$'s values in OOB data. Measure accuracy drop. More reliable than MDI for correlated features.

### OOB Error Estimate

For each sample $x_i$, let $\mathcal{B}_i = \{b : x_i \notin \text{bootstrap}_b\}$ be the set of trees for which $x_i$ is out-of-bag.

$$\hat{y}_i^{\text{OOB}} = \text{majority vote of } \{f_b(x_i) : b \in \mathcal{B}_i\}$$

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}[\hat{y}_i^{\text{OOB}} \neq y_i]$$

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np
from collections import Counter

# Assumes DecisionTree from 01-decision-trees.md
# (build_tree, predict functions)

def bootstrap_sample(X, y, rng):
    """Create a bootstrap sample."""
    # X: (N, D), y: (N,) -> bootstrap X, y, oob_indices
    N = X.shape[0]
    indices = rng.choice(N, size=N, replace=True)  # (N,) with repeats
    oob_mask = np.ones(N, dtype=bool)
    oob_mask[indices] = False
    oob_indices = np.where(oob_mask)[0]
    return X[indices], y[indices], oob_indices

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, max_features='sqrt', seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.oob_indices = []  # for OOB error

    def _get_max_features(self, D):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(D)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(D)))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return D

    def fit(self, X, y):
        """Fit the random forest."""
        # X: (N, D), y: (N,)
        N, D = X.shape
        rng = np.random.default_rng(self.seed)
        m = self._get_max_features(D)

        self.trees = []
        self.oob_predictions = {}  # {sample_idx: [predictions]}

        for b in range(self.n_trees):
            # Bootstrap sample
            X_boot, y_boot, oob_idx = bootstrap_sample(X, y, rng)

            # Random feature subset for this tree
            # (In practice, subsetting happens at each split, but this
            #  simplified version selects features per tree)
            feature_indices = rng.choice(D, size=m, replace=False)
            X_sub = X_boot[:, feature_indices]

            # Build tree on subset
            tree = build_tree(X_sub, y_boot, max_depth=self.max_depth)
            self.trees.append((tree, feature_indices))

            # Track OOB predictions
            for idx in oob_idx:
                x_sub = X[idx, feature_indices]
                pred = predict_one(tree, x_sub)
                if idx not in self.oob_predictions:
                    self.oob_predictions[idx] = []
                self.oob_predictions[idx].append(pred)

    def predict(self, X):
        """Predict by majority vote."""
        # X: (N, D) -> (N,)
        N = X.shape[0]
        all_preds = np.zeros((self.n_trees, N), dtype=object)

        for b, (tree, feat_idx) in enumerate(self.trees):
            X_sub = X[:, feat_idx]  # (N, m)
            all_preds[b] = predict(tree, X_sub)

        # Majority vote per sample
        result = np.empty(N, dtype=all_preds.dtype)
        for i in range(N):
            counter = Counter(all_preds[:, i])
            result[i] = counter.most_common(1)[0][0]
        return result

    def oob_error(self, y):
        """Compute out-of-bag error."""
        errors = 0
        count = 0
        for idx, preds in self.oob_predictions.items():
            oob_pred = Counter(preds).most_common(1)[0][0]
            if oob_pred != y[idx]:
                errors += 1
            count += 1
        return errors / count if count > 0 else float('nan')
```

### scikit-learn Equivalent

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

rf = RandomForestClassifier(
    n_estimators=100,        # number of trees
    max_depth=10,
    max_features='sqrt',     # feature subsampling
    oob_score=True,          # compute OOB error
    random_state=42
)
rf.fit(X, y)

print(f"OOB accuracy: {rf.oob_score_:.3f}")
print(f"Feature importances: {rf.feature_importances_}")

# Feature importance with names
import pandas as pd
importance = pd.Series(rf.feature_importances_, index=load_iris().feature_names)
print(importance.sort_values(ascending=False))
```

---

## Resources

- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.
- ISLR Chapter 8.2 — Bagging, Random Forests
- [scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

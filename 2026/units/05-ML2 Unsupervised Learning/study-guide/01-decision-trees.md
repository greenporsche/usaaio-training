# Decision Trees

**Prerequisites**: Basic probability (Unit 02), Python/NumPy (Unit 03), supervised ML concepts (Unit 04)
**USAAIO Relevance**: Foundation for random forests and boosting; tested directly in Round 1 (compute impurity, trace splits) and Round 2 (implement from scratch)

---

## Discovery

It's 1986, and you're Ross Quinlan at the University of Sydney. You've been working on an old AI problem: how to automatically learn classification rules from data. Expert systems are everywhere, but they require humans to hand-code rules — tedious and brittle. You want a machine to learn the rules itself.

You have a dataset of weather conditions and whether people played tennis:

```
Outlook    | Temp | Humidity | Wind   | Play?
-----------+------+----------+--------+------
Sunny      | Hot  | High     | Weak   | No
Sunny      | Hot  | High     | Strong | No
Overcast   | Hot  | High     | Weak   | Yes
Rain       | Mild | High     | Weak   | Yes
Rain       | Cool | Normal   | Weak   | Yes
Rain       | Cool | Normal   | Strong | No
Overcast   | Cool | Normal   | Strong | Yes
Sunny      | Mild | High     | Weak   | No
Sunny      | Cool | Normal   | Weak   | Yes
Rain       | Mild | Normal   | Weak   | Yes
Sunny      | Mild | Normal   | Strong | Yes
Overcast   | Mild | High     | Strong | Yes
Overcast   | Hot  | Normal   | Weak   | Yes
Rain       | Mild | High     | Strong | No
```

**The question**: Which feature should you ask about first to best separate "Yes" from "No"?

Think about it:
- If you split on Outlook, the "Overcast" branch is pure (all Yes). That seems good.
- If you split on Wind, you get a messier split.
- How do you *quantify* which split is better?

**Misconception trap**: Many students think "just pick the feature with the most categories" or "pick the feature with the highest accuracy." Neither is right. We need a principled measure of *impurity* — how mixed the classes are.

---

## Intuition

What you just discovered is exactly what Quinlan formalized in his ID3 algorithm (1986) and later C4.5 (1993). The key insight is:

**A good split reduces uncertainty about the class label.**

Think of it like a game of 20 Questions. Each question (split) should maximally reduce your uncertainty. The best first question eliminates the most possibilities.

### Impurity Measures — Visual Intuition

Imagine a bag of colored balls. Impurity measures how "mixed" the bag is.

```
Pure (impurity = 0)        Mixed (high impurity)      Perfectly mixed (max impurity)
[●●●●●]                   [●●●○○]                    [●●○○]
All same class             Some mixing                 50/50 split
```

Three common measures for a node with class proportions $p_1, p_2, \ldots, p_K$:

**Gini Impurity**: Probability that two randomly chosen samples have different classes.
```
G = 1 - sum(p_k^2)

For binary:  G = 2p(1-p)
             max at p = 0.5 → G = 0.5
             min at p = 0 or 1 → G = 0
```

**Entropy**: Information-theoretic uncertainty.
```
H = -sum(p_k * log2(p_k))

For binary:  H = -p*log2(p) - (1-p)*log2(1-p)
             max at p = 0.5 → H = 1.0
             min at p = 0 or 1 → H = 0
```

**Misclassification Error**: Fraction that would be wrong if we predict the majority class.
```
E = 1 - max(p_k)

For binary:  E = min(p, 1-p)
             max at p = 0.5 → E = 0.5
             min at p = 0 or 1 → E = 0
```

```
Impurity
  1.0 |        .--Entropy--.
      |      ./    ___      \.
      |    ./   .-' Gini`-.   \.
  0.5 |  ./  .'  .--ME--. `.  \.
      |  / .'  .'        `.  `. \
      | /.'  .'            `.  `.\
  0.0 |/__._'________________`_.__\
      0.0        0.5         1.0
                p (class 1 proportion)
```

**Why Gini and entropy but not misclassification error for splitting?** Misclassification error is piecewise linear — it doesn't distinguish between "barely mixed" and "very mixed" at the same majority proportion. Gini and entropy are strictly concave, making them more sensitive to impurity changes.

### The Splitting Process

```
        [9Y, 5N]  ← Root (impurity = ?)
        /        \
  Outlook=Sunny    Outlook=Overcast    Outlook=Rain
   [2Y, 3N]         [4Y, 0N]           [3Y, 2N]
                     PURE!
```

**Information Gain** = parent impurity - weighted average of children's impurities.

### Failure Cases

- **Overfitting**: A deep tree memorizes training data. A tree with one leaf per sample has zero training error but terrible generalization.
- **Bias toward many-valued features**: ID number has max information gain (each split is pure!) but is useless. Use gain ratio or restrict to binary splits.
- **Axis-aligned boundaries only**: Decision trees split on one feature at a time, creating rectangular decision regions. They struggle with diagonal boundaries.

---

## Math

### Gini Impurity

*Reasoning required for USAAIO.*

For a node with $K$ classes and class proportions $p_1, \ldots, p_K$ (where $p_k = n_k / n$):

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

**Interpretation**: If you randomly pick two samples (with replacement), $G$ is the probability they have different classes. Equivalently, $\sum p_k^2$ is the probability they match, so $G = 1 - P(\text{match})$.

For binary classification ($K=2$, proportions $p$ and $1-p$):

$$G = 1 - p^2 - (1-p)^2 = 2p(1-p)$$

### Entropy

*Reasoning required for USAAIO.*

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Convention: $0 \log 0 = 0$ (by continuity).

**Interpretation**: Expected number of bits needed to encode the class label. Maximum entropy = maximum surprise = most mixed.

### Information Gain

For a dataset $S$ split into subsets $S_1, \ldots, S_V$ by feature $A$:

$$IG(S, A) = H(S) - \sum_{v=1}^{V} \frac{|S_v|}{|S|} H(S_v)$$

Same formula works with Gini instead of entropy.

### Gain Ratio (C4.5)

To correct for bias toward many-valued features:

$$\text{SplitInfo}(S, A) = -\sum_{v=1}^{V} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

$$\text{GainRatio}(S, A) = \frac{IG(S, A)}{\text{SplitInfo}(S, A)}$$

### Cost-Complexity Pruning

*Reasoning not required for USAAIO, but understand the concept.*

For a subtree $T$ rooted at node $t$:

$$R_\alpha(T) = R(T) + \alpha |T|$$

where $R(T)$ is the misclassification rate and $|T|$ is the number of leaves. Increase $\alpha$ to favor smaller trees (more pruning).

### Continuous Features

For a continuous feature, sort values and consider all midpoints between consecutive distinct values as candidate thresholds. For $n$ unique values, there are at most $n-1$ candidate splits.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def gini(y):
    """Gini impurity of label array."""
    # y: (N,) array of class labels
    classes, counts = np.unique(y, return_counts=True)
    p = counts / len(y)  # (K,)
    return 1 - np.sum(p ** 2)

def entropy(y):
    """Shannon entropy (base 2) of label array."""
    classes, counts = np.unique(y, return_counts=True)
    p = counts / len(y)  # (K,)
    p = p[p > 0]  # avoid log(0)
    return -np.sum(p * np.log2(p))

def information_gain(X, y, feature_idx, threshold, criterion=gini):
    """Compute information gain for a binary split."""
    # X: (N, D), y: (N,)
    parent_impurity = criterion(y)

    left_mask = X[:, feature_idx] <= threshold  # (N,) boolean
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0.0

    n = len(y)
    n_left, n_right = left_mask.sum(), right_mask.sum()

    ig = parent_impurity - (
        (n_left / n) * criterion(y[left_mask]) +
        (n_right / n) * criterion(y[right_mask])
    )
    return ig

def best_split(X, y, criterion=gini):
    """Find the best feature and threshold to split on."""
    # X: (N, D), y: (N,)
    N, D = X.shape
    best_gain = -1
    best_feature, best_threshold = None, None

    for feature_idx in range(D):
        thresholds = np.unique(X[:, feature_idx])
        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            gain = information_gain(X, y, feature_idx, threshold, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # split feature index
        self.threshold = threshold    # split threshold
        self.left = left              # left child (<=)
        self.right = right            # right child (>)
        self.value = value            # leaf prediction (class label)

def build_tree(X, y, max_depth=10, min_samples=2, depth=0, criterion=gini):
    """Recursively build a decision tree."""
    # X: (N, D), y: (N,)

    # Base cases: pure node, max depth, or too few samples
    if len(np.unique(y)) == 1:
        return DecisionTreeNode(value=y[0])
    if depth >= max_depth or len(y) < min_samples:
        # Return majority class
        classes, counts = np.unique(y, return_counts=True)
        return DecisionTreeNode(value=classes[np.argmax(counts)])

    feature, threshold, gain = best_split(X, y, criterion)

    if gain <= 0:
        classes, counts = np.unique(y, return_counts=True)
        return DecisionTreeNode(value=classes[np.argmax(counts)])

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left_child = build_tree(X[left_mask], y[left_mask], max_depth, min_samples, depth + 1, criterion)
    right_child = build_tree(X[right_mask], y[right_mask], max_depth, min_samples, depth + 1, criterion)

    return DecisionTreeNode(feature=feature, threshold=threshold, left=left_child, right=right_child)

def predict_one(node, x):
    """Predict class for a single sample."""
    # x: (D,)
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

def predict(node, X):
    """Predict classes for multiple samples."""
    # X: (N, D) -> (N,)
    return np.array([predict_one(node, x) for x in X])
```

### scikit-learn Equivalent

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Gini impurity (default)
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
clf.fit(X, y)
predictions = clf.predict(X)

# Entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf_entropy.fit(X, y)

# Visualize
from sklearn.tree import export_text
print(export_text(clf, feature_names=load_iris().feature_names))
```

---

## Resources

- Quinlan, J.R. (1986). "Induction of Decision Trees." *Machine Learning*, 1(1), 81–106.
- Breiman, L. et al. (1984). *Classification and Regression Trees* (CART).
- ISLR Chapter 8.1 — Decision Trees
- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

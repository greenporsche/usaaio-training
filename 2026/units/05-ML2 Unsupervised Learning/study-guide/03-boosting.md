# Boosting

**Prerequisites**: Decision Trees (01), Random Forests (02), gradient descent (Unit 04)
**USAAIO Relevance**: AdaBoost weight updates and gradient boosting residual fitting tested in Round 1 (trace algorithm steps) and Round 2 (implement from scratch)

---

## Discovery

It's 1990, and you're Robert Schapire, a PhD student at MIT. Your advisor, Michael Kearns, posed a tantalizing question in computational learning theory:

**"Can a set of weak learners be combined to form a strong learner?"**

A **weak learner** is any classifier that does slightly better than random guessing — maybe 51% accuracy on a binary task. It's barely useful on its own. But Kearns suspects that if you could somehow *focus* each weak learner on the mistakes of the previous ones, the combination might be very powerful.

Think of it like tutoring: the first tutor teaches the easy stuff. The second tutor focuses on what the student still gets wrong. The third tutor targets the remaining gaps. Eventually, the student masters everything — even though no single tutor could teach it all.

**Socratic questions**:
- If each weak learner is only 51% accurate, how many do you need to get 99% accuracy?
- How should you *weight* the weak learners? Should a 90% accurate learner count more than a 55% accurate one?
- How do you *focus* subsequent learners on the mistakes? (Hint: change the data distribution.)

**Misconception trap**: Boosting is NOT the same as bagging. Bagging trains trees *independently* on different bootstrap samples. Boosting trains trees *sequentially*, where each depends on the previous ones' errors.

---

## Intuition

What you discovered is the core idea behind AdaBoost (Freund & Schapire, 1995) and later Gradient Boosting (Friedman, 2001). Schapire proved that yes, weak learners can be boosted into strong learners — answering Kearns' question affirmatively.

### AdaBoost — Reweighting Samples

The key idea: **give more weight to misclassified samples** so the next learner focuses on them.

```
Round 1:  All samples equally weighted
          Train weak learner h₁
          h₁ gets some wrong (marked with ✗)

          ○ ○ ○ ● ● ✗ ✗ ○ ● ○
          Equal weights: ▪▪▪▪▪▪▪▪▪▪

Round 2:  Increase weight on ✗ samples, decrease on correct ones
          Train weak learner h₂ (focuses on hard cases)

          ○ ○ ○ ● ● ✗ ✗ ○ ● ○
          Weights:        ▪▪▪▪▪███▪▪▪

Round 3:  Reweight again based on h₂'s errors
          ...

Final:    H(x) = sign(α₁h₁(x) + α₂h₂(x) + ... + αₜhₜ(x))
          where αₜ depends on hₜ's accuracy
```

Each learner $h_t$ gets a vote weight $\alpha_t$ proportional to its accuracy. A learner with 5% error gets a much louder vote than one with 45% error.

### Gradient Boosting — Fitting Residuals

An alternative (and more general) perspective: **each new learner fits the residual errors** of the current ensemble.

```
True function:    y = f(x)

Step 0: F₀(x) = average(y)
        Residual₀ = y - F₀(x)      ← what we still get wrong

Step 1: h₁ fits Residual₀
        F₁(x) = F₀(x) + η·h₁(x)
        Residual₁ = y - F₁(x)      ← smaller errors now

Step 2: h₂ fits Residual₁
        F₂(x) = F₁(x) + η·h₂(x)
        Residual₂ = y - F₂(x)      ← even smaller

...gradually "fills in" the remaining error
```

The learning rate $\eta \in (0, 1]$ controls how much of each learner we add. Smaller $\eta$ means more trees needed but often better generalization — a classic **regularization** technique.

### AdaBoost vs Gradient Boosting

| | AdaBoost | Gradient Boosting |
|---|---------|-------------------|
| How it adapts | Reweights samples | Fits residuals (negative gradient) |
| Loss function | Exponential loss | Any differentiable loss |
| Weak learner | Any (typically stumps) | Typically shallow trees |
| Connection | Special case of gradient boosting with exponential loss |

### XGBoost Key Ideas

XGBoost (Chen & Guestrin, 2016) is gradient boosting with several engineering improvements:
- **Regularized objective**: Adds $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ where $T$ = leaves, $w$ = leaf weights
- **Second-order approximation**: Uses both gradient and Hessian for splits
- **Sparsity-aware**: Handles missing values natively
- **Column subsampling**: Like random forests, subsample features
- **Parallel split-finding**: Efficient computation

### Failure Cases

- **Sensitive to noise**: Boosting keeps focusing on hard examples — if those examples are *noise* (mislabeled), it overfits to the noise.
- **Sequential training**: Cannot parallelize across boosting rounds (unlike random forests).
- **Requires careful tuning**: Learning rate, number of rounds, and tree depth all interact.

---

## Math

### AdaBoost Algorithm

*Reasoning required for USAAIO — you must trace weight updates.*

Given training data $(x_1, y_1), \ldots, (x_n, y_n)$ with $y_i \in \{-1, +1\}$:

**Initialize**: $w_i^{(1)} = \frac{1}{n}$ for all $i$.

**For** $t = 1, \ldots, T$:

1. Fit weak learner $h_t$ to data weighted by $w^{(t)}$.
2. Compute weighted error:
$$\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \mathbb{1}[h_t(x_i) \neq y_i] = \frac{\sum_{i: h_t(x_i) \neq y_i} w_i^{(t)}}{\sum_{i} w_i^{(t)}}$$

3. Compute learner weight:
$$\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$$

   Note: $\alpha_t > 0$ when $\epsilon_t < 0.5$ (better than random). $\alpha_t = 0$ when $\epsilon_t = 0.5$.

4. Update sample weights:
$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

   Then normalize: $w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_j w_j^{(t+1)}}$

   **Key insight**: If $h_t(x_i) = y_i$ (correct), the exponent is $-\alpha_t < 0$, so weight *decreases*. If wrong, exponent is $+\alpha_t > 0$, so weight *increases*.

**Final classifier**:
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

### AdaBoost as Gradient Descent on Exponential Loss

*Reasoning not required for USAAIO, but deepens understanding.*

AdaBoost minimizes the **exponential loss**:

$$L = \sum_{i=1}^{n} \exp(-y_i F(x_i)) \quad \text{where } F(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$$

The weight updates are exactly the negative gradient of this loss. This connection was discovered by Friedman, Hastie, and Tibshirani (2000).

### Gradient Boosting Algorithm

*Reasoning required for USAAIO — compute residuals and updates.*

For a differentiable loss $L(y, F(x))$:

1. **Initialize**: $F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$

2. **For** $m = 1, \ldots, M$:
   a. Compute pseudo-residuals:
   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

   b. Fit base learner $h_m$ to targets $(x_i, r_{im})$.

   c. Compute step size: $\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$

   d. Update: $F_m(x) = F_{m-1}(x) + \eta \cdot \gamma_m \cdot h_m(x)$

**For squared loss** $L = \frac{1}{2}(y - F)^2$: pseudo-residuals = $y_i - F_{m-1}(x_i)$ (actual residuals!).

**For logistic loss** $L = \log(1 + e^{-yF})$: pseudo-residuals = $\frac{y_i}{1 + e^{y_i F_{m-1}(x_i)}}$.

---

## Code

### AdaBoost From Scratch

```python
import numpy as np

class DecisionStump:
    """Weak learner: single-split decision tree."""
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # 1 or -1

    def fit(self, X, y, weights):
        # X: (N, D), y: (N,) in {-1, +1}, weights: (N,)
        N, D = X.shape
        best_err = float('inf')

        for j in range(D):
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(N)
                    if polarity == 1:
                        pred[X[:, j] < threshold] = -1
                    else:
                        pred[X[:, j] >= threshold] = -1

                    err = np.sum(weights[pred != y])

                    if err < best_err:
                        best_err = err
                        self.feature = j
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        # X: (N, D) -> (N,) in {-1, +1}
        N = X.shape[0]
        pred = np.ones(N)
        if self.polarity == 1:
            pred[X[:, self.feature] < self.threshold] = -1
        else:
            pred[X[:, self.feature] >= self.threshold] = -1
        return pred

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        """Fit AdaBoost with decision stumps."""
        # X: (N, D), y: (N,) in {-1, +1}
        N = X.shape[0]
        w = np.full(N, 1.0 / N)  # (N,) uniform weights

        for t in range(self.n_estimators):
            # Fit weak learner
            stump = DecisionStump()
            stump.fit(X, y, w)
            pred = stump.predict(X)  # (N,)

            # Weighted error
            err = np.sum(w[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # Learner weight
            alpha = 0.5 * np.log((1 - err) / err)

            # Update sample weights
            w *= np.exp(-alpha * y * pred)  # (N,)
            w /= np.sum(w)  # normalize

            self.alphas.append(alpha)
            self.stumps.append(stump)

    def predict(self, X):
        """Predict using weighted majority vote."""
        # X: (N, D) -> (N,) in {-1, +1}
        N = X.shape[0]
        F = np.zeros(N)  # (N,)
        for alpha, stump in zip(self.alphas, self.stumps):
            F += alpha * stump.predict(X)  # (N,)
        return np.sign(F)
```

### Gradient Boosting From Scratch (Regression)

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.f0 = None

    def fit(self, X, y):
        """Fit gradient boosting for squared loss."""
        # X: (N, D), y: (N,)
        self.f0 = np.mean(y)
        F = np.full(len(y), self.f0)  # (N,) current predictions

        for m in range(self.n_estimators):
            # Pseudo-residuals (negative gradient of squared loss)
            residuals = y - F  # (N,)

            # Fit tree to residuals
            tree = build_tree(X, residuals, max_depth=self.max_depth)
            pred = predict(tree, X)  # (N,)

            # Update predictions
            F += self.lr * pred  # (N,)
            self.trees.append(tree)

    def predict(self, X):
        """Predict by summing base learner outputs."""
        # X: (N, D) -> (N,)
        F = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            F += self.lr * predict(tree, X)
        return F
```

### scikit-learn / XGBoost Equivalents

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada.fit(X, y)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X, y)

# XGBoost
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_clf.fit(X, y)
```

---

## Resources

- Freund, Y. & Schapire, R.E. (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting." *JCSS*, 55(1), 119–139.
- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232.
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
- ISLR Chapter 8.2.3 — Boosting

# Classification Metrics

**Prerequisites**: Logistic regression (Topic 05), basic probability
**USAAIO Relevance**: Metric computation from confusion matrices is a staple exam question; understanding when accuracy fails; ROC-AUC interpretation; implementing metrics from scratch in coding rounds

---

## Discovery

### Why Accuracy Isn't Enough

You build a model to detect credit card fraud. Your dataset has 99.9% legitimate transactions and 0.1% fraud. A model that predicts "not fraud" for everything achieves 99.9% accuracy. Is it a good model?

Obviously not — it catches zero fraud. This is why we need metrics beyond accuracy: **precision**, **recall**, **F1-score**, and **ROC-AUC** each capture different aspects of classifier performance.

### Socratic Warm-Up

1. A medical test has 99% sensitivity (recall) and 5% false positive rate. If 1 in 1000 people has the disease, what's the probability someone who tests positive actually has it? (Hint: this is surprisingly low.)
2. Can you have precision = 1 and recall = 0.01 simultaneously? What kind of model would do that?
3. If you increase the classification threshold from 0.5 to 0.9, what happens to precision vs recall?

### Misconception Traps

- **"F1 is always the right metric."** — F1 weighs precision and recall equally. If false negatives are much worse than false positives (e.g., cancer screening), use recall or F$_\beta$ with $\beta > 1$.
- **"ROC-AUC is always informative."** — Under severe class imbalance, the precision-recall curve is more informative than ROC.
- **"Accuracy is useless."** — For balanced datasets, accuracy is perfectly fine and interpretable.

---

## Intuition

### The Confusion Matrix

```
                     Predicted
                  Pos        Neg
              ┌──────────┬──────────┐
Actual  Pos   │    TP     │    FN    │  ← Total Actual Positives = TP + FN
              ├──────────┼──────────┤
        Neg   │    FP     │    TN    │  ← Total Actual Negatives = FP + TN
              └──────────┴──────────┘
                   ↑           ↑
            Predicted    Predicted
             Positives    Negatives

TP = True Positive  (correct positive prediction)
FP = False Positive (Type I error — false alarm)
FN = False Negative (Type II error — missed detection)
TN = True Negative  (correct negative prediction)
```

### The Precision-Recall Tradeoff

```
High threshold (e.g., 0.9):        Low threshold (e.g., 0.1):
  • Predict positive only            • Predict positive for
    when very confident                almost everything
  • High precision, low recall       • High recall, low precision

Threshold:  0.9   0.7   0.5   0.3   0.1
Precision:  0.95  0.85  0.75  0.60  0.40
Recall:     0.30  0.55  0.70  0.85  0.95
            ←── conservative    aggressive ──→
```

### ROC Curve

```
  TPR (Recall)
  1.0 ┤              ●●●●●●●●●●
      │           ●●●
      │         ●●
  0.8 ┤        ●       ← Good model
      │       ●
      │      ●
  0.5 ┤    ●╱ ← Random (AUC = 0.5)
      │   ●╱
      │  ●╱
  0.2 ┤ ●╱
      │●╱
  0.0 ●───────────────────────
      0    0.2   0.5   0.8  1.0
               FPR →

  AUC = Area under the curve
  Perfect classifier: AUC = 1.0
  Random classifier:  AUC = 0.5
  Worthless:          AUC < 0.5
```

The ROC curve plots True Positive Rate (TPR = recall) vs False Positive Rate (FPR = FP/(FP+TN)) at all classification thresholds.

---

## Math

### Metric Definitions

*[Reasoning required for USAAIO]*

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} = P(\text{actually positive} \mid \text{predicted positive})$$

$$\text{Recall (Sensitivity, TPR)} = \frac{TP}{TP + FN} = P(\text{predicted positive} \mid \text{actually positive})$$

$$\text{Specificity} = \frac{TN}{TN + FP} = P(\text{predicted negative} \mid \text{actually negative})$$

$$\text{FPR} = 1 - \text{Specificity} = \frac{FP}{FP + TN}$$

### F1-Score and F-beta

The **F1-score** is the harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Why harmonic mean? It penalizes extreme imbalances. If $P = 1.0$ and $R = 0.01$:
- Arithmetic mean: $(1.0 + 0.01)/2 = 0.505$ (misleadingly high)
- Harmonic mean: $2(1.0)(0.01)/(1.0 + 0.01) = 0.0198$ (properly low)

The **F-beta score** generalizes with a parameter $\beta$:

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

- $\beta = 1$: Equal weight to precision and recall (standard F1)
- $\beta = 2$: Weights recall 2x more than precision (e.g., medical screening)
- $\beta = 0.5$: Weights precision 2x more than recall (e.g., spam filter)

### ROC-AUC

The **ROC curve** parametrically plots $(FPR(t), TPR(t))$ as threshold $t$ varies from $+\infty$ to $-\infty$.

**AUC interpretation**: The probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example:

$$\text{AUC} = P(f(x^+) > f(x^-))$$

where $x^+$ is a random positive and $x^-$ is a random negative.

### Multi-Class Extension

For $C$ classes, compute per-class metrics and aggregate:

**Macro-average**: Compute metric for each class, then average.

$$P_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} P_c$$

**Micro-average**: Aggregate TP, FP, FN across all classes, then compute metric.

$$P_{\text{micro}} = \frac{\sum_c TP_c}{\sum_c TP_c + \sum_c FP_c}$$

**Weighted-average**: Weight each class by its support (number of true instances).

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    y_true: (n,) integer labels
    y_pred: (n,) integer predictions
    Returns: (C, C) matrix where [i, j] = count of true=i, pred=j
    """
    C = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((C, C), dtype=int)

    # Vectorized: use np.add.at for sparse accumulation
    np.add.at(cm, (y_true, y_pred), 1)
    return cm  # (C, C)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Binary classification metrics.
    y_true, y_pred: (n,) binary labels
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute ROC curve.
    y_true: (n,) binary labels
    y_scores: (n,) predicted scores/probabilities
    Returns: (fpr, tpr, thresholds) — each is (m,)
    """
    # Sort by descending score
    sorted_idx = np.argsort(-y_scores)
    y_sorted = y_true[sorted_idx]
    scores_sorted = y_scores[sorted_idx]

    # Total positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Cumulative TP and FP
    tps = np.cumsum(y_sorted == 1)     # (n,)
    fps = np.cumsum(y_sorted == 0)     # (n,)

    tpr = tps / P                       # (n,)
    fpr = fps / N                       # (n,)

    # Prepend (0, 0)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    thresholds = np.concatenate([[scores_sorted[0] + 1], scores_sorted])

    return fpr, tpr, thresholds


def auc_trapezoidal(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC using trapezoidal rule."""
    return float(np.trapz(tpr, fpr))


def precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute precision-recall curve.
    Returns: (precision, recall, thresholds)
    """
    sorted_idx = np.argsort(-y_scores)
    y_sorted = y_true[sorted_idx]
    scores_sorted = y_scores[sorted_idx]

    P = np.sum(y_true == 1)

    tps = np.cumsum(y_sorted == 1)     # (n,)
    fps = np.cumsum(y_sorted == 0)     # (n,)

    precision = tps / (tps + fps)       # (n,)
    recall = tps / P                    # (n,)

    return precision, recall, scores_sorted


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> dict:
    """
    Multiclass precision, recall, F1.
    average: "macro" or "micro"
    """
    C = max(y_true.max(), y_pred.max()) + 1

    if average == "micro":
        tp_total = sum(np.sum((y_true == c) & (y_pred == c)) for c in range(C))
        fp_total = sum(np.sum((y_true != c) & (y_pred == c)) for c in range(C))
        fn_total = sum(np.sum((y_true == c) & (y_pred != c)) for c in range(C))

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "macro":
        precisions, recalls, f1s = [], [], []
        for c in range(C):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))

    return {"precision": precision, "recall": recall, "f1": f1}


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_scores = y_true * 0.6 + np.random.randn(n) * 0.3  # noisy scores
    y_pred = (y_scores > 0.5).astype(int)

    metrics = precision_recall_f1(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    print(f"AUC:       {auc_trapezoidal(fpr, tpr):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
```

### PyTorch Equivalent

```python
import torch

def accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float((y_true == y_pred).float().mean())

# For metrics, torchmetrics is the standard library:
# from torchmetrics import Precision, Recall, F1Score, AUROC
# metric = F1Score(task="binary")
# metric(y_pred_proba, y_true)
```

---

## Resources

- Hastie, Tibshirani, Friedman: *ESL*, Section 7.10 (Cross-Validation)
- Davis & Goadrich (2006): "The Relationship Between Precision-Recall and ROC Curves"
- Fawcett (2006): "An Introduction to ROC Analysis"

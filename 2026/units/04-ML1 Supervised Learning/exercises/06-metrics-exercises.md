# Classification Metrics — Exercises

> 5 exercises covering confusion matrices, precision/recall/F1, ROC-AUC, and metric tradeoffs

---

## Exercise 1: Compute Metrics from Confusion Matrix (Compute This)

A binary classifier produces the following confusion matrix:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 80 | 20 |
| **Actual Negative** | 30 | 870 |

**Tasks**:
1. Compute: accuracy, precision, recall, F1-score, specificity, FPR.
2. Is this dataset balanced or imbalanced? How can you tell?
3. A colleague says "99% accuracy is achievable!" What trivial model achieves this? Why is it useless?
4. Another colleague proposes using F2-score instead of F1. Under what circumstances would F2 be more appropriate?

<details>
<summary>Solution</summary>

From the matrix: TP = 80, FN = 20, FP = 30, TN = 870. Total = 1000.

**1.**

- **Accuracy** = (80 + 870) / 1000 = **0.950** (95.0%)
- **Precision** = 80 / (80 + 30) = 80/110 = **0.727** (72.7%)
- **Recall** = 80 / (80 + 20) = 80/100 = **0.800** (80.0%)
- **F1** = 2(0.727)(0.800) / (0.727 + 0.800) = 1.163 / 1.527 = **0.762**
- **Specificity** = 870 / (870 + 30) = 870/900 = **0.967** (96.7%)
- **FPR** = 1 - 0.967 = 30/900 = **0.033** (3.3%)

**2.** The dataset is **imbalanced**: 100 positives vs 900 negatives (1:9 ratio). You can tell because the actual positive total (TP + FN = 100) is much smaller than the actual negative total (FP + TN = 900).

**3.** A model that predicts "Negative" for everything achieves accuracy = 900/1000 = 90% (not 99%, my colleague was slightly off). But it has recall = 0/100 = 0 — it catches zero actual positives. A model predicting all positive has accuracy = 100/1000 = 10%. Neither trivial model is useful.

**4.** F2 weights recall twice as much as precision: $F_2 = 5 \cdot \frac{PR}{4P + R}$. Use F2 when missing a positive (false negative) is much worse than a false alarm (false positive). Examples:
- Medical screening (missing cancer is worse than an unnecessary follow-up)
- Security systems (missing an intrusion is worse than a false alarm)

</details>

---

## Exercise 2: ROC Curve Construction (Compute This)

A classifier produces the following scores for 10 samples:

| Sample | Score | True Label |
|---|---|---|
| 1 | 0.95 | 1 |
| 2 | 0.85 | 1 |
| 3 | 0.80 | 0 |
| 4 | 0.70 | 1 |
| 5 | 0.65 | 0 |
| 6 | 0.55 | 1 |
| 7 | 0.45 | 0 |
| 8 | 0.35 | 0 |
| 9 | 0.25 | 1 |
| 10 | 0.15 | 0 |

Total: 5 positives, 5 negatives.

**Tasks**:
1. Sort by score (descending) and compute cumulative TP, FP, TPR, FPR at each threshold.
2. Plot the ROC curve (as a table of (FPR, TPR) points).
3. Compute the AUC using the trapezoidal rule.
4. What is the score threshold that maximizes the Youden's J statistic ($J = TPR - FPR$)?

<details>
<summary>Solution</summary>

**1 & 2.** Sorted (already sorted):

| Threshold | Label | Cum TP | Cum FP | TPR | FPR |
|---|---|---|---|---|---|
| (start) | — | 0 | 0 | 0.0 | 0.0 |
| 0.95 | 1 | 1 | 0 | 0.2 | 0.0 |
| 0.85 | 1 | 2 | 0 | 0.4 | 0.0 |
| 0.80 | 0 | 2 | 1 | 0.4 | 0.2 |
| 0.70 | 1 | 3 | 1 | 0.6 | 0.2 |
| 0.65 | 0 | 3 | 2 | 0.6 | 0.4 |
| 0.55 | 1 | 4 | 2 | 0.8 | 0.4 |
| 0.45 | 0 | 4 | 3 | 0.8 | 0.6 |
| 0.35 | 0 | 4 | 4 | 0.8 | 0.8 |
| 0.25 | 1 | 5 | 4 | 1.0 | 0.8 |
| 0.15 | 0 | 5 | 5 | 1.0 | 1.0 |

ROC points: (0, 0) → (0, 0.2) → (0, 0.4) → (0.2, 0.4) → (0.2, 0.6) → (0.4, 0.6) → (0.4, 0.8) → (0.6, 0.8) → (0.8, 0.8) → (0.8, 1.0) → (1.0, 1.0)

**3.** AUC by trapezoidal rule (sum of rectangles):

$= 0.2 \times 0 + 0 \times 0.4 + 0.2 \times 0.4 + 0 \times 0.6 + 0.2 \times 0.6 + 0 \times 0.8 + 0.2 \times 0.8 + 0.2 \times 0.8 + 0 \times 1.0 + 0.2 \times 1.0$

More carefully using trapezoids between consecutive points:

| Segment | $\Delta$FPR | Avg TPR | Area |
|---|---|---|---|
| (0,0)→(0,0.2) | 0 | — | 0 |
| (0,0.2)→(0,0.4) | 0 | — | 0 |
| (0,0.4)→(0.2,0.4) | 0.2 | 0.4 | 0.08 |
| (0.2,0.4)→(0.2,0.6) | 0 | — | 0 |
| (0.2,0.6)→(0.4,0.6) | 0.2 | 0.6 | 0.12 |
| (0.4,0.6)→(0.4,0.8) | 0 | — | 0 |
| (0.4,0.8)→(0.6,0.8) | 0.2 | 0.8 | 0.16 |
| (0.6,0.8)→(0.8,0.8) | 0.2 | 0.8 | 0.16 |
| (0.8,0.8)→(0.8,1.0) | 0 | — | 0 |
| (0.8,1.0)→(1.0,1.0) | 0.2 | 1.0 | 0.20 |

AUC = 0.08 + 0.12 + 0.16 + 0.16 + 0.20 = **0.72**

**4.** Youden's J = TPR - FPR:

| Threshold | TPR | FPR | J |
|---|---|---|---|
| 0.95 | 0.2 | 0.0 | 0.2 |
| 0.85 | 0.4 | 0.0 | **0.4** |
| 0.70 | 0.6 | 0.2 | **0.4** |
| 0.55 | 0.8 | 0.4 | **0.4** |
| 0.25 | 1.0 | 0.8 | 0.2 |

Maximum J = 0.4, achieved at thresholds 0.85, 0.70, and 0.55.

</details>

---

## Exercise 3: Identify the Error (Debug)

```python
def precision(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (p + r) / 2                           # BUG?

def auc_score(y_true, y_scores):
    sorted_idx = np.argsort(y_scores)             # BUG?
    y_sorted = y_true[sorted_idx]
    # Count concordant pairs
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    cum_fp = np.cumsum(1 - y_sorted)
    auc = np.sum(y_sorted * cum_fp) / (n_pos * n_neg)
    return auc
```

Find **two bugs**.

<details>
<summary>Solution</summary>

**Bug 1 — `f1_score`**: The F1-score is the **harmonic mean**, not the **arithmetic mean** of precision and recall.

Current (wrong): `return (p + r) / 2`

Correct: `return 2 * p * r / (p + r)` (with a zero check: `if (p + r) == 0: return 0`)

The arithmetic mean $(p + r)/2$ can be misleadingly high when one metric is very low. For example, $p = 1.0, r = 0.01$: arithmetic mean = 0.505, harmonic mean = 0.0198.

**Bug 2 — `auc_score`**: `np.argsort(y_scores)` sorts in **ascending** order, but for AUC computation via concordant pairs, we need to process from highest to lowest score.

Fix: `sorted_idx = np.argsort(-y_scores)` (or equivalently `np.argsort(y_scores)[::-1]`)

With ascending order, the cumulative count of false positives is computed incorrectly — we'd be counting how many negatives have *lower* scores than each positive, when we should count how many negatives have lower scores.

</details>

---

## Exercise 4: When Accuracy Fails (Fill in the Analysis)

A medical screening test has the following results on 10,000 patients:

- Disease prevalence: 1% (100 patients have the disease, 9900 don't)
- Test sensitivity (recall): 99%
- Test specificity: 95%

**Tasks**:
1. Fill in the confusion matrix.
2. Compute the test's precision (positive predictive value).
3. Compute the test's accuracy.
4. Explain the paradox: the test has 99% sensitivity and 95% specificity, yet most positive results are false positives.

<details>
<summary>Solution</summary>

**1. Confusion matrix**:

- TP = 99% of 100 = 99
- FN = 1% of 100 = 1
- TN = 95% of 9900 = 9405
- FP = 5% of 9900 = 495

|  | Predicted + | Predicted - |
|---|---|---|
| **Actual +** | 99 | 1 |
| **Actual -** | 495 | 9405 |

**2.** Precision = TP / (TP + FP) = 99 / (99 + 495) = 99 / 594 = **0.167** (16.7%)

**3.** Accuracy = (99 + 9405) / 10000 = 9504 / 10000 = **95.04%**

**4. The Base Rate Paradox** (also known as the false positive paradox):

Even though the test has excellent sensitivity (99%) and good specificity (95%), the precision is only 16.7%. This means **83.3% of positive results are false positives**.

The reason: with only 1% prevalence, there are many more healthy people (9900) than sick people (100). Even a 5% false positive rate on 9900 people produces 495 false positives, which vastly outnumber the 99 true positives.

This is why:
- **Accuracy (95%) is misleading** — a model predicting "no disease" for everyone gets 99% accuracy
- **Recall alone is insufficient** — we also need high precision
- **Base rate matters** — metrics must be interpreted in the context of class prevalence
- This is essentially **Bayes' theorem** in action: $P(\text{disease} | \text{positive}) = \frac{P(\text{positive} | \text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$

</details>

---

## Exercise 5: True/False with Justification

1. **ROC-AUC of 0.5 means the model is worse than random.**
2. **Micro-average F1 equals accuracy for multiclass single-label classification.**
3. **A model with perfect precision can still have zero recall.**
4. **The precision-recall curve is more informative than ROC for imbalanced datasets.**
5. **Increasing the decision threshold always increases precision.**

<details>
<summary>Solution</summary>

1. **FALSE**. AUC = 0.5 means the model is *equivalent* to random (the ROC curve is the diagonal). A model *worse* than random has AUC < 0.5, which would mean flipping its predictions would give a better-than-random model. AUC = 0 means perfectly *anti*-correlated with the true labels.

2. **TRUE**. In single-label multiclass classification, each sample is classified into exactly one class. Micro-averaging sums TP, FP, FN across all classes. Since each sample is either correctly or incorrectly classified (no partial credit), micro-F1 = micro-precision = micro-recall = accuracy. This is because $\sum_c TP_c$ = total correct, and $\sum_c (TP_c + FP_c) = \sum_c (TP_c + FN_c) = n$ (total samples).

3. **TRUE but pathological**. A model that predicts positive for only one sample (and it's correct) has precision = 1.0. But if there are 1000 actual positives and it only found 1, recall = 1/1000 = 0.001. In the extreme, a model that predicts positive for *zero* samples has undefined precision (0/0) and recall = 0. So technically, precision = 1 with recall = 0 requires predicting positive for at least one (correct) sample while missing all others.

4. **TRUE**. Under severe class imbalance, the ROC curve can look deceptively good because specificity (TN/(TN+FP)) is high simply due to the large number of true negatives. The FPR stays low even with many false positives. The precision-recall curve focuses on the positive class and reveals the poor precision that ROC hides. For example, with 99.9% negatives, even 0.1% FPR yields many false positives relative to the few true positives.

5. **FALSE**. Increasing the threshold makes the model predict positive less often, which typically increases precision (fewer false positives). However, there are edge cases. If the next sample removed by raising the threshold was a true positive (but the samples just above the new threshold include false positives), precision can actually decrease. In general, the trend is toward higher precision, but it's not monotonic.

</details>

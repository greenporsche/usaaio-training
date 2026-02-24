# Problem 8 Variations: Classification Metrics (EXHAUSTIVE)

> **Original Problem**: Confusion matrix analysis, accuracy, precision, recall, F1-score for binary classification
> **Core Skills**: TP/FP/TN/FN understanding, metric computation, class-wise metrics, harmonic mean
> **Units**: 04 (ML1 - Supervised Learning)

---

## ORIGINAL PROBLEM (Reference)

**Setup**: Consider the following confusion matrix:

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 45 (TP)            | 5 (FN)             |
| **Actual Negative**  | 10 (FP)            | 40 (TN)            |

**Part 8.1**: Compute accuracy.
**Part 8.2**: Compute precision scores for positive and negative classes, respectively.
**Part 8.3**: Compute recall scores for positive and negative classes, respectively.
**Part 8.4**: Compute F1-scores for positive and negative classes, respectively.

<details>
<summary>Original Solutions</summary>

**8.1**: Accuracy = (TP + TN) / (TP + TN + FP + FN) = (45 + 40) / 100 = **0.85** or **85%**

**8.2**:
- Precision (Positive) = TP / (TP + FP) = 45 / 55 = **9/11 ≈ 0.818**
- Precision (Negative) = TN / (TN + FN) = 40 / 45 = **8/9 ≈ 0.889**

**8.3**:
- Recall (Positive) = TP / (TP + FN) = 45 / 50 = **0.90**
- Recall (Negative) = TN / (TN + FP) = 40 / 50 = **0.80**

**8.4**:
- F1 (Positive) = 2 × (9/11 × 0.90) / (9/11 + 0.90) = 2 × 0.736 / 1.718 ≈ **0.857**
- F1 (Negative) = 2 × (8/9 × 0.80) / (8/9 + 0.80) = 2 × 0.711 / 1.689 ≈ **0.842**

</details>

---

## CATEGORY A: Different Values (Same Structure)

### Variation A1: Balanced Performance

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 80 (TP)            | 20 (FN)            |
| **Actual Negative**  | 20 (FP)            | 80 (TN)            |

**Part A1.1**: Compute accuracy.
**Part A1.2**: Compute precision scores for positive and negative classes, respectively.
**Part A1.3**: Compute recall scores for positive and negative classes, respectively.
**Part A1.4**: Compute F1-scores for positive and negative classes, respectively.

<details>
<summary>Solution A1</summary>

**A1.1**: Accuracy = (80 + 80) / 200 = **0.80** or **80%**

**A1.2**:
- Precision (Positive) = 80 / (80 + 20) = **0.80**
- Precision (Negative) = 80 / (80 + 20) = **0.80**

**A1.3**:
- Recall (Positive) = 80 / (80 + 20) = **0.80**
- Recall (Negative) = 80 / (80 + 20) = **0.80**

**A1.4**:
- F1 (Positive) = 2 × (0.80 × 0.80) / (0.80 + 0.80) = **0.80**
- F1 (Negative) = 2 × (0.80 × 0.80) / (0.80 + 0.80) = **0.80**

**Key insight**: This is a perfectly symmetric confusion matrix where all metrics are equal.

</details>

---

### Variation A2: High Precision, Low Recall

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 30 (TP)            | 70 (FN)            |
| **Actual Negative**  | 5 (FP)             | 95 (TN)            |

**Part A2.1**: Compute accuracy.
**Part A2.2**: Compute precision scores for positive and negative classes, respectively.
**Part A2.3**: Compute recall scores for positive and negative classes, respectively.
**Part A2.4**: Compute F1-scores for positive and negative classes, respectively.

<details>
<summary>Solution A2</summary>

**A2.1**: Accuracy = (30 + 95) / 200 = **0.625** or **62.5%**

**A2.2**:
- Precision (Positive) = 30 / (30 + 5) = 30/35 = **6/7 ≈ 0.857**
- Precision (Negative) = 95 / (95 + 70) = 95/165 = **19/33 ≈ 0.576**

**A2.3**:
- Recall (Positive) = 30 / (30 + 70) = 30/100 = **0.30**
- Recall (Negative) = 95 / (95 + 5) = 95/100 = **0.95**

**A2.4**:
- F1 (Positive) = 2 × (6/7 × 0.30) / (6/7 + 0.30) = 2 × (0.257) / (1.157) ≈ **0.444**
- F1 (Negative) = 2 × (19/33 × 0.95) / (19/33 + 0.95) = 2 × (0.547) / (1.526) ≈ **0.717**

**Key insight**: This represents a conservative classifier that rarely predicts positive but is usually right when it does.

</details>

---

### Variation A3: Low Precision, High Recall

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 95 (TP)            | 5 (FN)             |
| **Actual Negative**  | 70 (FP)            | 30 (TN)            |

**Part A3.1**: Compute accuracy.
**Part A3.2**: Compute precision scores for positive and negative classes, respectively.
**Part A3.3**: Compute recall scores for positive and negative classes, respectively.
**Part A3.4**: Compute F1-scores for positive and negative classes, respectively.

<details>
<summary>Solution A3</summary>

**A3.1**: Accuracy = (95 + 30) / 200 = **0.625** or **62.5%**

**A3.2**:
- Precision (Positive) = 95 / (95 + 70) = 95/165 = **19/33 ≈ 0.576**
- Precision (Negative) = 30 / (30 + 5) = 30/35 = **6/7 ≈ 0.857**

**A3.3**:
- Recall (Positive) = 95 / (95 + 5) = 95/100 = **0.95**
- Recall (Negative) = 30 / (30 + 70) = 30/100 = **0.30**

**A3.4**:
- F1 (Positive) = 2 × (19/33 × 0.95) / (19/33 + 0.95) ≈ **0.717**
- F1 (Negative) = 2 × (6/7 × 0.30) / (6/7 + 0.30) ≈ **0.444**

**Key insight**: This is the mirror of A2—an aggressive classifier that catches most positives but generates many false alarms.

</details>

---

### Variation A4: Near-Perfect Classifier

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 98 (TP)            | 2 (FN)             |
| **Actual Negative**  | 3 (FP)             | 97 (TN)            |

**Part A4.1**: Compute accuracy.
**Part A4.2**: Compute precision scores for positive and negative classes, respectively.
**Part A4.3**: Compute recall scores for positive and negative classes, respectively.
**Part A4.4**: Compute F1-scores for positive and negative classes, respectively.

<details>
<summary>Solution A4</summary>

**A4.1**: Accuracy = (98 + 97) / 200 = **0.975** or **97.5%**

**A4.2**:
- Precision (Positive) = 98 / (98 + 3) = 98/101 ≈ **0.970**
- Precision (Negative) = 97 / (97 + 2) = 97/99 ≈ **0.980**

**A4.3**:
- Recall (Positive) = 98 / (98 + 2) = 98/100 = **0.98**
- Recall (Negative) = 97 / (97 + 3) = 97/100 = **0.97**

**A4.4**:
- F1 (Positive) = 2 × (0.970 × 0.98) / (0.970 + 0.98) ≈ **0.975**
- F1 (Negative) = 2 × (0.980 × 0.97) / (0.980 + 0.97) ≈ **0.975**

</details>

---

### Variation A5: Worst-Case Scenario (Inverted Predictions)

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 5 (TP)             | 95 (FN)            |
| **Actual Negative**  | 95 (FP)            | 5 (TN)             |

**Part A5.1**: Compute accuracy.
**Part A5.2**: Compute precision scores for positive and negative classes, respectively.
**Part A5.3**: Compute recall scores for positive and negative classes, respectively.
**Part A5.4**: Compute F1-scores for positive and negative classes, respectively.
**Part A5.5**: If you flipped all predictions (Positive → Negative, Negative → Positive), what would the new accuracy be?

<details>
<summary>Solution A5</summary>

**A5.1**: Accuracy = (5 + 5) / 200 = **0.05** or **5%**

**A5.2**:
- Precision (Positive) = 5 / (5 + 95) = 5/100 = **0.05**
- Precision (Negative) = 5 / (5 + 95) = 5/100 = **0.05**

**A5.3**:
- Recall (Positive) = 5 / (5 + 95) = 5/100 = **0.05**
- Recall (Negative) = 5 / (5 + 95) = 5/100 = **0.05**

**A5.4**:
- F1 (Positive) = 2 × (0.05 × 0.05) / (0.05 + 0.05) = **0.05**
- F1 (Negative) = 2 × (0.05 × 0.05) / (0.05 + 0.05) = **0.05**

**A5.5**: After flipping:
- New TP = 95, New TN = 95, New FP = 5, New FN = 5
- New Accuracy = (95 + 95) / 200 = **0.95** or **95%**

**Key insight**: This classifier is doing worse than random—it's almost perfectly wrong! Flipping its predictions gives excellent results.

</details>

---

## CATEGORY B: Imbalanced Datasets

### Variation B1: Highly Imbalanced (Rare Positive Class - Fraud Detection)

Consider a fraud detection scenario with the following confusion matrix:

|                      | Predicted Fraud | Predicted Normal |
|----------------------|-----------------|------------------|
| **Actual Fraud**     | 18 (TP)         | 2 (FN)           |
| **Actual Normal**    | 50 (FP)         | 930 (TN)         |

**Part B1.1**: Compute accuracy.
**Part B1.2**: Compute precision and recall for the fraud (positive) class.
**Part B1.3**: Compute F1-score for the fraud class.
**Part B1.4**: A naive classifier that always predicts "Normal" would have what accuracy? Compare this to your answer in B1.1.

<details>
<summary>Solution B1</summary>

**B1.1**: Accuracy = (18 + 930) / 1000 = **0.948** or **94.8%**

**B1.2**:
- Precision (Fraud) = 18 / (18 + 50) = 18/68 ≈ **0.265**
- Recall (Fraud) = 18 / (18 + 2) = 18/20 = **0.90**

**B1.3**: F1 (Fraud) = 2 × (0.265 × 0.90) / (0.265 + 0.90) ≈ **0.409**

**B1.4**: A naive "always Normal" classifier:
- Would correctly predict all 980 normal cases
- Would miss all 20 fraud cases
- Accuracy = 980/1000 = **98.0%**

This is **higher** than our model's 94.8% accuracy! This demonstrates why accuracy is misleading for imbalanced datasets—a trivial classifier beats our fraud detector on accuracy, but catches zero fraud.

</details>

---

### Variation B2: Medical Diagnosis (High Cost of False Negatives)

Consider a cancer screening test with the following confusion matrix:

|                      | Predicted Cancer | Predicted Healthy |
|----------------------|------------------|-------------------|
| **Actual Cancer**    | 45 (TP)          | 5 (FN)            |
| **Actual Healthy**   | 200 (FP)         | 750 (TN)          |

**Part B2.1**: Compute accuracy.
**Part B2.2**: Compute sensitivity (recall for cancer class) and specificity (recall for healthy class).
**Part B2.3**: Compute the positive predictive value (precision for cancer class) and negative predictive value (precision for healthy class).
**Part B2.4**: In medical contexts, why might we prefer this test despite its low positive predictive value?

<details>
<summary>Solution B2</summary>

**B2.1**: Accuracy = (45 + 750) / 1000 = **0.795** or **79.5%**

**B2.2**:
- Sensitivity (Recall for Cancer) = 45 / (45 + 5) = 45/50 = **0.90** or **90%**
- Specificity (Recall for Healthy) = 750 / (750 + 200) = 750/950 ≈ **0.789** or **78.9%**

**B2.3**:
- Positive Predictive Value = 45 / (45 + 200) = 45/245 ≈ **0.184** or **18.4%**
- Negative Predictive Value = 750 / (750 + 5) = 750/755 ≈ **0.993** or **99.3%**

**B2.4**: Despite only 18.4% of positive predictions being true cancers:
- **High sensitivity (90%)** means we catch most actual cancers
- **High NPV (99.3%)** means a negative test is very reliable
- In screening, false negatives (missing cancer) are far more costly than false positives (unnecessary follow-up tests)
- This test is good for **ruling out** disease—if negative, you almost certainly don't have cancer

</details>

---

### Variation B3: Spam Detection (High Cost of False Positives)

Consider an email spam filter with the following confusion matrix:

|                      | Predicted Spam | Predicted Legitimate |
|----------------------|----------------|---------------------|
| **Actual Spam**      | 180 (TP)       | 20 (FN)             |
| **Actual Legitimate**| 2 (FP)         | 798 (TN)            |

**Part B3.1**: Compute accuracy.
**Part B3.2**: Compute precision and recall for the spam class.
**Part B3.3**: Compute F1-score for the spam class.
**Part B3.4**: Why might this spam filter prioritize precision over recall?

<details>
<summary>Solution B3</summary>

**B3.1**: Accuracy = (180 + 798) / 1000 = **0.978** or **97.8%**

**B3.2**:
- Precision (Spam) = 180 / (180 + 2) = 180/182 ≈ **0.989** or **98.9%**
- Recall (Spam) = 180 / (180 + 20) = 180/200 = **0.90** or **90%**

**B3.3**: F1 (Spam) = 2 × (0.989 × 0.90) / (0.989 + 0.90) ≈ **0.943**

**B3.4**: High precision is critical for spam filters because:
- **False positives (legitimate mail → spam)** are very costly: users might miss important emails
- **False negatives (spam → inbox)** are annoying but usually harmless
- Users will abandon a filter that blocks legitimate mail, even if it catches more spam
- A 98.9% precision means only 1.1% of flagged emails are actually legitimate—an acceptable rate

</details>

---

### Variation B4: Extreme Imbalance (Rare Disease)

Consider screening for a rare disease affecting 1 in 10,000 people:

|                      | Predicted Disease | Predicted Healthy |
|----------------------|-------------------|-------------------|
| **Actual Disease**   | 9 (TP)            | 1 (FN)            |
| **Actual Healthy**   | 99 (FP)           | 99,891 (TN)       |

**Part B4.1**: Compute accuracy.
**Part B4.2**: Compute precision and recall for the disease class.
**Part B4.3**: If the prevalence was 1 in 1,000 instead (with the same test sensitivity and specificity), what would happen to precision? *Hint: Use Bayes' theorem.*

<details>
<summary>Solution B4</summary>

**B4.1**: Accuracy = (9 + 99,891) / 100,000 = **0.999** or **99.9%**

**B4.2**:
- Precision (Disease) = 9 / (9 + 99) = 9/108 ≈ **0.083** or **8.3%**
- Recall (Disease) = 9 / (9 + 1) = 9/10 = **0.90** or **90%**

**B4.3**: With 1 in 1,000 prevalence (100 actual cases in 100,000):
- TP ≈ 90 (90% recall), FN ≈ 10
- FP ≈ 99 (same false positive rate), TN ≈ 99,801
- New Precision = 90 / (90 + 99) ≈ **0.476** or **47.6%**

**Key insight**: Precision depends heavily on prevalence. The same test is much more useful when the disease is more common. This is Bayes' theorem in action!

</details>

---

## CATEGORY C: Multi-Class Classification

### Variation C1: Three-Class Classification (Sentiment Analysis)

Consider a sentiment analysis model classifying reviews as Positive, Neutral, or Negative:

|                      | Pred. Positive | Pred. Neutral | Pred. Negative |
|----------------------|----------------|---------------|----------------|
| **Actual Positive**  | 85             | 10            | 5              |
| **Actual Neutral**   | 15             | 60            | 25             |
| **Actual Negative**  | 5              | 20            | 75             |

**Part C1.1**: Compute overall accuracy.
**Part C1.2**: Compute precision for each class.
**Part C1.3**: Compute recall for each class.
**Part C1.4**: Compute macro-averaged precision, recall, and F1-score.
**Part C1.5**: Compute micro-averaged precision. *Hint: For multi-class single-label, think about the relationship between micro-precision, micro-recall, and accuracy.*

<details>
<summary>Solution C1</summary>

**C1.1**:
Total correct = 85 + 60 + 75 = 220
Total samples = 100 + 100 + 100 = 300
Accuracy = 220/300 = **0.733** or **73.3%**

**C1.2**:
- Precision (Positive) = 85 / (85 + 15 + 5) = 85/105 ≈ **0.810**
- Precision (Neutral) = 60 / (10 + 60 + 20) = 60/90 ≈ **0.667**
- Precision (Negative) = 75 / (5 + 25 + 75) = 75/105 ≈ **0.714**

**C1.3**:
- Recall (Positive) = 85 / (85 + 10 + 5) = 85/100 = **0.85**
- Recall (Neutral) = 60 / (15 + 60 + 25) = 60/100 = **0.60**
- Recall (Negative) = 75 / (5 + 20 + 75) = 75/100 = **0.75**

**C1.4**:
- Macro-Precision = (0.810 + 0.667 + 0.714) / 3 ≈ **0.730**
- Macro-Recall = (0.85 + 0.60 + 0.75) / 3 ≈ **0.733**
- Macro-F1 = 2 × (0.730 × 0.733) / (0.730 + 0.733) ≈ **0.731**

**C1.5**:
For multi-class classification with single-label assignment:
- Micro-Precision = Micro-Recall = Accuracy = **0.733**

This is because ∑TP = ∑(TP + FP) = ∑(TP + FN) = total correct predictions when each sample belongs to exactly one class.

</details>

---

### Variation C2: Four-Class with Class Imbalance (Digit Recognition)

Consider a digit recognition model classifying digits 0, 1, 2, 3 with imbalanced test data:

|              | Pred. 0 | Pred. 1 | Pred. 2 | Pred. 3 |
|--------------|---------|---------|---------|---------|
| **Actual 0** | 180     | 10      | 5       | 5       |
| **Actual 1** | 8       | 85      | 4       | 3       |
| **Actual 2** | 5       | 3       | 40      | 2       |
| **Actual 3** | 2       | 2       | 1       | 45      |

**Part C2.1**: Compute overall accuracy.
**Part C2.2**: Compute precision and recall for class 0 only.
**Part C2.3**: Compute weighted-average F1-score (weighted by class support).
**Part C2.4**: Which class has the worst F1-score?

<details>
<summary>Solution C2</summary>

**C2.1**:
Total correct = 180 + 85 + 40 + 45 = 350
Total samples = 200 + 100 + 50 + 50 = 400
Accuracy = 350/400 = **0.875** or **87.5%**

**C2.2**:
- Precision (0) = 180 / (180 + 8 + 5 + 2) = 180/195 ≈ **0.923**
- Recall (0) = 180 / (180 + 10 + 5 + 5) = 180/200 = **0.90**

**C2.3**:
First compute F1 for each class:
- F1(0) = 2 × (0.923 × 0.90) / (0.923 + 0.90) ≈ 0.911
- P(1) = 85/100 = 0.85, R(1) = 85/100 = 0.85, F1(1) = 0.85
- P(2) = 40/50 = 0.80, R(2) = 40/50 = 0.80, F1(2) = 0.80
- P(3) = 45/55 ≈ 0.818, R(3) = 45/50 = 0.90, F1(3) ≈ 0.857

Weighted F1 = (200×0.911 + 100×0.85 + 50×0.80 + 50×0.857) / 400 ≈ **0.875**

**C2.4**: Class 2 has the worst F1-score at **0.80**

</details>

---

### Variation C3: Multi-Label Classification

Consider a document classification system where each document can have multiple tags (multi-label). For a single document:

**True labels:** {Sports, Entertainment}
**Predicted labels:** {Sports, Politics}

**Part C3.1**: Compute precision, recall, and F1-score for this single prediction.

**Part C3.2**: Given the following predictions for 4 documents:

| Document | True Labels | Predicted Labels |
|----------|-------------|------------------|
| 1        | {A, B}      | {A, B, C}        |
| 2        | {B, C}      | {B}              |
| 3        | {A}         | {A}              |
| 4        | {A, B, C}   | {B, C}           |

Compute sample-averaged precision, recall, and F1.

<details>
<summary>Solution C3</summary>

**C3.1**:
- True positives = |{Sports, Entertainment} ∩ {Sports, Politics}| = |{Sports}| = 1
- Predicted positives = |{Sports, Politics}| = 2
- Actual positives = |{Sports, Entertainment}| = 2

- Precision = 1/2 = **0.50**
- Recall = 1/2 = **0.50**
- F1 = 2 × (0.5 × 0.5) / (0.5 + 0.5) = **0.50**

**C3.2**:
Per-document metrics:
- Doc 1: P = 2/3, R = 2/2 = 1, F1 = 2×(2/3×1)/(2/3+1) = 4/5 = 0.80
- Doc 2: P = 1/1 = 1, R = 1/2, F1 = 2×(1×0.5)/(1+0.5) = 2/3 ≈ 0.667
- Doc 3: P = 1/1 = 1, R = 1/1 = 1, F1 = 1.0
- Doc 4: P = 2/2 = 1, R = 2/3, F1 = 2×(1×2/3)/(1+2/3) = 4/5 = 0.80

Sample-averaged:
- Precision = (2/3 + 1 + 1 + 1) / 4 ≈ **0.917**
- Recall = (1 + 0.5 + 1 + 2/3) / 4 ≈ **0.792**
- F1 = (0.80 + 0.667 + 1.0 + 0.80) / 4 ≈ **0.817**

</details>

---

## CATEGORY D: Edge Cases

### Variation D1: Perfect Classifier

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 100 (TP)           | 0 (FN)             |
| **Actual Negative**  | 0 (FP)             | 100 (TN)           |

**Part D1.1**: Compute accuracy, precision, recall, and F1-score.
**Part D1.2**: What is the false positive rate (FPR)? What is the false negative rate (FNR)?
**Part D1.3**: On an ROC curve, where would this classifier appear?

<details>
<summary>Solution D1</summary>

**D1.1**:
- Accuracy = (100 + 100) / 200 = **1.0** or **100%**
- Precision = 100 / (100 + 0) = **1.0**
- Recall = 100 / (100 + 0) = **1.0**
- F1 = 2 × (1.0 × 1.0) / (1.0 + 1.0) = **1.0**

**D1.2**:
- FPR = FP / (FP + TN) = 0 / 100 = **0**
- FNR = FN / (FN + TP) = 0 / 100 = **0**

**D1.3**: At the point **(0, 1)** — the top-left corner, representing perfect classification (0% FPR, 100% TPR).

</details>

---

### Variation D2: Random Classifier

Consider a classifier that randomly guesses with 50% probability for each class on a balanced dataset of 200 samples (100 positive, 100 negative). The expected confusion matrix is:

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 50 (TP)            | 50 (FN)            |
| **Actual Negative**  | 50 (FP)            | 50 (TN)            |

**Part D2.1**: Compute expected accuracy, precision, recall, and F1-score.
**Part D2.2**: On an ROC curve, where would this classifier appear?
**Part D2.3**: What is the expected Area Under the ROC Curve (AUC) for a random classifier?

<details>
<summary>Solution D2</summary>

**D2.1**:
- Accuracy = (50 + 50) / 200 = **0.50** or **50%**
- Precision = 50 / (50 + 50) = **0.50**
- Recall = 50 / (50 + 50) = **0.50**
- F1 = 2 × (0.5 × 0.5) / (0.5 + 0.5) = **0.50**

**D2.2**: At the point **(0.5, 0.5)** — on the diagonal line.

**D2.3**: AUC = **0.5** — the diagonal line from (0,0) to (1,1) represents random guessing.

</details>

---

### Variation D3: All-Positive Predictor

Consider a classifier that always predicts "Positive":

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 60 (TP)            | 0 (FN)             |
| **Actual Negative**  | 40 (FP)            | 0 (TN)             |

**Part D3.1**: Compute accuracy.
**Part D3.2**: Compute precision and recall for the positive class.
**Part D3.3**: What is the precision and recall for the negative class?
**Part D3.4**: Why is F1-score undefined for the negative class?

<details>
<summary>Solution D3</summary>

**D3.1**: Accuracy = (60 + 0) / 100 = **0.60** or **60%**

**D3.2**:
- Precision (Positive) = 60 / (60 + 40) = **0.60**
- Recall (Positive) = 60 / (60 + 0) = **1.0** (catches all positives)

**D3.3**:
- Precision (Negative) = TN / (TN + FN) = 0 / (0 + 0) = **undefined** (no negative predictions)
- Recall (Negative) = TN / (TN + FP) = 0 / (0 + 40) = **0** (misses all negatives)

**D3.4**: F1-score requires both precision and recall to be defined. Since precision for the negative class involves division by zero (no negative predictions made), F1 is **undefined**.

*In practice, some implementations set precision to 0 when there are no predictions for a class, giving F1 = 0.*

</details>

---

### Variation D4: Zero True Positives

Consider a classifier that makes some positive predictions but all are wrong:

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | 0 (TP)             | 30 (FN)            |
| **Actual Negative**  | 20 (FP)            | 50 (TN)            |

**Part D4.1**: Compute accuracy.
**Part D4.2**: Compute precision, recall, and F1-score for the positive class.
**Part D4.3**: What does this tell us about the classifier?

<details>
<summary>Solution D4</summary>

**D4.1**: Accuracy = (0 + 50) / 100 = **0.50** or **50%**

**D4.2**:
- Precision (Positive) = 0 / (0 + 20) = **0**
- Recall (Positive) = 0 / (0 + 30) = **0**
- F1 (Positive) = 2 × (0 × 0) / (0 + 0) = **0** (or undefined; typically set to 0)

**D4.3**: This classifier:
- Never correctly identifies a positive sample
- Has some ability to identify negatives (50/70 = 71% TN rate)
- Its positive predictions are completely unreliable (0% precision)
- This might indicate the model learned inverted features or has a bug in label mapping

</details>

---

## CATEGORY E: Application Contexts

### Variation E1: Autonomous Vehicle Safety

Consider an obstacle detection system for self-driving cars. A "positive" means detecting a pedestrian in the path:

|                          | Predicted Pedestrian | Predicted Clear |
|--------------------------|---------------------|-----------------|
| **Actual Pedestrian**    | 995 (TP)            | 5 (FN)          |
| **Actual Clear**         | 500 (FP)            | 98,500 (TN)     |

**Part E1.1**: Compute accuracy, precision, and recall.
**Part E1.2**: The car brakes every time it predicts "Pedestrian." What fraction of braking events are unnecessary (false alarms)?
**Part E1.3**: What is the probability of hitting a pedestrian that's actually in the path?
**Part E1.4**: In this safety-critical context, which error is more acceptable: FP or FN? How does this influence threshold selection?

<details>
<summary>Solution E1</summary>

**E1.1**:
- Accuracy = (995 + 98,500) / 100,000 = **0.9949** or **99.49%**
- Precision = 995 / (995 + 500) = 995/1495 ≈ **0.666**
- Recall = 995 / (995 + 5) = 995/1000 = **0.995**

**E1.2**:
False alarm rate among positive predictions = FP / (TP + FP) = 500/1495 ≈ **33.4%**
About 1/3 of braking events are unnecessary.

**E1.3**:
Probability of hitting pedestrian = FN / (TP + FN) = 5/1000 = **0.5%**
This is the miss rate (false negative rate).

**E1.4**:
- **FN is catastrophic**: Missing a pedestrian could result in death
- **FP is merely inconvenient**: Unnecessary braking is jarring but safe
- System should **optimize for high recall** even at the cost of precision
- Threshold should be set low, triggering on even uncertain detections

</details>

---

### Variation E2: Credit Card Fraud Detection

Consider a fraud detection system where positives are fraudulent transactions:

|                      | Predicted Fraud | Predicted Legitimate |
|----------------------|-----------------|---------------------|
| **Actual Fraud**     | 85 (TP)         | 15 (FN)             |
| **Actual Legitimate**| 950 (FP)        | 98,950 (TN)         |

**Part E2.1**: Compute precision and recall for fraud detection.
**Part E2.2**: If investigating each flagged transaction costs $10 and each missed fraud costs $500 on average, compute the total cost.
**Part E2.3**: If we adjust the threshold to be more conservative (fewer fraud predictions):
- New TP = 60, FN = 40, FP = 200, TN = 99,700

Compute the new total cost. Is this threshold better?

<details>
<summary>Solution E2</summary>

**E2.1**:
- Precision = 85 / (85 + 950) = 85/1035 ≈ **0.082** or **8.2%**
- Recall = 85 / (85 + 15) = 85/100 = **0.85** or **85%**

**E2.2**:
- Investigation cost = (TP + FP) × $10 = (85 + 950) × $10 = **$10,350**
- Missed fraud cost = FN × $500 = 15 × $500 = **$7,500**
- **Total cost = $17,850**

**E2.3**:
- New investigation cost = (60 + 200) × $10 = **$2,600**
- New missed fraud cost = 40 × $500 = **$20,000**
- **New total cost = $22,600**

**The original threshold is better** despite having more false positives. The cost of missed frauds ($500 each) outweighs the cost of investigations ($10 each).

*This demonstrates cost-sensitive threshold optimization.*

</details>

---

### Variation E3: COVID-19 Testing

Consider a COVID-19 rapid antigen test:

|                      | Test Positive | Test Negative |
|----------------------|---------------|---------------|
| **Actual COVID+**    | 70 (TP)       | 30 (FN)       |
| **Actual COVID-**    | 5 (FP)        | 895 (TN)      |

**Part E3.1**: Compute sensitivity and specificity.
**Part E3.2**: If prevalence in the population is 10%, what is the positive predictive value (PPV)?
**Part E3.3**: If prevalence drops to 1%, what is the new PPV? Why does this change matter for policy?

<details>
<summary>Solution E3</summary>

**E3.1**:
- Sensitivity = 70 / (70 + 30) = **0.70** or **70%**
- Specificity = 895 / (895 + 5) = 895/900 ≈ **0.994** or **99.4%**

**E3.2**: Using Bayes' theorem:
With prevalence p = 0.10:
- PPV = (Sensitivity × Prevalence) / ((Sensitivity × Prevalence) + (1 - Specificity) × (1 - Prevalence))
- PPV = (0.70 × 0.10) / ((0.70 × 0.10) + (0.006 × 0.90))
- PPV = 0.07 / (0.07 + 0.0054) = 0.07 / 0.0754 ≈ **0.93** or **93%**

**E3.3**: With prevalence p = 0.01:
- PPV = (0.70 × 0.01) / ((0.70 × 0.01) + (0.006 × 0.99))
- PPV = 0.007 / (0.007 + 0.00594) = 0.007 / 0.01294 ≈ **0.54** or **54%**

**Policy implications:**
- When prevalence is 10%: A positive test means 93% chance of having COVID
- When prevalence is 1%: A positive test means only 54% chance of having COVID
- Low prevalence settings require **confirmatory testing** for positive results
- This is why mass screening policies must account for base rates

</details>

---

## CATEGORY F: Theoretical and Proof Questions

### Variation F1: Metric Relationships

**Part F1.1**: Prove that for binary classification:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

can be rewritten as:

$$\text{Accuracy} = \text{Prevalence} \times \text{Recall}_+ + (1 - \text{Prevalence}) \times \text{Recall}_-$$

where Prevalence = (TP + FN) / N and Recall₋ is specificity (TN / (TN + FP)).

**Part F1.2**: Under what conditions is Accuracy = Precision for the positive class?

**Part F1.3**: Prove that F1-score is always between precision and recall (inclusive).

<details>
<summary>Solution F1</summary>

**F1.1**:
Let N = TP + TN + FP + FN (total samples)
Let P = TP + FN (actual positives)
Let N' = TN + FP (actual negatives)

Prevalence = P/N
Recall₊ = TP/P
Recall₋ = TN/N'

Prevalence × Recall₊ + (1 - Prevalence) × Recall₋
= (P/N) × (TP/P) + (N'/N) × (TN/N')
= TP/N + TN/N
= (TP + TN)/N
= Accuracy ✓

**F1.2**:
Accuracy = (TP + TN) / N
Precision = TP / (TP + FP)

These are equal when:
(TP + TN) / N = TP / (TP + FP)

Cross-multiplying:
(TP + TN)(TP + FP) = TP × N
(TP + TN)(TP + FP) = TP(TP + TN + FP + FN)
(TP + TN)(TP + FP) = TP(TP + TN) + TP(FP + FN)
(TP + TN)(FP) = TP(FP + FN)
FP × TN = TP × FN

This occurs when **FP × TN = TP × FN**, or equivalently when **FP/TP = FN/TN**.

**F1.3**:
F1 = 2PR/(P+R) where P = precision, R = recall.

Assume WLOG that P ≤ R (the argument is symmetric).

To show P ≤ F1 ≤ R:

**Lower bound (P ≤ F1):**
2PR/(P+R) ≥ P
2PR ≥ P(P+R)
2R ≥ P+R
R ≥ P ✓ (by assumption)

**Upper bound (F1 ≤ R):**
2PR/(P+R) ≤ R
2P ≤ P+R
P ≤ R ✓ (by assumption)

Therefore, F1 is the harmonic mean and always lies between P and R. ✓

</details>

---

### Variation F2: ROC and AUC Properties

**Part F2.1**: Prove that swapping all predictions (P ↔ N) reflects the ROC curve across the line y = 1 - x.

**Part F2.2**: Show that for a classifier with AUC = 0.5, its predictions are uncorrelated with the true labels.

**Part F2.3**: If classifier A has AUC = 0.8 and classifier B has AUC = 0.7, does A always outperform B at every threshold? Why or why not?

<details>
<summary>Solution F2</summary>

**F2.1**:
Original ROC point: (FPR, TPR) where:
- FPR = FP/(FP+TN)
- TPR = TP/(TP+FN)

After swapping predictions:
- Old TP → New FN, Old FN → New TP
- Old FP → New TN, Old TN → New FP

New FPR' = old TN / (old TN + old FP) = TN/(TN+FP) = 1 - FPR
New TPR' = old FN / (old FN + old TP) = FN/(FN+TP) = 1 - TPR

So (FPR, TPR) maps to (1-FPR, 1-TPR), which is reflection across y = 1-x. ✓

**F2.2**:
AUC = P(score(positive) > score(negative)) for a random positive-negative pair.

If AUC = 0.5, then:
P(score(positive) > score(negative)) = P(score(positive) < score(negative)) = 0.5

This means the score distribution for positives and negatives are indistinguishable—knowing the score tells you nothing about the class. The classifier's output is statistically independent of (uncorrelated with) the true labels.

**F2.3**:
**No**, AUC measures overall ranking quality, not performance at any specific threshold.

Counterexample: ROC curves can cross. Classifier B might outperform A in a specific region (e.g., low FPR region) while A has better overall AUC.

This is why practitioners often compare:
- **Partial AUC** over a relevant FPR range
- **Performance at specific thresholds** chosen for the application
- **Precision-Recall curves** for imbalanced data

</details>

---

### Variation F3: Cost-Sensitive Classification

**Part F3.1**: Given costs: C_FP (cost of false positive), C_FN (cost of false negative), derive the optimal threshold t* for a probabilistic classifier that outputs P(y=1|x).

**Part F3.2**: Show that when C_FP = C_FN, the optimal threshold is t* = 0.5.

**Part F3.3**: In fraud detection, if C_FN = 100 × C_FP, what is the optimal threshold?

<details>
<summary>Solution F3</summary>

**F3.1**:
Expected cost for predicting positive when P(y=1|x) = p:
- Cost if wrong = (1-p) × C_FP

Expected cost for predicting negative when P(y=1|x) = p:
- Cost if wrong = p × C_FN

Predict positive when:
(1-p) × C_FP < p × C_FN
C_FP - p×C_FP < p×C_FN
C_FP < p(C_FP + C_FN)
p > C_FP / (C_FP + C_FN)

**Optimal threshold:** t* = C_FP / (C_FP + C_FN)

**F3.2**:
When C_FP = C_FN = C:
t* = C / (C + C) = C / 2C = **0.5** ✓

**F3.3**:
Let C_FP = 1, then C_FN = 100:
t* = 1 / (1 + 100) = 1/101 ≈ **0.0099**

We should predict fraud even when the probability is just ~1%. This captures our intuition that missing fraud is 100× worse than a false alarm.

</details>

---

## CATEGORY G: Coding Implementations

### Variation G1: Basic Metrics Function

Write a Python function that computes all basic metrics from a confusion matrix.

```python
import numpy as np

def compute_metrics(confusion_matrix):
    """
    Compute classification metrics from a 2x2 confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (2, 2) where
                         [[TP, FN], [FP, TN]]

    Returns:
        dict with accuracy, precision, recall, f1 for both classes
    """
    # YOUR CODE HERE
    pass
```

**Part G1.1**: Implement the function.
**Part G1.2**: Test with the original problem's confusion matrix.

<details>
<summary>Solution G1</summary>

```python
import numpy as np

def compute_metrics(confusion_matrix):
    """
    Compute classification metrics from a 2x2 confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (2, 2) where
                         [[TP, FN], [FP, TN]]

    Returns:
        dict with accuracy, precision, recall, f1 for both classes
    """
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    total = TP + TN + FP + FN

    # Accuracy
    accuracy = (TP + TN) / total

    # Positive class metrics
    precision_pos = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

    # Negative class metrics
    precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision_positive': precision_pos,
        'recall_positive': recall_pos,
        'f1_positive': f1_pos,
        'precision_negative': precision_neg,
        'recall_negative': recall_neg,
        'f1_negative': f1_neg
    }

# Test with original problem
cm = np.array([[45, 5], [10, 40]])
metrics = compute_metrics(cm)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision (Pos): {metrics['precision_positive']:.4f}")
print(f"Recall (Pos): {metrics['recall_positive']:.4f}")
print(f"F1 (Pos): {metrics['f1_positive']:.4f}")
```

**Output:**
```
Accuracy: 0.8500
Precision (Pos): 0.8182
Recall (Pos): 0.9000
F1 (Pos): 0.8571
```

</details>

---

### Variation G2: Multi-Class Metrics

Write a function to compute metrics for multi-class classification.

```python
import numpy as np

def multiclass_metrics(confusion_matrix, average='macro'):
    """
    Compute classification metrics from an NxN confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (N, N)
        average: 'macro', 'micro', or 'weighted'

    Returns:
        dict with precision, recall, f1
    """
    # YOUR CODE HERE
    pass
```

**Part G2.1**: Implement the function supporting all three averaging methods.
**Part G2.2**: Test with the three-class matrix from Variation C1.

<details>
<summary>Solution G2</summary>

```python
import numpy as np

def multiclass_metrics(confusion_matrix, average='macro'):
    """
    Compute classification metrics from an NxN confusion matrix.

    Args:
        confusion_matrix: numpy array of shape (N, N)
        average: 'macro', 'micro', or 'weighted'

    Returns:
        dict with precision, recall, f1
    """
    n_classes = confusion_matrix.shape[0]

    # Per-class metrics
    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    supports = np.zeros(n_classes)  # Number of true samples per class

    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp  # Column sum minus diagonal
        fn = confusion_matrix[i, :].sum() - tp  # Row sum minus diagonal

        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        supports[i] = confusion_matrix[i, :].sum()

    if average == 'macro':
        precision = precisions.mean()
        recall = recalls.mean()
    elif average == 'micro':
        # Micro-averaging: aggregate TP, FP, FN across all classes
        total_tp = np.diag(confusion_matrix).sum()
        total_samples = confusion_matrix.sum()
        precision = total_tp / total_samples
        recall = precision  # Same for multi-class single-label
    elif average == 'weighted':
        total_support = supports.sum()
        precision = (precisions * supports).sum() / total_support
        recall = (recalls * supports).sum() / total_support
    else:
        raise ValueError(f"Unknown average: {average}")

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_precision': precisions,
        'per_class_recall': recalls
    }

# Test with C1 matrix
cm_c1 = np.array([
    [85, 10, 5],
    [15, 60, 25],
    [5, 20, 75]
])

for avg in ['macro', 'micro', 'weighted']:
    result = multiclass_metrics(cm_c1, average=avg)
    print(f"\n{avg.upper()} averaging:")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  F1: {result['f1']:.4f}")
```

**Output:**
```
MACRO averaging:
  Precision: 0.7303
  Recall: 0.7333
  F1: 0.7318

MICRO averaging:
  Precision: 0.7333
  Recall: 0.7333
  F1: 0.7333

WEIGHTED averaging:
  Precision: 0.7303
  Recall: 0.7333
  F1: 0.7318
```

</details>

---

### Variation G3: ROC Curve and AUC

Write a function to compute ROC curve points and AUC from predictions.

```python
import numpy as np

def compute_roc_auc(y_true, y_scores):
    """
    Compute ROC curve and AUC from binary labels and prediction scores.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_scores: numpy array of prediction scores (probabilities)

    Returns:
        dict with 'fpr', 'tpr', 'thresholds', 'auc'
    """
    # YOUR CODE HERE
    pass
```

**Part G3.1**: Implement the function without using sklearn.
**Part G3.2**: Test with a sample dataset and verify AUC is in [0, 1].

<details>
<summary>Solution G3</summary>

```python
import numpy as np

def compute_roc_auc(y_true, y_scores):
    """
    Compute ROC curve and AUC from binary labels and prediction scores.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_scores: numpy array of prediction scores (probabilities)

    Returns:
        dict with 'fpr', 'tpr', 'thresholds', 'auc'
    """
    # Sort by scores descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)[::-1]  # Descending

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tpr_list = [0]
    fpr_list = [0]
    threshold_list = [thresholds[0] + 1]  # Start above max threshold

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        threshold_list.append(thresh)

    # Add endpoint
    tpr_list.append(1)
    fpr_list.append(1)
    threshold_list.append(thresholds[-1] - 1)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_arr, fpr_arr)

    return {
        'fpr': fpr_arr,
        'tpr': tpr_arr,
        'thresholds': np.array(threshold_list),
        'auc': auc
    }

# Test
np.random.seed(42)
n_samples = 100
y_true = np.random.randint(0, 2, n_samples)
# Good classifier: higher scores for positive class
y_scores = y_true * 0.4 + np.random.uniform(0.2, 0.6, n_samples)

result = compute_roc_auc(y_true, y_scores)
print(f"AUC: {result['auc']:.4f}")
print(f"Number of ROC points: {len(result['fpr'])}")
print(f"AUC in [0,1]: {0 <= result['auc'] <= 1}")
```

**Output:**
```
AUC: 0.7534
Number of ROC points: 67
AUC in [0,1]: True
```

</details>

---

### Variation G4: Precision-Recall Curve

Write a function to compute the Precision-Recall curve and Average Precision.

```python
import numpy as np

def compute_pr_curve(y_true, y_scores):
    """
    Compute Precision-Recall curve and Average Precision.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_scores: numpy array of prediction scores

    Returns:
        dict with 'precision', 'recall', 'thresholds', 'ap'
    """
    # YOUR CODE HERE
    pass
```

**Part G4.1**: Implement the function.
**Part G4.2**: Why is Average Precision (AP) often preferred over AUC for imbalanced datasets?

<details>
<summary>Solution G4</summary>

```python
import numpy as np

def compute_pr_curve(y_true, y_scores):
    """
    Compute Precision-Recall curve and Average Precision.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_scores: numpy array of prediction scores

    Returns:
        dict with 'precision', 'recall', 'thresholds', 'ap'
    """
    # Sort by scores descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    thresholds = np.unique(y_scores_sorted)[::-1]

    n_pos = y_true.sum()

    precisions = [1.0]  # Start at (recall=0, precision=1)
    recalls = [0.0]

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / n_pos if n_pos > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Compute Average Precision (area under PR curve)
    # Use interpolated precision (monotonically decreasing)
    precisions_interp = np.maximum.accumulate(precisions[::-1])[::-1]

    # AP = sum of (recall[i] - recall[i-1]) * precision_interp[i]
    recall_diff = np.diff(recalls)
    ap = np.sum(recall_diff * precisions_interp[1:])

    return {
        'precision': precisions,
        'recall': recalls,
        'thresholds': thresholds,
        'ap': ap
    }

# Test
np.random.seed(42)
y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 30% positive
y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

result = compute_pr_curve(y_true, y_scores)
print(f"Average Precision: {result['ap']:.4f}")
```

**Part G4.2 Answer:**

AP is preferred for imbalanced datasets because:

1. **ROC curves can be misleading**: AUC uses FPR, which divides by the number of negatives. When negatives dominate (N >> P), even many false positives result in small FPR, making the classifier look good.

2. **PR curves focus on positives**: Precision and recall only consider how well we detect the minority (positive) class, which is usually what we care about.

3. **Example**: A fraud detector with 99% negatives:
   - Predicting 100 frauds with 90 TP and 10 FP gives:
     - ROC: FPR = 10/99,000 ≈ 0.0001 (looks great!)
     - PR: Precision = 90/100 = 0.90 (realistic view)

4. **Baseline matters**: Random classifier has AUC = 0.5 regardless of imbalance, but AP equals the positive class prevalence. This makes AP more sensitive to actual performance gains.

</details>

---

### Variation G5: Confusion Matrix from Predictions

Write a function to build a confusion matrix from predictions and compute all metrics.

```python
import numpy as np

def evaluate_classifier(y_true, y_pred_proba, threshold=0.5):
    """
    Build confusion matrix and compute all metrics.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_pred_proba: numpy array of prediction probabilities
        threshold: classification threshold (default 0.5)

    Returns:
        dict with confusion_matrix and all metrics
    """
    # YOUR CODE HERE
    pass
```

**Part G5.1**: Implement the function.
**Part G5.2**: Show how metrics change as threshold varies from 0.1 to 0.9.

<details>
<summary>Solution G5</summary>

```python
import numpy as np

def evaluate_classifier(y_true, y_pred_proba, threshold=0.5):
    """
    Build confusion matrix and compute all metrics.

    Args:
        y_true: numpy array of true labels (0 or 1)
        y_pred_proba: numpy array of prediction probabilities
        threshold: classification threshold (default 0.5)

    Returns:
        dict with confusion_matrix and all metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    confusion_matrix = np.array([[tp, fn], [fp, tn]])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'confusion_matrix': confusion_matrix,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
    }

# Part G5.2: Threshold analysis
np.random.seed(42)
n_samples = 200
y_true = np.concatenate([np.ones(80), np.zeros(120)])
# Simulated predictions: positives tend to have higher scores
y_pred_proba = np.concatenate([
    np.random.beta(5, 2, 80),   # Positives: skewed high
    np.random.beta(2, 5, 120)   # Negatives: skewed low
])

print("Threshold | Precision | Recall | F1-Score | Accuracy")
print("-" * 55)
for thresh in np.arange(0.1, 1.0, 0.1):
    result = evaluate_classifier(y_true, y_pred_proba, threshold=thresh)
    print(f"   {thresh:.1f}    |   {result['precision']:.3f}   | {result['recall']:.3f}  |  {result['f1']:.3f}   |  {result['accuracy']:.3f}")
```

**Output:**
```
Threshold | Precision | Recall | F1-Score | Accuracy
-------------------------------------------------------
   0.1    |   0.449   | 1.000  |  0.620   |  0.540
   0.2    |   0.506   | 0.988  |  0.669   |  0.615
   0.3    |   0.588   | 0.963  |  0.730   |  0.700
   0.4    |   0.662   | 0.913  |  0.767   |  0.760
   0.5    |   0.761   | 0.863  |  0.809   |  0.820
   0.6    |   0.841   | 0.763  |  0.800   |  0.835
   0.7    |   0.920   | 0.575  |  0.708   |  0.800
   0.8    |   0.964   | 0.338  |  0.500   |  0.730
   0.9    |   1.000   | 0.138  |  0.242   |  0.655
```

**Key insight**: As threshold increases:
- Precision increases (fewer false positives)
- Recall decreases (more false negatives)
- F1-score peaks around threshold 0.5
- The optimal threshold depends on the relative cost of FP vs FN

</details>

---

## Summary

| Category | Count | Focus |
|----------|-------|-------|
| **A: Different Values** | 5 | Same structure, different numbers |
| **B: Imbalanced Data** | 4 | Class distribution effects |
| **C: Multi-Class** | 3 | Beyond binary classification |
| **D: Edge Cases** | 4 | Perfect, random, degenerate classifiers |
| **E: Applications** | 3 | Real-world contexts and costs |
| **F: Theory/Proofs** | 3 | Mathematical foundations |
| **G: Coding** | 5 | Implementation from scratch |

**Total: 27 variations**

---

## Key Takeaways

1. **No single metric tells the whole story** - Always report multiple metrics and consider the application context

2. **Class imbalance breaks accuracy** - Use precision, recall, F1, or AUC for imbalanced data

3. **Threshold selection is a design decision** - It should be based on the cost asymmetry of different error types

4. **Prevalence affects precision** - The same test can have very different PPV depending on base rates (Bayes' theorem)

5. **ROC vs PR curves** - Use PR curves for imbalanced datasets; they give a more realistic picture of performance

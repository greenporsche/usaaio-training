# kNN & Cross-Validation — Exercises

> 5 exercises covering kNN prediction, distance computation, cross-validation, and the curse of dimensionality

---

## Exercise 1: Compute kNN Prediction (Compute This)

Given 6 training points in $\mathbb{R}^2$:

| Point | $x_1$ | $x_2$ | Label |
|---|---|---|---|
| A | 1 | 2 | 0 |
| B | 2 | 1 | 0 |
| C | 3 | 3 | 1 |
| D | 5 | 4 | 1 |
| E | 4 | 2 | 1 |
| F | 1 | 4 | 0 |

Query point: $x_q = (3, 2)$.

**Tasks**:
1. Compute the Euclidean distance from $x_q$ to each training point.
2. Find the 3-nearest neighbors.
3. Predict the label using $k = 3$ (majority vote).
4. What would the prediction be for $k = 1$? For $k = 5$?

<details>
<summary>Solution</summary>

**1. Distances**:

| Point | $\|x_q - x_i\|_2$ |
|---|---|
| A | $\sqrt{(3-1)^2 + (2-2)^2} = \sqrt{4} = 2.0$ |
| B | $\sqrt{(3-2)^2 + (2-1)^2} = \sqrt{2} \approx 1.414$ |
| C | $\sqrt{(3-3)^2 + (2-3)^2} = \sqrt{1} = 1.0$ |
| D | $\sqrt{(3-5)^2 + (2-4)^2} = \sqrt{8} \approx 2.828$ |
| E | $\sqrt{(3-4)^2 + (2-2)^2} = \sqrt{1} = 1.0$ |
| F | $\sqrt{(3-1)^2 + (2-4)^2} = \sqrt{8} \approx 2.828$ |

**2. Sorted**: C (1.0), E (1.0), B (1.414), A (2.0), D (2.828), F (2.828)

3-nearest neighbors: **C** (label 1), **E** (label 1), **B** (label 0)

**3.** $k = 3$: Labels are {1, 1, 0}. Majority vote: **1**. (2 votes for class 1, 1 vote for class 0)

**4.**
- $k = 1$: Nearest is C (or E, both at distance 1.0). Label: **1**.
- $k = 5$: Neighbors are C, E, B, A, D (or F — tied). Labels: {1, 1, 0, 0, 1}. Majority: **1** (3 vs 2).

</details>

---

## Exercise 2: Cross-Validation Error (Compute This)

You perform 5-fold CV on a classifier and get the following per-fold accuracy:

| Fold | Train Size | Test Size | Accuracy |
|---|---|---|---|
| 1 | 80 | 20 | 0.85 |
| 2 | 80 | 20 | 0.90 |
| 3 | 80 | 20 | 0.80 |
| 4 | 80 | 20 | 0.95 |
| 5 | 80 | 20 | 0.85 |

**Tasks**:
1. Compute the 5-fold CV accuracy.
2. Compute the standard deviation of the fold accuracies.
3. If you were reporting this result, what would you write? (e.g., "$X \pm Y$")
4. Is this CV estimate biased high or low relative to the true performance of a model trained on all 100 points?

<details>
<summary>Solution</summary>

**1.** CV accuracy $= \frac{0.85 + 0.90 + 0.80 + 0.95 + 0.85}{5} = \frac{4.35}{5} = 0.87$

**2.** Standard deviation:

Deviations from mean: $(-0.02, 0.03, -0.07, 0.08, -0.02)$

Squared deviations: $(0.0004, 0.0009, 0.0049, 0.0064, 0.0004)$

Variance $= \frac{0.0004 + 0.0009 + 0.0049 + 0.0064 + 0.0004}{5} = \frac{0.013}{5} = 0.0026$

Standard deviation $= \sqrt{0.0026} \approx 0.051$

**3.** Report: **$0.87 \pm 0.05$** (or more precisely, $0.870 \pm 0.051$).

**4.** The CV estimate is **biased low** (pessimistic). Each fold trains on only 80 points instead of 100. Since more training data generally improves performance, the model trained on all 100 points would likely perform better than 0.87. This pessimistic bias is inherent to cross-validation — you always train on less data than available.

</details>

---

## Exercise 3: Curse of Dimensionality (Derive That)

Data is uniformly distributed in the unit hypercube $[0, 1]^d$.

1. To capture a fraction $f = 0.1$ (10%) of the data in a hypercube neighborhood, what edge length $\ell$ is needed? Express as a function of $d$.
2. Compute $\ell$ for $d = 1, 2, 5, 10, 50, 100$.
3. What fraction of the range $[0, 1]$ does this edge length span in each dimension?
4. Explain why this makes kNN ineffective in high dimensions.

<details>
<summary>Solution</summary>

**1.** The volume of a hypercube with edge length $\ell$ in $[0, 1]^d$ is $\ell^d$. To capture a fraction $f$ of the data:

$$\ell^d = f \implies \ell = f^{1/d}$$

**2. & 3.** With $f = 0.1$:

| $d$ | $\ell = 0.1^{1/d}$ | % of range |
|---|---|---|
| 1 | 0.100 | 10% |
| 2 | 0.316 | 31.6% |
| 5 | 0.631 | 63.1% |
| 10 | 0.794 | 79.4% |
| 50 | 0.955 | 95.5% |
| 100 | 0.977 | 97.7% |

**4.** In high dimensions, even to capture a small fraction of the data (10%), the neighborhood must span nearly the entire range of each dimension (97.7% at $d=100$). This means:

- "Local" neighborhoods are not local at all — they cover almost the entire space
- All points become approximately equidistant from each other
- The concept of "nearest" becomes meaningless because $\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$
- kNN's core assumption — that nearby points have similar labels — breaks down because there are no meaningfully "nearby" points

This is why kNN requires either low dimensions or exponentially large datasets to work well.

</details>

---

## Exercise 4: LOOCV Shortcut (Compute This)

For linear regression, LOOCV can be computed efficiently:

$$\text{CV}_n = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$$

Given:

$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$$

1. Compute the hat matrix $H = X(X^TX)^{-1}X^T$.
2. Compute the diagonal entries $h_{11}, h_{22}, h_{33}$.
3. Compute $\hat{y} = Hy$ (the fitted values).
4. Compute the LOOCV error using the shortcut formula.

<details>
<summary>Solution</summary>

**1.** First compute $X^TX$ and its inverse:

$X^TX = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$

$(X^TX)^{-1} = \frac{1}{3(14) - 6(6)}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}$

$H = X(X^TX)^{-1}X^T$

$(X^TX)^{-1}X^T = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 8 & 2 & -4 \\ -3 & 0 & 3 \end{bmatrix}$

$H = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} \cdot \frac{1}{6}\begin{bmatrix} 8 & 2 & -4 \\ -3 & 0 & 3 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{bmatrix}$

**2.** Diagonal entries: $h_{11} = 5/6$, $h_{22} = 2/6 = 1/3$, $h_{33} = 5/6$.

**3.** $\hat{y} = Hy = \frac{1}{6}\begin{bmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix} = \frac{1}{6}\begin{bmatrix} 7 \\ 10 \\ 13 \end{bmatrix} = \begin{bmatrix} 7/6 \\ 10/6 \\ 13/6 \end{bmatrix}$

Residuals: $y - \hat{y} = \begin{bmatrix} 1 - 7/6 \\ 2 - 10/6 \\ 2 - 13/6 \end{bmatrix} = \begin{bmatrix} -1/6 \\ 2/6 \\ -1/6 \end{bmatrix}$

**4.** LOOCV:

$$\text{CV}_3 = \frac{1}{3}\left[\left(\frac{-1/6}{1 - 5/6}\right)^2 + \left(\frac{2/6}{1 - 1/3}\right)^2 + \left(\frac{-1/6}{1 - 5/6}\right)^2\right]$$

$$= \frac{1}{3}\left[\left(\frac{-1/6}{1/6}\right)^2 + \left(\frac{1/3}{2/3}\right)^2 + \left(\frac{-1/6}{1/6}\right)^2\right]$$

$$= \frac{1}{3}\left[(-1)^2 + (1/2)^2 + (-1)^2\right] = \frac{1}{3}\left[1 + 0.25 + 1\right] = \frac{2.25}{3} = 0.75$$

</details>

---

## Exercise 5: True/False with Justification

1. **kNN with $k = n$ always predicts the majority class.**
2. **LOOCV always has lower bias than 5-fold CV.**
3. **Feature scaling (standardization) doesn't affect kNN's predictions.**
4. **Cross-validation eliminates the need for a separate test set.**
5. **Weighted kNN (inverse-distance weighting) always outperforms unweighted kNN.**

<details>
<summary>Solution</summary>

1. **TRUE**. With $k = n$, every training point is a neighbor. For classification, the majority vote over all $n$ labels is the majority class. For regression, it predicts the mean of all training targets. This is the maximum-bias, minimum-variance extreme.

2. **TRUE**. LOOCV trains on $n-1$ points (out of $n$), while 5-fold trains on $\frac{4n}{5}$ points. Since $n-1 > \frac{4n}{5}$ (for $n > 5$), LOOCV uses more training data per fold, making it a less pessimistic (lower bias) estimate of the true error. However, LOOCV has **higher variance** because the $n$ training sets are highly correlated (differing by only one point).

3. **FALSE**. Feature scaling dramatically affects kNN. Consider features "height in cm" (range 150-200) and "weight in kg" (range 50-100). Without scaling, the Euclidean distance is dominated by the height feature. After standardization (zero mean, unit variance), both features contribute equally to the distance. This changes which points are "nearest" and therefore changes predictions.

4. **FALSE**. Cross-validation is used for **model selection** (choosing hyperparameters, comparing models). You still need a held-out test set to get an unbiased estimate of the final model's generalization error. Using CV for both selection and evaluation leads to optimistic estimates because the "test" data influenced the model choice.

5. **FALSE**. Weighted kNN can be worse when the closest neighbors happen to be outliers or noise. The inverse-distance weighting amplifies the influence of the single nearest point, which may be mislabeled. In noisy data, unweighted kNN with a moderate $k$ can be more robust. The best approach depends on the data.

</details>

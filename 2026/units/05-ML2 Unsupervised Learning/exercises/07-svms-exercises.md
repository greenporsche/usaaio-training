# Support Vector Machines — Exercises

**Target time**: 2–5 minutes each | **Total**: 5 exercises

---

## Exercise 1: Compute the Margin

A 2D SVM has learned the decision boundary $2x_1 + x_2 - 3 = 0$ (i.e., $w = [2, 1]^T$, $b = -3$).

1. Compute $\|w\|$.
2. What is the margin width?
3. What are the equations of the two margin boundaries?
4. Classify the point $x = (2, 1)$. Is it inside or outside the margin?
5. Classify the point $x = (1, 0)$. Is it a support vector candidate?

<details>
<summary>Solution</summary>

**Part 1**: $\|w\| = \sqrt{2^2 + 1^2} = \sqrt{5} \approx 2.236$

**Part 2**: Margin width $= \frac{2}{\|w\|} = \frac{2}{\sqrt{5}} = \frac{2\sqrt{5}}{5} \approx 0.894$

**Part 3**: The margin boundaries are:

$w^T x + b = +1$: $2x_1 + x_2 - 3 = +1 \Rightarrow 2x_1 + x_2 = 4$

$w^T x + b = -1$: $2x_1 + x_2 - 3 = -1 \Rightarrow 2x_1 + x_2 = 2$

**Part 4**: $f(2, 1) = 2(2) + 1(1) - 3 = 4 + 1 - 3 = 2$

Since $f > 0$, classify as **positive** (+1). Since $f = 2 > 1$, the point is **outside the margin** (well into the positive region).

**Part 5**: $f(1, 0) = 2(1) + 1(0) - 3 = -1$

Since $f = -1 < 0$, classify as **negative** (-1). Since $f = -1$, the point is exactly **on the negative margin boundary** ($w^T x + b = -1$). This point IS a support vector candidate — it lies exactly on the margin.

</details>

---

## Exercise 2: Soft Margin SVM

Consider a soft margin SVM with $C = 10$. The following points are in the training set:

| Point | $x_1$ | $x_2$ | $y$ | $w^Tx + b$ |
|-------|--------|--------|-----|------------|
| A | 1 | 2 | +1 | 1.5 |
| B | 2 | 1 | +1 | 0.8 |
| C | 0 | 1 | -1 | -1.2 |
| D | 1 | 0 | -1 | -0.3 |

1. Compute the slack variable $\xi_i$ for each point.
2. Which points are support vectors? (Recall: support vectors have $\alpha_i > 0$, meaning they're on or inside the margin.)
3. Which points are misclassified?
4. What is the total hinge loss: $\sum_i \max(0, 1 - y_i f(x_i))$?
5. If $C$ is increased to 1000, what would happen to the margin width? To the number of support vectors?

<details>
<summary>Solution</summary>

**Part 1**: Slack $\xi_i = \max(0, 1 - y_i(w^Tx_i + b))$:

| Point | $y_i f(x_i)$ | $\xi_i = \max(0, 1 - y_i f)$ |
|-------|-------------|-------------------------------|
| A | $(+1)(1.5) = 1.5$ | $\max(0, 1-1.5) = 0$ |
| B | $(+1)(0.8) = 0.8$ | $\max(0, 1-0.8) = 0.2$ |
| C | $(-1)(-1.2) = 1.2$ | $\max(0, 1-1.2) = 0$ |
| D | $(-1)(-0.3) = 0.3$ | $\max(0, 1-0.3) = 0.7$ |

**Part 2**: Support vectors are points with $y_i f(x_i) \leq 1$ (on or inside the margin):

- A: $y_i f = 1.5 > 1$ → NOT a support vector (beyond margin)
- B: $y_i f = 0.8 < 1$ → **Support vector** (inside margin)
- C: $y_i f = 1.2 > 1$ → NOT a support vector
- D: $y_i f = 0.3 < 1$ → **Support vector** (inside margin)

**Part 3**: Misclassified points have $y_i f(x_i) < 0$ (equivalently $\xi_i > 1$):

- D: $y_i f = 0.3 > 0$ → correctly classified but inside margin ($0 < \xi < 1$)
- No points are misclassified in this example.

**Part 4**: Total hinge loss = $0 + 0.2 + 0 + 0.7 = 0.9$

**Part 5**: Increasing $C$ to 1000:
- **Margin width decreases**: Higher $C$ penalizes violations more, forcing the margin to shrink to reduce $\xi_i$.
- **Fewer support vectors**: With a narrower margin, fewer points fall within or on the margin boundary.
- The model approaches a hard-margin SVM, potentially overfitting.

</details>

---

## Exercise 3: Kernel Computations

1. Compute the polynomial kernel $K(x, z) = (x^T z + 1)^2$ for $x = [1, 2]^T$ and $z = [3, 4]^T$.

2. Show that this kernel corresponds to a feature map. Expand $(x^T z + 1)^2$ and identify $\phi(x)$.

3. Compute the RBF kernel $K(x, z) = \exp(-\gamma\|x-z\|^2)$ with $\gamma = 0.5$ for the same points.

4. What happens to the RBF kernel as $\gamma \to \infty$? As $\gamma \to 0$?

<details>
<summary>Solution</summary>

**Part 1**: $x^T z = 1(3) + 2(4) = 3 + 8 = 11$

$K(x, z) = (11 + 1)^2 = 144$

**Part 2**: Expanding $(x^T z + 1)^2$ for 2D inputs $x = [x_1, x_2]$ and $z = [z_1, z_2]$:

$(x_1 z_1 + x_2 z_2 + 1)^2$

$= x_1^2 z_1^2 + x_2^2 z_2^2 + 1 + 2x_1 x_2 z_1 z_2 + 2x_1 z_1 + 2x_2 z_2$

This equals $\phi(x)^T \phi(z)$ where:

$\phi(x) = [x_1^2, x_2^2, \sqrt{2}x_1 x_2, \sqrt{2}x_1, \sqrt{2}x_2, 1]^T$

Verification: $\phi(x)^T\phi(z) = x_1^2 z_1^2 + x_2^2 z_2^2 + 2x_1x_2z_1z_2 + 2x_1z_1 + 2x_2z_2 + 1$ ✓

The kernel maps from 2D to 6D! And we never needed to compute $\phi$ explicitly.

**Part 3**: $\|x - z\|^2 = (1-3)^2 + (2-4)^2 = 4 + 4 = 8$

$K(x, z) = \exp(-0.5 \times 8) = \exp(-4) \approx 0.0183$

The points are relatively far apart, so the RBF kernel gives a small value (low similarity).

**Part 4**:

- $\gamma \to \infty$: $K(x, z) \to 0$ for $x \neq z$ and $K(x, x) = 1$. The kernel matrix becomes the identity matrix. Each point is its own cluster, and the SVM memorizes the training data (extreme overfitting). The decision boundary becomes very complex, wrapping around individual points.

- $\gamma \to 0$: $K(x, z) \to \exp(0) = 1$ for all pairs. The kernel matrix becomes all 1's. All points look identical, and the SVM can't distinguish between any of them (extreme underfitting). The decision boundary becomes a single hyperplane in the original space.

</details>

---

## Exercise 4: SVM Dual and Support Vectors

A trained SVM has the following support vectors and dual variables:

| SV | $x_i$ | $y_i$ | $\alpha_i$ |
|----|--------|--------|-----------|
| 1 | (1, 1) | +1 | 0.5 |
| 2 | (2, 0) | +1 | 0.3 |
| 3 | (0, 0) | -1 | 0.8 |

The bias is $b = 0.2$.

1. Compute the weight vector $w = \sum_i \alpha_i y_i x_i$.
2. Compute $f(x)$ for $x = (1.5, 0.5)$.
3. Classify $x = (1.5, 0.5)$.
4. Verify that the sum constraint holds: $\sum_i \alpha_i y_i = 0$.
5. Compute the margin width.

<details>
<summary>Solution</summary>

**Part 1**: $w = \alpha_1 y_1 x_1 + \alpha_2 y_2 x_2 + \alpha_3 y_3 x_3$

$= 0.5(+1)(1,1) + 0.3(+1)(2,0) + 0.8(-1)(0,0)$

$= (0.5, 0.5) + (0.6, 0) + (0, 0)$

$= (1.1, 0.5)$

**Part 2**: $f(x) = w^T x + b = 1.1(1.5) + 0.5(0.5) + 0.2 = 1.65 + 0.25 + 0.2 = 2.1$

**Part 3**: $f(x) = 2.1 > 0$, so classify as **positive (+1)**. Since $f > 1$, the point is outside the margin (confidently positive).

**Part 4**: $\sum_i \alpha_i y_i = 0.5(+1) + 0.3(+1) + 0.8(-1) = 0.5 + 0.3 - 0.8 = 0$ ✓

This constraint always holds in the SVM dual formulation.

**Part 5**: $\|w\| = \sqrt{1.1^2 + 0.5^2} = \sqrt{1.21 + 0.25} = \sqrt{1.46} \approx 1.208$

Margin width $= \frac{2}{\|w\|} = \frac{2}{1.208} \approx 1.656$

</details>

---

## Exercise 5: SVM vs. Logistic Regression

Consider the following comparison between SVM (hinge loss) and logistic regression (logistic loss):

| Property | SVM | Logistic Regression |
|----------|-----|-------------------|
| Loss function | $\max(0, 1 - yf)$ | $\log(1 + e^{-yf})$ |
| Output | Distance to boundary | Probability |
| Sparsity | Only SVs matter | All points matter |

1. For a point with $y = +1$ and $f(x) = 2$, compute the hinge loss and the logistic loss.
2. For a point with $y = +1$ and $f(x) = 0.5$, compute both losses.
3. For a point with $y = +1$ and $f(x) = -1$, compute both losses.
4. At what value of $yf$ does the hinge loss become exactly zero? What about logistic loss?
5. Why does the "zero loss beyond margin" property of SVM lead to sparsity (few support vectors)?

<details>
<summary>Solution</summary>

**Part 1**: $yf = (+1)(2) = 2$

Hinge: $\max(0, 1 - 2) = \max(0, -1) = 0$

Logistic: $\log(1 + e^{-2}) = \log(1 + 0.135) = \log(1.135) = 0.127$

The point is well-classified. Hinge loss is exactly zero; logistic loss is small but nonzero.

**Part 2**: $yf = 0.5$

Hinge: $\max(0, 1 - 0.5) = 0.5$

Logistic: $\log(1 + e^{-0.5}) = \log(1 + 0.607) = \log(1.607) = 0.475$

Both losses are positive — the point is correctly classified but inside the margin.

**Part 3**: $yf = -1$

Hinge: $\max(0, 1 - (-1)) = \max(0, 2) = 2$

Logistic: $\log(1 + e^{1}) = \log(1 + 2.718) = \log(3.718) = 1.313$

Both losses are large — the point is misclassified. Hinge loss is actually larger here.

**Part 4**:
- Hinge loss: $\max(0, 1 - yf) = 0$ when $yf \geq 1$. It's exactly zero for all points beyond the margin.
- Logistic loss: $\log(1 + e^{-yf}) > 0$ for all finite $yf$. It **never** reaches exactly zero (only approaches 0 as $yf \to \infty$).

**Part 5**: Because hinge loss is exactly zero for all points with $yf > 1$ (beyond the margin), these points contribute zero gradient during optimization. Only points with $yf \leq 1$ (on or inside the margin) have nonzero gradient and influence the solution. These points are the support vectors.

In logistic regression, every point has a (small) nonzero gradient, so every point pulls the boundary slightly. This means ALL points matter, and the solution depends on the entire dataset. SVMs, by contrast, depend only on the support vectors — typically a small fraction of the data — making them sparse and efficient.

</details>

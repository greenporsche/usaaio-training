# Logistic Regression — Exercises

> 5 exercises covering sigmoid, cross-entropy, gradient derivation, softmax, and decision boundaries

---

## Exercise 1: Compute Sigmoid and Cross-Entropy (Compute This)

Given:
- Weight vector: $w = [1, -2]^T$, bias $b = 0.5$
- Data point: $x = [1, 1]^T$, true label $y = 1$

**Tasks**:
1. Compute the logit $z = w^Tx + b$.
2. Compute the predicted probability $\hat{y} = \sigma(z)$.
3. Compute the binary cross-entropy loss for this single point.
4. If we had a second point $x_2 = [-1, 0]^T$ with $y_2 = 0$, compute the average BCE loss over both points.

<details>
<summary>Solution</summary>

**1.** $z = w^Tx + b = 1(1) + (-2)(1) + 0.5 = 1 - 2 + 0.5 = -0.5$

**2.** $\hat{y} = \sigma(-0.5) = \frac{1}{1 + e^{0.5}} = \frac{1}{1 + 1.6487} = \frac{1}{2.6487} \approx 0.3775$

**3.** $\text{BCE} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$

$= -[1 \cdot \log(0.3775) + 0 \cdot \log(0.6225)]$

$= -\log(0.3775) \approx -(-0.9741) = 0.974$

**4.** For point 2: $z_2 = 1(-1) + (-2)(0) + 0.5 = -0.5$

$\hat{y}_2 = \sigma(-0.5) \approx 0.3775$

$\text{BCE}_2 = -[0 \cdot \log(0.3775) + 1 \cdot \log(1 - 0.3775)]$

$= -\log(0.6225) \approx -(-0.4740) = 0.474$

Average: $\frac{0.974 + 0.474}{2} = 0.724$

</details>

---

## Exercise 2: Derive the Gradient of Cross-Entropy (Derive That)

Starting from the binary cross-entropy loss:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\sigma(z_i) + (1-y_i)\log(1 - \sigma(z_i))\right]$$

where $z_i = w^Tx_i$.

1. Compute $\frac{\partial \mathcal{L}}{\partial z_i}$ using the chain rule and $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
2. Then compute $\nabla_w \mathcal{L} = \frac{1}{n}\sum_i \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial w}$.
3. Express the result in matrix form.
4. Compare this gradient to the gradient of MSE for linear regression. What's similar?

<details>
<summary>Solution</summary>

**1.** Let $\hat{y}_i = \sigma(z_i)$.

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i}$$

First: $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)$

Second: $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$

Multiply:

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)\hat{y}_i(1-\hat{y}_i)$$

$$= -\frac{1}{n}\left(y_i(1-\hat{y}_i) - (1-y_i)\hat{y}_i\right)$$

$$= -\frac{1}{n}(y_i - y_i\hat{y}_i - \hat{y}_i + y_i\hat{y}_i)$$

$$= -\frac{1}{n}(y_i - \hat{y}_i) = \frac{1}{n}(\hat{y}_i - y_i)$$

**2.** Since $\frac{\partial z_i}{\partial w} = x_i$:

$$\nabla_w \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_i$$

**3.** In matrix form:

$$\boxed{\nabla_w \mathcal{L} = \frac{1}{n}X^T(\sigma(Xw) - y)}$$

where $X$ is $(n, d)$, $\sigma(Xw)$ is $(n,)$, and the result is $(d,)$.

**4.** For linear regression MSE: $\nabla_w \mathcal{L} = \frac{2}{n}X^T(Xw - y)$.

The structure is identical: $X^T \times (\text{prediction} - \text{truth})$. The only differences are:
- The prediction function: $\sigma(Xw)$ vs $Xw$
- The scaling factor: $1/n$ vs $2/n$

This elegant similarity arises because both are generalized linear models (GLMs).

</details>

---

## Exercise 3: Identify the Error (Debug)

```python
def softmax(Z):
    exp_Z = np.exp(Z)                           # (n, C)
    return exp_Z / np.sum(exp_Z, axis=0)         # BUG?

def cross_entropy_loss(Y_true, Y_pred):
    return -np.mean(Y_true * np.log(Y_pred))     # BUG?

def train_softmax(X, Y, lr=0.01, n_steps=1000):
    n, d = X.shape
    C = Y.shape[1]
    W = np.random.randn(d, C)                    # BUG?
    for _ in range(n_steps):
        Z = X @ W                                 # (n, C)
        Y_hat = softmax(Z)                        # (n, C)
        grad = (1/n) * X.T @ (Y_hat - Y)          # (d, C)
        W = W - lr * grad
    return W
```

Find **three bugs**.

<details>
<summary>Solution</summary>

**Bug 1 — `softmax`**: `np.sum(exp_Z, axis=0)` sums over the **wrong axis**. The sum should be over classes (axis=1) for each sample, not over samples for each class.

Fix: `return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)`

Also missing numerical stability (log-sum-exp trick):
```python
Z_shifted = Z - Z.max(axis=1, keepdims=True)
exp_Z = np.exp(Z_shifted)
```

**Bug 2 — `cross_entropy_loss`**: `np.mean(Y_true * np.log(Y_pred))` takes the mean over *all* elements of the $(n, C)$ matrix. But cross-entropy sums over classes and averages over samples:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic}\log\hat{y}_{ic}$$

Fix: `return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))`

The `np.mean` averages over the $n$ samples (after `np.sum` aggregates over $C$ classes).

**Bug 3 — Weight initialization**: `np.random.randn(d, C)` initializes with standard normal values, which may be too large for gradient descent, causing numerical overflow in softmax (especially with `np.exp`). This isn't strictly a "bug" but will cause numerical issues.

Better: `W = np.zeros((d, C))` or `W = 0.01 * np.random.randn(d, C)`

With the incorrect softmax (Bug 1), the initialization doesn't matter much, but fixing Bug 1 while keeping large random init can lead to NaN from overflow.

</details>

---

## Exercise 4: Decision Boundary Geometry (Fill in the Shape)

For logistic regression with $w = [2, -1]^T$ and $b = 3$:

1. Write the equation of the decision boundary.
2. What is the slope and intercept of this line (in $x_1$-$x_2$ coordinates)?
3. Classify the following points: $(0, 0)$, $(1, 5)$, $(-2, -1)$, $(0, 4)$.
4. Which direction (in feature space) does the positive class lie?

<details>
<summary>Solution</summary>

**1.** Decision boundary: $w^Tx + b = 0$, i.e., $2x_1 - x_2 + 3 = 0$.

**2.** Rearranging: $x_2 = 2x_1 + 3$. Slope = 2, intercept = 3.

**3.** Compute $z = 2x_1 - x_2 + 3$:

| Point | $z$ | $\sigma(z)$ | Class |
|---|---|---|---|
| $(0, 0)$ | $0 - 0 + 3 = 3$ | $\sigma(3) \approx 0.95$ | **1** |
| $(1, 5)$ | $2 - 5 + 3 = 0$ | $\sigma(0) = 0.50$ | **tie** (on boundary) |
| $(-2, -1)$ | $-4 + 1 + 3 = 0$ | $\sigma(0) = 0.50$ | **tie** (on boundary) |
| $(0, 4)$ | $0 - 4 + 3 = -1$ | $\sigma(-1) \approx 0.27$ | **0** |

**4.** The positive class lies in the direction of $w = [2, -1]^T$. Specifically, points where $w^Tx + b > 0$ (above the decision threshold) are classified as positive. This is the half-space on the side of $w$ (pointing toward increasing $x_1$ and decreasing $x_2$), shifted by the bias.

Geometrically: the normal vector $w = [2, -1]$ points from the negative region to the positive region. The boundary line $x_2 = 2x_1 + 3$ separates the two classes, with class 1 **below** the line (where $z > 0$).

</details>

---

## Exercise 5: True/False with Justification

1. **The sigmoid function can output exactly 0 or 1.**
2. **Logistic regression's cross-entropy loss is convex in the weights $w$.**
3. **Softmax with $C = 2$ classes is equivalent to sigmoid.**
4. **Logistic regression can perfectly separate any linearly separable dataset.**
5. **Adding L2 regularization to logistic regression makes the loss strictly convex.**

<details>
<summary>Solution</summary>

1. **FALSE**. $\sigma(z) = \frac{1}{1 + e^{-z}}$ is always strictly between 0 and 1 for any finite $z$. It approaches 0 as $z \to -\infty$ and 1 as $z \to +\infty$, but never reaches these limits. This is why we clip predictions in code to avoid $\log(0)$.

2. **TRUE**. The Hessian of the cross-entropy loss is $H = \frac{1}{n}X^TSX$ where $S = \text{diag}(\hat{y}_i(1-\hat{y}_i))$ has positive diagonal entries. For any $v \neq 0$: $v^THv = \frac{1}{n}\|S^{1/2}Xv\|^2 \geq 0$. This makes the loss convex (and strictly convex if $X$ has full column rank).

3. **TRUE**. With $C = 2$, softmax gives:

$$\text{softmax}(z)_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

So the probability of class 1 is $\sigma$ applied to the difference of logits. This is why binary classification only needs one logit (one weight vector), not two.

4. **This is subtle — effectively TRUE** but with caveats. For linearly separable data, the logistic regression weights will grow unbounded ($\|w\| \to \infty$) to make the sigmoid outputs approach 0 and 1. Gradient descent will make the loss arbitrarily close to 0 but never exactly reach 0. The MLE does not exist in the traditional sense (no finite $w$ achieves zero loss), but in the limit, the decision boundary perfectly separates the classes. In practice, we stop at some iteration and get near-perfect separation.

5. **TRUE**. Without regularization, the Hessian $\frac{1}{n}X^TSX$ may be singular (if $X$ doesn't have full rank). Adding L2 regularization $\lambda\|w\|^2$ adds $2\lambda I$ to the Hessian, giving $\frac{1}{n}X^TSX + 2\lambda I$. Since $2\lambda I$ is positive definite (for $\lambda > 0$), the sum is positive definite, making the loss strictly convex. This guarantees a unique global minimum.

</details>

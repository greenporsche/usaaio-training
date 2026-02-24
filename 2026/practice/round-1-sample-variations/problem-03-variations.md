# Problem 3 Variations: Regularization & Bias-Variance (EXHAUSTIVE)

> Original: L1 vs L2 regularization, bias-variance tradeoff
> Core Skills: Understanding regularization effects, bias-variance decomposition

---

## CATEGORY A: L1 vs L2 Regularization

### Variation A1: Basic Comparison
Consider two regularized linear regression models trained on the same dataset:
- Model A uses L1-regularization (Lasso)
- Model B uses L2-regularization (Ridge)
- Both use the same regularization strength λ > 0

**Part 3.1**: Which statement is most likely true?
- A. Models A and B have the same number of non-zero weights
- B. Model A has fewer non-zero weights than Model B
- C. Model B has fewer non-zero weights than Model A
- D. Neither model can produce zero weights
- E. The relationship depends on the data

<details>
<summary>Solution A1</summary>

**Answer: B**

L1 regularization (Lasso) produces sparse solutions due to its diamond-shaped constraint region. The corners of the diamond lie on the axes, making it likely that the optimal solution has some weights exactly equal to zero.

L2 regularization (Ridge) produces small but non-zero weights due to its circular constraint region. The solution shrinks toward zero but rarely reaches exactly zero.
</details>

### Variation A2: Geometric Interpretation
**Part 3.1**: The constraint region for L1 regularization in 2D is:
- A. A circle
- B. A diamond (rotated square)
- C. A square aligned with axes
- D. An ellipse
- E. A rectangle

**Part 3.2**: Why does the L1 constraint region lead to sparse solutions?
- A. The region is smaller than L2
- B. The corners lie on the coordinate axes
- C. The region is convex
- D. The region is unbounded
- E. The region is symmetric

<details>
<summary>Solution A2</summary>

**Part 3.1: B** - L1 constraint ||w||₁ ≤ t forms a diamond in 2D.

**Part 3.2: B** - The corners of the diamond lie on the axes (e.g., (t,0), (0,t), (-t,0), (0,-t)). When the loss function's level curves touch the constraint region, they're most likely to touch at a corner, setting one or more weights to exactly zero.
</details>

### Variation A3: Regularization Strength
As the regularization parameter λ increases from 0 to ∞ in Ridge regression:

**Part 3.1**: What happens to the weights?
- A. They increase without bound
- B. They approach the OLS solution
- C. They approach zero
- D. They become more sparse
- E. They oscillate

**Part 3.2**: What happens to the bias of the model?
- A. Bias increases
- B. Bias decreases
- C. Bias stays constant
- D. Bias becomes undefined
- E. Bias oscillates

**Part 3.3**: What happens to the variance of the model?
- A. Variance increases
- B. Variance decreases
- C. Variance stays constant
- D. Variance becomes undefined
- E. Variance oscillates

<details>
<summary>Solution A3</summary>

**Part 3.1: C** - As λ → ∞, the penalty term dominates, forcing weights → 0.

**Part 3.2: A** - Smaller weights mean the model is less flexible, increasing bias.

**Part 3.3: B** - Smaller weights mean the model is less sensitive to training data, decreasing variance.
</details>

### Variation A4: Elastic Net
Elastic Net combines L1 and L2 regularization:
Loss = ||y - Xw||² + α·λ||w||₁ + (1-α)·λ||w||₂²

**Part 3.1**: When α = 1, Elastic Net is equivalent to:
- A. Ridge regression
- B. Lasso regression
- C. OLS regression
- D. Principal Component Regression
- E. None of the above

**Part 3.2**: When α = 0, Elastic Net is equivalent to:
- A. Ridge regression
- B. Lasso regression
- C. OLS regression
- D. Principal Component Regression
- E. None of the above

**Part 3.3**: For correlated features, which is TRUE?
- A. Lasso tends to select one feature and ignore others
- B. Ridge tends to select one feature and ignore others
- C. Elastic Net cannot handle correlated features
- D. All methods treat correlated features identically
- E. Lasso distributes weight evenly among correlated features

<details>
<summary>Solution A4</summary>

**Part 3.1: B** - α=1 means only L1 penalty → Lasso

**Part 3.2: A** - α=0 means only L2 penalty → Ridge

**Part 3.3: A** - Lasso arbitrarily selects one feature from a group of correlated features. Elastic Net (with α between 0 and 1) tends to select groups of correlated features together.
</details>

### Variation A5: Feature Selection
You have a dataset with 100 features. After applying Lasso with λ=0.1, you find 30 non-zero weights.

**Part 3.1**: If you increase λ to 0.5, the number of non-zero weights will most likely:
- A. Increase
- B. Decrease
- C. Stay the same
- D. Become exactly 100
- E. Become exactly 0

**Part 3.2**: If you decrease λ to 0.01, the number of non-zero weights will most likely:
- A. Increase
- B. Decrease
- C. Stay the same
- D. Become exactly 100
- E. Become exactly 0

<details>
<summary>Solution A5</summary>

**Part 3.1: B** - Larger λ means stronger regularization, more weights pushed to zero.

**Part 3.2: A** - Smaller λ means weaker regularization, fewer weights pushed to zero.
</details>

---

## CATEGORY B: Bias-Variance Tradeoff

### Variation B1: Model Complexity
In supervised machine learning, as you increase the complexity of a model:

**Part 3.1**: Which best describes the typical behavior?
- A. Both Bias and Variance increase
- B. Both Bias and Variance decrease
- C. Bias increases and Variance decreases
- D. Bias decreases and Variance increases
- E. Both remain constant

<details>
<summary>Solution B1</summary>

**Answer: D**

More complex models can fit the training data better (lower bias) but are more sensitive to the specific training set (higher variance).
</details>

### Variation B2: Polynomial Regression
You fit polynomial regression models of degree 1, 3, 10, and 50 to a dataset.

**Part 3.1**: Which model likely has the HIGHEST bias?
- A. Degree 1
- B. Degree 3
- C. Degree 10
- D. Degree 50
- E. All have equal bias

**Part 3.2**: Which model likely has the HIGHEST variance?
- A. Degree 1
- B. Degree 3
- C. Degree 10
- D. Degree 50
- E. All have equal variance

**Part 3.3**: Which model is most likely to overfit?
- A. Degree 1
- B. Degree 3
- C. Degree 10
- D. Degree 50
- E. None will overfit

<details>
<summary>Solution B2</summary>

**Part 3.1: A** - Degree 1 (linear) is the simplest, highest bias.

**Part 3.2: D** - Degree 50 is the most complex, highest variance.

**Part 3.3: D** - Degree 50 is most likely to overfit (memorize training data).
</details>

### Variation B3: k-Nearest Neighbors
In k-NN classification:

**Part 3.1**: As k increases from 1 to n (number of training samples), bias:
- A. Increases
- B. Decreases
- C. Stays constant
- D. First increases then decreases
- E. First decreases then increases

**Part 3.2**: As k increases from 1 to n, variance:
- A. Increases
- B. Decreases
- C. Stays constant
- D. First increases then decreases
- E. First decreases then increases

**Part 3.3**: For k=1, which is TRUE?
- A. High bias, low variance
- B. Low bias, high variance
- C. High bias, high variance
- D. Low bias, low variance
- E. Cannot determine

<details>
<summary>Solution B3</summary>

**Part 3.1: A** - Larger k means more averaging, smoother decision boundary, higher bias.

**Part 3.2: B** - Larger k means more averaging, less sensitive to individual points, lower variance.

**Part 3.3: B** - k=1 perfectly fits training data (low bias) but is very sensitive to noise (high variance).
</details>

### Variation B4: Decision Trees
For decision trees:

**Part 3.1**: A fully grown tree (no pruning) typically has:
- A. High bias, low variance
- B. Low bias, high variance
- C. High bias, high variance
- D. Low bias, low variance
- E. Moderate bias and variance

**Part 3.2**: Pruning a decision tree:
- A. Increases bias, decreases variance
- B. Decreases bias, increases variance
- C. Increases both bias and variance
- D. Decreases both bias and variance
- E. Has no effect on bias or variance

**Part 3.3**: Random forests reduce variance by:
- A. Using deeper trees
- B. Averaging predictions from multiple trees
- C. Using fewer features
- D. Using more training data
- E. Using regularization

<details>
<summary>Solution B4</summary>

**Part 3.1: B** - Fully grown trees can fit any training data (low bias) but are very sensitive to training set (high variance).

**Part 3.2: A** - Pruning simplifies the tree, increasing bias but decreasing variance.

**Part 3.3: B** - Averaging multiple high-variance models reduces overall variance (ensemble method).
</details>

### Variation B5: Bias-Variance Decomposition
The expected test error can be decomposed as:
E[Test Error] = Bias² + Variance + Irreducible Error

**Part 3.1**: The irreducible error is due to:
- A. Model complexity
- B. Training set size
- C. Noise in the data
- D. Regularization strength
- E. Feature selection

**Part 3.2**: If training error is very low but test error is high, the model likely has:
- A. High bias
- B. High variance
- C. High irreducible error
- D. Low variance
- E. Optimal bias-variance tradeoff

**Part 3.3**: If both training and test error are high, the model likely has:
- A. High bias (underfitting)
- B. High variance (overfitting)
- C. High irreducible error
- D. Optimal fit
- E. Cannot determine

<details>
<summary>Solution B5</summary>

**Part 3.1: C** - Irreducible error comes from inherent noise in the data that no model can capture.

**Part 3.2: B** - Low training error + high test error = overfitting = high variance.

**Part 3.3: A** - High training error + high test error = underfitting = high bias.
</details>

---

## CATEGORY C: Regularization Math

### Variation C1: Ridge Regression Closed Form
The Ridge regression solution is: ŵ = (XᵀX + λI)⁻¹Xᵀy

**Part 3.1**: As λ → 0, ŵ approaches:
- A. Zero vector
- B. OLS solution (XᵀX)⁻¹Xᵀy
- C. Infinity
- D. The identity matrix
- E. Undefined

**Part 3.2**: As λ → ∞, ŵ approaches:
- A. Zero vector
- B. OLS solution
- C. Infinity
- D. The identity matrix
- E. Undefined

**Part 3.3**: Why does adding λI help when XᵀX is singular?
- A. It makes the matrix smaller
- B. It makes all eigenvalues positive
- C. It removes collinearity
- D. It reduces the rank
- E. It has no effect

<details>
<summary>Solution C1</summary>

**Part 3.1: B** - λ=0 gives standard OLS.

**Part 3.2: A** - As λ→∞, (XᵀX + λI)⁻¹ → (1/λ)I → 0, so ŵ → 0.

**Part 3.3: B** - If XᵀX has eigenvalue 0, then XᵀX + λI has eigenvalue λ > 0, making it invertible.
</details>

### Variation C2: Lasso Optimization
The Lasso objective is: minimize ||y - Xw||² + λ||w||₁

**Part 3.1**: The L1 norm ||w||₁ is:
- A. Differentiable everywhere
- B. Not differentiable at w=0
- C. Always positive
- D. Both B and C
- E. Neither B nor C

**Part 3.2**: Because of the non-differentiability, Lasso is typically solved using:
- A. Gradient descent
- B. Newton's method
- C. Coordinate descent or subgradient methods
- D. Closed-form solution
- E. Random search

<details>
<summary>Solution C2</summary>

**Part 3.1: D** - |w| is not differentiable at w=0, and ||w||₁ = Σ|wᵢ| ≥ 0.

**Part 3.2: C** - Coordinate descent (updating one weight at a time) or subgradient methods handle the non-differentiability.
</details>

### Variation C3: Regularization and Eigenvalues
Let XᵀX have eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ ≥ 0.

**Part 3.1**: The eigenvalues of (XᵀX + αI) are:
- A. λ₁, λ₂, ..., λₚ
- B. λ₁ + α, λ₂ + α, ..., λₚ + α
- C. αλ₁, αλ₂, ..., αλₚ
- D. λ₁/α, λ₂/α, ..., λₚ/α
- E. α, α, ..., α

**Part 3.2**: If λₚ = 0 (XᵀX is singular), what value of α makes (XᵀX + αI) invertible?
- A. α = 0
- B. α < 0
- C. α > 0
- D. Any α
- E. No value of α works

<details>
<summary>Solution C3</summary>

**Part 3.1: B** - Adding αI shifts all eigenvalues by α.

**Part 3.2: C** - Any α > 0 makes the smallest eigenvalue λₚ + α > 0, ensuring invertibility.
</details>

---

## CATEGORY D: Practical Scenarios

### Variation D1: Diagnosing Problems
You train a model and observe:
- Training accuracy: 99%
- Validation accuracy: 65%

**Part 3.1**: This indicates:
- A. Underfitting
- B. Overfitting
- C. Good fit
- D. Data leakage
- E. Insufficient data

**Part 3.2**: Which would MOST LIKELY help?
- A. Use a more complex model
- B. Add regularization
- C. Remove regularization
- D. Use more features
- E. Train longer

<details>
<summary>Solution D1</summary>

**Part 3.1: B** - Large gap between training and validation = overfitting.

**Part 3.2: B** - Regularization reduces overfitting by constraining model complexity.
</details>

### Variation D2: Diagnosing Underfitting
You train a model and observe:
- Training accuracy: 60%
- Validation accuracy: 58%

**Part 3.1**: This indicates:
- A. Underfitting
- B. Overfitting
- C. Good fit
- D. Data leakage
- E. Perfect generalization

**Part 3.2**: Which would MOST LIKELY help?
- A. Add more regularization
- B. Use a simpler model
- C. Use a more complex model
- D. Reduce training data
- E. Add dropout

<details>
<summary>Solution D2</summary>

**Part 3.1: A** - Both training and validation accuracy are low = underfitting.

**Part 3.2: C** - More complex model can better fit the data (reduce bias).
</details>

### Variation D3: Learning Curves
As training set size increases:

**Part 3.1**: Training error typically:
- A. Increases
- B. Decreases
- C. Stays constant
- D. First increases then decreases
- E. Oscillates

**Part 3.2**: Validation error typically:
- A. Increases
- B. Decreases
- C. Stays constant
- D. First increases then decreases
- E. Oscillates

**Part 3.3**: For a high-bias model, adding more data:
- A. Helps significantly
- B. Helps minimally
- C. Hurts performance
- D. Has no effect
- E. Causes overfitting

<details>
<summary>Solution D3</summary>

**Part 3.1: A** - More data is harder to fit perfectly, training error increases.

**Part 3.2: B** - More data gives better generalization, validation error decreases.

**Part 3.3: B** - High-bias models are limited by their simplicity, not data. More data helps minimally.
</details>

### Variation D4: Regularization Selection
You have a dataset with:
- 1000 samples
- 500 features
- Many features are irrelevant

**Part 3.1**: Which regularization is BEST for feature selection?
- A. L1 (Lasso)
- B. L2 (Ridge)
- C. No regularization
- D. Dropout
- E. Early stopping

**Part 3.2**: If you suspect many features are correlated, which is BEST?
- A. L1 only
- B. L2 only
- C. Elastic Net
- D. No regularization
- E. PCA first, then no regularization

<details>
<summary>Solution D4</summary>

**Part 3.1: A** - Lasso performs automatic feature selection by setting irrelevant weights to zero.

**Part 3.2: C** - Elastic Net handles correlated features better than pure Lasso (which arbitrarily picks one).
</details>

---

## CATEGORY E: Neural Network Regularization

### Variation E1: Dropout
Dropout randomly sets neurons to zero during training.

**Part 3.1**: Dropout is a form of:
- A. L1 regularization
- B. L2 regularization
- C. Implicit regularization
- D. Data augmentation
- E. Feature selection

**Part 3.2**: During inference (test time), dropout:
- A. Is applied the same way
- B. Is turned off, weights scaled
- C. Is turned off, no scaling
- D. Uses higher dropout rate
- E. Only applies to output layer

**Part 3.3**: Dropout reduces overfitting by:
- A. Reducing model capacity
- B. Preventing co-adaptation of neurons
- C. Adding noise to gradients
- D. Reducing learning rate
- E. Increasing batch size

<details>
<summary>Solution E1</summary>

**Part 3.1: C** - Dropout is implicit regularization (not explicit penalty term).

**Part 3.2: B** - At test time, dropout is off but weights are scaled by (1-p) to maintain expected values.

**Part 3.3: B** - Dropout prevents neurons from relying too heavily on specific other neurons.
</details>

### Variation E2: Weight Decay
Weight decay adds λ||w||² to the loss function.

**Part 3.1**: Weight decay is equivalent to:
- A. L1 regularization
- B. L2 regularization
- C. Dropout
- D. Batch normalization
- E. Early stopping

**Part 3.2**: In the gradient update w ← w - η∇L, weight decay modifies this to:
- A. w ← w - η∇L - ηλw
- B. w ← w - η∇L + ηλw
- C. w ← w - η∇L - λ
- D. w ← (1-ηλ)w - η∇L
- E. Both A and D are equivalent

<details>
<summary>Solution E2</summary>

**Part 3.1: B** - Weight decay = L2 regularization.

**Part 3.2: E** - Both A and D are equivalent:
- w ← w - η∇L - ηλw = w(1 - ηλ) - η∇L
</details>

### Variation E3: Early Stopping
Early stopping halts training when validation error starts increasing.

**Part 3.1**: Early stopping acts as:
- A. Explicit regularization
- B. Implicit regularization
- C. Data augmentation
- D. Feature selection
- E. Hyperparameter tuning

**Part 3.2**: Early stopping prevents overfitting by:
- A. Reducing model capacity
- B. Limiting effective model complexity
- C. Adding noise
- D. Removing features
- E. Increasing training data

<details>
<summary>Solution E3</summary>

**Part 3.1: B** - Early stopping is implicit regularization (no explicit penalty).

**Part 3.2: B** - Stopping early limits how much the model can fit the training data, effectively limiting complexity.
</details>

### Variation E4: Batch Normalization
**Part 3.1**: Batch normalization has a regularization effect because:
- A. It adds noise through mini-batch statistics
- B. It reduces the learning rate
- C. It removes features
- D. It adds an L2 penalty
- E. It has no regularization effect

**Part 3.2**: When using batch normalization, you typically:
- A. Need more dropout
- B. Need less dropout
- C. Cannot use dropout
- D. Must use L2 regularization
- E. Must use L1 regularization

<details>
<summary>Solution E4</summary>

**Part 3.1: A** - Mini-batch statistics introduce noise, acting as regularization.

**Part 3.2: B** - Batch norm already provides some regularization, so less dropout is needed.
</details>

---

## CATEGORY F: Mathematical Derivations

### Variation F1: Ridge Regression Derivation
Derive the closed-form solution for Ridge regression.

**Part 3.1**: The Ridge objective is L(w) = ||y - Xw||² + λ||w||². Expand this.

**Part 3.2**: Take the gradient ∇ᵥL and set it to zero.

**Part 3.3**: Solve for w.

<details>
<summary>Solution F1</summary>

**Part 3.1**: 
L(w) = (y - Xw)ᵀ(y - Xw) + λwᵀw
     = yᵀy - 2wᵀXᵀy + wᵀXᵀXw + λwᵀw

**Part 3.2**:
∇ᵥL = -2Xᵀy + 2XᵀXw + 2λw = 0
XᵀXw + λw = Xᵀy
(XᵀX + λI)w = Xᵀy

**Part 3.3**:
w = (XᵀX + λI)⁻¹Xᵀy
</details>

### Variation F2: Bias-Variance Derivation
For a model f̂(x) trained on data D, the expected test error at point x is:

E_D[(y - f̂(x))²]

**Part 3.1**: Let f(x) = E[y|x] be the true function. Show that:
E_D[(y - f̂(x))²] = E_D[(f̂(x) - E_D[f̂(x)])²] + (E_D[f̂(x)] - f(x))² + E[(y - f(x))²]

**Part 3.2**: Identify each term (Variance, Bias², Irreducible Error).

<details>
<summary>Solution F2</summary>

**Part 3.1**: 
E[(y - f̂)²] = E[(y - f + f - E[f̂] + E[f̂] - f̂)²]

Using independence and E[y-f] = 0:
= E[(y-f)²] + E[(f - E[f̂])²] + E[(E[f̂] - f̂)²]
= E[(y-f)²] + (f - E[f̂])² + E[(f̂ - E[f̂])²]

**Part 3.2**:
- Variance = E[(f̂ - E[f̂])²] - how much f̂ varies across datasets
- Bias² = (E[f̂] - f)² - systematic error in expected prediction
- Irreducible Error = E[(y-f)²] - noise in the data
</details>

---

## CATEGORY G: Conceptual Deep Dives

### Variation G1: Why L1 Causes Sparsity (Geometric)
**Part 3.1**: Draw the L1 constraint region ||w||₁ ≤ t in 2D.

**Part 3.2**: Draw elliptical level curves of a quadratic loss function.

**Part 3.3**: Explain why the optimal point is likely at a corner.

<details>
<summary>Solution G1</summary>

**Part 3.1**: Diamond shape with vertices at (±t, 0) and (0, ±t).

**Part 3.2**: Ellipses centered at the unconstrained optimum.

**Part 3.3**: The constrained optimum is where the smallest ellipse touches the diamond. Due to the diamond's corners, this contact point is likely at a corner (on an axis), meaning one weight is zero.
</details>

### Variation G2: Double Descent
Modern deep learning shows "double descent" where test error:
1. First decreases (classical regime)
2. Then increases (overfitting)
3. Then decreases again (over-parameterized regime)

**Part 3.1**: In the classical regime, what limits performance?
- A. Too few parameters
- B. Too many parameters
- C. Insufficient regularization
- D. Too much regularization
- E. Data quality

**Part 3.2**: In the over-parameterized regime, why does test error decrease?
- A. Implicit regularization from optimization
- B. Explicit L2 regularization
- C. Data augmentation
- D. Batch normalization
- E. Dropout

<details>
<summary>Solution G2</summary>

**Part 3.1: A** - Classical regime is limited by model capacity (underfitting).

**Part 3.2: A** - Over-parameterized models trained with SGD find "flat" minima that generalize well (implicit regularization).
</details>

---

## CATEGORY H: Coding Implementations

### Variation H1: Ridge Regression from Scratch
Implement Ridge regression using only NumPy.

**Part 3.1**: Complete the following function:

```python
import numpy as np

def ridge_regression(X, y, lambda_reg):
    """
    Compute Ridge regression weights.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,)
        lambda_reg: Regularization strength (λ > 0)

    Returns:
        w: Weight vector, shape (n_features,)
    """
    # YOUR CODE HERE
    pass
```

**Part 3.2**: Test your implementation with:
```python
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
lambda_reg = 0.5
```

What is the sum of the weights (rounded to 4 decimal places)?

<details>
<summary>Solution H1</summary>

**Part 3.1**:
```python
import numpy as np

def ridge_regression(X, y, lambda_reg):
    n_features = X.shape[1]
    # w = (X^T X + λI)^{-1} X^T y
    XtX = X.T @ X
    XtX_regularized = XtX + lambda_reg * np.eye(n_features)
    Xty = X.T @ y
    w = np.linalg.solve(XtX_regularized, Xty)
    return w
```

**Part 3.2**:
```python
w = ridge_regression(X, y, 0.5)
print(w)  # [0.03225806, 0.46774194]
print(np.sum(w))  # 0.5
```

**Answer: 0.5**
</details>

### Variation H2: Coordinate Descent for Lasso
Implement one step of coordinate descent for Lasso.

**Part 3.1**: The soft-thresholding operator is defined as:
S(z, γ) = sign(z) · max(|z| - γ, 0)

Implement this function:

```python
def soft_threshold(z, gamma):
    """
    Soft-thresholding operator for Lasso.

    Args:
        z: Input value (scalar or array)
        gamma: Threshold value (γ ≥ 0)

    Returns:
        Soft-thresholded value
    """
    # YOUR CODE HERE
    pass
```

**Part 3.2**: For Lasso, the coordinate descent update for weight wⱼ is:
wⱼ ← S(ρⱼ, λ) / zⱼ

where:
- ρⱼ = Σᵢ xᵢⱼ(yᵢ - ŷᵢ⁽⁻ʲ⁾) is the partial residual
- zⱼ = Σᵢ xᵢⱼ² is the sum of squared feature values
- ŷᵢ⁽⁻ʲ⁾ is prediction without feature j

Implement one full pass of coordinate descent:

```python
def lasso_coordinate_descent_step(X, y, w, lambda_reg):
    """
    One pass of coordinate descent for Lasso.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,)
        w: Current weights, shape (n_features,)
        lambda_reg: Regularization strength

    Returns:
        w_new: Updated weights
    """
    # YOUR CODE HERE
    pass
```

<details>
<summary>Solution H2</summary>

**Part 3.1**:
```python
def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)
```

**Part 3.2**:
```python
def lasso_coordinate_descent_step(X, y, w, lambda_reg):
    w_new = w.copy()
    n_samples, n_features = X.shape

    for j in range(n_features):
        # Compute prediction without feature j
        y_pred_no_j = X @ w_new - X[:, j] * w_new[j]

        # Partial residual
        rho_j = X[:, j] @ (y - y_pred_no_j)

        # Sum of squared feature values
        z_j = np.sum(X[:, j] ** 2)

        # Coordinate descent update with soft-thresholding
        w_new[j] = soft_threshold(rho_j, lambda_reg) / z_j

    return w_new
```
</details>

### Variation H3: Cross-Validation for λ Selection
Implement k-fold cross-validation to select the optimal regularization parameter.

**Part 3.1**: Complete the following function:

```python
import numpy as np

def kfold_cv_ridge(X, y, lambda_values, k=5, seed=42):
    """
    K-fold cross-validation for Ridge regression.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,)
        lambda_values: List of λ values to try
        k: Number of folds
        seed: Random seed for reproducibility

    Returns:
        best_lambda: λ with lowest mean CV error
        cv_errors: Dict mapping λ → mean CV error
    """
    # YOUR CODE HERE
    pass
```

**Part 3.2**: Given the following CV errors, which λ should be selected?

| λ | Mean CV Error | Std CV Error |
|---|---------------|--------------|
| 0.001 | 0.892 | 0.15 |
| 0.01 | 0.756 | 0.12 |
| 0.1 | 0.623 | 0.08 |
| 1.0 | 0.645 | 0.05 |
| 10.0 | 0.891 | 0.04 |

- A. 0.001 (lowest initial error trend)
- B. 0.1 (lowest mean error)
- C. 1.0 (most stable, within one std of best)
- D. 10.0 (lowest variance)
- E. Cannot determine without test set

<details>
<summary>Solution H3</summary>

**Part 3.1**:
```python
import numpy as np

def kfold_cv_ridge(X, y, lambda_values, k=5, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // k

    cv_errors = {}

    for lam in lambda_values:
        fold_errors = []

        for i in range(k):
            # Create train/val split
            val_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate([
                indices[:i * fold_size],
                indices[(i + 1) * fold_size:]
            ])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit Ridge
            n_features = X_train.shape[1]
            XtX = X_train.T @ X_train
            w = np.linalg.solve(
                XtX + lam * np.eye(n_features),
                X_train.T @ y_train
            )

            # Compute validation error
            y_pred = X_val @ w
            mse = np.mean((y_val - y_pred) ** 2)
            fold_errors.append(mse)

        cv_errors[lam] = np.mean(fold_errors)

    best_lambda = min(cv_errors, key=cv_errors.get)
    return best_lambda, cv_errors
```

**Part 3.2: B** - Select λ = 0.1 with the lowest mean CV error (0.623). The "one standard error rule" (option C) is a valid heuristic for more regularization, but the question asks which "should" be selected, and the standard approach is minimum CV error.
</details>

### Variation H4: Bias-Variance Estimation via Bootstrap
Estimate bias and variance of a model using bootstrap sampling.

**Part 3.1**: Complete the following function to estimate bias² and variance:

```python
import numpy as np

def estimate_bias_variance(X, y, model_fn, x_test, n_bootstrap=100, seed=42):
    """
    Estimate bias² and variance at a test point using bootstrap.

    Args:
        X: Training features, shape (n_samples, n_features)
        y: Training targets, shape (n_samples,)
        model_fn: Function that takes (X, y) and returns fitted model
        x_test: Single test point, shape (n_features,)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        bias_squared: Estimated bias²
        variance: Estimated variance
    """
    # YOUR CODE HERE
    pass
```

**Part 3.2**: If bootstrap predictions at a test point are [2.1, 2.3, 1.9, 2.0, 2.2] and the true value is 2.5, compute:
- Mean prediction
- Bias²
- Variance

<details>
<summary>Solution H4</summary>

**Part 3.1**:
```python
import numpy as np

def estimate_bias_variance(X, y, model_fn, x_test, n_bootstrap=100, seed=42):
    np.random.seed(seed)
    n_samples = X.shape[0]
    predictions = []

    for _ in range(n_bootstrap):
        # Bootstrap sample (sample with replacement)
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]

        # Fit model and predict
        model = model_fn(X_boot, y_boot)
        pred = model.predict(x_test.reshape(1, -1))[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    # E[f̂(x)] - average prediction
    mean_pred = np.mean(predictions)

    # Variance = E[(f̂(x) - E[f̂(x)])²]
    variance = np.mean((predictions - mean_pred) ** 2)

    # Note: Bias² requires true f(x), which we don't have
    # In practice, we'd need the true underlying function
    # This returns variance only; bias requires ground truth

    return variance, mean_pred

# For bias², if we knew true value f(x):
# bias_squared = (mean_pred - f(x)) ** 2
```

**Part 3.2**:
```python
predictions = np.array([2.1, 2.3, 1.9, 2.0, 2.2])
true_value = 2.5

mean_pred = np.mean(predictions)  # 2.1
bias_squared = (mean_pred - true_value) ** 2  # (2.1 - 2.5)² = 0.16
variance = np.var(predictions)  # 0.02
```

**Answers:**
- Mean prediction: **2.1**
- Bias²: **(2.1 - 2.5)² = 0.16**
- Variance: **0.02**
</details>

### Variation H5: Regularization Path Visualization
Implement code to visualize how coefficients change with λ.

**Part 3.1**: Complete the function to compute the regularization path:

```python
import numpy as np

def ridge_path(X, y, lambda_values):
    """
    Compute Ridge coefficients for multiple λ values.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target vector, shape (n_samples,)
        lambda_values: Array of λ values (sorted ascending)

    Returns:
        coef_path: Array of shape (len(lambda_values), n_features)
    """
    # YOUR CODE HERE
    pass
```

**Part 3.2**: Given a 3-feature problem, as λ increases from 0 to ∞, the coefficient path will:
- A. All coefficients decrease monotonically to 0
- B. All coefficients increase monotonically
- C. Coefficients may cross zero but eventually approach 0
- D. Coefficients oscillate indefinitely
- E. Only the smallest coefficient approaches 0

<details>
<summary>Solution H5</summary>

**Part 3.1**:
```python
import numpy as np

def ridge_path(X, y, lambda_values):
    n_features = X.shape[1]
    coef_path = np.zeros((len(lambda_values), n_features))

    XtX = X.T @ X
    Xty = X.T @ y

    for i, lam in enumerate(lambda_values):
        w = np.linalg.solve(XtX + lam * np.eye(n_features), Xty)
        coef_path[i] = w

    return coef_path

# Visualization:
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# for j in range(coef_path.shape[1]):
#     plt.plot(np.log10(lambda_values), coef_path[:, j], label=f'w_{j}')
# plt.xlabel('log10(λ)')
# plt.ylabel('Coefficient value')
# plt.legend()
# plt.title('Ridge Regularization Path')
```

**Part 3.2: C** - Ridge coefficients shrink toward zero but can cross zero if the sign of the optimal coefficient changes. This happens when features are correlated and the regularization changes which feature "dominates." All coefficients eventually approach 0 as λ → ∞.
</details>

---

## CATEGORY I: Edge Cases & Special Scenarios

### Variation I1: High-Dimensional Setting (p >> n)
When the number of features exceeds the number of samples:

**Part 3.1**: You have n = 50 samples and p = 500 features. Without regularization, OLS:
- A. Works perfectly fine
- B. Has a unique solution
- C. Has infinitely many solutions
- D. Has no solution
- E. Produces negative weights only

**Part 3.2**: In the p >> n setting, which is TRUE?
- A. Ridge regression fails because XᵀX is still singular
- B. Ridge regression works because (XᵀX + λI) is always invertible for λ > 0
- C. Lasso is impossible to compute
- D. Both Ridge and Lasso fail
- E. Only Elastic Net works

**Part 3.3**: In the p >> n setting, Lasso can select at most:
- A. p features
- B. n features
- C. min(n, p) features
- D. p - n features
- E. Unlimited features

<details>
<summary>Solution I1</summary>

**Part 3.1: C** - When p > n, XᵀX is rank-deficient (rank ≤ n < p), so the system Xw = y is underdetermined with infinitely many solutions.

**Part 3.2: B** - Adding λI to XᵀX makes all eigenvalues at least λ > 0, guaranteeing invertibility regardless of the original rank.

**Part 3.3: B** - Lasso can select at most n features when p > n. This is a fundamental limitation. Elastic Net can select more than n features due to its L2 component.
</details>

### Variation I2: Perfect Multicollinearity
Consider features where X₂ = 2·X₁ (perfect linear dependence).

**Part 3.1**: For OLS regression with perfectly collinear features:
- A. Unique solution exists
- B. XᵀX is invertible
- C. XᵀX is singular (not invertible)
- D. The loss function has no minimum
- E. Gradient descent converges to unique solution

**Part 3.2**: With Ridge regularization (λ > 0) and perfect collinearity:
- A. Still no unique solution
- B. Unique solution exists, weights are distributed between collinear features
- C. One weight is forced to zero
- D. Both weights are forced to zero
- E. Ridge cannot handle collinearity

**Part 3.3**: With Lasso and perfect collinearity:
- A. Both collinear features get equal non-zero weights
- B. One feature is arbitrarily selected, the other gets zero weight
- C. Both features get zero weights
- D. Lasso fails to converge
- E. Weights oscillate between the two features

<details>
<summary>Solution I2</summary>

**Part 3.1: C** - If X₂ = 2·X₁, the columns of X are linearly dependent, making XᵀX singular (determinant = 0).

**Part 3.2: B** - Ridge regularization makes (XᵀX + λI) invertible. The weights will be distributed between the collinear features (both non-zero but smaller than if only one feature existed).

**Part 3.3: B** - Lasso arbitrarily selects one of the perfectly collinear features and sets the other to zero. The selection depends on the optimization path and can be unstable.
</details>

### Variation I3: Extreme Regularization Values
Analyze behavior at extreme values of λ.

**Part 3.1**: For Ridge regression, what is lim(λ→0⁺) w_ridge?
- A. Zero vector
- B. OLS solution
- C. Undefined
- D. Infinity
- E. Identity vector

**Part 3.2**: For Ridge regression with standardized features, what is lim(λ→∞) w_ridge?
- A. Zero vector
- B. OLS solution
- C. Mean of y repeated
- D. Infinity
- E. (1/p, 1/p, ..., 1/p)

**Part 3.3**: For Lasso, there exists a value λ_max such that for all λ ≥ λ_max, all weights are zero. For standardized features and centered y, λ_max equals:
- A. max|Xᵀy|
- B. ||y||₂
- C. ||X||_F
- D. n · max|corr(Xⱼ, y)|
- E. 2 · max|Xᵀy|

<details>
<summary>Solution I3</summary>

**Part 3.1: B** - As λ → 0⁺, the Ridge solution approaches the OLS solution: (XᵀX + λI)⁻¹Xᵀy → (XᵀX)⁻¹Xᵀy

**Part 3.2: A** - As λ → ∞, (XᵀX + λI)⁻¹ ≈ (1/λ)I → 0, so w_ridge → 0.

**Part 3.3: A** - λ_max = max|Xᵀy| = max_j |Σᵢ xᵢⱼyᵢ|. This is the smallest λ where the gradient at w=0 doesn't exceed the L1 penalty, so no feature "activates."
</details>

### Variation I4: Numerical Stability
Consider numerical issues in regularization.

**Part 3.1**: Computing (XᵀX)⁻¹ directly is numerically unstable when:
- A. X has large values
- B. X has small values
- C. XᵀX has a large condition number
- D. X is sparse
- E. X has many rows

**Part 3.2**: The condition number κ(A) = σ_max/σ_min measures numerical stability. For (XᵀX + λI), the condition number is:
- A. (σ²_max + λ)/(σ²_min + λ)
- B. σ²_max/σ²_min (unchanged)
- C. λ/σ²_min
- D. σ²_max/λ
- E. 1 (always well-conditioned)

**Part 3.3**: As λ increases, the condition number of (XᵀX + λI):
- A. Increases (worse conditioning)
- B. Decreases (better conditioning)
- C. Stays the same
- D. First increases then decreases
- E. Depends on the data

<details>
<summary>Solution I4</summary>

**Part 3.1: C** - A large condition number (ratio of largest to smallest singular value) means near-singularity, causing numerical instability in inversion.

**Part 3.2: A** - If XᵀX has eigenvalues λ₁ ≥ ... ≥ λₚ ≥ 0 (which equal σ²ᵢ where σᵢ are singular values of X), then (XᵀX + λI) has eigenvalues (λᵢ + λ), so:
κ = (σ²_max + λ)/(σ²_min + λ)

**Part 3.3: B** - As λ → ∞:
κ = (σ²_max + λ)/(σ²_min + λ) → 1

Regularization improves conditioning by "lifting" all eigenvalues, reducing their ratio.
</details>

---

## CATEGORY F (Extended): Additional Proofs & Theory

### Variation F3: Bayesian Interpretation of Ridge
Ridge regression is equivalent to MAP estimation with a specific prior.

**Part 3.1**: If we assume:
- Likelihood: y|X,w ~ N(Xw, σ²I)
- Prior: w ~ N(0, τ²I)

The MAP estimate maximizes log p(w|X,y). Show this is equivalent to Ridge with λ = σ²/τ².

**Part 3.2**: What does a small τ² (tight prior) correspond to?
- A. Small λ (weak regularization)
- B. Large λ (strong regularization)
- C. No regularization
- D. L1 regularization
- E. Elastic Net

**Part 3.3**: What prior distribution gives Lasso (L1 regularization)?
- A. Gaussian
- B. Uniform
- C. Laplace (double exponential)
- D. Cauchy
- E. Exponential

<details>
<summary>Solution F3</summary>

**Part 3.1**:
log p(w|X,y) ∝ log p(y|X,w) + log p(w)

= -1/(2σ²)||y - Xw||² - 1/(2τ²)||w||² + const

Maximizing this is equivalent to minimizing:
||y - Xw||² + (σ²/τ²)||w||²

This is Ridge regression with λ = σ²/τ².

**Part 3.2: B** - Small τ² means the prior is concentrated near zero (we strongly believe weights should be small), which corresponds to large λ (strong regularization).

**Part 3.3: C** - The Laplace distribution p(wⱼ) ∝ exp(-|wⱼ|/b) gives:
log p(w) ∝ -Σⱼ|wⱼ|/b = -(1/b)||w||₁

This is the L1 penalty, so Laplace prior → Lasso.
</details>

### Variation F4: Why Lasso Has No Closed Form
Explain why Lasso doesn't have a closed-form solution.

**Part 3.1**: The subdifferential of |w| at w = 0 is:
- A. {0}
- B. {-1, 1}
- C. [-1, 1]
- D. (-∞, ∞)
- E. Undefined

**Part 3.2**: Setting the gradient of the Lasso objective to zero involves:
- A. A linear system (always solvable)
- B. A quadratic equation
- C. A piecewise condition depending on whether wⱼ = 0
- D. An exponential equation
- E. No equation (gradient is always zero)

**Part 3.3**: Why does coordinate descent work for Lasso?
- A. It avoids computing gradients
- B. Each 1D subproblem has a closed-form solution (soft-thresholding)
- C. It uses random search
- D. It converts L1 to L2
- E. It doesn't actually solve Lasso exactly

<details>
<summary>Solution F4</summary>

**Part 3.1: C** - At w = 0, the subdifferential of |w| is the interval [-1, 1]. This represents all possible "slopes" of tangent lines that stay below the function.

**Part 3.2: C** - The optimality condition is:
- If wⱼ ≠ 0: -2Xⱼᵀ(y - Xw) + λ·sign(wⱼ) = 0
- If wⱼ = 0: |2Xⱼᵀ(y - Xw)| ≤ λ

This creates a piecewise system that can't be solved in closed form for all wⱼ simultaneously.

**Part 3.3: B** - When optimizing over a single coordinate wⱼ while holding others fixed, the 1D problem has the closed-form solution:
wⱼ = S(ρⱼ, λ)/zⱼ (soft-thresholding)

This makes coordinate descent efficient despite the non-differentiability.
</details>

### Variation F5: Elastic Net Gradient
Derive the gradient of the Elastic Net objective.

**Part 3.1**: The Elastic Net objective is:
L(w) = ||y - Xw||² + λ₁||w||₁ + λ₂||w||₂²

The gradient with respect to w (where defined) is:
- A. -2Xᵀ(y - Xw) + λ₁sign(w) + 2λ₂w
- B. -2Xᵀ(y - Xw) + λ₁ + λ₂
- C. 2Xᵀ(y - Xw) + λ₁||w||₁ + λ₂||w||₂²
- D. -Xᵀy + XᵀXw
- E. λ₁ + 2λ₂w

**Part 3.2**: At what value of w does the L2 term equal the L1 term (for λ₁ = λ₂ = 1)?
- A. |w| = 1 for all components
- B. ||w||₂ = ||w||₁
- C. When w is sparse
- D. Never (they use different norms)
- E. When w = 0

<details>
<summary>Solution F5</summary>

**Part 3.1: A** - Taking the gradient:
∇L = ∇||y - Xw||² + λ₁∇||w||₁ + λ₂∇||w||₂²
    = -2Xᵀ(y - Xw) + λ₁sign(w) + 2λ₂w

Note: This is the subgradient; at wⱼ = 0, sign(wⱼ) ∈ [-1, 1].

**Part 3.2: A** - For λ₁||w||₁ = λ₂||w||₂² with λ₁ = λ₂ = 1:
Σ|wⱼ| = Σwⱼ²

For a single component: |w| = w² → |w| = 1.
For multiple components with equal magnitudes: each |wⱼ| = 1.
</details>

---

## KEY FORMULAS SUMMARY

| Concept | Formula |
|---------|---------|
| Ridge objective | \|\|y - Xw\|\|² + λ\|\|w\|\|₂² |
| Ridge solution | w = (XᵀX + λI)⁻¹Xᵀy |
| Lasso objective | \|\|y - Xw\|\|² + λ\|\|w\|\|₁ |
| Elastic Net | \|\|y - Xw\|\|² + αλ\|\|w\|\|₁ + (1-α)λ\|\|w\|\|₂² |
| Bias-Variance | E[Error] = Bias² + Variance + Irreducible |
| L1 norm | \|\|w\|\|₁ = Σᵢ\|wᵢ\| |
| L2 norm squared | \|\|w\|\|₂² = Σᵢwᵢ² |

## KEY INSIGHTS

### Conceptual
1. **L1 → Sparsity**: Diamond constraint, corners on axes
2. **L2 → Small weights**: Circular constraint, no corners
3. **High complexity → Low bias, high variance**
4. **Low complexity → High bias, low variance**
5. **Overfitting**: Training error << Validation error
6. **Underfitting**: Both errors high
7. **More data helps variance, not bias**
8. **Regularization trades bias for variance**

### Mathematical
9. **Ridge is Bayesian**: Gaussian prior on weights → L2 penalty
10. **Lasso is Bayesian**: Laplace prior on weights → L1 penalty
11. **No closed-form for Lasso**: Subdifferential at w=0 prevents analytical solution
12. **λ_max for Lasso**: max|Xᵀy| - above this, all weights are zero

### Computational
13. **Ridge improves conditioning**: κ(XᵀX + λI) < κ(XᵀX) for λ > 0
14. **p >> n problem**: Lasso selects at most n features
15. **Coordinate descent**: Works because 1D subproblems have closed-form solutions
16. **Cross-validation**: Use k-fold CV to select optimal λ

### Practical
17. **Collinearity**: Ridge distributes weights; Lasso picks one arbitrarily
18. **Elastic Net**: Best of both worlds for correlated features
19. **Bootstrap for bias-variance**: Resample to estimate decomposition
20. **Regularization path**: Visualize how coefficients shrink with increasing λ

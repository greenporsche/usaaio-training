# Backpropagation Exercises

**Topic**: Chain rule, gradient computation, backward pass by hand
**Difficulty**: Intermediate → Advanced

---

## Exercise 1: Single Neuron Backward Pass

Given a single neuron with sigmoid activation:

$$\hat{y} = \sigma(wx + b), \quad L = (\hat{y} - y)^2$$

With $w = 0.5$, $b = -0.1$, $x = 2.0$, $y = 1.0$:

1. Compute the forward pass: $z$, $\hat{y}$, $L$
2. Compute all gradients: $\frac{\partial L}{\partial \hat{y}}$, $\frac{\partial L}{\partial z}$, $\frac{\partial L}{\partial w}$, $\frac{\partial L}{\partial b}$

<details>
<summary>Solution</summary>

**Forward**:
- $z = wx + b = 0.5(2.0) + (-0.1) = 0.9$
- $\hat{y} = \sigma(0.9) = \frac{1}{1 + e^{-0.9}} \approx 0.7109$
- $L = (0.7109 - 1.0)^2 = (-0.2891)^2 \approx 0.0836$

**Backward**:
- $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2(0.7109 - 1.0) = -0.5782$
- $\sigma'(z) = \sigma(z)(1-\sigma(z)) = 0.7109 \times 0.2891 = 0.2055$
- $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z) = -0.5782 \times 0.2055 = -0.1188$
- $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot x = -0.1188 \times 2.0 = -0.2376$
- $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot 1 = -0.1188$

**Key insight**: All gradients are negative, meaning we should increase both $w$ and $b$ to reduce the loss. This makes sense: the output (0.71) is below the target (1.0), so we need to increase the pre-activation.
</details>

---

## Exercise 2: Two-Layer Network Backward Pass

Network: 2→2→1, ReLU hidden, no output activation, MSE loss.

```
W₁ = [[0.1, 0.3],    b₁ = [0.0, 0.0]
      [0.2, 0.4]]

W₂ = [[0.5, 0.6]]    b₂ = [0.0]

x = [1.0, 2.0],  y = [1.0]
```

1. Compute the full forward pass
2. Compute $\frac{\partial L}{\partial W_2}$ and $\frac{\partial L}{\partial b_2}$
3. Compute $\frac{\partial L}{\partial W_1}$ and $\frac{\partial L}{\partial b_1}$

<details>
<summary>Solution</summary>

**Forward**:
- $z^{[1]} = W_1 x + b_1 = [0.1(1)+0.3(2), 0.2(1)+0.4(2)] = [0.7, 1.0]$
- $a^{[1]} = \text{ReLU}([0.7, 1.0]) = [0.7, 1.0]$ (both positive)
- $z^{[2]} = W_2 a^{[1]} + b_2 = 0.5(0.7) + 0.6(1.0) + 0 = 0.95$
- $\hat{y} = 0.95$
- $L = (0.95 - 1.0)^2 = 0.0025$

**Backward from output**:
- $\frac{\partial L}{\partial \hat{y}} = 2(0.95 - 1.0) = -0.10$
- $\delta^{[2]} = \frac{\partial L}{\partial z^{[2]}} = -0.10$ (no activation on output)

**Gradients for layer 2**:
- $\frac{\partial L}{\partial W_2} = \delta^{[2]} \cdot (a^{[1]})^T = -0.10 \cdot [0.7, 1.0] = [-0.07, -0.10]$
- $\frac{\partial L}{\partial b_2} = \delta^{[2]} = -0.10$

**Propagate to hidden layer**:
- $\frac{\partial L}{\partial a^{[1]}} = W_2^T \cdot \delta^{[2]} = [0.5, 0.6]^T \cdot (-0.10) = [-0.05, -0.06]$
- ReLU derivative: both $z^{[1]}$ values are positive, so $\sigma'(z^{[1]}) = [1, 1]$
- $\delta^{[1]} = [-0.05, -0.06] \odot [1, 1] = [-0.05, -0.06]$

**Gradients for layer 1**:
- $\frac{\partial L}{\partial W_1} = \delta^{[1]} \cdot x^T = \begin{bmatrix}-0.05 \\ -0.06\end{bmatrix} \begin{bmatrix}1.0 & 2.0\end{bmatrix} = \begin{bmatrix}-0.05 & -0.10 \\ -0.06 & -0.12\end{bmatrix}$
- $\frac{\partial L}{\partial b_1} = [-0.05, -0.06]$

**Key insight**: The gradient of $W_1$ is the outer product of $\delta^{[1]}$ and $x$. Each weight $W_{ij}$ gets a gradient proportional to the error signal ($\delta_i$) times the input ($x_j$). This is the Hebb-like rule: connections between active neurons and error signals get the strongest updates.
</details>

---

## Exercise 3: ReLU Gradient Gating

Consider a 1→3→1 network with ReLU:

```
W₁ = [[2.0], [-1.0], [0.5]]    b₁ = [-1.0, 0.5, -0.3]
W₂ = [[1.0, 1.0, 1.0]]         b₂ = [0.0]

Input: x = [0.8]
```

1. Compute the forward pass, noting which neurons are active
2. Compute the backward pass
3. Which weight gradients are exactly zero? Why?

<details>
<summary>Solution</summary>

**Forward**:
- $z^{[1]} = [2(0.8)-1, -1(0.8)+0.5, 0.5(0.8)-0.3] = [0.6, -0.3, 0.1]$
- $a^{[1]} = \text{ReLU}([0.6, -0.3, 0.1]) = [0.6, 0.0, 0.1]$
- Neuron 2 is DEAD (pre-activation $-0.3 < 0$)
- $z^{[2]} = 1.0(0.6) + 1.0(0.0) + 1.0(0.1) + 0 = 0.7$

Assume target $y = 1.0$, MSE loss:
- $L = (0.7 - 1.0)^2 = 0.09$

**Backward**:
- $\delta^{[2]} = 2(0.7-1.0) = -0.6$
- $\frac{\partial L}{\partial a^{[1]}} = W_2^T \delta^{[2]} = [1,1,1]^T \cdot (-0.6) = [-0.6, -0.6, -0.6]$
- ReLU gradient: $\sigma'(z^{[1]}) = [1, 0, 1]$ (neuron 2 blocked!)
- $\delta^{[1]} = [-0.6, -0.6, -0.6] \odot [1, 0, 1] = [-0.6, 0.0, -0.6]$

**Gradients**:
- $\frac{\partial L}{\partial W_1} = \delta^{[1]} \cdot x^T = [-0.6(0.8), 0.0(0.8), -0.6(0.8)] = [-0.48, 0.0, -0.48]$
- $\frac{\partial L}{\partial b_1} = [-0.6, 0.0, -0.6]$

**Zero gradients**: The weight $W_{1,2}$ (connecting input to neuron 2) and bias $b_{1,2}$ have ZERO gradients because neuron 2 was killed by ReLU. No matter what the error signal is, the ReLU gate blocks it.

**Key insight**: Dead neurons (negative pre-activation) have zero gradients through ReLU, meaning they cannot recover through gradient descent. This is the "dying ReLU" problem. If a neuron becomes dead for ALL training inputs, it is permanently dead.
</details>

---

## Exercise 4: Softmax + Cross-Entropy Gradient

Given logits $z = [2.0, 1.0, 0.1]$ and true label $y = 0$ (first class):

1. Compute the softmax probabilities $p = \text{softmax}(z)$
2. Compute the cross-entropy loss $L = -\log(p_y)$
3. Compute the gradient $\frac{\partial L}{\partial z}$
4. Verify the elegant formula: $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y]$

<details>
<summary>Solution</summary>

**Softmax**:
- $e^z = [e^{2.0}, e^{1.0}, e^{0.1}] = [7.389, 2.718, 1.105]$
- Sum: $7.389 + 2.718 + 1.105 = 11.212$
- $p = [7.389/11.212, 2.718/11.212, 1.105/11.212] = [0.6590, 0.2424, 0.0986]$

**Cross-entropy loss**:
- $L = -\log(p_0) = -\log(0.6590) = 0.4170$

**Gradient** (using the elegant formula):
- True label is class 0, so $\mathbb{1}[i=0] = [1, 0, 0]$
- $\frac{\partial L}{\partial z} = p - \text{one\_hot}(y) = [0.6590-1, 0.2424-0, 0.0986-0] = [-0.3410, 0.2424, 0.0986]$

**Verification**: The gradient for the correct class is negative (push logit up), and for incorrect classes is positive (push logits down). Magnitudes are proportional to the softmax probabilities.

Note that the gradients sum to zero: $-0.3410 + 0.2424 + 0.0986 = 0$. This always holds for softmax + CE.

**Key insight**: The $p - y$ formula is one of the most beautiful results in deep learning. The complex derivatives of softmax and log-likelihood cancel to give this simple expression. This is why softmax + cross-entropy is the standard pairing — clean gradients and numerical stability.
</details>

---

## Exercise 5: Vanishing Gradient Calculation

Consider a 5-layer network where every layer uses sigmoid activation and has a single neuron (for simplicity). All weights are $w = 1.0$ and all biases are $b = 0$.

1. Compute the forward pass for input $x = 0$
2. What is $\sigma'(0)$?
3. Compute $\frac{\partial L}{\partial w_1}$ (the gradient of the first layer's weight), assuming $\frac{\partial L}{\partial z^{[5]}} = 1$
4. What would this gradient be in a 20-layer version?

<details>
<summary>Solution</summary>

**Forward pass** ($x = 0$, all $w=1, b=0$):
- $z^{[1]} = 1 \cdot 0 + 0 = 0$, $a^{[1]} = \sigma(0) = 0.5$
- $z^{[2]} = 1 \cdot 0.5 + 0 = 0.5$, $a^{[2]} = \sigma(0.5) = 0.6225$
- $z^{[3]} = 0.6225$, $a^{[3]} = \sigma(0.6225) = 0.6506$
- $z^{[4]} = 0.6506$, $a^{[4]} = \sigma(0.6506) = 0.6572$
- $z^{[5]} = 0.6572$, $a^{[5]} = \sigma(0.6572) = 0.6588$

$\sigma'(z) = \sigma(z)(1-\sigma(z))$:
- $\sigma'(z^{[1]}) = 0.5 \times 0.5 = 0.25$
- $\sigma'(z^{[2]}) = 0.6225 \times 0.3775 = 0.2350$
- $\sigma'(z^{[3]}) = 0.6506 \times 0.3494 = 0.2273$
- $\sigma'(z^{[4]}) = 0.6572 \times 0.3428 = 0.2253$
- $\sigma'(z^{[5]}) = 0.6588 \times 0.3412 = 0.2248$

**Gradient** at layer 1 (chain rule through 5 layers):

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z^{[5]}} \cdot \prod_{l=1}^{4} w \cdot \sigma'(z^{[l]}) \cdot \sigma'(z^{[5]}) \cdot a^{[0]}$$

Wait — let me be more careful. With each layer being one neuron:

$$\frac{\partial z^{[5]}}{\partial w_1} = \sigma'(z^{[5]}) \cdot w_5 \cdot \sigma'(z^{[4]}) \cdot w_4 \cdot \sigma'(z^{[3]}) \cdot w_3 \cdot \sigma'(z^{[2]}) \cdot w_2 \cdot \sigma'(z^{[1]}) \cdot x$$

Hmm, $x=0$ so this would be zero. Let's use $x = 1$ instead:

With $x=1$: all weights and biases same.
- $z^{[1]} = 1$, $a^{[1]} = \sigma(1) = 0.7311$, $\sigma'(1) = 0.7311 \times 0.2689 = 0.1966$

The key point: the gradient at layer 1 involves multiplying 4 sigmoid derivatives together (each $\leq 0.25$):

$$\frac{\partial L}{\partial w_1} \propto \prod_{l=2}^{5} \sigma'(z^{[l]}) \leq 0.25^4 = 0.0039$$

**For 20 layers**: gradient $\propto 0.25^{19} \approx 2.7 \times 10^{-12}$

The gradient is essentially zero — the first layer cannot learn.

**Key insight**: Sigmoid's maximum derivative is 0.25, so each layer reduces the gradient by at least 4x. After 20 layers, the gradient is $10^{-12}$, which is numerically zero in float32. This is why sigmoid networks deeper than ~5 layers are practically untrainable, and why ReLU (with derivative 1 for positive inputs) revolutionized deep learning.
</details>

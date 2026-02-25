# Activation Function Exercises

**Topic**: ReLU, sigmoid, tanh, GELU, derivatives, vanishing/dying gradients
**Difficulty**: Foundational → Intermediate

---

## Exercise 1: Compute Derivatives by Hand

For each activation, compute $\sigma(x)$ and $\sigma'(x)$ at the given values:

1. Sigmoid at $x = 0$, $x = 2$, $x = -3$
2. Tanh at $x = 0$, $x = 1$
3. ReLU at $x = -2$, $x = 0$, $x = 3$
4. Leaky ReLU ($\alpha = 0.01$) at $x = -5$, $x = 2$

<details>
<summary>Solution</summary>

**1. Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$, $\sigma'(x) = \sigma(x)(1-\sigma(x))$

| $x$ | $\sigma(x)$ | $\sigma'(x)$ |
|---|---|---|
| 0 | 0.5 | $0.5 \times 0.5 = 0.25$ |
| 2 | $\frac{1}{1+e^{-2}} = \frac{1}{1.1353} = 0.8808$ | $0.8808 \times 0.1192 = 0.1050$ |
| -3 | $\frac{1}{1+e^{3}} = \frac{1}{21.086} = 0.0474$ | $0.0474 \times 0.9526 = 0.0452$ |

**2. Tanh**: $\tanh'(x) = 1 - \tanh^2(x)$

| $x$ | $\tanh(x)$ | $\tanh'(x)$ |
|---|---|---|
| 0 | 0 | $1 - 0 = 1$ |
| 1 | 0.7616 | $1 - 0.7616^2 = 1 - 0.5800 = 0.4200$ |

**3. ReLU**: $\max(0, x)$

| $x$ | ReLU($x$) | ReLU$'(x)$ |
|---|---|---|
| -2 | 0 | 0 |
| 0 | 0 | 0 (by convention) |
| 3 | 3 | 1 |

**4. Leaky ReLU** ($\alpha = 0.01$):

| $x$ | LeakyReLU($x$) | LeakyReLU$'(x)$ |
|---|---|---|
| -5 | $0.01 \times (-5) = -0.05$ | 0.01 |
| 2 | 2 | 1 |

**Key insight**: Notice how sigmoid's derivative at $x = -3$ is only 0.045 — already shrinking gradients even with one layer. At $x=0$ (the maximum), it's only 0.25. ReLU's derivative is either 0 or 1, making gradient flow much simpler.
</details>

---

## Exercise 2: Activation Range Analysis

For each activation function, answer:
1. What is the output range?
2. Is the output zero-centered?
3. What is the maximum gradient?
4. Can the gradient vanish? Under what conditions?

Functions: Sigmoid, Tanh, ReLU, GELU, Swish

<details>
<summary>Solution</summary>

| Activation | Range | Zero-centered? | Max gradient | Vanishing? |
|---|---|---|---|---|
| Sigmoid | $(0, 1)$ | No (always positive) | 0.25 (at $x=0$) | Yes, for $|x| \gg 0$ |
| Tanh | $(-1, 1)$ | Yes | 1.0 (at $x=0$) | Yes, for $|x| \gg 0$ |
| ReLU | $[0, \infty)$ | No (always $\geq 0$) | 1.0 (for $x > 0$) | Yes, for $x < 0$ (dead) |
| GELU | $\approx(-0.17, \infty)$ | Approximately yes | $\approx 1.08$ | Soft vanishing for $x \ll 0$ |
| Swish | $\approx(-0.28, \infty)$ | Approximately yes | $\approx 1.10$ | Soft vanishing for $x \ll 0$ |

**Key insight**: No activation is perfect. Sigmoid and tanh vanish for extreme inputs. ReLU is hard zero for negative inputs. GELU and Swish offer smooth transitions — their negative tails approach zero smoothly, allowing tiny but nonzero gradients. This is why they work well in transformers where gradient flow through many layers is critical.
</details>

---

## Exercise 3: Dying ReLU Diagnosis

A 3-layer MLP with ReLU activations is trained on a classification task. After 100 epochs, you observe:

```python
# Check neuron activity
with torch.no_grad():
    activations = model.fc1(X_train)
    active_fraction = (activations > 0).float().mean(dim=0)
    print(f"Dead neurons (0% active): {(active_fraction == 0).sum()}")
    print(f"Rarely active (<5%): {(active_fraction < 0.05).sum()}")
    print(f"Active fraction per neuron: min={active_fraction.min():.3f}, "
          f"mean={active_fraction.mean():.3f}")
```

Output:
```
Dead neurons (0% active): 47
Rarely active (<5%): 83
Active fraction per neuron: min=0.000, mean=0.412
```

The hidden layer has 256 neurons. Answer:
1. What percentage of neurons are completely dead?
2. What could have caused this?
3. Propose three solutions.

<details>
<summary>Solution</summary>

1. **Dead neurons**: 47/256 = 18.4% completely dead. 83/256 = 32.4% rarely active.

2. **Causes**:
   - **Learning rate too high**: Large gradient updates pushed biases very negative, making pre-activations permanently negative
   - **Poor initialization**: If initial weights produce mostly negative pre-activations, neurons start dead
   - **Large negative gradients**: A single bad batch can push a neuron into the dead zone permanently

3. **Solutions**:
   - **Use Leaky ReLU**: $\max(0.01x, x)$ — ensures a small gradient even for negative pre-activations, preventing permanent death
   - **Use He initialization**: $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$ — specifically designed for ReLU to keep pre-activations centered
   - **Reduce learning rate**: Smaller updates reduce the risk of catastrophic weight changes. Or use learning rate warmup.
   - **Use batch normalization**: Normalizes pre-activations to have zero mean, centering them around the active region of ReLU

**Key insight**: 18% dead neurons is significant but not catastrophic — the network may still function. However, it represents wasted capacity. For competition, using Leaky ReLU or GELU is a simple, zero-cost way to eliminate this failure mode entirely.
</details>

---

## Exercise 4: Softmax Properties

Given logits $z = [3, 1, -2, 0]$:

1. Compute $\text{softmax}(z)$
2. Compute $\text{softmax}(z + 10)$ (add constant 10 to all logits)
3. Compute $\text{softmax}(2z)$ (multiply all logits by 2)
4. What properties of softmax do parts 2 and 3 illustrate?

<details>
<summary>Solution</summary>

**1.** $e^z = [e^3, e^1, e^{-2}, e^0] = [20.086, 2.718, 0.135, 1.000]$

Sum $= 23.939$

$\text{softmax}(z) = [0.839, 0.114, 0.006, 0.042]$

**2.** $z + 10 = [13, 11, 8, 10]$

$e^{z+10} = [e^{13}, e^{11}, e^{8}, e^{10}] = [442413, 59874, 2981, 22026]$

Sum $= 527294$

$\text{softmax}(z+10) = [0.839, 0.114, 0.006, 0.042]$

**Same as part 1!**

**3.** $2z = [6, 2, -4, 0]$

$e^{2z} = [403.4, 7.389, 0.0183, 1.000]$

Sum $= 411.8$

$\text{softmax}(2z) = [0.980, 0.018, 0.000, 0.002]$

**More peaked!**

**4. Properties illustrated**:
- **Translation invariance** (part 2): $\text{softmax}(z + c) = \text{softmax}(z)$ for any scalar $c$. Adding a constant to all logits does not change the output. This is the property exploited for numerical stability (subtracting the max).
- **Temperature scaling** (part 3): Multiplying logits by $\alpha > 1$ makes the distribution more peaked (lower "temperature"). Dividing by $\alpha > 1$ makes it more uniform (higher "temperature"). This is the basis for the temperature parameter in softmax: $\text{softmax}(z/T)$.

**Key insight**: Temperature $T \to 0$ makes softmax approach argmax (one-hot). $T \to \infty$ makes softmax approach uniform. This is used in knowledge distillation and sampling from language models.
</details>

---

## Exercise 5: When to Use Which Activation

For each scenario, choose the best activation function and explain why:

1. Hidden layers of a CNN for image classification
2. Output layer for predicting probability of a single event (e.g., "is this email spam?")
3. Output layer for predicting one of 100 classes
4. Hidden layers in a Transformer encoder
5. Output layer for regression (predicting a continuous value that can be any real number)

<details>
<summary>Solution</summary>

1. **CNN hidden layers**: **ReLU** (or Leaky ReLU). Simple, fast, no vanishing gradient. The standard choice for vision networks since AlexNet (2012). He initialization pairs well with ReLU.

2. **Binary probability output**: **Sigmoid**. Maps any real number to $(0, 1)$, directly interpretable as a probability. Used with `BCEWithLogitsLoss` (which applies sigmoid internally).

3. **Multi-class output**: **Softmax**. Maps a vector of logits to a probability distribution that sums to 1. In PyTorch, `CrossEntropyLoss` applies log-softmax internally, so you typically do NOT apply softmax in the forward pass.

4. **Transformer hidden layers**: **GELU**. Used in BERT, GPT, and most modern transformers. Its smooth approximation of ReLU with a probabilistic interpretation (soft gating) works well with the attention mechanism. The non-monotonic behavior (slightly negative for inputs around $-1$) provides a form of regularization.

5. **Regression output**: **No activation (identity)**. The output should be unconstrained to represent any real number. Applying sigmoid or ReLU would limit the output range.

**Key insight**: The choice of output activation depends on the task. The choice of hidden activation is more about training dynamics (gradient flow, computation speed). For competition, use ReLU/GELU for hidden layers and match the output activation to your loss function.
</details>

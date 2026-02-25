# MLP Exercises

**Topic**: Multi-layer perceptrons, universal approximation, XOR problem
**Difficulty**: Foundational → Intermediate

---

## Exercise 1: Parameter Counting

Compute the total number of learnable parameters for each MLP architecture. Include both weights and biases.

```
1. Input: 10 → Hidden: 32 → Output: 5
2. Input: 784 → Hidden: 512 → Hidden: 256 → Output: 10
3. Input: 100 → Hidden: 64 → Hidden: 64 → Hidden: 64 → Output: 1
4. Input: 3 → Hidden: 2 → Output: 1 (the XOR network)
```

<details>
<summary>Solution</summary>

```
1.  (10+1)×32 + (32+1)×5 = 352 + 165 = 517
2.  (784+1)×512 + (512+1)×256 + (256+1)×10 = 401,920 + 131,328 + 2,570 = 535,818
3.  (100+1)×64 + (64+1)×64 + (64+1)×64 + (64+1)×1 = 6,464 + 4,160 + 4,160 + 65 = 14,849
4.  (3+1)×2 + (2+1)×1 = 8 + 3 = 11

Wait — the XOR network has input dimension 2, not 3:
4.  (2+1)×2 + (2+1)×1 = 6 + 3 = 9
```

**Key insight**: The "+1" in each term accounts for the bias. Each layer with $n_{in}$ inputs and $n_{out}$ outputs has $(n_{in} + 1) \times n_{out}$ parameters.
</details>

---

## Exercise 2: XOR by Hand

Given the following 2→2→1 MLP with step function activation ($\sigma(x) = 1$ if $x > 0$, else $0$):

```
W₁ = [[1, 1],     b₁ = [-0.5, -1.5]
      [1, 1]]

W₂ = [[1, -2]]    b₂ = [0.5]
```

Compute the output for all four XOR inputs by hand:

| Input $(x_1, x_2)$ | $z_1 = W_1 x + b_1$ | $h = \sigma(z_1)$ | $z_2 = W_2 h + b_2$ | $\hat{y} = \sigma(z_2)$ |
|---|---|---|---|---|
| (0, 0) | ? | ? | ? | ? |
| (0, 1) | ? | ? | ? | ? |
| (1, 0) | ? | ? | ? | ? |
| (1, 1) | ? | ? | ? | ? |

<details>
<summary>Solution</summary>

| Input | $z_1$ | $h$ | $z_2$ | $\hat{y}$ |
|---|---|---|---|---|
| (0,0) | $[0+0-0.5, 0+0-1.5] = [-0.5, -1.5]$ | $[0, 0]$ | $[0+0+0.5] = [0.5]$ | $1$... |

Let me redo this more carefully:

| Input | $z_1 = W_1 x + b_1$ | $h = \sigma(z_1)$ | $z_2 = W_2 h + b_2$ | $\hat{y} = \sigma(z_2)$ |
|---|---|---|---|---|
| (0,0) | $[0-0.5,\; 0-1.5] = [-0.5,\; -1.5]$ | $[0, 0]$ | $1(0) + (-2)(0) + 0.5 = 0.5$ | $1$ |

Hmm, that gives 1 for (0,0), which is wrong for XOR. Let me check the threshold.

With the convention $\sigma(x) = 1$ if $x > 0$ (strictly greater), $0.5 > 0$ so output is 1.

Try $\sigma(x) = 1$ if $x \geq 0.5$:

| Input | $z_1$ | $h$ | $z_2$ | $\hat{y}$ |
|---|---|---|---|---|
| (0,0) | $[-0.5, -1.5]$ | $[0, 0]$ | $0.5$ | $1$ |

Still wrong. The issue is the threshold. With bias $b_2 = [0.0]$ instead:

Actually, let us use $b_2 = [-0.5]$:

| Input | $z_1$ | $h$ | $z_2$ | $\hat{y}$ |
|---|---|---|---|---|
| (0,0) | $[-0.5, -1.5]$ | $[0, 0]$ | $0-0-0.5=-0.5$ | $0$ |
| (0,1) | $[0.5, -0.5]$ | $[1, 0]$ | $1-0-0.5=0.5$ | $1$ |
| (1,0) | $[0.5, -0.5]$ | $[1, 0]$ | $1-0-0.5=0.5$ | $1$ |
| (1,1) | $[1.5, 0.5]$ | $[1, 1]$ | $1-2-0.5=-1.5$ | $0$ |

With $b_2 = [-0.5]$, XOR is correct. (The original problem had $b_2 = [0.5]$.)

**Corrected weights**: $b_2 = [-0.5]$.

**Key insight**: Hidden neuron $h_1$ computes OR (fires when at least one input is 1), and $h_2$ computes AND (fires when both inputs are 1). The output layer computes $h_1 \text{ AND NOT } h_2$, which is XOR.
</details>

---

## Exercise 3: Linear Networks Cannot Learn XOR

Prove that no single-layer network (no hidden layer) can compute XOR.

Specifically, show that there exist no $w_1, w_2, b$ such that:

$$\text{sign}(w_1 x_1 + w_2 x_2 + b) = \text{XOR}(x_1, x_2) \quad \forall (x_1, x_2) \in \{0,1\}^2$$

<details>
<summary>Solution</summary>

For XOR, we need:
- $(0,0) \to 0$: $b \leq 0$
- $(0,1) \to 1$: $w_2 + b > 0$
- $(1,0) \to 1$: $w_1 + b > 0$
- $(1,1) \to 0$: $w_1 + w_2 + b \leq 0$

From constraints 2 and 3: $w_1 > -b$ and $w_2 > -b$.

Adding: $w_1 + w_2 > -2b$.

From constraint 1: $b \leq 0$, so $-2b \geq 0$, so $w_1 + w_2 > 0$.

But constraint 4 requires $w_1 + w_2 \leq -b$. Since $b \leq 0$, $-b \geq 0$.

From constraint 2: $w_2 > -b$, so $w_1 + w_2 > w_1 - b$.
From constraint 3: $w_1 > -b$, so $w_1 - b > -2b \geq 0$.

So $w_1 + w_2 > -b$, but constraint 4 requires $w_1 + w_2 + b \leq 0 \Rightarrow w_1 + w_2 \leq -b$.

**Contradiction**: $w_1 + w_2 > -b$ and $w_1 + w_2 \leq -b$ cannot both hold.

Therefore, no single-layer perceptron can compute XOR. QED.

**Key insight**: This is exactly Minsky & Papert's 1969 result. XOR requires at least one hidden layer because the positive and negative examples are not linearly separable.
</details>

---

## Exercise 4: Universal Approximation — Staircase Construction

Consider approximating the function $f(x) = x^2$ on $[0, 1]$ using a ReLU network with one hidden layer.

A single ReLU neuron computes $\max(0, wx + b)$ — a "hinge" function. By combining multiple hinges, we can build a staircase approximation.

Show that 3 hidden neurons can approximate $f(x) = x^2$ at the points $x = 0, 0.5, 1$ with the values $f = 0, 0.25, 1$.

Hint: Think of the output as $y = v_1 \text{ReLU}(x) + v_2 \text{ReLU}(x - 0.5) + v_3 \text{ReLU}(x - 1) + c$.

<details>
<summary>Solution</summary>

We want a piecewise linear function that passes through $(0, 0)$, $(0.5, 0.25)$, $(1, 1)$.

Using ReLU basis functions $r_k(x) = \max(0, x - a_k)$:

$$y = v_1 \max(0, x) + v_2 \max(0, x - 0.5) + c$$

At $x = 0$: $y = 0 + 0 + c = c$. Need $c = 0$.

At $x = 0.5$: $y = 0.5v_1 + 0 + 0 = 0.5v_1$. Need $0.5v_1 = 0.25$, so $v_1 = 0.5$.

At $x = 1$: $y = 1 \cdot 0.5 + 0.5 \cdot v_2 = 0.5 + 0.5v_2$. Need $0.5 + 0.5v_2 = 1$, so $v_2 = 1$.

So: $y = 0.5 \max(0, x) + 1.0 \max(0, x - 0.5)$.

This is a piecewise linear approximation of $x^2$ with breakpoints at $0$ and $0.5$.

**Key insight**: With $N$ hidden neurons, we get $N$ breakpoints and can approximate any continuous function on a compact set with error $\sim O(1/N)$. This is a constructive sketch of the universal approximation theorem for ReLU networks.
</details>

---

## Exercise 5: Depth vs. Width

Consider two networks for a binary classification task:

- **Wide**: 10 → 1000 → 1 (one hidden layer, 1000 neurons)
- **Deep**: 10 → 100 → 100 → 100 → 1 (three hidden layers, 100 neurons each)

1. Compute the parameter count for each.
2. Which has more parameters?
3. Which is likely to generalize better on a moderately complex task? Why?

<details>
<summary>Solution</summary>

**Wide network**: $(10+1) \times 1000 + (1000+1) \times 1 = 11{,}000 + 1{,}001 = 12{,}001$

**Deep network**: $(10+1) \times 100 + (100+1) \times 100 + (100+1) \times 100 + (100+1) \times 1 = 1{,}100 + 10{,}100 + 10{,}100 + 101 = 21{,}401$

The deep network has MORE parameters (21,401 vs 12,001), but:

**Generalization**: The deep network is likely to generalize better because:
1. **Hierarchical features**: Each layer builds on the previous, learning increasingly abstract representations. A single wide layer must learn everything at once.
2. **Parameter efficiency**: Deep networks can represent exponentially more complex functions per parameter (due to compositionality).
3. **Regularization through depth**: Each layer acts as a feature extraction step, providing an implicit regularization.

**Key insight**: Depth > width for most practical tasks. This is why modern architectures are deep (100+ layers) rather than wide. However, very deep networks require skip connections to train effectively.
</details>

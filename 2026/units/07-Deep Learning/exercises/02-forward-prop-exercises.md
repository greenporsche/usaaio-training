# Forward Propagation Exercises

**Topic**: Layer-by-layer computation, shape tracking, matrix operations
**Difficulty**: Foundational → Intermediate

---

## Exercise 1: Trace the Forward Pass

Given the following 3→2→1 network with ReLU activation on the hidden layer:

```
W₁ = [[0.5, -0.3, 0.8],    b₁ = [0.1, -0.2]
      [0.2,  0.6, -0.4]]

W₂ = [[0.7, -0.5]]          b₂ = [0.3]

Input: x = [1.0, -1.0, 0.5]
```

Compute each intermediate value step by step:
1. $z^{[1]} = W_1 x + b_1$
2. $a^{[1]} = \text{ReLU}(z^{[1]})$
3. $z^{[2]} = W_2 a^{[1]} + b_2$
4. Final output $\hat{y} = z^{[2]}$

<details>
<summary>Solution</summary>

**Step 1**: $z^{[1]} = W_1 x + b_1$

$$z^{[1]} = \begin{bmatrix}0.5 & -0.3 & 0.8 \\ 0.2 & 0.6 & -0.4\end{bmatrix} \begin{bmatrix}1.0 \\ -1.0 \\ 0.5\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.2\end{bmatrix}$$

$$= \begin{bmatrix}0.5(1) + (-0.3)(-1) + 0.8(0.5) \\ 0.2(1) + 0.6(-1) + (-0.4)(0.5)\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.2\end{bmatrix}$$

$$= \begin{bmatrix}0.5 + 0.3 + 0.4 \\ 0.2 - 0.6 - 0.2\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.2\end{bmatrix} = \begin{bmatrix}1.2 \\ -0.6\end{bmatrix} + \begin{bmatrix}0.1 \\ -0.2\end{bmatrix} = \begin{bmatrix}1.3 \\ -0.8\end{bmatrix}$$

**Step 2**: $a^{[1]} = \text{ReLU}(z^{[1]}) = [\max(0, 1.3), \max(0, -0.8)] = [1.3, 0.0]$

**Step 3**: $z^{[2]} = W_2 a^{[1]} + b_2 = [0.7, -0.5] \cdot [1.3, 0.0] + 0.3 = 0.91 + 0 + 0.3 = 1.21$

**Step 4**: $\hat{y} = 1.21$

**Key insight**: The second hidden neuron has a negative pre-activation ($-0.8$), so ReLU kills it. This neuron contributes nothing to the output for this input. If this happens for ALL inputs, the neuron is "dead."
</details>

---

## Exercise 2: Shape Tracking Through a Network

For each layer, write the output shape. Input batch: $X$ of shape $(32, 784)$.

```
Layer 1:  nn.Linear(784, 512)
Layer 2:  nn.ReLU()
Layer 3:  nn.BatchNorm1d(512)
Layer 4:  nn.Dropout(0.3)
Layer 5:  nn.Linear(512, 256)
Layer 6:  nn.ReLU()
Layer 7:  nn.Linear(256, 128)
Layer 8:  nn.ReLU()
Layer 9:  nn.Linear(128, 10)
```

<details>
<summary>Solution</summary>

```
Input:               (32, 784)
After Linear(784,512): (32, 512)    — matrix multiply + bias
After ReLU:          (32, 512)    — element-wise, shape unchanged
After BatchNorm1d:   (32, 512)    — normalizes features, shape unchanged
After Dropout:       (32, 512)    — zeros some elements, shape unchanged
After Linear(512,256): (32, 256)    — matrix multiply + bias
After ReLU:          (32, 256)    — element-wise
After Linear(256,128): (32, 128)    — matrix multiply + bias
After ReLU:          (32, 128)    — element-wise
After Linear(128,10):  (32, 10)     — final logits
```

**Key insight**: Only `nn.Linear` changes the shape. ReLU, BatchNorm, and Dropout all preserve the input shape. This makes shape tracking through MLPs straightforward — you only need to track the linear layers.
</details>

---

## Exercise 3: Batched vs. Single Forward Pass

Given weights $W = [[1, 2], [3, 4], [5, 6]]$ (shape $3 \times 2$) and bias $b = [0.1, 0.2, 0.3]$ (shape $3$):

1. Compute the forward pass for a single input $x = [1, -1]$ (shape $2$).
2. Compute the batched forward pass for $X = [[1, -1], [2, 0], [0, 3]]$ (shape $3 \times 2$).
3. Verify that row $i$ of the batched result equals the single-sample result for input $X[i]$.

<details>
<summary>Solution</summary>

**Single input**: $z = Wx + b$

$$z = \begin{bmatrix}1 & 2 \\ 3 & 4 \\ 5 & 6\end{bmatrix}\begin{bmatrix}1 \\ -1\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.2 \\ 0.3\end{bmatrix} = \begin{bmatrix}-1 \\ -1 \\ -1\end{bmatrix} + \begin{bmatrix}0.1 \\ 0.2 \\ 0.3\end{bmatrix} = \begin{bmatrix}-0.9 \\ -0.8 \\ -0.7\end{bmatrix}$$

**Batched**: $Z = XW^T + b$ (note the transpose!)

$$Z = \begin{bmatrix}1 & -1 \\ 2 & 0 \\ 0 & 3\end{bmatrix}\begin{bmatrix}1 & 3 & 5 \\ 2 & 4 & 6\end{bmatrix} + \begin{bmatrix}0.1 & 0.2 & 0.3\end{bmatrix}$$

$$= \begin{bmatrix}-1 & -1 & -1 \\ 2 & 6 & 10 \\ 6 & 12 & 18\end{bmatrix} + \begin{bmatrix}0.1 & 0.2 & 0.3\end{bmatrix} = \begin{bmatrix}-0.9 & -0.8 & -0.7 \\ 2.1 & 6.2 & 10.3 \\ 6.1 & 12.2 & 18.3\end{bmatrix}$$

Row 0 matches the single input result: $[-0.9, -0.8, -0.7]$. Verified.

**Key insight**: PyTorch's `nn.Linear(2, 3)` stores $W$ as shape $(3, 2)$ and computes $xW^T + b$. The batched version naturally parallelizes: each row of $X$ is an independent sample processed simultaneously.
</details>

---

## Exercise 4: Forward Pass with Sigmoid

Repeat Exercise 1 but with sigmoid activation instead of ReLU. Same weights and input.

```
W₁ = [[0.5, -0.3, 0.8],    b₁ = [0.1, -0.2]
      [0.2,  0.6, -0.4]]

W₂ = [[0.7, -0.5]]          b₂ = [0.3]

Input: x = [1.0, -1.0, 0.5]
```

Compute all values and compare the final output with the ReLU version.

<details>
<summary>Solution</summary>

$z^{[1]}$ is the same as before: $[1.3, -0.8]$.

$a^{[1]} = \sigma(z^{[1]}) = [\sigma(1.3), \sigma(-0.8)]$

$\sigma(1.3) = \frac{1}{1+e^{-1.3}} = \frac{1}{1+0.2725} \approx 0.7858$

$\sigma(-0.8) = \frac{1}{1+e^{0.8}} = \frac{1}{1+2.2255} \approx 0.3100$

$z^{[2]} = 0.7(0.7858) + (-0.5)(0.3100) + 0.3 = 0.5501 - 0.1550 + 0.3 = 0.6951$

**Comparison**:
- ReLU output: $1.21$ (killed the negative neuron, amplified the positive one)
- Sigmoid output: $0.6951$ (both neurons contributed, but with compressed values)

**Key insight**: Sigmoid squashes ALL values into $(0, 1)$, so both neurons contribute to the output. ReLU zeros the negative one completely. This is why ReLU creates sparser representations — many neurons are inactive for any given input.
</details>

---

## Exercise 5: Memory Requirements for Forward Pass

A network has architecture: $1024 \to 2048 \to 2048 \to 512 \to 10$. Batch size is 64. Using float32 (4 bytes per number).

Compute the memory required to store:
1. All weight matrices and biases
2. All intermediate activations (needed for backprop) for one forward pass
3. The total memory (parameters + activations)

<details>
<summary>Solution</summary>

**Parameters** (weights + biases):
- $W_1$: $1024 \times 2048 = 2{,}097{,}152$ values, $b_1$: $2048$
- $W_2$: $2048 \times 2048 = 4{,}194{,}304$ values, $b_2$: $2048$
- $W_3$: $2048 \times 512 = 1{,}048{,}576$ values, $b_3$: $512$
- $W_4$: $512 \times 10 = 5{,}120$ values, $b_4$: $10$

Total parameter values: $2{,}097{,}152 + 2{,}048 + 4{,}194{,}304 + 2{,}048 + 1{,}048{,}576 + 512 + 5{,}120 + 10 = 7{,}349{,}770$

Parameter memory: $7{,}349{,}770 \times 4 = 29{,}399{,}080$ bytes $\approx 28$ MB

**Activations** (batch of 64, need to cache $z$ and $a$ for each layer):
- $a^{[0]}$ (input): $64 \times 1024 = 65{,}536$
- $z^{[1]}$, $a^{[1]}$: $64 \times 2048 \times 2 = 262{,}144$
- $z^{[2]}$, $a^{[2]}$: $64 \times 2048 \times 2 = 262{,}144$
- $z^{[3]}$, $a^{[3]}$: $64 \times 512 \times 2 = 65{,}536$
- $z^{[4]}$ (output): $64 \times 10 = 640$

Total activation values: $65{,}536 + 262{,}144 + 262{,}144 + 65{,}536 + 640 = 656{,}000$

Activation memory: $656{,}000 \times 4 = 2{,}624{,}000$ bytes $\approx 2.5$ MB

**Total**: $\approx 28 + 2.5 = 30.5$ MB

Note: During training, we also need gradients (same size as parameters) + optimizer state (e.g., Adam stores 2x parameter size). Total training memory $\approx 28 \times 4 + 2.5 = 114.5$ MB.

**Key insight**: Parameters dominate memory for MLPs, but for CNNs with large spatial dimensions, activations often dominate (because the spatial size multiplies the batch size).
</details>

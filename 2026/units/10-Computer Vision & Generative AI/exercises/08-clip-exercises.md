# CLIP Exercises

**5 exercises** | Covers: cosine similarity, InfoNCE loss computation, temperature scaling, zero-shot classification, similarity matrix analysis
**CRITICAL for Round 2**: The 2025 Round 2 Problem 3 was entirely about CLIP.

---

## Exercise 1: Compute Cosine Similarity Matrix

**Target time**: 5 minutes

A batch of $N=3$ image-text pairs produces the following L2-normalized embeddings ($D=3$):

Image embeddings $V$:
$$v_1 = [0.6, 0.8, 0.0], \quad v_2 = [0.0, 0.6, 0.8], \quad v_3 = [0.8, 0.0, 0.6]$$

Text embeddings $T$:
$$t_1 = [0.5, 0.7, 0.5], \quad t_2 = [0.1, 0.5, 0.9], \quad t_3 = [0.7, 0.1, 0.7]$$

(Note: $t_j$ are NOT pre-normalized.)

**Part 1**: First, L2-normalize each text embedding $t_j$.

**Part 2**: Compute the full $3 \times 3$ cosine similarity matrix $S$ where $S_{ij} = v_i \cdot t_j$.

**Part 3**: For each image, which text has the highest similarity? Does the diagonal dominate?

<details>
<summary>Solution</summary>

**Part 1**: Normalize text embeddings:
- $\|t_1\| = \sqrt{0.25 + 0.49 + 0.25} = \sqrt{0.99} = 0.995$. $\hat{t}_1 = [0.503, 0.703, 0.503]$
- $\|t_2\| = \sqrt{0.01 + 0.25 + 0.81} = \sqrt{1.07} = 1.034$. $\hat{t}_2 = [0.097, 0.483, 0.870]$
- $\|t_3\| = \sqrt{0.49 + 0.01 + 0.49} = \sqrt{0.99} = 0.995$. $\hat{t}_3 = [0.703, 0.100, 0.703]$

**Part 2**: $S_{ij} = v_i \cdot \hat{t}_j$:

$S_{11} = 0.6(0.503) + 0.8(0.703) + 0.0(0.503) = 0.302 + 0.562 + 0 = 0.864$
$S_{12} = 0.6(0.097) + 0.8(0.483) + 0.0(0.870) = 0.058 + 0.387 + 0 = 0.445$
$S_{13} = 0.6(0.703) + 0.8(0.100) + 0.0(0.703) = 0.422 + 0.080 + 0 = 0.502$

$S_{21} = 0.0(0.503) + 0.6(0.703) + 0.8(0.503) = 0 + 0.422 + 0.402 = 0.824$
$S_{22} = 0.0(0.097) + 0.6(0.483) + 0.8(0.870) = 0 + 0.290 + 0.696 = 0.986$
$S_{23} = 0.0(0.703) + 0.6(0.100) + 0.8(0.703) = 0 + 0.060 + 0.562 = 0.622$

$S_{31} = 0.8(0.503) + 0.0(0.703) + 0.6(0.503) = 0.402 + 0 + 0.302 = 0.704$
$S_{32} = 0.8(0.097) + 0.0(0.483) + 0.6(0.870) = 0.078 + 0 + 0.522 = 0.600$
$S_{33} = 0.8(0.703) + 0.0(0.100) + 0.6(0.703) = 0.562 + 0 + 0.422 = 0.984$

$$S = \begin{bmatrix} 0.864 & 0.445 & 0.502 \\ 0.824 & 0.986 & 0.622 \\ 0.704 & 0.600 & 0.984 \end{bmatrix}$$

**Part 3**:
- Image 1: highest sim with text 1 (0.864) -- correct!
- Image 2: highest sim with text 2 (0.986) -- correct!
- Image 3: highest sim with text 3 (0.984) -- correct!

The diagonal dominates, which is what we want for matched pairs.

</details>

---

## Exercise 2: Compute InfoNCE Loss

**Target time**: 6 minutes

Using the similarity matrix from Exercise 1 with temperature $\tau = 0.1$:

$$S = \begin{bmatrix} 0.864 & 0.445 & 0.502 \\ 0.824 & 0.986 & 0.622 \\ 0.704 & 0.600 & 0.984 \end{bmatrix}$$

**Part 1**: Compute the logits matrix $L = S / \tau$.

**Part 2**: Compute the image-to-text loss $\mathcal{L}^{i2t}$ for image 1:
$$\mathcal{L}_1^{i2t} = -\log \frac{\exp(L_{11})}{\sum_{k=1}^{3} \exp(L_{1k})}$$

**Part 3**: Compute the full symmetric CLIP loss (average of all $\mathcal{L}_i^{i2t}$ and $\mathcal{L}_j^{t2i}$).

<details>
<summary>Solution</summary>

**Part 1**: $L = S / 0.1$:
$$L = \begin{bmatrix} 8.64 & 4.45 & 5.02 \\ 8.24 & 9.86 & 6.22 \\ 7.04 & 6.00 & 9.84 \end{bmatrix}$$

**Part 2**: For image 1:
$\exp(L_{11}) = e^{8.64} = 5{,}659$
$\exp(L_{12}) = e^{4.45} = 85.6$
$\exp(L_{13}) = e^{5.02} = 151.4$

$\sum = 5{,}659 + 85.6 + 151.4 = 5{,}896$

$\mathcal{L}_1^{i2t} = -\log\frac{5{,}659}{5{,}896} = -\log(0.9598) = 0.041$

**Part 3**: Image-to-text losses (row-wise softmax):

$\mathcal{L}_1^{i2t} = 0.041$ (computed above)

For image 2: $e^{8.24} = 3{,}777$, $e^{9.86} = 19{,}106$, $e^{6.22} = 502$. Sum = $23{,}385$.
$\mathcal{L}_2^{i2t} = -\log(19{,}106 / 23{,}385) = -\log(0.817) = 0.202$

For image 3: $e^{7.04} = 1{,}139$, $e^{6.00} = 403$, $e^{9.84} = 18{,}728$. Sum = $20{,}270$.
$\mathcal{L}_3^{i2t} = -\log(18{,}728 / 20{,}270) = -\log(0.924) = 0.079$

$\overline{\mathcal{L}}^{i2t} = (0.041 + 0.202 + 0.079)/3 = 0.107$

Text-to-image losses (column-wise softmax):

Column 1: $e^{8.64} = 5{,}659$, $e^{8.24} = 3{,}777$, $e^{7.04} = 1{,}139$. Sum = $10{,}575$.
$\mathcal{L}_1^{t2i} = -\log(5{,}659/10{,}575) = -\log(0.535) = 0.625$

Column 2: $e^{4.45} = 85.6$, $e^{9.86} = 19{,}106$, $e^{6.00} = 403$. Sum = $19{,}595$.
$\mathcal{L}_2^{t2i} = -\log(19{,}106/19{,}595) = -\log(0.975) = 0.025$

Column 3: $e^{5.02} = 151$, $e^{6.22} = 502$, $e^{9.84} = 18{,}728$. Sum = $19{,}381$.
$\mathcal{L}_3^{t2i} = -\log(18{,}728/19{,}381) = -\log(0.966) = 0.035$

$\overline{\mathcal{L}}^{t2i} = (0.625 + 0.025 + 0.035)/3 = 0.228$

**Symmetric CLIP loss**: $\mathcal{L} = (\overline{\mathcal{L}}^{i2t} + \overline{\mathcal{L}}^{t2i})/2 = (0.107 + 0.228)/2 = 0.168$

</details>

---

## Exercise 3: Temperature Analysis

**Target time**: 4 minutes

Using $S_{11} = 0.85$, $S_{12} = 0.80$, $S_{13} = 0.75$ for image 1:

**Part 1**: Compute the softmax probabilities $P(t_k | v_1) = \frac{\exp(S_{1k}/\tau)}{\sum_j \exp(S_{1j}/\tau)}$ for $\tau = 1.0$, $\tau = 0.1$, and $\tau = 0.01$.

**Part 2**: How does the probability assigned to the correct match ($t_1$) change with $\tau$?

**Part 3**: CLIP initializes $\tau = 0.07$ and learns it. Why not just fix $\tau = 0.01$?

<details>
<summary>Solution</summary>

**Part 1**:

$\tau = 1.0$: logits = $[0.85, 0.80, 0.75]$
$\exp = [2.340, 2.226, 2.117]$, sum = $6.683$
$P = [0.350, 0.333, 0.317]$

$\tau = 0.1$: logits = $[8.5, 8.0, 7.5]$
$\exp = [4{,}915, 2{,}981, 1{,}808]$, sum = $9{,}704$
$P = [0.506, 0.307, 0.186]$

$\tau = 0.01$: logits = $[85, 80, 75]$
$\exp = [e^{85}, e^{80}, e^{75}]$. Ratio: $e^{85}/e^{80} = e^5 = 148.4$, $e^{85}/e^{75} = e^{10} = 22{,}026$.
$P \approx [0.9933, 0.0067, 0.000003]$

**Part 2**: As $\tau$ decreases:
- $\tau = 1.0$: $P(\text{correct}) = 35\%$ (nearly uniform)
- $\tau = 0.1$: $P(\text{correct}) = 50.6\%$ (moderate preference)
- $\tau = 0.01$: $P(\text{correct}) = 99.3\%$ (very confident)

Lower temperature makes the distribution sharper, concentrating probability on the most similar item.

**Part 3**: Fixing $\tau = 0.01$ from the start would make the loss extremely easy once the model gets the ordering roughly right (all probability mass on the correct match). This means:
1. Gradients for negative pairs nearly vanish — the model stops learning fine-grained similarities
2. Early in training when similarities are random, the very sharp softmax creates extreme gradient magnitudes
3. A learnable $\tau$ lets the model adapt: use higher $\tau$ early (when embeddings are noisy) and lower $\tau$ later (for fine-grained discrimination)

</details>

---

## Exercise 4: Zero-Shot Classification

**Target time**: 4 minutes

A CLIP model encodes an image of a tabby cat and 4 class prompts. After L2 normalization:

Image embedding: $v = [0.5, 0.7, 0.3, 0.4]$ (already normalized, $\|v\|=1$)

Text embeddings:
- "a photo of a cat": $t_1 = [0.48, 0.68, 0.32, 0.45]$ (normalized)
- "a photo of a dog": $t_2 = [0.3, 0.4, 0.7, 0.5]$ (normalized)
- "a photo of a car": $t_3 = [0.8, 0.1, 0.2, 0.6]$ (normalized)
- "a photo of a bird": $t_4 = [0.4, 0.5, 0.6, 0.5]$ (normalized)

**Part 1**: Compute cosine similarity between $v$ and each $t_k$.

**Part 2**: What class does CLIP predict? With what confidence (softmax with $\tau = 0.01$)?

**Part 3**: If we change the prompt from "a photo of a cat" to "a cute tabby cat sitting", how might this affect the similarity? Why does prompt engineering matter?

<details>
<summary>Solution</summary>

**Part 1**: Since all vectors are already normalized, cosine sim = dot product:
- $\cos(v, t_1) = 0.5(0.48) + 0.7(0.68) + 0.3(0.32) + 0.4(0.45) = 0.24 + 0.476 + 0.096 + 0.18 = 0.992$
- $\cos(v, t_2) = 0.5(0.3) + 0.7(0.4) + 0.3(0.7) + 0.4(0.5) = 0.15 + 0.28 + 0.21 + 0.20 = 0.840$
- $\cos(v, t_3) = 0.5(0.8) + 0.7(0.1) + 0.3(0.2) + 0.4(0.6) = 0.40 + 0.07 + 0.06 + 0.24 = 0.770$
- $\cos(v, t_4) = 0.5(0.4) + 0.7(0.5) + 0.3(0.6) + 0.4(0.5) = 0.20 + 0.35 + 0.18 + 0.20 = 0.930$

**Part 2**: CLIP predicts **"cat"** ($t_1$) with highest similarity 0.992.

With $\tau = 0.01$: logits = $[99.2, 84.0, 77.0, 93.0]$. The differences are so large that softmax gives essentially $P(\text{cat}) \approx 1.0$. The model is very confident.

**Part 3**: A more descriptive prompt "a cute tabby cat sitting" would produce a text embedding closer to the specific image content (tabby cat), potentially increasing the similarity further. Prompt engineering matters because CLIP learned image-text associations from natural language captions — the text encoder produces different embeddings for different phrasings. Using templates that match the training distribution (e.g., "a photo of a {class}") tends to work better than bare class names.

</details>

---

## Exercise 5: CLIP Training — Gradient Analysis

**Target time**: 5 minutes

In a batch of $N=2$ image-text pairs, the similarity matrix (scaled by $1/\tau$) is:

$$L = \begin{bmatrix} 10 & 3 \\ 4 & 9 \end{bmatrix}$$

**Part 1**: Compute the softmax probabilities for the image-to-text direction (each row is a distribution).

**Part 2**: The cross-entropy loss for image 1 with label 0 is $\mathcal{L}_1 = -\log P_{10}$. The gradient of cross-entropy w.r.t. logits is $\frac{\partial \mathcal{L}}{\partial L_{1k}} = P_{1k} - \mathbb{1}[k=0]$. Compute the gradient for all entries in row 1.

**Part 3**: Interpret: which direction does the gradient push the similarity between image 1 and text 1? Between image 1 and text 2?

<details>
<summary>Solution</summary>

**Part 1**: Row 1: $\exp(10) = 22{,}026$, $\exp(3) = 20.1$. Sum = $22{,}046$.
$P_{10} = 22{,}026/22{,}046 = 0.9991$, $P_{11} = 20.1/22{,}046 = 0.0009$.

Row 2: $\exp(4) = 54.6$, $\exp(9) = 8{,}103$. Sum = $8{,}158$.
$P_{20} = 54.6/8{,}158 = 0.0067$, $P_{21} = 8{,}103/8{,}158 = 0.9933$.

**Part 2**: Gradient for row 1 (label = 0):
$\frac{\partial \mathcal{L}_1}{\partial L_{10}} = P_{10} - 1 = 0.9991 - 1 = -0.0009$
$\frac{\partial \mathcal{L}_1}{\partial L_{11}} = P_{11} - 0 = 0.0009$

**Part 3**:
- Gradient for $L_{10}$ is $-0.0009$ (negative) → the loss wants to *increase* the similarity between image 1 and text 1 (its positive pair). However, the magnitude is tiny because this pair already has very high similarity.
- Gradient for $L_{11}$ is $+0.0009$ (positive) → the loss wants to *decrease* the similarity between image 1 and text 2 (a negative pair). Also tiny.

The gradients are small because the model is already very confident about the correct pairings. In early training when the model is uncertain, gradients would be much larger.

</details>

---

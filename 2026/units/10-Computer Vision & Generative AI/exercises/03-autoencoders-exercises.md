# Autoencoder Exercises

**5 exercises** | Covers: reconstruction loss, bottleneck analysis, linear autoencoder = PCA, latent space, denoising autoencoder

---

## Exercise 1: Compute Reconstruction Loss

**Target time**: 3 minutes

An autoencoder with 4-dimensional input produces:
- Input: $x = [0.8, 0.2, 0.5, 0.9]$
- Reconstruction: $\hat{x} = [0.7, 0.3, 0.4, 0.85]$

**Part 1**: Compute the MSE reconstruction loss.

**Part 2**: Compute the BCE reconstruction loss (assuming inputs are in $[0, 1]$).

**Part 3**: Which loss is more appropriate for MNIST images normalized to $[0, 1]$? Why?

<details>
<summary>Solution</summary>

**Part 1**: MSE:
$\mathcal{L} = \frac{1}{4}[(0.8-0.7)^2 + (0.2-0.3)^2 + (0.5-0.4)^2 + (0.9-0.85)^2]$
$= \frac{1}{4}[0.01 + 0.01 + 0.01 + 0.0025] = \frac{0.0325}{4} = 0.008125$

**Part 2**: BCE:
$\mathcal{L} = -\frac{1}{4}\sum_j [x_j \log \hat{x}_j + (1-x_j)\log(1-\hat{x}_j)]$

$= -\frac{1}{4}[0.8\log(0.7) + 0.2\log(0.3) + 0.5\log(0.4) + 0.2\log(0.3) + 0.5\log(0.6) + 0.9\log(0.85) + 0.1\log(0.15)]$

$\approx -\frac{1}{4}[-0.2853 - 0.2408 - 0.4581 - 0.2408 - 0.2554 - 0.1625 - 0.1897]$

Wait, let me redo this carefully:
- $j=1$: $0.8\ln(0.7) + 0.2\ln(0.3) = 0.8(-0.3567) + 0.2(-1.2040) = -0.2853 - 0.2408 = -0.5261$
- $j=2$: $0.2\ln(0.3) + 0.8\ln(0.7) = -0.2408 + (-0.2853) = -0.5261$
- $j=3$: $0.5\ln(0.4) + 0.5\ln(0.6) = 0.5(-0.9163) + 0.5(-0.5108) = -0.4581 - 0.2554 = -0.7135$
- $j=4$: $0.9\ln(0.85) + 0.1\ln(0.15) = 0.9(-0.1625) + 0.1(-1.8971) = -0.1463 - 0.1897 = -0.3360$

$\mathcal{L} = -\frac{1}{4}(-0.5261 - 0.5261 - 0.7135 - 0.3360) = -\frac{1}{4}(-2.1017) = 0.5254$

**Part 3**: BCE is more appropriate because MNIST pixel values represent probabilities (how "on" each pixel is). BCE treats the decoder output as a Bernoulli parameter, which matches the binary nature of the data. MSE also works well in practice for normalized image data.

</details>

---

## Exercise 2: Bottleneck Analysis

**Target time**: 3 minutes

Consider an autoencoder for 28x28 grayscale images (784 dimensions).

**Part 1**: If the latent dimension is $d = 784$, what does the autoencoder learn? Is this useful?

**Part 2**: If $d = 2$, what compression ratio is achieved? What might be lost?

**Part 3**: If the data truly lies on a 10-dimensional manifold (e.g., 10 MNIST digit classes with continuous variations), what is the ideal latent dimension?

<details>
<summary>Solution</summary>

**Part 1**: With $d = 784$ (same as input), the autoencoder can learn the identity function: $f(x) = x$. The reconstruction loss is zero, but the latent space has no useful structure. This is not useful for learning representations.

**Part 2**: Compression ratio = $784/2 = 392:1$. With only 2 dimensions, the autoencoder must discard most information. It will keep the most distinguishing features (e.g., digit class, rough shape) but lose fine details (exact stroke width, small variations). Good for visualization but poor reconstruction.

**Part 3**: Ideally, $d$ should be at least the intrinsic dimensionality of the data. For MNIST with 10 classes plus continuous variations (rotation, thickness, slant), a latent dimension of $d = 10\text{-}20$ would capture most of the important variation while still compressing significantly ($784/20 \approx 40:1$).

In practice, slightly overcomplete ($d = 32$ or $d = 64$) often works better because it gives the model more room to organize the latent space.

</details>

---

## Exercise 3: Linear Autoencoder and PCA

**Target time**: 4 minutes

A dataset has 3 data points in $\mathbb{R}^3$:
$$X = \begin{bmatrix} 2 & 1 & 0 \\ 0 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix}$$

(each row is a data point, already mean-centered)

**Part 1**: A linear autoencoder with $d=1$ learns $\hat{x} = W_2 W_1 x$ where $W_1 \in \mathbb{R}^{1 \times 3}$ and $W_2 \in \mathbb{R}^{3 \times 1}$. What is the covariance matrix $C = \frac{1}{n}X^T X$?

**Part 2**: The top eigenvector of $C$ is the direction that the linear autoencoder will learn (same as PCA). Find the eigenvector corresponding to the largest eigenvalue.

<details>
<summary>Solution</summary>

**Part 1**:
$$C = \frac{1}{3}\begin{bmatrix} 2 & 0 & 1 \\ 1 & 1 & 1 \\ 0 & 2 & 1 \end{bmatrix}^T \begin{bmatrix} 2 & 1 & 0 \\ 0 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 5 & 3 & 1 \\ 3 & 3 & 3 \\ 1 & 3 & 5 \end{bmatrix}$$

**Part 2**: By inspection or computation, the eigenvalues are $\lambda_1 = 3, \lambda_2 = 1, \lambda_3 = 0$ (approximately).

The eigenvector for $\lambda_1 = 3$: $v_1 = \frac{1}{\sqrt{3}}[1, 1, 1]^T$ (the direction of maximum variance).

Actually, let's verify: $Cv_1 = \frac{1}{3}\begin{bmatrix}5+3+1\\3+3+3\\1+3+5\end{bmatrix} \cdot \frac{1}{\sqrt{3}} = \frac{1}{3\sqrt{3}}\begin{bmatrix}9\\9\\9\end{bmatrix} = \frac{3}{\sqrt{3}}[1,1,1]^T/\sqrt{3} = 3 v_1$. Confirmed.

The linear autoencoder with $d=1$ will learn to project onto $v_1 = \frac{1}{\sqrt{3}}[1,1,1]^T$.

</details>

---

## Exercise 4: Latent Space Traversal

**Target time**: 3 minutes

An autoencoder with $d=2$ has been trained. Two test images encode to:
- Digit "3": $z_3 = [-1.5, 0.5]$
- Digit "8": $z_8 = [1.5, 0.5]$

**Part 1**: Compute the latent code for the midpoint interpolation: $z_{mid} = \frac{1}{2}(z_3 + z_8)$.

**Part 2**: Compute the latent codes for 5 evenly-spaced points along the interpolation path from $z_3$ to $z_8$ (including endpoints).

**Part 3**: Why might this interpolation NOT produce smooth transitions with a standard autoencoder but WOULD with a VAE?

<details>
<summary>Solution</summary>

**Part 1**: $z_{mid} = \frac{1}{2}([-1.5, 0.5] + [1.5, 0.5]) = [0.0, 0.5]$

**Part 2**: Using $z(\alpha) = (1-\alpha)z_3 + \alpha z_8$ for $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$:
- $\alpha=0$: $[-1.5, 0.5]$ (digit "3")
- $\alpha=0.25$: $[-0.75, 0.5]$
- $\alpha=0.5$: $[0.0, 0.5]$
- $\alpha=0.75$: $[0.75, 0.5]$
- $\alpha=1.0$: $[1.5, 0.5]$ (digit "8")

**Part 3**: In a standard autoencoder, the latent space has no constraint on its structure. The region between $z_3$ and $z_8$ might be a "dead zone" — the decoder has never seen latent codes from this region during training, so it might produce garbage. In a VAE, the KL divergence term encourages all latent codes to stay near $\mathcal{N}(0, I)$, creating a smooth, densely-populated latent space where interpolation produces meaningful outputs.

</details>

---

## Exercise 5: Denoising Autoencoder

**Target time**: 3 minutes

A denoising autoencoder receives corrupted input $\tilde{x} = x + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.1^2 I)$ and must reconstruct the clean $x$.

**Part 1**: Write the loss function for the denoising autoencoder.

**Part 2**: If $x = [0.8, 0.2, 0.5]$ and $\epsilon = [0.05, -0.03, 0.08]$, what is $\tilde{x}$? What is the target for reconstruction?

**Part 3**: How does training with corrupted inputs prevent the autoencoder from learning the identity function, even when the bottleneck is the same size as the input?

<details>
<summary>Solution</summary>

**Part 1**: $\mathcal{L} = \|x - f(\tilde{x})\|^2 = \|x - g_\theta(f_\phi(x + \epsilon))\|^2$

The model receives $\tilde{x}$ but is evaluated against clean $x$.

**Part 2**:
$\tilde{x} = [0.8 + 0.05, 0.2 - 0.03, 0.5 + 0.08] = [0.85, 0.17, 0.58]$

The reconstruction target is the **clean** input: $x = [0.8, 0.2, 0.5]$.

**Part 3**: If the autoencoder learns the identity $f(\tilde{x}) = \tilde{x}$, it outputs the noisy input, not the clean one. The loss would be $\|\epsilon\|^2 > 0$. To minimize loss, it must learn to *denoise* — extract the underlying structure and ignore the noise. This forces the autoencoder to learn meaningful features even without a bottleneck, because it must distinguish signal from noise.

</details>

---

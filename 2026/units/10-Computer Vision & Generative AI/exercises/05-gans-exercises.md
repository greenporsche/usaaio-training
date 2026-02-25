# GAN Exercises

**5 exercises** | Covers: minimax objective, optimal discriminator, training dynamics, mode collapse, Wasserstein distance

---

## Exercise 1: Compute GAN Losses

**Target time**: 4 minutes

A discriminator outputs the following for a batch of 4 samples:
- Real images: $D(x_1) = 0.9$, $D(x_2) = 0.7$
- Fake images: $D(G(z_1)) = 0.3$, $D(G(z_2)) = 0.1$

**Part 1**: Compute the discriminator loss:
$$\mathcal{L}_D = -\frac{1}{2}\sum_i \log D(x_i) - \frac{1}{2}\sum_j \log(1 - D(G(z_j)))$$

**Part 2**: Compute the generator loss (non-saturating):
$$\mathcal{L}_G = -\frac{1}{2}\sum_j \log D(G(z_j))$$

**Part 3**: Is the discriminator doing a good job? Is the generator?

<details>
<summary>Solution</summary>

**Part 1**:
$\mathcal{L}_D = -\frac{1}{2}[\log(0.9) + \log(0.7)] - \frac{1}{2}[\log(1-0.3) + \log(1-0.1)]$

$= -\frac{1}{2}[-0.1054 + (-0.3567)] - \frac{1}{2}[-0.3567 + (-0.1054)]$

$= -\frac{1}{2}(-0.4621) - \frac{1}{2}(-0.4621)$

$= 0.2311 + 0.2311 = 0.4621$

**Part 2**:
$\mathcal{L}_G = -\frac{1}{2}[\log(0.3) + \log(0.1)]$

$= -\frac{1}{2}[-1.2040 + (-2.3026)] = -\frac{1}{2}(-3.5066) = 1.7533$

**Part 3**: The discriminator is doing well — it outputs high values (0.9, 0.7) for real and low values (0.3, 0.1) for fake. The generator is doing poorly — its samples are easily detected as fake ($D(G(z))$ is low). The high $\mathcal{L}_G$ confirms this.

</details>

---

## Exercise 2: Optimal Discriminator

**Target time**: 4 minutes

The optimal discriminator is:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

At a point $x_0$ where $p_{data}(x_0) = 0.6$ and $p_g(x_0) = 0.2$:

**Part 1**: What is $D^*(x_0)$?

**Part 2**: At convergence ($p_g = p_{data}$), what is $D^*(x)$ for all $x$?

**Part 3**: Substitute $D^*$ when $p_g = p_{data}$ into the value function:
$$V(D^*, G) = \mathbb{E}[\log D^*(x)] + \mathbb{E}[\log(1-D^*(G(z)))]$$
What value does this give?

<details>
<summary>Solution</summary>

**Part 1**:
$D^*(x_0) = \frac{0.6}{0.6 + 0.2} = \frac{0.6}{0.8} = 0.75$

The discriminator assigns 75% probability that $x_0$ is real, which makes sense since $p_{data}$ is 3x larger than $p_g$ at this point.

**Part 2**: When $p_g = p_{data}$:
$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{data}(x)} = \frac{1}{2}$ for all $x$.

The discriminator can't tell real from fake — it outputs 0.5 everywhere.

**Part 3**:
$V = \mathbb{E}[\log(1/2)] + \mathbb{E}[\log(1 - 1/2)]$
$= \log(1/2) + \log(1/2) = -\log 2 - \log 2 = -2\log 2 = -\log 4 \approx -1.386$

This is the global minimum of the GAN objective.

</details>

---

## Exercise 3: Training Instability Analysis

**Target time**: 3 minutes

During GAN training, you observe:

| Epoch | $\mathcal{L}_D$ | $\mathcal{L}_G$ | D(real) avg | D(fake) avg |
|---|---|---|---|---|
| 1 | 0.70 | 5.00 | 0.50 | 0.50 |
| 10 | 0.02 | 8.00 | 0.99 | 0.01 |
| 20 | 0.01 | 12.0 | 1.00 | 0.00 |
| 30 | 0.01 | 15.0 | 1.00 | 0.00 |

**Part 1**: What's going wrong? Why is the generator loss increasing?

**Part 2**: Compute the gradient of $\log(1-D(G(z)))$ w.r.t. $G$ when $D(G(z)) \approx 0$. What does this tell you?

**Part 3**: How would the non-saturating loss $-\log D(G(z))$ help here?

<details>
<summary>Solution</summary>

**Part 1**: The discriminator has become too strong — it perfectly classifies all samples. The generator can't learn because the discriminator provides no useful gradient signal. $D(G(z)) \approx 0$ means $\log(1 - D(G(z))) \approx \log(1) = 0$ — the loss is flat, so the gradient vanishes.

**Part 2**: $\frac{\partial}{\partial G}\log(1 - D(G(z))) = \frac{-D'(G(z))}{1 - D(G(z))}$

When $D(G(z)) \approx 0$: $\frac{-D'}{1-0} = -D'$ — the gradient is small because $D$ is already at its flat region near 0.

More precisely, the $\log(1-x)$ function has slope $-1/(1-x)$. At $x \approx 0$: slope $\approx -1$ (weak). At $x \approx 1$: slope $\to -\infty$ (strong). So the generator gets weak signal when it's doing badly.

**Part 3**: The non-saturating loss $-\log D(G(z))$ has gradient $\frac{-D'(G(z))}{D(G(z))}$.

When $D(G(z)) \approx 0$: $\frac{-D'}{0^+} \to -\infty$ — the gradient is LARGE, providing strong signal to the generator exactly when it needs it most. This flips the asymmetry and gives the generator stronger gradients early in training.

</details>

---

## Exercise 4: Mode Collapse Detection

**Target time**: 3 minutes

A GAN generates 100 MNIST-like samples. You compute statistics:
- Mean pixel intensity across all 100 samples: 0.142
- Standard deviation across samples: 0.003 (very low)
- All samples look like the digit "1"

The real MNIST dataset has:
- Mean pixel intensity: 0.131
- Standard deviation across images: 0.308
- 10 digit classes roughly equally distributed

**Part 1**: What type of failure has occurred?

**Part 2**: The discriminator loss is low ($\approx 0.3$) and the generator loss is also low ($\approx 0.3$). Why doesn't the loss indicate a problem?

**Part 3**: Propose a metric (beyond visual inspection) that would detect this failure.

<details>
<summary>Solution</summary>

**Part 1**: **Mode collapse**. The generator has learned to produce only one type of output (digit "1") that consistently fools the discriminator. It has "collapsed" to a single mode of the data distribution instead of covering all 10 digit classes.

**Part 2**: The losses look fine because the discriminator sees realistic-looking "1"s and can't easily distinguish them from real "1"s in the dataset. The GAN loss only measures whether individual samples look real — it doesn't measure whether the generator covers the full data distribution. A generator producing perfect "1"s can achieve low loss even though it completely misses 9 out of 10 modes.

**Part 3**: Several metrics:
- **Inception Score (IS)**: Measures both quality (sharp class predictions) AND diversity (uniform class distribution). Mode collapse → low entropy of marginal $p(y)$ → low IS.
- **Frechet Inception Distance (FID)**: Compares statistics (mean and covariance of features) between real and generated distributions. Mode collapse → generated feature covariance is much smaller → high FID.
- **Sample diversity**: Compute pairwise distances between generated samples. Mode collapse → very small pairwise distances.

</details>

---

## Exercise 5: Wasserstein Distance Intuition

**Target time**: 3 minutes

Consider two 1D distributions:
- $P$: point mass at $x = 0$
- $Q_\theta$: point mass at $x = \theta$

**Part 1**: Compute the JS divergence $D_{JS}(P \| Q_\theta)$ when $\theta \neq 0$ and when $\theta = 0$.

**Part 2**: Compute the Wasserstein-1 distance $W_1(P, Q_\theta)$.

**Part 3**: Why does the Wasserstein distance give a better training signal than JS divergence?

<details>
<summary>Solution</summary>

**Part 1**: When $\theta \neq 0$: $P$ and $Q_\theta$ have disjoint supports. The mixture $M = (P + Q)/2$ has probability 0.5 at both $x=0$ and $x=\theta$.

$D_{JS} = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M) = \frac{1}{2}\log\frac{1}{0.5} + \frac{1}{2}\log\frac{1}{0.5} = \log 2 \approx 0.693$

This is a CONSTANT for all $\theta \neq 0$. When $\theta = 0$: $D_{JS} = 0$.

So $D_{JS}$ jumps discontinuously from $\log 2$ to $0$ — no useful gradient!

**Part 2**:
$W_1(P, Q_\theta) = |\theta|$

This is simply the distance between the two point masses. It varies smoothly and continuously with $\theta$.

**Part 3**: The Wasserstein distance provides a gradient everywhere: $\frac{\partial W_1}{\partial \theta} = \text{sign}(\theta)$. This gradient tells the generator to move $Q_\theta$ toward $P$ regardless of how far apart they are.

JS divergence is flat ($= \log 2$) for all $\theta \neq 0$, giving zero gradient — the generator has no signal about which direction to move. This is why WGAN training is more stable.

</details>

---

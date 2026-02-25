# Variational Autoencoder Exercises

**5 exercises** | Covers: ELBO derivation, KL divergence computation, reparameterization trick, VAE loss, posterior collapse

---

## Exercise 1: Derive ELBO from Scratch

**Target time**: 8 minutes

Starting from $\log p(x)$, derive the Evidence Lower Bound (ELBO) step by step.

**Part 1**: Show that $\log p(x) = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] + D_{KL}(q(z|x) \| p(z|x))$.

**Part 2**: Since $D_{KL} \geq 0$, conclude that $\log p(x) \geq \text{ELBO}$.

**Part 3**: Decompose the ELBO into reconstruction + KL terms:
$$\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

<details>
<summary>Solution</summary>

**Part 1**:
$$\log p(x) = \log p(x) \cdot 1 = \log p(x) \cdot \int q(z|x) dz = \int q(z|x) \log p(x) \, dz$$

$$= \int q(z|x) \log \frac{p(x,z)}{p(z|x)} dz = \int q(z|x) \log \frac{p(x,z) \cdot q(z|x)}{p(z|x) \cdot q(z|x)} dz$$

$$= \int q(z|x) \log \frac{p(x,z)}{q(z|x)} dz + \int q(z|x) \log \frac{q(z|x)}{p(z|x)} dz$$

$$= \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] + D_{KL}(q(z|x) \| p(z|x))$$

**Part 2**: Since $D_{KL}(q \| p) \geq 0$ always:
$$\log p(x) \geq \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] = \text{ELBO}$$

**Part 3**:
$$\text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x|z)p(z)}{q(z|x)}\right]$$

$$= \mathbb{E}_{q(z|x)}[\log p(x|z)] + \mathbb{E}_{q(z|x)}\left[\log \frac{p(z)}{q(z|x)}\right]$$

$$= \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{Regularization}}$$

</details>

---

## Exercise 2: Compute KL Divergence for Gaussians

**Target time**: 5 minutes

A VAE encoder outputs for a single data point:
- $\mu = [1.5, -0.5]$
- $\log \sigma^2 = [0.4, -0.6]$ (i.e., $\sigma^2 = [e^{0.4}, e^{-0.6}] \approx [1.492, 0.549]$)

Prior: $p(z) = \mathcal{N}(0, I)$

**Part 1**: Write the KL divergence formula for diagonal Gaussian $q = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ vs. $p = \mathcal{N}(0, I)$.

**Part 2**: Compute $D_{KL}$ numerically for the given $\mu$ and $\log \sigma^2$.

**Part 3**: Which dimension contributes more to the KL? Why?

<details>
<summary>Solution</summary>

**Part 1**:
$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**Part 2**: For each dimension:

Dimension 1 ($\mu_1 = 1.5, \log\sigma_1^2 = 0.4, \sigma_1^2 = 1.492$):
$$-\frac{1}{2}(1 + 0.4 - 1.5^2 - 1.492) = -\frac{1}{2}(1 + 0.4 - 2.25 - 1.492) = -\frac{1}{2}(-2.342) = 1.171$$

Dimension 2 ($\mu_2 = -0.5, \log\sigma_2^2 = -0.6, \sigma_2^2 = 0.549$):
$$-\frac{1}{2}(1 + (-0.6) - (-0.5)^2 - 0.549) = -\frac{1}{2}(1 - 0.6 - 0.25 - 0.549) = -\frac{1}{2}(-0.399) = 0.200$$

$$D_{KL} = 1.171 + 0.200 = 1.371$$

**Part 3**: Dimension 1 contributes much more (1.171 vs 0.200). This is primarily because $|\mu_1| = 1.5$ is much larger than $|\mu_2| = 0.5$ — the mean is far from the prior mean of 0. Additionally, $\sigma_1^2 = 1.492$ is larger than the prior variance of 1, adding a penalty.

</details>

---

## Exercise 3: Reparameterization Trick

**Target time**: 4 minutes

**Part 1**: A VAE encoder outputs $\mu = [2.0, -1.0]$ and $\log \sigma^2 = [0, -1]$. A random sample $\epsilon = [0.5, -0.3]$ is drawn from $\mathcal{N}(0, I)$. Compute $z$ using the reparameterization trick.

**Part 2**: Compute $\frac{\partial z}{\partial \mu}$ and $\frac{\partial z}{\partial \sigma}$. Show that gradients can flow.

**Part 3**: If instead we sampled $z$ directly from $\mathcal{N}(\mu, \sigma^2 I)$, why can't we compute $\frac{\partial z}{\partial \mu}$?

<details>
<summary>Solution</summary>

**Part 1**:
First, compute $\sigma$:
$\sigma = \exp(0.5 \cdot \log \sigma^2) = [\exp(0), \exp(-0.5)] = [1.0, 0.6065]$

Then: $z = \mu + \sigma \odot \epsilon = [2.0 + 1.0 \times 0.5, \; -1.0 + 0.6065 \times (-0.3)]$
$= [2.5, -1.182]$

**Part 2**:
$z_j = \mu_j + \sigma_j \epsilon_j$

$\frac{\partial z_j}{\partial \mu_j} = 1$ (gradients flow directly through $\mu$)

$\frac{\partial z_j}{\partial \sigma_j} = \epsilon_j$ (gradients flow through $\sigma$, scaled by the fixed noise)

Both are well-defined and non-zero, so backpropagation works.

**Part 3**: If $z \sim \mathcal{N}(\mu, \sigma^2 I)$ is sampled directly (e.g., using `torch.normal(mu, sigma)`), the sampling operation is a stochastic node with no deterministic path from $\mu, \sigma$ to $z$. There is no differentiable function connecting the parameters to the output — the gradient $\frac{\partial z}{\partial \mu}$ is undefined because $z$ is a random variable, not a deterministic function of $\mu$. The reparameterization trick makes $z = \mu + \sigma \epsilon$ a deterministic function of $(\mu, \sigma, \epsilon)$, and since $\epsilon$ is fixed (independent of parameters), gradients are well-defined.

</details>

---

## Exercise 4: VAE Loss Computation

**Target time**: 5 minutes

A VAE processes a single 4-pixel image $x = [1, 0, 1, 0]$ with latent dimension $d=2$.

The encoder outputs: $\mu = [0.8, -0.3]$, $\log\sigma^2 = [-0.5, 0.2]$

After reparameterization with $\epsilon = [0.1, 0.4]$, the decoder outputs: $\hat{x} = [0.9, 0.1, 0.8, 0.2]$

**Part 1**: Compute the BCE reconstruction loss.

**Part 2**: Compute the KL divergence.

**Part 3**: Compute the total VAE loss.

<details>
<summary>Solution</summary>

**Part 1**: BCE (summed, not averaged):
$$\mathcal{L}_{\text{recon}} = -\sum_{j=1}^{4}[x_j \log \hat{x}_j + (1-x_j)\log(1-\hat{x}_j)]$$

- $j=1$: $1 \cdot \log(0.9) + 0 \cdot \log(0.1) = -0.1054$
- $j=2$: $0 \cdot \log(0.1) + 1 \cdot \log(0.9) = -0.1054$
- $j=3$: $1 \cdot \log(0.8) + 0 \cdot \log(0.2) = -0.2231$
- $j=4$: $0 \cdot \log(0.2) + 1 \cdot \log(0.8) = -0.2231$

$\mathcal{L}_{\text{recon}} = -(-0.1054 - 0.1054 - 0.2231 - 0.2231) = 0.6570$

**Part 2**: KL:
$\sigma_1^2 = e^{-0.5} = 0.6065$, $\sigma_2^2 = e^{0.2} = 1.2214$

$D_{KL} = -\frac{1}{2}[(1 + (-0.5) - 0.64 - 0.6065) + (1 + 0.2 - 0.09 - 1.2214)]$

$= -\frac{1}{2}[(-0.7465) + (-0.1114)] = -\frac{1}{2}(-0.8579) = 0.4290$

**Part 3**: Total loss $= 0.6570 + 0.4290 = 1.086$

</details>

---

## Exercise 5: Posterior Collapse

**Target time**: 3 minutes

During VAE training, you observe the following:
- Epoch 1: Recon loss = 200, KL = 15
- Epoch 10: Recon loss = 55, KL = 8
- Epoch 50: Recon loss = 42, KL = 0.01
- The decoder ignores $z$ and produces the dataset mean for all inputs.

**Part 1**: What has happened? Why is $D_{KL} \approx 0$?

**Part 2**: What does $D_{KL} \approx 0$ imply about $q(z|x)$ for every $x$?

**Part 3**: Name two strategies to prevent this problem.

<details>
<summary>Solution</summary>

**Part 1**: This is **posterior collapse**. The KL term has driven $q(z|x)$ to match $p(z) = \mathcal{N}(0, I)$ for all inputs, meaning the encoder outputs $\mu \approx 0, \sigma^2 \approx 1$ regardless of the input. The decoder has learned to ignore $z$ (since $z$ carries no information about $x$) and outputs the unconditional mean of the data.

**Part 2**: $D_{KL}(q(z|x) \| p(z)) \approx 0$ means $q(z|x) \approx p(z) = \mathcal{N}(0, I)$ for every input $x$. The latent code $z$ is independent of $x$ — the encoder has "collapsed" to producing the prior regardless of what it sees.

**Part 3**:
1. **KL annealing (warm-up)**: Start with KL weight $\beta = 0$ and gradually increase to $\beta = 1$ over training. This lets the encoder first learn useful representations before the KL penalty kicks in.
2. **Free bits / minimum KL**: Set a minimum KL per dimension. If $D_{KL,j} < \lambda$, don't penalize it. This ensures the model uses the latent space.
3. **$\beta$-VAE with $\beta < 1$**: Reduce the weight of the KL term: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}$ with $\beta < 1$.

</details>

---

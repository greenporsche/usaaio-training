# 06 — Probability and Statistics

**Prerequisites**: Basic algebra. Helpful: `01-vector-spaces.md` (for multivariate distributions), `03-eigenvalues-eigenvectors.md` (for PSD covariance matrices)
**USAAIO Relevance**: **High priority.** Bayes' theorem, Gaussian distributions, and MLE appear on nearly every USAAIO exam. Understanding probability is essential for loss functions, generative models, and Bayesian reasoning.

---

## Discovery

It's 1763, and the Reverend Thomas Bayes has just passed away. His friend Richard Price discovers a manuscript that poses a deceptively simple question: if you observe some evidence, how should you update your beliefs?

**Motivating challenge**: A medical test for a rare disease has:
- Sensitivity (true positive rate): 99% — if you're sick, the test catches it 99% of the time
- Specificity (true negative rate): 95% — if you're healthy, the test correctly says "negative" 95% of the time
- Disease prevalence: 1% of the population has the disease

You test positive. What's the probability you actually have the disease?

Most people guess ~99%. The actual answer is about 17%. This shocking gap is the essence of Bayes' theorem — and it's the foundation of probabilistic reasoning in AI.

**Socratic questions**:
1. If 1000 people are tested, how many are sick? (10) How many of those test positive? (about 10)
2. How many of the 990 healthy people test positive? (about 50 — false positives!)
3. Of the ~60 total positive results, what fraction are truly sick? (10/60 $\approx$ 17%)

**Misconception trap**: Confusing $P(\text{positive} | \text{sick})$ with $P(\text{sick} | \text{positive})$. These are fundamentally different quantities! Bayes' theorem connects them.

---

## Intuition

What you just worked through is **Bayes' theorem** in action. The key insight: the probability of a hypothesis given evidence depends not just on how well the evidence matches the hypothesis, but also on how likely the hypothesis was *a priori* (the **prior**).

### The Bayesian Update Cycle

```
  Prior belief ──> Observe evidence ──> Updated belief (Posterior)
     P(H)              P(E|H)              P(H|E)
```

$$P(\text{posterior}) = \frac{P(\text{evidence} | \text{hypothesis}) \cdot P(\text{prior})}{P(\text{evidence})}$$

In ML, this is everywhere:
- **Prior**: what we believe about model parameters before seeing data
- **Likelihood**: how well the data fits given those parameters
- **Posterior**: what we believe after seeing the data

### Distributions as "Shape Templates"

Think of probability distributions as templates describing the shape of uncertainty:

```
  Bernoulli:     Gaussian:          Uniform:
    |              ___
  p |*            /   \            ________
    |            /     \          |        |
  1-p  |*      /       \         |        |
    +--+--+   +----+----+       +----+----+
    0  1       μ-2σ μ μ+2σ       a       b
```

### What Goes Wrong Without Probability?

- Can't reason about uncertainty in predictions
- Can't define loss functions (cross-entropy loss IS negative log-likelihood)
- Can't do Bayesian inference, which is the foundation of modern ML
- Can't understand why neural networks work (they implicitly approximate posteriors)

---

## Math

### Probability Axioms (Kolmogorov)

For a sample space $\Omega$ and events $A, B \subseteq \Omega$:
1. $P(A) \geq 0$
2. $P(\Omega) = 1$
3. If $A \cap B = \emptyset$: $P(A \cup B) = P(A) + P(B)$

### Key Rules

**Complement**: $P(A^c) = 1 - P(A)$

**Union**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

**Conditional probability**:

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

**Chain rule**: $P(A \cap B) = P(A|B) P(B) = P(B|A) P(A)$

**Law of total probability**: If $\{B_1, \ldots, B_n\}$ partition $\Omega$:

$$P(A) = \sum_{i=1}^{n} P(A | B_i) P(B_i)$$

**Independence**: $A$ and $B$ are independent iff $P(A \cap B) = P(A)P(B)$, equivalently $P(A|B) = P(A)$.

### Bayes' Theorem

$$P(H | E) = \frac{P(E | H) P(H)}{P(E)} = \frac{P(E|H) P(H)}{\sum_i P(E|H_i) P(H_i)}$$

| Term | Name | Interpretation |
|------|------|---------------|
| $P(H)$ | Prior | Belief before evidence |
| $P(E\|H)$ | Likelihood | How probable is evidence given hypothesis |
| $P(H\|E)$ | Posterior | Updated belief after evidence |
| $P(E)$ | Evidence / Marginal likelihood | Normalizing constant |

*Reasoning required*: Deriving Bayes' theorem from the definition of conditional probability.

### Discrete Distributions

**Bernoulli** ($X \in \{0, 1\}$):

$$P(X = x) = p^x (1-p)^{1-x}, \quad \mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Categorical** ($X \in \{1, \ldots, K\}$):

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

**Binomial** ($X$ = number of successes in $n$ trials):

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad \mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

### Continuous Distributions

**Gaussian (Normal)** ($X \sim \mathcal{N}(\mu, \sigma^2)$):

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**68-95-99.7 rule**: $P(\mu - k\sigma \leq X \leq \mu + k\sigma)$ for $k = 1, 2, 3$ is approximately 68%, 95%, 99.7%.

**Multivariate Gaussian** ($\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, $\mathbf{X} \in \mathbb{R}^d$):

$$f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

The covariance matrix $\Sigma$ must be positive semi-definite. Its eigenvectors define the principal axes of the elliptical contours, and its eigenvalues define the spread along each axis.

*Reasoning not required*: Proof that this normalizes to 1.

### Expectation, Variance, Covariance

**Expectation** (linear operator):

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

**Variance**:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

**Covariance**:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

If $X, Y$ are independent: $\text{Cov}(X, Y) = 0$ (but NOT the converse!).

**Covariance matrix** for random vector $\mathbf{X} = [X_1, \ldots, X_d]^\top$:

$$\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top] \in \mathbb{R}^{d \times d}$$

$\Sigma_{ij} = \text{Cov}(X_i, X_j)$. This matrix is always symmetric and PSD.

*Reasoning required*: Prove $\Sigma$ is PSD: $\mathbf{a}^\top \Sigma \mathbf{a} = \text{Var}(\mathbf{a}^\top \mathbf{X}) \geq 0$ for all $\mathbf{a}$.

### Maximum Likelihood Estimation (MLE)

Given data $\{x_1, \ldots, x_N\}$ drawn i.i.d. from distribution $f(x; \theta)$, the MLE is:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^{N} f(x_i; \theta) = \arg\max_\theta \sum_{i=1}^{N} \log f(x_i; \theta)$$

**MLE for Gaussian**:

$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

$$\hat{\sigma}^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{\mu})^2$$

**Derivation** (for $\mu$, given known $\sigma^2$):

$$\ell(\mu) = \sum_{i=1}^{N} \log f(x_i; \mu, \sigma^2) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i - \mu)^2$$

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i - \mu) = 0 \implies \hat{\mu} = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad \blacksquare$$

*Reasoning required*: MLE derivation for Gaussian is a classic USAAIO problem.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """Apply Bayes' theorem.

    Args:
        prior: P(H)
        likelihood: P(E|H)
        evidence: P(E)
    Returns:
        posterior: P(H|E)
    """
    return (likelihood * prior) / evidence  # scalar

# Medical test example
prior_sick = 0.01  # P(sick)
sensitivity = 0.99  # P(positive | sick)
specificity = 0.95  # P(negative | healthy)
false_positive_rate = 1 - specificity  # P(positive | healthy) = 0.05

# P(positive) via law of total probability
p_positive = sensitivity * prior_sick + false_positive_rate * (1 - prior_sick)  # scalar
# = 0.99 * 0.01 + 0.05 * 0.99 = 0.0594

posterior = bayes_theorem(prior_sick, sensitivity, p_positive)  # scalar
print(f"P(sick | positive) = {posterior:.4f}")  # ~0.1667

def gaussian_pdf(x: np.ndarray, mu: float, sigma_sq: float) -> np.ndarray:
    """Univariate Gaussian PDF.

    Args:
        x: shape (N,) — points to evaluate
        mu: mean
        sigma_sq: variance
    Returns:
        pdf: shape (N,)
    """
    coeff = 1.0 / np.sqrt(2 * np.pi * sigma_sq)  # scalar
    exponent = -0.5 * (x - mu)**2 / sigma_sq  # (N,)
    return coeff * np.exp(exponent)  # (N,)

def multivariate_gaussian_pdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Multivariate Gaussian PDF.

    Args:
        X: shape (N, d) — data points
        mu: shape (d,) — mean
        Sigma: shape (d, d) — covariance matrix
    Returns:
        pdf: shape (N,)
    """
    N, d = X.shape
    diff = X - mu  # (N, d)
    Sigma_inv = np.linalg.inv(Sigma)  # (d, d)
    det_Sigma = np.linalg.det(Sigma)  # scalar

    coeff = 1.0 / np.sqrt((2 * np.pi)**d * det_Sigma)  # scalar
    # Mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu) for each row
    exponent = -0.5 * np.sum(diff @ Sigma_inv * diff, axis=1)  # (N,)

    return coeff * np.exp(exponent)  # (N,)

def mle_gaussian(X: np.ndarray) -> tuple:
    """MLE for univariate Gaussian parameters.

    Args:
        X: shape (N,) — observed data
    Returns:
        mu_hat: MLE mean
        sigma_sq_hat: MLE variance
    """
    N = len(X)
    mu_hat = np.mean(X)  # scalar
    sigma_sq_hat = np.mean((X - mu_hat)**2)  # scalar (biased MLE, not N-1)
    return mu_hat, sigma_sq_hat

def mle_gaussian_multivariate(X: np.ndarray) -> tuple:
    """MLE for multivariate Gaussian.

    Args:
        X: shape (N, d)
    Returns:
        mu_hat: shape (d,)
        Sigma_hat: shape (d, d)
    """
    N, d = X.shape
    mu_hat = np.mean(X, axis=0)  # (d,)
    diff = X - mu_hat  # (N, d)
    Sigma_hat = (diff.T @ diff) / N  # (d, d) = (d, N) @ (N, d) / scalar
    return mu_hat, Sigma_hat

# --- Demo ---
np.random.seed(42)

# Generate Gaussian data
true_mu, true_sigma_sq = 5.0, 4.0
X = np.random.normal(true_mu, np.sqrt(true_sigma_sq), size=1000)  # (1000,)

mu_hat, sigma_sq_hat = mle_gaussian(X)
print(f"True: mu={true_mu}, sigma^2={true_sigma_sq}")
print(f"MLE:  mu={mu_hat:.4f}, sigma^2={sigma_sq_hat:.4f}")

# Covariance matrix
X_2d = np.random.multivariate_normal([1, 2], [[3, 1], [1, 2]], size=500)  # (500, 2)
mu_hat_2d, Sigma_hat_2d = mle_gaussian_multivariate(X_2d)
print(f"\n2D MLE mean: {mu_hat_2d}")
print(f"2D MLE covariance:\n{Sigma_hat_2d}")

# Verify covariance is PSD (all eigenvalues >= 0)
eigvals = np.linalg.eigvalsh(Sigma_hat_2d)  # (2,)
print(f"Covariance eigenvalues: {eigvals} (all >= 0: {np.all(eigvals >= 0)})")
```

### PyTorch Equivalent

```python
import torch
import torch.distributions as dist

# Define a Gaussian distribution
normal = dist.Normal(loc=5.0, scale=2.0)  # mean=5, std=2

# Sample
samples = normal.sample((1000,))  # (1000,)

# Log probability (used in loss functions)
log_probs = normal.log_prob(samples)  # (1000,)

# Multivariate Gaussian
mvn = dist.MultivariateNormal(
    loc=torch.tensor([1.0, 2.0]),  # (2,)
    covariance_matrix=torch.tensor([[3.0, 1.0], [1.0, 2.0]])  # (2, 2)
)
samples_2d = mvn.sample((500,))  # (500, 2)
log_probs_2d = mvn.log_prob(samples_2d)  # (500,)

# Bernoulli
bern = dist.Bernoulli(probs=0.7)
binary_samples = bern.sample((100,))  # (100,)

# Categorical (one-hot)
cat = dist.Categorical(probs=torch.tensor([0.2, 0.3, 0.5]))
class_samples = cat.sample((100,))  # (100,) — values in {0, 1, 2}
```

---

## Resources

- [StatQuest: Bayes' Theorem](https://www.youtube.com/watch?v=9wCnvr7Xw4E)
- [3Blue1Brown: Bayes' theorem](https://www.3blue1brown.com/lessons/bayes-theorem) — visual explanation
- MML Book, Chapter 6: Probability and Distributions
- Bishop, *Pattern Recognition and Machine Learning*, Chapter 2

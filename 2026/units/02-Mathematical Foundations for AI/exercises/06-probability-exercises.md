# Exercises: Probability and Statistics

**Target time**: 2-5 minutes per exercise | **Total**: 6 exercises

---

## Exercise 6.1 — Bayes' Theorem Application

A spam filter classifies emails. Given:
- $P(\text{spam}) = 0.3$
- $P(\text{"free"} | \text{spam}) = 0.8$
- $P(\text{"free"} | \text{not spam}) = 0.1$

An email contains the word "free." What is the probability it is spam?

<details><summary>Solution</summary>

By Bayes' theorem:

$$P(\text{spam} | \text{"free"}) = \frac{P(\text{"free"} | \text{spam}) \cdot P(\text{spam})}{P(\text{"free"})}$$

First compute $P(\text{"free"})$ using the law of total probability:

$$P(\text{"free"}) = P(\text{"free"}|\text{spam})P(\text{spam}) + P(\text{"free"}|\text{not spam})P(\text{not spam})$$
$$= 0.8 \times 0.3 + 0.1 \times 0.7 = 0.24 + 0.07 = 0.31$$

$$P(\text{spam} | \text{"free"}) = \frac{0.8 \times 0.3}{0.31} = \frac{0.24}{0.31} \approx \mathbf{0.774}$$

So a 30% prior probability of spam jumps to ~77% after observing "free."

</details>

---

## Exercise 6.2 — Gaussian MLE Derivation

Derive the maximum likelihood estimator for $\sigma^2$ of a Gaussian $\mathcal{N}(\mu, \sigma^2)$ given i.i.d. samples $x_1, \ldots, x_N$ (assume $\mu$ is known).

<details><summary>Solution</summary>

Log-likelihood:

$$\ell(\sigma^2) = \sum_{i=1}^{N} \log f(x_i; \mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i - \mu)^2$$

Take derivative with respect to $\sigma^2$ (treating $\sigma^2$ as a single variable):

$$\frac{\partial \ell}{\partial(\sigma^2)} = -\frac{N}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{N}(x_i - \mu)^2 = 0$$

$$\frac{N}{2\sigma^2} = \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{N}(x_i - \mu)^2$$

$$N\sigma^2 = \sum_{i=1}^{N}(x_i - \mu)^2$$

$$\boxed{\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$

Note: This is a *biased* estimator. The unbiased estimator uses $N-1$ instead of $N$.

</details>

---

## Exercise 6.3 — Covariance Matrix Properties

Let $\mathbf{X} = \begin{bmatrix}X_1 \\ X_2\end{bmatrix}$ be a random vector with covariance matrix $\Sigma = \begin{bmatrix}4 & 2 \\ 2 & 3\end{bmatrix}$.

**(a)** What is $\text{Var}(X_1)$? $\text{Var}(X_2)$? $\text{Cov}(X_1, X_2)$?

**(b)** Compute the correlation $\rho(X_1, X_2) = \frac{\text{Cov}(X_1, X_2)}{\sqrt{\text{Var}(X_1)\text{Var}(X_2)}}$.

**(c)** Is $\Sigma$ positive definite? Verify using eigenvalues.

**(d)** Compute $\text{Var}(2X_1 - X_2)$.

<details><summary>Solution</summary>

**(a)** $\text{Var}(X_1) = \Sigma_{11} = 4$, $\text{Var}(X_2) = \Sigma_{22} = 3$, $\text{Cov}(X_1, X_2) = \Sigma_{12} = 2$.

**(b)** $\rho = \frac{2}{\sqrt{4 \cdot 3}} = \frac{2}{2\sqrt{3}} = \frac{1}{\sqrt{3}} \approx \mathbf{0.577}$

**(c)** Eigenvalues: $(4-\lambda)(3-\lambda) - 4 = \lambda^2 - 7\lambda + 8 = 0$

$\lambda = \frac{7 \pm \sqrt{49-32}}{2} = \frac{7 \pm \sqrt{17}}{2}$

$\lambda_1 = \frac{7+\sqrt{17}}{2} \approx 5.56 > 0$, $\lambda_2 = \frac{7-\sqrt{17}}{2} \approx 1.44 > 0$

Both positive, so $\Sigma$ is **positive definite**. $\checkmark$

**(d)** Let $Y = 2X_1 - X_2 = \mathbf{a}^\top\mathbf{X}$ where $\mathbf{a} = \begin{bmatrix}2\\-1\end{bmatrix}$.

$\text{Var}(Y) = \mathbf{a}^\top \Sigma \mathbf{a} = \begin{bmatrix}2&-1\end{bmatrix}\begin{bmatrix}4&2\\2&3\end{bmatrix}\begin{bmatrix}2\\-1\end{bmatrix} = \begin{bmatrix}2&-1\end{bmatrix}\begin{bmatrix}6\\1\end{bmatrix} = 12 - 1 = \mathbf{11}$

</details>

---

## Exercise 6.4 — Independence vs Uncorrelated

**(a)** If $X$ and $Y$ are independent, prove that $\text{Cov}(X, Y) = 0$.

**(b)** Give an example where $\text{Cov}(X, Y) = 0$ but $X$ and $Y$ are NOT independent.

<details><summary>Solution</summary>

**(a)** If $X, Y$ are independent: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]\mathbb{E}[Y] = 0$ $\blacksquare$

**(b)** Classic example: Let $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$.

$\mathbb{E}[X] = 0$ (symmetric distribution)

$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] - 0 = 0$ (since $X^3$ is an odd function of $X$ with symmetric distribution)

But $X$ and $Y$ are clearly **not** independent: knowing $X$ completely determines $Y = X^2$!

**Takeaway**: Uncorrelated ($\text{Cov} = 0$) does NOT imply independence. However, for jointly Gaussian random variables, uncorrelated DOES imply independent.

</details>

---

## Exercise 6.5 — Law of Total Expectation

A factory has two machines. Machine A produces 60% of items, Machine B produces 40%. Machine A has a 2% defect rate, Machine B has a 5% defect rate.

**(a)** What is the overall defect rate?

**(b)** If an item is defective, what is the probability it came from Machine B?

<details><summary>Solution</summary>

**(a)** By the law of total probability:

$$P(\text{defect}) = P(\text{defect}|A)P(A) + P(\text{defect}|B)P(B) = 0.02 \times 0.6 + 0.05 \times 0.4 = 0.012 + 0.02 = \mathbf{0.032}$$

Overall defect rate: 3.2%.

**(b)** By Bayes' theorem:

$$P(B | \text{defect}) = \frac{P(\text{defect}|B)P(B)}{P(\text{defect})} = \frac{0.05 \times 0.4}{0.032} = \frac{0.02}{0.032} = \mathbf{0.625}$$

So 62.5% of defective items come from Machine B, despite it producing only 40% of all items.

</details>

---

## Exercise 6.6 — USAAIO Competition Style

Let $X_1, X_2, \ldots, X_N$ be i.i.d. samples from $\mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$.

**(a)** Write the log-likelihood function $\ell(\mu)$.

**(b)** Show that $\hat{\mu}_{MLE} = \bar{X} = \frac{1}{N}\sum X_i$.

**(c)** Compute $\text{Var}(\hat{\mu}_{MLE})$. What happens as $N \to \infty$?

**(d)** Show that $\hat{\mu}_{MLE}$ is unbiased: $\mathbb{E}[\hat{\mu}_{MLE}] = \mu$.

<details><summary>Solution</summary>

**(a)** $\ell(\mu) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(X_i - \mu)^2$

**(b)** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{N}(X_i - \mu) = \frac{1}{\sigma^2}\left(\sum X_i - N\mu\right) = 0$

$\implies \hat{\mu}_{MLE} = \frac{1}{N}\sum_{i=1}^{N} X_i = \bar{X}$ $\blacksquare$

**(c)** $\text{Var}(\hat{\mu}) = \text{Var}\left(\frac{1}{N}\sum X_i\right) = \frac{1}{N^2}\sum \text{Var}(X_i) = \frac{1}{N^2} \cdot N\sigma^2 = \frac{\sigma^2}{N}$

As $N \to \infty$: $\text{Var}(\hat{\mu}) \to 0$. The estimate becomes more precise with more data. This is the **law of large numbers** in action.

**(d)** $\mathbb{E}[\hat{\mu}] = \mathbb{E}\left[\frac{1}{N}\sum X_i\right] = \frac{1}{N}\sum \mathbb{E}[X_i] = \frac{1}{N} \cdot N\mu = \mu$ $\blacksquare$

$\hat{\mu}_{MLE}$ is unbiased: its expected value equals the true parameter.

</details>

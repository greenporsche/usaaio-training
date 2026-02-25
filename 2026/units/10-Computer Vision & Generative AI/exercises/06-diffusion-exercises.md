# Diffusion Model Exercises

**5 exercises** | Covers: noise schedule, forward process, closed-form computation, training objective, sampling algorithm

---

## Exercise 1: Compute the Noise Schedule

**Target time**: 4 minutes

A DDPM uses $T = 5$ timesteps with linear beta schedule: $\beta_1 = 0.01, \beta_5 = 0.05$.

**Part 1**: Compute $\beta_t$ for $t = 1, 2, 3, 4, 5$.

**Part 2**: Compute $\alpha_t = 1 - \beta_t$ for each $t$.

**Part 3**: Compute $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ for each $t$. How close is $\bar{\alpha}_5$ to 0?

<details>
<summary>Solution</summary>

**Part 1**: Linear interpolation: $\beta_t = 0.01 + \frac{t-1}{4}(0.05 - 0.01)$

| $t$ | $\beta_t$ |
|---|---|
| 1 | 0.010 |
| 2 | 0.020 |
| 3 | 0.030 |
| 4 | 0.040 |
| 5 | 0.050 |

**Part 2**:

| $t$ | $\alpha_t = 1 - \beta_t$ |
|---|---|
| 1 | 0.990 |
| 2 | 0.980 |
| 3 | 0.970 |
| 4 | 0.960 |
| 5 | 0.950 |

**Part 3**:

| $t$ | $\bar{\alpha}_t$ |
|---|---|
| 1 | 0.990 |
| 2 | 0.990 × 0.980 = 0.9702 |
| 3 | 0.9702 × 0.970 = 0.9411 |
| 4 | 0.9411 × 0.960 = 0.9034 |
| 5 | 0.9034 × 0.950 = 0.8583 |

$\bar{\alpha}_5 = 0.858$ — still quite far from 0. With only $T=5$ steps, significant signal remains. In DDPM, $T=1000$ ensures $\bar{\alpha}_T \approx 0$ (pure noise). With our schedule, you'd need many more steps.

</details>

---

## Exercise 2: Forward Diffusion Computation

**Target time**: 5 minutes

Given a 1D "image" $x_0 = 3.0$ and noise schedule $\bar{\alpha}_t$ from Exercise 1.

**Part 1**: Using the closed-form $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ with $\epsilon = 0.5$, compute $x_t$ for $t = 1, 3, 5$.

**Part 2**: What is the signal-to-noise ratio (SNR) at each $t$? Define SNR as $\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$.

**Part 3**: Verify $x_1$ by computing it step-by-step: $x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$ with $\epsilon_1 = 0.5$. Does it match the closed-form?

<details>
<summary>Solution</summary>

**Part 1**:

$t=1$: $x_1 = \sqrt{0.990} \times 3.0 + \sqrt{0.010} \times 0.5 = 0.995 \times 3.0 + 0.100 \times 0.5 = 2.985 + 0.050 = 3.035$

$t=3$: $x_3 = \sqrt{0.9411} \times 3.0 + \sqrt{0.0589} \times 0.5 = 0.9701 \times 3.0 + 0.2427 \times 0.5 = 2.910 + 0.121 = 3.031$

$t=5$: $x_5 = \sqrt{0.8583} \times 3.0 + \sqrt{0.1417} \times 0.5 = 0.9265 \times 3.0 + 0.3764 \times 0.5 = 2.779 + 0.188 = 2.968$

**Part 2**: SNR = $\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$:

| $t$ | $\bar{\alpha}_t$ | $1-\bar{\alpha}_t$ | SNR |
|---|---|---|---|
| 1 | 0.990 | 0.010 | 99.0 |
| 3 | 0.941 | 0.059 | 15.9 |
| 5 | 0.858 | 0.142 | 6.05 |

SNR decreases as $t$ increases — noise becomes more dominant relative to signal.

**Part 3**: Step-by-step: $x_1 = \sqrt{0.99} \times 3.0 + \sqrt{0.01} \times 0.5 = 0.9950 \times 3.0 + 0.1 \times 0.5 = 2.985 + 0.05 = 3.035$

This matches the closed-form result! The closed-form is correct because for $t=1$, $\bar{\alpha}_1 = \alpha_1$, so they are equivalent.

</details>

---

## Exercise 3: Training Objective Derivation

**Target time**: 4 minutes

The DDPM training objective samples $(x_0, t, \epsilon)$ and minimizes $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$.

**Part 1**: Given $x_0 = [1.0, 2.0]$, $t = 3$ (from Exercise 1, $\bar{\alpha}_3 = 0.941$), $\epsilon = [0.5, -0.3]$, compute $x_t$.

**Part 2**: Suppose the model predicts $\epsilon_\theta(x_t, 3) = [0.4, -0.2]$. Compute the loss.

**Part 3**: In what direction should the model update to reduce this loss? What would the perfect prediction be?

<details>
<summary>Solution</summary>

**Part 1**:
$x_3 = \sqrt{0.941} \cdot [1.0, 2.0] + \sqrt{0.059} \cdot [0.5, -0.3]$

$= 0.9701 \cdot [1.0, 2.0] + 0.2427 \cdot [0.5, -0.3]$

$= [0.970, 1.940] + [0.121, -0.073]$

$= [1.091, 1.867]$

**Part 2**:
$\mathcal{L} = \|\epsilon - \epsilon_\theta\|^2 = \|[0.5 - 0.4, -0.3 - (-0.2)]\|^2 = \|[0.1, -0.1]\|^2 = 0.01 + 0.01 = 0.02$

**Part 3**: The gradient of MSE loss w.r.t. $\epsilon_\theta$ is $2(\epsilon_\theta - \epsilon)$:

$\nabla_{\epsilon_\theta} \mathcal{L} = 2 \cdot ([0.4, -0.2] - [0.5, -0.3]) = 2 \cdot [-0.1, 0.1] = [-0.2, 0.2]$

The model should update to increase its first component and decrease its second. The perfect prediction is $\epsilon_\theta = \epsilon = [0.5, -0.3]$, giving loss = 0.

</details>

---

## Exercise 4: Sampling Step

**Target time**: 5 minutes

Given a trained model, we're sampling at timestep $t = 3$ (going from $x_3$ to $x_2$).

From Exercise 1: $\alpha_3 = 0.97$, $\beta_3 = 0.03$, $\bar{\alpha}_3 = 0.941$.

Current noisy sample: $x_3 = [1.5, -0.5]$

Model prediction: $\epsilon_\theta(x_3, 3) = [0.3, 0.1]$

**Part 1**: Compute the predicted mean using:
$$\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\right)$$

**Part 2**: With noise $z = [0.2, -0.1]$ and $\sigma_t = \sqrt{\beta_t}$, compute $x_2 = \mu + \sigma_t z$.

**Part 3**: Why do we add noise $z$ during sampling (for $t > 1$) but not at $t = 1$?

<details>
<summary>Solution</summary>

**Part 1**:
$\frac{1}{\sqrt{0.97}} = \frac{1}{0.9849} = 1.0153$

$\frac{\beta_3}{\sqrt{1-\bar{\alpha}_3}} = \frac{0.03}{\sqrt{0.059}} = \frac{0.03}{0.2427} = 0.1236$

$\mu = 1.0153 \cdot ([1.5, -0.5] - 0.1236 \cdot [0.3, 0.1])$
$= 1.0153 \cdot ([1.5, -0.5] - [0.0371, 0.0124])$
$= 1.0153 \cdot [1.4629, -0.5124]$
$= [1.4853, -0.5203]$

**Part 2**:
$\sigma_3 = \sqrt{0.03} = 0.1732$

$x_2 = [1.4853, -0.5203] + 0.1732 \cdot [0.2, -0.1]$
$= [1.4853 + 0.0346, -0.5203 - 0.0173]$
$= [1.520, -0.538]$

**Part 3**: The reverse process $p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu, \sigma_t^2 I)$ is a Gaussian distribution. For $t > 1$, we sample from this distribution by adding noise $\sigma_t z$. For $t = 1$ (final step), we want the clean image $x_0$, not a noisy version, so we take just the mean $\mu$ without adding noise. This gives the sharpest, most likely reconstruction.

</details>

---

## Exercise 5: Predict $x_0$ from $x_t$

**Target time**: 3 minutes

From the forward equation $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, we can rearrange to predict $x_0$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Part 1**: Given $x_5 = [2.0, -1.0]$, $\bar{\alpha}_5 = 0.858$, and $\epsilon_\theta = [0.6, -0.4]$, compute $\hat{x}_0$.

**Part 2**: Does this formula work well at large $t$ (near pure noise)? Why or why not?

**Part 3**: In DDIM (Denoising Diffusion Implicit Models), this $\hat{x}_0$ prediction is used at every step. What advantage does this give over standard DDPM sampling?

<details>
<summary>Solution</summary>

**Part 1**:
$\hat{x}_0 = \frac{[2.0, -1.0] - \sqrt{1-0.858} \cdot [0.6, -0.4]}{\sqrt{0.858}}$

$= \frac{[2.0, -1.0] - 0.3768 \cdot [0.6, -0.4]}{0.9264}$

$= \frac{[2.0 - 0.226, -1.0 + 0.151]}{0.9264}$

$= \frac{[1.774, -0.849]}{0.9264}$

$= [1.914, -0.916]$

**Part 2**: At large $t$, $\bar{\alpha}_t \approx 0$, so $\sqrt{\bar{\alpha}_t} \approx 0$. We divide by a very small number, amplifying any error in $\epsilon_\theta$:

$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$

A small error $\delta$ in $\epsilon_\theta$ causes error $\frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \delta$ in $\hat{x}_0$, which blows up as $\bar{\alpha}_t \to 0$.

**Part 3**: DDIM uses the $\hat{x}_0$ prediction to take larger steps in the reverse process, allowing sampling with fewer steps (e.g., 50 instead of 1000). DDIM can also be made deterministic (no noise added), making the generation process reproducible. The key tradeoff: DDPM needs all $T$ steps; DDIM can skip steps by predicting $\hat{x}_0$ at each stage.

</details>

---

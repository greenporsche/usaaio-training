# Stable Diffusion Exercises

**5 exercises** | Covers: latent space compression, cross-attention, classifier-free guidance, pipeline tracing, conditioning

---

## Exercise 1: Latent Space Compression

**Target time**: 3 minutes

Stable Diffusion uses a pretrained autoencoder that compresses images from pixel space to latent space.

**Part 1**: An image has shape $(B, 3, 512, 512)$. The autoencoder compresses with factor 8 spatially and outputs 4 latent channels. What is the latent shape?

**Part 2**: Compute the compression ratio (number of elements in image / number of elements in latent).

**Part 3**: The UNet in pixel-space DDPM would process a tensor of size $3 \times 512 \times 512 = 786{,}432$ per image. What size tensor does the latent-space UNet process? How much faster is each UNet forward pass (roughly)?

<details>
<summary>Solution</summary>

**Part 1**: Spatial: $512/8 = 64$. Latent shape: $(B, 4, 64, 64)$.

**Part 2**: Image elements: $3 \times 512 \times 512 = 786{,}432$. Latent elements: $4 \times 64 \times 64 = 16{,}384$. Compression ratio: $786{,}432 / 16{,}384 = 48\times$.

**Part 3**: The latent UNet processes tensors of size $16{,}384$ per image vs. $786{,}432$ in pixel space. Since UNet cost scales roughly as $O(n)$ to $O(n \log n)$ in the number of elements, this is approximately $48\times$ faster per step. In practice, the speedup is even larger because attention layers in the UNet scale quadratically with spatial dimensions — $(64 \times 64)^2 = 16M$ vs. $(512 \times 512)^2 = 68B$ — a $4096\times$ difference for self-attention.

</details>

---

## Exercise 2: Cross-Attention Shapes

**Target time**: 4 minutes

In a Stable Diffusion UNet block:
- Spatial feature map: $(B, 320, 64, 64)$ reshaped to $(B, 4096, 320)$ for attention
- Text embeddings from CLIP: $(B, 77, 768)$ (77 tokens, 768-dim)
- Cross-attention with $d_{model} = 320$, $d_k = 64$, 5 heads

**Part 1**: What are the shapes of $W_Q, W_K, W_V$?

**Part 2**: What are the shapes of $Q, K, V$?

**Part 3**: What is the shape of the attention weight matrix $\text{softmax}(QK^T/\sqrt{d_k})$? What does each entry represent?

<details>
<summary>Solution</summary>

**Part 1**:
- $W_Q$: $(320, 320)$ — projects spatial features to query space
- $W_K$: $(768, 320)$ — projects text embeddings to key space
- $W_V$: $(768, 320)$ — projects text embeddings to value space

**Part 2** (per head, $d_k = 320/5 = 64$):
- $Q = \text{spatial} \cdot W_Q$, then reshaped: $(B, 5, 4096, 64)$
- $K = \text{text} \cdot W_K$, then reshaped: $(B, 5, 77, 64)$
- $V = \text{text} \cdot W_V$, then reshaped: $(B, 5, 77, 64)$

**Part 3**:
$QK^T$: $(B, 5, 4096, 64) \times (B, 5, 64, 77) = (B, 5, 4096, 77)$

After softmax: $(B, 5, 4096, 77)$.

Entry $(b, h, i, j)$ = how much spatial position $i$ attends to text token $j$ (in head $h$, batch $b$). This is how the image generation is guided by specific words — a spatial location that should depict "cat" will have high attention to the text token "cat".

</details>

---

## Exercise 3: Classifier-Free Guidance

**Target time**: 4 minutes

At timestep $t$, the UNet produces:
- Unconditional noise prediction: $\epsilon_\theta(x_t, t, \emptyset) = [0.5, -0.3, 0.2]$
- Conditional noise prediction (with text): $\epsilon_\theta(x_t, t, c) = [0.8, -0.1, 0.6]$

**Part 1**: Compute the guided prediction with guidance scale $s = 7.5$:
$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

**Part 2**: What happens to $\tilde{\epsilon}$ as $s \to 0$? As $s \to 1$?

**Part 3**: Why does increasing $s$ beyond 1 improve adherence to the text prompt, but eventually cause artifacts?

<details>
<summary>Solution</summary>

**Part 1**:
Direction = $[0.8, -0.1, 0.6] - [0.5, -0.3, 0.2] = [0.3, 0.2, 0.4]$

$\tilde{\epsilon} = [0.5, -0.3, 0.2] + 7.5 \times [0.3, 0.2, 0.4]$
$= [0.5 + 2.25, -0.3 + 1.5, 0.2 + 3.0]$
$= [2.75, 1.2, 3.2]$

Note: these values are much larger than the original predictions! The guidance amplifies the conditioning signal.

**Part 2**:
- $s = 0$: $\tilde{\epsilon} = \epsilon_\theta(x_t, t, \emptyset)$ — purely unconditional (ignores text completely)
- $s = 1$: $\tilde{\epsilon} = \epsilon_\theta(x_t, t, c)$ — standard conditional prediction (no guidance amplification)

**Part 3**: At $s > 1$, the model *overshoots* in the direction from unconditional to conditional. This amplifies the text-relevant features: if the text says "cat", the image becomes more "cat-like". However, extremely high $s$ pushes the noise prediction far from the model's training distribution — the denoised samples become oversaturated, high-contrast, and exhibit artifacts because the model is being asked to extrapolate beyond what it learned.

</details>

---

## Exercise 4: Training with Conditioning Dropout

**Target time**: 3 minutes

During Stable Diffusion training, the text conditioning is randomly dropped with probability $p_{drop} = 0.1$ (replaced with a null embedding $\emptyset$).

**Part 1**: In a batch of 32 image-text pairs, how many samples (in expectation) will have their text conditioning dropped?

**Part 2**: Why is this dropout necessary? What would happen if we always conditioned on text?

**Part 3**: The model is trained on the same objective regardless of whether conditioning is present:
$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$$
where $c$ is either the real text embedding or $\emptyset$. How does a single model learn both conditional and unconditional generation?

<details>
<summary>Solution</summary>

**Part 1**: Expected dropped samples = $32 \times 0.1 = 3.2$ samples per batch.

**Part 2**: Without conditioning dropout, the model only learns $\epsilon_\theta(x_t, t, c)$ — it always expects text input. At inference time, classifier-free guidance requires *both* $\epsilon_\theta(x_t, t, c)$ and $\epsilon_\theta(x_t, t, \emptyset)$. If the model was never trained with $c = \emptyset$, the unconditional prediction would be garbage, and guidance would fail.

**Part 3**: The model learns a single function $\epsilon_\theta(x_t, t, c)$ that handles both cases:
- When $c = \text{text embedding}$: the model learns to predict noise given the text context (text-conditional denoising)
- When $c = \emptyset$: the model learns to predict noise without any text guidance (unconditional denoising)

The conditioning $c$ enters through cross-attention layers. When $c = \emptyset$, the cross-attention essentially becomes a no-op (attending to a null signal), so the model falls back to unconditional behavior. Both modes share the same convolutional weights, allowing joint training.

</details>

---

## Exercise 5: Full Pipeline Trace

**Target time**: 4 minutes

Trace the Stable Diffusion image generation pipeline for the prompt "a golden retriever on a beach":

**Part 1**: List every component involved and what each does (text encoder, UNet, autoencoder decoder).

**Part 2**: How many UNet forward passes are needed if $T = 50$ (DDIM) and guidance scale $s = 7.5$?

**Part 3**: What is the output shape at each stage for a 512x512 output image?

<details>
<summary>Solution</summary>

**Part 1**:
1. **CLIP Text Encoder** (frozen): Tokenizes "a golden retriever on a beach" → encodes to text embeddings $c \in \mathbb{R}^{77 \times 768}$.
2. **Random noise**: Sample $z_T \sim \mathcal{N}(0, I)$ in latent space.
3. **UNet** (trained): At each timestep $t$, predicts noise $\epsilon_\theta(z_t, t, c)$ conditioned on text via cross-attention. Also predicts unconditional $\epsilon_\theta(z_t, t, \emptyset)$ for guidance.
4. **DDIM scheduler**: Computes $z_{t-1}$ from $z_t$ using the guided noise prediction.
5. **VAE Decoder** (frozen): Decodes final latent $z_0$ to pixel-space image $x$.

**Part 2**: At each of 50 timesteps, we need TWO UNet forward passes (one conditional, one unconditional) for classifier-free guidance.

Total: $50 \times 2 = 100$ UNet forward passes.

(In practice, both can be batched into a single forward pass with batch size 2.)

**Part 3**:
1. Text embeddings: $(1, 77, 768)$
2. Initial latent noise: $(1, 4, 64, 64)$
3. Each UNet step: input $(1, 4, 64, 64)$ → output $(1, 4, 64, 64)$ (predicted noise, same shape)
4. Final latent: $(1, 4, 64, 64)$
5. After VAE decoder: $(1, 3, 512, 512)$ — the output image!

</details>

---

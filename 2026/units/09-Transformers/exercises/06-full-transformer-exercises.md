# Exercises 06 — Full Transformer Architecture

> 5 competition-level exercises

---

## Exercise 6.1 — Parameter Counting

A transformer encoder-decoder has:
- $D = 512, H = 8, D_{qk} = D_v = 64, D_{ff} = 2048$
- $N_{\text{enc}} = 6$ encoder layers, $N_{\text{dec}} = 6$ decoder layers
- Source vocabulary: 32,000 tokens, target vocabulary: 32,000 tokens

**(a)** Compute parameters per encoder layer: MHA + 2 LayerNorm + FFN.

**(b)** Compute parameters per decoder layer: masked self-attention + cross-attention + 3 LayerNorm + FFN.

**(c)** Compute embedding parameters (source + target + positional, assuming learned positional with max_len = 5000).

**(d)** Compute the output projection (linear + softmax over target vocabulary).

**(e)** Total parameters for the full model. Express in millions.

---

## Exercise 6.2 — Pre-Norm vs. Post-Norm

**Post-norm** encoder block:
```
x = LayerNorm(x + MHA(x, x))
x = LayerNorm(x + FFN(x))
```

**Pre-norm** encoder block:
```
x = x + MHA(LayerNorm(x), LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**(a)** In post-norm, what is the input to LayerNorm in the first sub-layer? What is its expected scale?

**(b)** In pre-norm, the residual path is $x + f(\text{LayerNorm}(x))$. Show that as $f$ approaches zero (early in training), the output is approximately $x$. Why is this good for gradient flow?

**(c)** For a network with $N$ layers using post-norm, the gradient of the loss with respect to the input must pass through $N$ LayerNorm operations. Why might this cause training instability?

**(d)** Pre-norm has been shown to NOT require learning rate warmup, while post-norm does. Propose an explanation based on gradient flow.

---

## Exercise 6.3 — FFN Analysis

The feed-forward network: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$

**(a)** For $D = 512$ and $D_{ff} = 2048$, what are the shapes of $W_1, b_1, W_2, b_2$? Total parameters?

**(b)** The FFN is applied independently to each position. If we have $L = 100$ tokens, how many times is the FFN applied? Is there any interaction between positions in the FFN?

**(c)** After ReLU, what fraction of the $D_{ff}$-dimensional hidden layer is expected to be zero (for random inputs near zero mean)?

**(d)** Some recent models replace ReLU with GELU or SwiGLU. Write the formulas for each and explain the advantage.

**(e)** The expansion ratio $D_{ff}/D = 4$ is a design choice. What happens if we use a ratio of 1? Of 16? Discuss the trade-offs.

---

## Exercise 6.4 — Residual Stream View

The "residual stream" interpretation (Elhage et al.) views the transformer as maintaining a residual stream $x$ that gets incrementally updated:

```
x_0 = embedding + positional_encoding
x_1 = x_0 + MHA_1(x_0)
x_2 = x_1 + FFN_1(x_1)
x_3 = x_2 + MHA_2(x_2)
x_4 = x_3 + FFN_2(x_3)
...
```

**(a)** Express $x_4$ entirely in terms of $x_0$ and the sublayer outputs. How many terms are there?

**(b)** The gradient $\partial x_{2N} / \partial x_0$ has a "skip" path that goes directly through all residual connections. What is this gradient?

**(c)** Why does this skip path help prevent vanishing gradients even in deep transformers?

**(d)** In a 48-layer transformer, the residual stream must carry information from embedding to output through 96 sublayers. What dimension constraint does this impose?

---

## Exercise 6.5 — Architecture Design

You are designing a transformer for a USAAIO competition problem. The task requires:
- Processing sequences up to $L = 1024$
- Model must fit in 100M parameters
- Maximum latency of 50ms on a single GPU

**(a)** Given the constraint of 100M parameters (excluding embeddings), and using $D_{ff} = 4D$ and $D_{qk} = D_v = D/H$, express the parameters per layer as a function of $D$.

**(b)** If $D = 768$ and $H = 12$, how many layers can you fit in 100M parameters?

**(c)** The attention computation is $O(L^2 D)$ per layer. For $L = 1024, D = 768$, and 12 layers, estimate the total FLOPs.

**(d)** If the GPU can do $10^{12}$ FLOPs/second, can you meet the 50ms latency requirement?

**(e)** If not, what architectural modifications would you propose?

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

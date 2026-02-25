# Exercises: Paper-to-Implementation

## Exercise 1: Notation-to-Shapes

**Difficulty:** Introductory

Translate the following paper-style descriptions into shape-annotated PyTorch code. You do not need to run the code — just write it with shape comments.

**(a)** "The input embedding $E \in \mathbb{R}^{V \times d}$ maps token indices $x \in \mathbb{Z}^{B \times L}$ to dense representations $H_0 = E[x] \in \mathbb{R}^{B \times L \times d}$."

**(b)** "We compute the query, key, and value matrices as $Q = H W_Q$, $K = H W_K$, $V = H W_V$ where $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$."

**(c)** "The attention output is $\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$."

**(d)** "For multi-head attention with $h$ heads, we split $Q$ into $h$ heads of dimension $d_k / h$, compute attention independently per head, concatenate the outputs, and project with $W_O \in \mathbb{R}^{d_v \times d}$."

---

## Exercise 2: Identify the Key Equation

**Difficulty:** Intermediate

Read each mini-description and identify: (1) the key equation, (2) the learnable parameters, (3) the input and output shapes.

**(a)** "Squeeze-and-Excitation: Given feature map $U \in \mathbb{R}^{B \times C \times H \times W}$, compute channel statistics $z = \text{GAP}(U) \in \mathbb{R}^{B \times C}$ via global average pooling. Then compute channel weights $s = \sigma(W_2 \, \delta(W_1 z))$ where $W_1 \in \mathbb{R}^{C/r \times C}$, $W_2 \in \mathbb{R}^{C \times C/r}$, $\delta = \text{ReLU}$, $\sigma = \text{sigmoid}$. The output is $\tilde{U} = s \cdot U$ (channel-wise scaling)."

**(b)** "Rotary Position Embedding (RoPE): For a query vector $q \in \mathbb{R}^{d}$, apply rotation: $(q_{2i}, q_{2i+1}) \mapsto (q_{2i}\cos\theta_i - q_{2i+1}\sin\theta_i, \, q_{2i}\sin\theta_i + q_{2i+1}\cos\theta_i)$ where $\theta_i = 10000^{-2i/d} \cdot m$ and $m$ is the position index. There are no learnable parameters."

**(c)** "Cross-Attention: Query comes from decoder $Q = H_{\text{dec}} W_Q$, key and value come from encoder $K = H_{\text{enc}} W_K$, $V = H_{\text{enc}} W_V$. Otherwise identical to self-attention."

---

## Exercise 3: Shape-First Implementation

**Difficulty:** Advanced

Implement the following from description. Use the shape-first method: write all shapes as comments first, then fill in the code.

**Description: "Additive Attention" (Bahdanau Attention)**

> "Given query $q \in \mathbb{R}^{B \times d_q}$ and keys $K \in \mathbb{R}^{B \times L \times d_k}$ and values $V \in \mathbb{R}^{B \times L \times d_v}$:
> 1. Project: $q' = qW_q \in \mathbb{R}^{B \times d}$, $K' = KW_k \in \mathbb{R}^{B \times L \times d}$
> 2. Score: $e = v^T \tanh(q'.\text{unsqueeze}(1) + K') \in \mathbb{R}^{B \times L}$ where $v \in \mathbb{R}^{d}$
> 3. Attend: $\alpha = \text{softmax}(e) \in \mathbb{R}^{B \times L}$
> 4. Output: $c = \alpha V = \sum_j \alpha_j V_j \in \mathbb{R}^{B \times d_v}$"

**Tasks:**

(a) Write the `__init__` method with all learnable parameters.

(b) Write the `forward` method with shape comments at every step.

(c) Write a smoke test that verifies the output shape.

(d) What is the computational complexity of this attention mechanism compared to dot-product attention?

---

## Exercise 4: Speed Reading Challenge

**Difficulty:** Advanced (Timed)

**Set a timer for 10 minutes.** Read the following description and implement it.

**Description: "Gated Residual Network (GRN)"**

> "The GRN processes input $a \in \mathbb{R}^{B \times d}$ with optional context $c \in \mathbb{R}^{B \times d_c}$:
> 1. $\eta_1 = W_1 a + b_1 \in \mathbb{R}^{B \times d_h}$
> 2. If context is provided: $\eta_1 = \eta_1 + W_c c$ where $W_c \in \mathbb{R}^{d_c \times d_h}$  (note: this is added to $\eta_1$ before the activation)
> 3. $\eta_2 = \text{ELU}(\eta_1) \in \mathbb{R}^{B \times d_h}$
> 4. $\eta_3 = W_2 \eta_2 + b_2 \in \mathbb{R}^{B \times d}$
> 5. Apply Gated Linear Unit: $\eta_4 = \text{GLU}(\eta_3) = \sigma(\eta_{3a}) \odot \eta_{3b}$
>    where $\eta_3$ is split in half along the last dimension, so $W_2 \in \mathbb{R}^{d_h \times 2d}$
> 6. Layer norm + residual: output $= \text{LayerNorm}(\eta_4 + a)$"

Wait — step 5 says split in half, so $W_2$ maps to $2d$, and GLU halves it back to $d$. Make sure you handle this.

**Tasks:**

(a) Implement `GatedResidualNetwork` as an `nn.Module`.

(b) Run a smoke test with and without context.

(c) What is the purpose of the gating mechanism (GLU) in this architecture?

(d) How many learnable parameters does this module have (in terms of $d$, $d_h$, and $d_c$)?

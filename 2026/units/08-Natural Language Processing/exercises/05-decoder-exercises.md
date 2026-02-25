# Decoder Transformer (GPT) Exercises

**Topic**: GPT architecture, causal attention, autoregressive generation, sampling strategies
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: Causal Attention Mask

Given a sequence of 4 tokens: `["The", "cat", "is", "happy"]`.

**Tasks**:
1. Write the causal attention mask matrix (4x4) using 0 for "allowed" and $-\infty$ for "blocked."
2. Why are positions above the diagonal set to $-\infty$?
3. After applying the mask to attention scores and then softmax, what does row 1 (for "cat") look like? (In terms of which tokens have nonzero attention weight.)
4. Compare the mask shape for BERT vs GPT for the same 4 tokens.
5. If we added a fifth token, what changes in the mask? Which existing entries remain the same?

<details>
<summary>Solution</summary>

**1.** Causal mask $M$:
```
         The    cat    is    happy
The    [  0     -∞     -∞     -∞  ]
cat    [  0      0     -∞     -∞  ]
is     [  0      0      0     -∞  ]
happy  [  0      0      0      0  ]
```

**2.** Positions above the diagonal ($M_{ij}$ where $j > i$) represent future tokens. Setting them to $-\infty$ before softmax ensures $\text{softmax}(-\infty) = 0$, so token $i$ assigns zero attention weight to any token after it. This prevents information leakage from the future during autoregressive generation.

**3.** Row 1 (for "cat") after softmax:
```
cat → [w₀, w₁, 0, 0]  where w₀ + w₁ = 1, w₀ > 0, w₁ > 0
```
"cat" can only attend to "The" and itself. The exact values of $w_0$ and $w_1$ depend on the attention scores $QK^T$.

**4.** Comparison:
```
BERT (full):           GPT (causal):
[ 0  0  0  0 ]        [  0  -∞  -∞  -∞ ]
[ 0  0  0  0 ]        [  0   0  -∞  -∞ ]
[ 0  0  0  0 ]        [  0   0   0  -∞ ]
[ 0  0  0  0 ]        [  0   0   0   0 ]
```
BERT: every token sees every other token. GPT: each token sees only previous tokens (and itself).

**5.** Adding a fifth token: the mask becomes 5x5. The existing 4x4 entries remain **exactly the same** — we just add a new row (the fifth token attends to all 5) and a new column (all existing tokens block the fifth with $-\infty$). This is what makes GPT generation efficient — previously computed attention patterns do not change when new tokens are appended.

</details>

---

## Exercise 2: Autoregressive Generation Trace

A tiny GPT model has vocabulary: `{a: 0, b: 1, c: 2, <eos>: 3}` and produces these logit vectors at each step.

Starting with prompt: `"a"`

Step 1 — input `[a]`, model outputs logits for position 1: `[1.0, 2.5, 0.5, -1.0]`
Step 2 — input `[a, b]`, model outputs logits for position 2: `[0.2, 0.8, 2.0, 0.1]`
Step 3 — input `[a, b, c]`, model outputs logits for position 3: `[-0.5, 0.3, 0.5, 3.0]`

**Tasks**:
1. Using **greedy decoding** (always pick highest logit), what is the generated sequence?
2. Compute the softmax probabilities for step 1. What is the probability of the greedy choice?
3. At step 2, what is $P(\text{c})$? If we used sampling instead of greedy, what is the probability we would generate "c"?
4. What is the full generated sequence if we stop at `<eos>`?
5. Compute the **perplexity** of the generated sequence (assuming greedy decoding gives us `a b c <eos>`).

<details>
<summary>Solution</summary>

**1.** Greedy decoding:
- Step 1: max logit = 2.5 at index 1 → token "b"
- Step 2: max logit = 2.0 at index 2 → token "c"
- Step 3: max logit = 3.0 at index 3 → token `<eos>`

Generated sequence: `"a b c <eos>"`

**2.** Step 1 softmax:
$e^{1.0} = 2.718, \; e^{2.5} = 12.182, \; e^{0.5} = 1.649, \; e^{-1.0} = 0.368$

Sum = 16.917

$P = [0.161, 0.720, 0.097, 0.022]$

Probability of greedy choice ("b"): 0.720

**3.** Step 2 softmax:
$e^{0.2} = 1.221, \; e^{0.8} = 2.226, \; e^{2.0} = 7.389, \; e^{0.1} = 1.105$

Sum = 11.941

$P = [0.102, 0.186, 0.619, 0.093]$

$P(\text{c}) = 0.619$. With sampling, we would generate "c" with probability 0.619 — not guaranteed, unlike greedy.

**4.** Full generated sequence: `a b c <eos>` (stop at `<eos>`).

**5.** Perplexity over the 3 generated tokens (b, c, <eos>):

Step 3 softmax:
$e^{-0.5} = 0.607, \; e^{0.3} = 1.350, \; e^{0.5} = 1.649, \; e^{3.0} = 20.086$

Sum = 23.692

$P = [0.026, 0.057, 0.070, 0.848]$

$$\text{PPL} = \exp\left(-\frac{1}{3}[\log(0.720) + \log(0.619) + \log(0.848)]\right)$$
$$= \exp\left(-\frac{1}{3}[-0.329 + (-0.480) + (-0.165)]\right)$$
$$= \exp\left(-\frac{-0.974}{3}\right) = \exp(0.325) = 1.384$$

Perplexity = 1.384, which is very low (good). The model is quite confident about its predictions.

</details>

---

## Exercise 3: Temperature and Sampling

Given logits at the final position: `[3.0, 1.5, 1.0, 0.5, -2.0]` for a 5-word vocabulary: `{cat, dog, fish, bird, rock}`.

**Tasks**:
1. Compute softmax probabilities with $T = 1.0$ (default).
2. Compute softmax probabilities with $T = 0.5$ (sharper).
3. Compute softmax probabilities with $T = 2.0$ (flatter).
4. For top-$k$ with $k = 3$ at $T = 1.0$: which tokens are kept? What are the renormalized probabilities?
5. For top-$p$ with $p = 0.9$ at $T = 1.0$: which tokens are kept?

<details>
<summary>Solution</summary>

**1.** $T = 1.0$: $z/T = [3.0, 1.5, 1.0, 0.5, -2.0]$

$e^{3.0} = 20.086, \; e^{1.5} = 4.482, \; e^{1.0} = 2.718, \; e^{0.5} = 1.649, \; e^{-2.0} = 0.135$

Sum = 29.070

$P = [0.691, 0.154, 0.094, 0.057, 0.005]$

**2.** $T = 0.5$: $z/T = [6.0, 3.0, 2.0, 1.0, -4.0]$

$e^{6.0} = 403.4, \; e^{3.0} = 20.09, \; e^{2.0} = 7.389, \; e^{1.0} = 2.718, \; e^{-4.0} = 0.018$

Sum = 433.6

$P = [0.930, 0.046, 0.017, 0.006, 0.000]$

Much sharper — "cat" dominates with 93%.

**3.** $T = 2.0$: $z/T = [1.5, 0.75, 0.5, 0.25, -1.0]$

$e^{1.5} = 4.482, \; e^{0.75} = 2.117, \; e^{0.5} = 1.649, \; e^{0.25} = 1.284, \; e^{-1.0} = 0.368$

Sum = 9.900

$P = [0.453, 0.214, 0.167, 0.130, 0.037]$

Much flatter — probability is spread more evenly.

**4.** Top-3 at $T = 1.0$: keep {cat, dog, fish} (top 3 by probability).

Original probs: [0.691, 0.154, 0.094]

Renormalized: sum = 0.939

$P_{\text{top-3}} = [0.691/0.939, \; 0.154/0.939, \; 0.094/0.939] = [0.736, 0.164, 0.100]$

**5.** Top-$p$ with $p = 0.9$ at $T = 1.0$:

Sorted by probability: cat (0.691), dog (0.154), fish (0.094), bird (0.057), rock (0.005)

Cumulative: 0.691, 0.845, 0.939, 0.996, 1.000

$p = 0.9$: We need cumulative $\geq 0.9$. After including fish (0.939 > 0.9), we stop.

Keep: {cat, dog, fish} — same as top-3 in this case, but the cutoff adapts to the distribution. If "cat" had probability 0.95, top-$p$ would keep only "cat."

</details>

---

## Exercise 4: GPT Training Objective

Consider training GPT on the sentence: `"the cat sat on the mat"` (6 tokens).

**Tasks**:
1. What are the 5 prediction tasks that GPT learns from this single sentence?
2. For each prediction task, what tokens are in the context (visible to the model)?
3. If the model's predicted probability for each correct next token is: $P(\text{cat}|\text{the}) = 0.05$, $P(\text{sat}|\text{the cat}) = 0.10$, $P(\text{on}|\text{the cat sat}) = 0.15$, $P(\text{the}|\text{the cat sat on}) = 0.20$, $P(\text{mat}|\text{the cat sat on the}) = 0.08$. Compute the total loss and perplexity.
4. All 5 predictions are computed in a **single forward pass**. Explain how the causal mask makes this possible.
5. Compare this to BERT's training: how many tokens would BERT predict in one pass over this sentence (with 15% masking)?

<details>
<summary>Solution</summary>

**1.** Five prediction tasks:
1. Given "the" → predict "cat"
2. Given "the cat" → predict "sat"
3. Given "the cat sat" → predict "on"
4. Given "the cat sat on" → predict "the"
5. Given "the cat sat on the" → predict "mat"

**2.** Visible context:
1. "the" → predict "cat"
2. "the", "cat" → predict "sat"
3. "the", "cat", "sat" → predict "on"
4. "the", "cat", "sat", "on" → predict "the"
5. "the", "cat", "sat", "on", "the" → predict "mat"

**3.** Loss:
$$\mathcal{L} = -\frac{1}{5}[\log(0.05) + \log(0.10) + \log(0.15) + \log(0.20) + \log(0.08)]$$
$$= -\frac{1}{5}[-3.00 + (-2.30) + (-1.90) + (-1.61) + (-2.53)]$$
$$= -\frac{-11.34}{5} = 2.268$$

Perplexity:
$$\text{PPL} = \exp(2.268) = 9.66$$

The model is "about as surprised as if it had to choose between ~10 equally likely words" on average.

**4.** The causal mask ensures that in a single forward pass:
- Position 0 ("the") computes attention only using position 0
- Position 1 ("cat") computes attention using positions 0-1
- Position 2 ("sat") computes attention using positions 0-2
- etc.

Each position's hidden state only depends on previous positions. So the logits at position $t$ predict token $t+1$ using only context $x_{\leq t}$. All 5 predictions happen simultaneously in the same matrix computation — the mask just prevents information from flowing right-to-left.

**5.** BERT with 15% masking: $0.15 \times 6 = 0.9$, rounded to 1 token predicted per pass. GPT predicts 5 tokens per pass ($L-1 = 5$). GPT extracts more training signal per forward pass, but GPT only uses left context while BERT uses full bidirectional context.

</details>

---

## Exercise 5: Encoder vs. Decoder Architecture Selection

For each of the following NLP tasks, decide whether you would use an **encoder model** (BERT-like), a **decoder model** (GPT-like), or an **encoder-decoder model** (T5-like). Justify your choice.

**Tasks**:
1. **Sentiment analysis**: Given a movie review, classify it as positive/negative.
2. **Text summarization**: Given an article, generate a shorter summary.
3. **Named entity recognition**: Given a sentence, label each token as person/organization/location/other.
4. **Chatbot / dialogue**: Given a conversation history, generate the next response.
5. **Semantic similarity**: Given two sentences, compute a similarity score.

<details>
<summary>Solution</summary>

**1. Sentiment analysis** → **Encoder (BERT)**

Sentiment analysis is a **classification** task — we need to understand the full input and map it to a label. BERT excels because:
- Bidirectional attention captures the meaning of the entire review
- The [CLS] token provides a natural sequence-level representation
- No text generation is needed

**2. Text summarization** → **Encoder-decoder (T5, BART)** or **Decoder (GPT)**

Summarization requires both **understanding** the input (comprehension) and **generating** the output (production):
- Encoder-decoder is ideal: encoder understands the article, decoder generates the summary
- GPT can also work via "Article: ... Summary:" prompting, but encoder-decoder architectures were designed for this seq-to-seq pattern

**3. Named entity recognition** → **Encoder (BERT)**

NER is a **token-level classification** task:
- Each token needs its own label (B-PER, I-PER, O, etc.)
- BERT provides a contextualized representation for each token
- Bidirectional context is critical: "Apple launched a new iPhone" — knowing "iPhone" is after "Apple" helps classify "Apple" as ORG, not a fruit
- A linear head on each token's hidden state produces per-token predictions

**4. Chatbot / dialogue** → **Decoder (GPT)**

Dialogue generation is an **autoregressive generation** task:
- The model needs to generate a response token by token
- GPT's causal attention naturally models the sequential generation process
- The conversation history serves as the prompt/context
- In-context learning allows adapting to conversation style without fine-tuning

**5. Semantic similarity** → **Encoder (BERT)**

Semantic similarity is an **understanding** task:
- Encode both sentences with BERT (using [CLS] representations)
- Compute cosine similarity between the two [CLS] vectors
- Or: encode the concatenated pair `[CLS] sent_A [SEP] sent_B [SEP]` and classify into similarity buckets
- Bidirectional attention is essential for understanding the meaning of both sentences
- Sentence-BERT (SBERT) was specifically designed for this

</details>

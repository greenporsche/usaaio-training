# Encoder Transformer (BERT) Exercises

**Topic**: BERT architecture, masked language modeling, bidirectional attention, classification heads
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: BERT Input Representation

Given the sentence pair:
- Sentence A: `"The cat sat"`
- Sentence B: `"It slept"`

BERT's tokenizer produces:
```
[CLS] the cat sat [SEP] it slept [SEP]
```

**Tasks**:
1. What is the sequence length $L$?
2. Write out the segment (token type) IDs for each position.
3. Write out the position IDs for each position.
4. If BERT-base has $D = 768$, what are the shapes of the token embedding, segment embedding, and position embedding lookups?
5. How are the three embeddings combined to form the input?

<details>
<summary>Solution</summary>

**1.** $L = 8$ tokens: [CLS], the, cat, sat, [SEP], it, slept, [SEP]

**2.** Segment IDs:
```
[CLS]  the  cat  sat  [SEP]  it  slept  [SEP]
  0      0    0    0     0     1    1      1
```
Sentence A tokens (including [CLS] and first [SEP]) get segment 0, sentence B tokens get segment 1.

**3.** Position IDs:
```
[CLS]  the  cat  sat  [SEP]  it  slept  [SEP]
  0      1    2    3     4     5    6      7
```

**4.** Shapes:
- Token embedding lookup: input $(B, 8)$ → output $(B, 8, 768)$
- Segment embedding lookup: input $(B, 8)$ → output $(B, 8, 768)$
- Position embedding lookup: input $(B, 8)$ → output $(B, 8, 768)$

(The embedding tables are: token $(30522, 768)$, segment $(2, 768)$, position $(512, 768)$.)

**5.** Element-wise addition: $\text{input} = \text{token\_emb} + \text{segment\_emb} + \text{position\_emb}$, result shape $(B, 8, 768)$. Followed by layer normalization and dropout.

</details>

---

## Exercise 2: Masked Language Modeling

Given the sentence: `"The quick brown fox jumps over the lazy dog"`

BERT's MLM procedure selects 15% of tokens for prediction.

**Tasks**:
1. How many tokens should be selected for masking (round to nearest integer)?
2. Suppose positions 2 ("brown") and 6 ("the") are selected. Apply the 80/10/10 rule: give one possible masked input for each combination.
3. What are the targets (labels) for the masked positions?
4. What are the labels for the non-masked positions? How does the loss function handle them?
5. Why does BERT use the 80/10/10 strategy instead of always replacing with `[MASK]`?

<details>
<summary>Solution</summary>

**1.** 9 tokens in the sentence. 15% of 9 = 1.35, rounded to 1 or 2 tokens. (In practice, BERT selects exactly 15% across the batch, so for this sentence, 1-2 tokens.)

**2.** For position 2 ("brown") and position 6 ("the"):

80% MASK, 10% random, 10% keep:
- Possible input 1 (both masked): `"The quick [MASK] fox jumps over [MASK] lazy dog"`
- Possible input 2 (one masked, one random): `"The quick [MASK] fox jumps over purple lazy dog"`
- Possible input 3 (one masked, one kept): `"The quick [MASK] fox jumps over the lazy dog"`
- Possible input 4 (both random): `"The quick elephant fox jumps over purple lazy dog"`

Each selected position independently gets 80/10/10 treatment.

**3.** Targets: position 2 → "brown", position 6 → "the". The model must predict the **original** token regardless of how it was corrupted.

**4.** Non-masked positions get a special ignore label (typically -100 in PyTorch). The cross-entropy loss function ignores these positions (via `ignore_index=-100`), so the model is only trained to predict the masked tokens.

**5.** If always `[MASK]`:
- During fine-tuning, there are no `[MASK]` tokens in the input
- The model would only learn to produce good representations when `[MASK]` is present (a distribution mismatch)
- The 10% random and 10% unchanged force the model to produce good representations for **all** positions, not just masked ones
- This bridges the gap between pretraining (with masks) and fine-tuning (without masks)

</details>

---

## Exercise 3: BERT Attention Patterns

Consider BERT processing: `[CLS] The cat sat [SEP]` (5 tokens, $L = 5$).

BERT-base has 12 heads per layer. Assume one attention head produces these attention weights (after softmax):

```
         [CLS]   The    cat    sat   [SEP]
[CLS]  [ 0.10   0.30   0.25   0.20   0.15 ]
The    [ 0.05   0.15   0.40   0.30   0.10 ]
cat    [ 0.08   0.35   0.12   0.35   0.10 ]
sat    [ 0.07   0.20   0.40   0.13   0.20 ]
[SEP]  [ 0.30   0.10   0.15   0.15   0.30 ]
```

**Tasks**:
1. Verify that each row sums to approximately 1. Why must this be the case?
2. What does "cat" attend to most strongly? Interpret this linguistically.
3. What does "[CLS]" attend to? Why might this pattern emerge?
4. If this were a GPT model with causal masking, which entries would be zero? Rewrite the matrix.
5. BERT has 12 heads. Why might different heads learn different attention patterns? Give examples of what different heads might specialize in.

<details>
<summary>Solution</summary>

**1.** Each row sums to 1.0 (by construction — softmax normalizes each row). This is because attention weights represent a probability distribution: "how much should this token attend to each other token?" The weights must be non-negative and sum to 1.

**2.** "cat" attends most to "The" (0.35) and "sat" (0.35) equally. Linguistically, "The" is the determiner of "cat" (syntactic relationship), and "sat" is the verb that "cat" performs (subject-verb relationship). This head may be capturing syntactic dependencies.

**3.** "[CLS]" distributes attention fairly broadly: The=0.30, cat=0.25, sat=0.20, [SEP]=0.15, self=0.10. Since [CLS] is used for classification, it needs to aggregate information from the entire sequence. Broad attention allows it to collect a summary representation.

**4.** With causal masking, token $i$ can only attend to positions $\leq i$:
```
         [CLS]   The    cat    sat   [SEP]
[CLS]  [ 1.00   0.00   0.00   0.00   0.00 ]
The    [ 0.12   0.88   0.00   0.00   0.00 ]
cat    [ 0.10   0.45   0.45   0.00   0.00 ]
sat    [ 0.08   0.22   0.44   0.26   0.00 ]
[SEP]  [ 0.30   0.10   0.15   0.15   0.30 ]
```
Upper-triangular entries become 0, and remaining entries are renormalized to sum to 1. (Exact values would differ since they depend on pre-softmax scores; shown here as illustration.)

**5.** Different heads can specialize in different linguistic relationships:
- **Syntactic head**: Attends to syntactic dependencies (subject → verb, determiner → noun)
- **Positional head**: Attends to adjacent tokens (local context)
- **[SEP] head**: Attends heavily to separator tokens (boundary detection)
- **Coreference head**: Attends to coreferent mentions (pronouns → antecedents)
- **Rare word head**: Distributes attention broadly to gather context for rare tokens

Research (Clark et al., 2019) has confirmed these specialization patterns in pretrained BERT.

</details>

---

## Exercise 4: BERT for Classification

You are fine-tuning BERT-base for a 3-class sentiment classification task (positive, neutral, negative).

**Tasks**:
1. What is the shape of the classification head weight matrix and bias? What is the total number of new parameters added?
2. The [CLS] hidden state for an example is $h_{[\text{CLS}]} = [0.5, -0.2, 0.8, \dots] \in \mathbb{R}^{768}$. If the classifier weights are $W \in \mathbb{R}^{3 \times 768}$ and bias $b \in \mathbb{R}^3$, what operations produce the final prediction?
3. The model outputs logits $[2.1, 0.5, -1.3]$. Compute the softmax probabilities and the predicted class.
4. If the true label is class 0 (positive), compute the cross-entropy loss.
5. You have only 500 labeled training examples. Should you fine-tune all of BERT or freeze the backbone? Justify your answer.

<details>
<summary>Solution</summary>

**1.**
- Weight matrix: $W \in \mathbb{R}^{3 \times 768}$, parameters: $3 \times 768 = 2{,}304$
- Bias: $b \in \mathbb{R}^3$, parameters: 3
- Total new parameters: 2,307
- Compare to BERT-base total: 110M parameters. The head adds only 0.002% new parameters.

**2.** Operations:
1. Extract [CLS]: $h = \text{outputs.last\_hidden\_state}[:, 0, :]$ → shape $(B, 768)$
2. Apply dropout: $h' = \text{Dropout}(h)$ → shape $(B, 768)$
3. Linear projection: $\text{logits} = h' W^T + b$ → shape $(B, 3)$
4. Softmax: $\text{probs} = \text{softmax}(\text{logits})$ → shape $(B, 3)$
5. Prediction: $\hat{y} = \arg\max(\text{probs})$

**3.** Softmax:
$$P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

$e^{2.1} = 8.166, \quad e^{0.5} = 1.649, \quad e^{-1.3} = 0.273$

Sum = 10.088

$P = [8.166/10.088, \; 1.649/10.088, \; 0.273/10.088] = [0.810, \; 0.163, \; 0.027]$

Predicted class: 0 (positive) with probability 0.810.

**4.** Cross-entropy loss (true label = class 0):
$$\mathcal{L} = -\log P(\text{class 0}) = -\log(0.810) = 0.211$$

**5.** With only 500 examples, **freeze the backbone** and only train the classification head:
- 500 examples are far too few to meaningfully update 110M parameters without severe overfitting
- The pretrained BERT representations are already rich and general
- Training only 2,307 parameters (the head) requires very few examples
- Alternative: Gradually unfreeze top 1-2 layers with a very small learning rate after the head converges

If accuracy is insufficient with frozen backbone, try unfreezing the last 2-4 transformer layers with discriminative learning rates ($10^{-6}$ for bottom layers, $10^{-4}$ for the head).

</details>

---

## Exercise 5: BERT vs. ELMo vs. Traditional

Compare three approaches to NLP representation:

| Approach | Model | Representation |
|---|---|---|
| Traditional | Word2Vec/GloVe | Static embedding per word |
| ELMo | Bidirectional LSTM | Contextualized (left+right concatenated) |
| BERT | Transformer encoder | Deeply contextualized (bidirectional) |

Consider the word "bank" in these sentences:
1. "I went to the **bank** to deposit money."
2. "The river **bank** was covered in wildflowers."

**Tasks**:
1. How would Word2Vec represent "bank" in sentences 1 and 2? What is the problem?
2. How does ELMo address this? What is its limitation?
3. How does BERT address this? Why is it better than ELMo?
4. In BERT, the hidden state of "bank" in sentence 1 should be similar to which words? Different from which words? Explain.
5. If you extracted BERT embeddings for "bank" from both sentences and computed cosine similarity, would you expect it to be high or low? Why?

<details>
<summary>Solution</summary>

**1.** Word2Vec gives "bank" a **single static vector** regardless of context. The same embedding is used for the financial institution and the river bank. This is a fundamental limitation — the embedding is the average of all meanings, which is close to neither specific meaning. Polysemy (multiple word meanings) is not handled.

**2.** ELMo computes contextualized embeddings using two LSTMs:
- Forward LSTM reads left-to-right: "I went to the" → hidden state for "bank"
- Backward LSTM reads right-to-left: "deposit money" → hidden state for "bank"
- Final: concatenation of both hidden states

**Limitation**: The forward and backward LSTMs are trained **independently** — the forward LSTM does not know what the backward LSTM sees, and vice versa. They are only combined after the fact. This is "shallow" bidirectionality.

**3.** BERT's self-attention allows **every token to attend to every other token simultaneously** at every layer. When processing "bank" in sentence 1:
- "bank" can attend to "deposit" and "money" (future context) AND "went to the" (past context) **jointly**
- At each layer, the representation of "bank" is refined using all available context
- This is "deep" bidirectionality — information flows in both directions within each layer

**4.** In sentence 1, "bank" (financial) should have a hidden state similar to:
- "deposit", "money", "account", "financial" (financial context words)
- Other instances of "bank" used in financial contexts

It should be dissimilar to:
- "river", "shore", "water" (natural landscape words)
- Other instances of "bank" used in geographical contexts

**5.** The cosine similarity would be **relatively low** (perhaps 0.3-0.6) — much lower than 1.0. BERT's deep contextualization gives "bank" very different representations based on context. The financial "bank" lives in a different part of the representation space than the river "bank." This is exactly the desirable behavior — the model disambiguates word senses through context.

</details>

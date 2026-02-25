# Word Embeddings Exercises

**Topic**: Skip-gram, CBOW, negative sampling, cosine similarity
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: Skip-gram Training Pairs

Given the sentence: `"the dog chased the cat"` and a window size of 2.

**Tasks**:
1. List all (center word, context word) training pairs.
2. How many training pairs are generated?
3. If the window size were 1, how many pairs would there be?
4. If the vocabulary size is 4 (the, dog, chased, cat), what is the shape of the input one-hot vector? The center embedding matrix $W_{\text{in}}$? The context embedding matrix $W_{\text{out}}$? (Assume embedding dimension $d = 3$.)

<details>
<summary>Solution</summary>

**1.** With window size 2, for each center word we look 2 positions left and 2 positions right:

| Center (position) | Context words |
|---|---|
| the (0) | dog (1), chased (2) |
| dog (1) | the (0), chased (2), the (3) |
| chased (2) | the (0), dog (1), the (3), cat (4) |
| the (3) | dog (1), chased (2), cat (4) |
| cat (4) | chased (2), the (3) |

Training pairs:
```
(the, dog), (the, chased),
(dog, the), (dog, chased), (dog, the),
(chased, the), (chased, dog), (chased, the), (chased, cat),
(the, dog), (the, chased), (the, cat),
(cat, chased), (cat, the)
```

**2.** 14 training pairs

**3.** With window size 1:
- the(0): dog(1) → 1 pair
- dog(1): the(0), chased(2) → 2 pairs
- chased(2): dog(1), the(3) → 2 pairs
- the(3): chased(2), cat(4) → 2 pairs
- cat(4): the(3) → 1 pair
Total: 8 pairs

**4.** Shapes:
- Input one-hot: $(4,)$ or $(1, 4)$ — one-hot vector over vocabulary
- $W_{\text{in}}$: $(4, 3)$ — 4 words, each with 3-dimensional embedding
- $W_{\text{out}}$: $(3, 4)$ or equivalently $(4, 3)$ — context embedding matrix
- Output logits: $(4,)$ — probability distribution over vocabulary

</details>

---

## Exercise 2: Cosine Similarity

Given three word embeddings (dimension 3):
```
cat  = [0.8, 0.5, 0.1]
dog  = [0.7, 0.6, 0.2]
car  = [0.1, 0.2, 0.9]
```

**Tasks**:
1. Compute $\cos(\text{cat}, \text{dog})$.
2. Compute $\cos(\text{cat}, \text{car})$.
3. Which pair is more similar? Does this match your intuition?
4. What would the cosine similarity be between a word and itself?
5. Can cosine similarity be negative? What would that mean?

<details>
<summary>Solution</summary>

**1.**
$$\cos(\text{cat}, \text{dog}) = \frac{0.8 \times 0.7 + 0.5 \times 0.6 + 0.1 \times 0.2}{\sqrt{0.8^2 + 0.5^2 + 0.1^2} \times \sqrt{0.7^2 + 0.6^2 + 0.2^2}}$$
$$= \frac{0.56 + 0.30 + 0.02}{\sqrt{0.64 + 0.25 + 0.01} \times \sqrt{0.49 + 0.36 + 0.04}}$$
$$= \frac{0.88}{\sqrt{0.90} \times \sqrt{0.89}} = \frac{0.88}{0.9487 \times 0.9434} = \frac{0.88}{0.8950} \approx 0.983$$

**2.**
$$\cos(\text{cat}, \text{car}) = \frac{0.08 + 0.10 + 0.09}{\sqrt{0.90} \times \sqrt{0.01 + 0.04 + 0.81}}$$
$$= \frac{0.27}{0.9487 \times \sqrt{0.86}} = \frac{0.27}{0.9487 \times 0.9274} = \frac{0.27}{0.8798} \approx 0.307$$

**3.** cat and dog are much more similar (0.983 vs 0.307). This matches intuition — cats and dogs are both animals/pets, while a car is unrelated.

**4.** $\cos(v, v) = \frac{v \cdot v}{\|v\| \|v\|} = \frac{\|v\|^2}{\|v\|^2} = 1$. A word is maximally similar to itself.

**5.** Yes, cosine similarity ranges from $[-1, 1]$. Negative similarity means vectors point in opposite directions. In word embeddings, this would indicate words with "opposite" semantic properties (though in practice, most word pairs have non-negative similarity since embeddings tend to have mostly positive components).

</details>

---

## Exercise 3: Negative Sampling

You are training a skip-gram model with negative sampling ($K = 2$ negatives). The vocabulary is:
```
{the: 0, cat: 1, sat: 2, on: 3, mat: 4}
```

Word frequencies: the=10, cat=3, sat=2, on=5, mat=1. Total = 21.

The current training pair is (center=sat, context=cat).

**Tasks**:
1. Compute the noise distribution $P_n(w) \propto \text{count}(w)^{3/4}$ for each word.
2. Suppose you sample negatives: $w_1^- = \text{the}$, $w_2^- = \text{on}$. Write out the negative sampling loss function for this specific example.
3. Assume current embeddings (dimension 2):
   - $v_{\text{sat}} = [1, 0]$, $u_{\text{cat}} = [0.5, 0.5]$, $u_{\text{the}} = [0.3, -0.2]$, $u_{\text{on}} = [-0.1, 0.4]$

   Compute the loss value.
4. Compute $\nabla_{v_{\text{sat}}} J_{\text{neg}}$.

<details>
<summary>Solution</summary>

**1.** Compute $\text{count}(w)^{3/4}$:
- the: $10^{0.75} = 5.623$
- cat: $3^{0.75} = 2.280$
- sat: $2^{0.75} = 1.682$
- on: $5^{0.75} = 3.344$
- mat: $1^{0.75} = 1.000$

Total = 13.929

$P_n$: the=0.404, cat=0.164, sat=0.121, on=0.240, mat=0.072

Note the 3/4 power makes the distribution flatter — "mat" gets 0.072 instead of 1/21 = 0.048.

**2.** Negative sampling loss:
$$J = -\log \sigma(u_{\text{cat}}^T v_{\text{sat}}) - \log \sigma(-u_{\text{the}}^T v_{\text{sat}}) - \log \sigma(-u_{\text{on}}^T v_{\text{sat}})$$

**3.** Compute dot products:
- $u_{\text{cat}}^T v_{\text{sat}} = 0.5 \times 1 + 0.5 \times 0 = 0.5$
- $u_{\text{the}}^T v_{\text{sat}} = 0.3 \times 1 + (-0.2) \times 0 = 0.3$
- $u_{\text{on}}^T v_{\text{sat}} = -0.1 \times 1 + 0.4 \times 0 = -0.1$

Loss:
$$J = -\log \sigma(0.5) - \log \sigma(-0.3) - \log \sigma(0.1)$$
$$= -\log(0.6225) - \log(0.4256) - \log(0.5250)$$
$$= 0.4741 + 0.8544 + 0.6444 = 1.973$$

**4.** Gradient:
$$\nabla_{v_{\text{sat}}} J = -(1 - \sigma(0.5)) \cdot u_{\text{cat}} + \sigma(0.3) \cdot u_{\text{the}} + \sigma(-0.1) \cdot u_{\text{on}}$$

$\sigma(0.5) = 0.6225$, $\sigma(0.3) = 0.5744$, $\sigma(-0.1) = 0.4750$

$$= -(0.3775)[0.5, 0.5] + (0.5744)[0.3, -0.2] + (0.4750)[-0.1, 0.4]$$
$$= [-0.1888, -0.1888] + [0.1723, -0.1149] + [-0.0475, 0.1900]$$
$$= [-0.0640, -0.1137]$$

</details>

---

## Exercise 4: Embedding Properties

Given a trained Word2Vec model, the following cosine similarities are measured:

```
sim(king, queen)    = 0.65
sim(king, man)      = 0.55
sim(king, woman)    = 0.30
sim(queen, woman)   = 0.58
sim(man, woman)     = 0.60
sim(king, apple)    = 0.05
```

**Tasks**:
1. Verify intuitively: do these similarities make sense?
2. For the analogy "king is to man as queen is to ___", compute the vector $v = v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$. Without actual vectors, explain conceptually what direction this vector points.
3. Why does the analogy task use **subtraction** followed by **addition**? What does $v_{\text{king}} - v_{\text{man}}$ represent geometrically?
4. Name two limitations of word embedding analogies.

<details>
<summary>Solution</summary>

**1.** The similarities make intuitive sense:
- king and queen are very related (royalty) → 0.65
- king and man share gender → 0.55
- king and woman have less in common → 0.30
- queen and woman share gender → 0.58
- man and woman are related (both human, gender pair) → 0.60
- king and apple are unrelated → 0.05

**2.** $v_{\text{king}} - v_{\text{man}}$ isolates the "royalty" concept (subtracting the "male" component). Adding $v_{\text{woman}}$ adds back a "female" component. So $v$ points toward the concept of "female royalty" — which should be closest to $v_{\text{queen}}$.

**3.** $v_{\text{king}} - v_{\text{man}}$ represents the **direction** from "man" to "king" in embedding space — a vector that encodes the transformation "make royal." Geometrically, it is a displacement vector. The analogy assumes this displacement is consistent: the same "royalty" direction applied to "woman" should give "queen."

This works because embeddings organize semantic dimensions as approximately linear subspaces. The "gender" dimension and "royalty" dimension are somewhat independent, allowing vector arithmetic.

**4.** Limitations:
- **Only works for simple analogies**: Complex relationships (e.g., "doctor is to hospital as teacher is to ___") often fail
- **Bias amplification**: If the training corpus contains gender stereotypes, the embeddings encode them (e.g., "man:computer programmer :: woman:homemaker")
- **Frequency bias**: Very frequent words and very rare words have poorer analogical relationships
- **Polysemy**: Words with multiple meanings (e.g., "bank") have a single embedding that averages all meanings

</details>

---

## Exercise 5: Skip-gram Gradient Derivation

Derive the gradient of the full softmax skip-gram objective with respect to the center word embedding $v_{w_c}$.

The probability is:
$$P(w_o \mid w_c) = \frac{\exp(u_{w_o}^T v_{w_c})}{\sum_{w=1}^{|V|} \exp(u_w^T v_{w_c})}$$

The loss for a single (center, context) pair is:
$$\mathcal{L} = -\log P(w_o \mid w_c)$$

**Tasks**:
1. Expand $\mathcal{L}$ by substituting the softmax probability.
2. Compute $\frac{\partial \mathcal{L}}{\partial v_{w_c}}$.
3. Interpret the gradient: what is the "observed" term and what is the "expected" term?
4. Explain why this gradient is expensive to compute and how negative sampling addresses this.

<details>
<summary>Solution</summary>

**1.** Expanding:
$$\mathcal{L} = -\log \frac{\exp(u_{w_o}^T v_{w_c})}{\sum_{w} \exp(u_w^T v_{w_c})}$$
$$= -u_{w_o}^T v_{w_c} + \log \sum_{w=1}^{|V|} \exp(u_w^T v_{w_c})$$

**2.** Taking the gradient with respect to $v_{w_c}$:

For the first term:
$$\frac{\partial}{\partial v_{w_c}} (-u_{w_o}^T v_{w_c}) = -u_{w_o}$$

For the second term (using the log-sum-exp gradient):
$$\frac{\partial}{\partial v_{w_c}} \log \sum_w \exp(u_w^T v_{w_c}) = \frac{\sum_w \exp(u_w^T v_{w_c}) \cdot u_w}{\sum_w \exp(u_w^T v_{w_c})} = \sum_w P(w \mid w_c) \cdot u_w$$

Combining:
$$\frac{\partial \mathcal{L}}{\partial v_{w_c}} = -u_{w_o} + \sum_{w=1}^{|V|} P(w \mid w_c) \cdot u_w$$

**3.** Interpretation:
- **Observed term** ($-u_{w_o}$): Push $v_{w_c}$ **toward** the actual context word $w_o$
- **Expected term** ($\sum_w P(w|w_c) u_w$): Push $v_{w_c}$ **away from** the expected (average) context word under the current model

The gradient updates the center embedding to be more aligned with the observed context and less aligned with what the model currently predicts.

**4.** The sum $\sum_{w=1}^{|V|} P(w|w_c) u_w$ requires computing $P(w|w_c)$ for all $|V|$ words — this means a forward pass through the entire vocabulary (the softmax denominator). With $|V| = 50{,}000+$, this is prohibitively expensive.

Negative sampling replaces this sum with just $K$ randomly sampled negative words (typically $K = 5$-$20$). Instead of pushing away from all words weighted by their probabilities, we push away from a small random sample. This reduces the per-example cost from $O(|V|)$ to $O(K)$.

</details>

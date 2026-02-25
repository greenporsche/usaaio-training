# Fine-Tuning Exercises

**Topic**: Transfer learning, task heads, learning rate scheduling, freezing strategies
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: Parameter Counting

You are fine-tuning BERT-base (110M parameters, $D = 768$) for different tasks.

**Tasks**:
1. **Binary sentiment classification**: What is the shape of the task head? How many new parameters?
2. **5-class topic classification**: What is the shape of the task head? How many new parameters?
3. **NER with 9 entity types (BIO scheme)**: What is the shape of the task head? How many new parameters? Why is this applied to every token?
4. **Extractive QA (SQuAD-style)**: The model predicts start and end positions. What is the shape of the task head?
5. For each task, compute the ratio of new parameters to total BERT parameters. What does this tell you about the efficiency of fine-tuning?

<details>
<summary>Solution</summary>

**1.** Binary sentiment:
- Head: $W \in \mathbb{R}^{2 \times 768}$, $b \in \mathbb{R}^2$
- New parameters: $2 \times 768 + 2 = 1{,}538$

**2.** 5-class topic:
- Head: $W \in \mathbb{R}^{5 \times 768}$, $b \in \mathbb{R}^5$
- New parameters: $5 \times 768 + 5 = 3{,}845$

**3.** NER with 9 types:
- Head: $W \in \mathbb{R}^{9 \times 768}$, $b \in \mathbb{R}^9$
- New parameters: $9 \times 768 + 9 = 6{,}921$
- Applied per token because NER requires a label for **each token** in the sequence. The same linear head is shared across all positions (parameter sharing), but it receives different hidden states at each position.

**4.** Extractive QA:
- Head: $W \in \mathbb{R}^{2 \times 768}$, $b \in \mathbb{R}^2$
- New parameters: $2 \times 768 + 2 = 1{,}538$
- The output at each token position is a 2-dim vector: (start score, end score). The predicted answer span is the highest-scoring (start, end) pair where start ≤ end.

**5.** Ratios:
- Sentiment: $1{,}538 / 110{,}000{,}000 = 0.0014\%$
- Topic: $3{,}845 / 110{,}000{,}000 = 0.0035\%$
- NER: $6{,}921 / 110{,}000{,}000 = 0.0063\%$
- QA: $1{,}538 / 110{,}000{,}000 = 0.0014\%$

The task head adds a **negligible** number of parameters. The heavy lifting is done by the 110M pretrained BERT parameters — fine-tuning slightly adjusts them while the tiny task head projects from the learned representation space to the task-specific output space.

</details>

---

## Exercise 2: Learning Rate Scheduling

You are fine-tuning BERT for 3 epochs on a dataset with 10,000 examples, batch size 32.

**Tasks**:
1. How many training steps per epoch? How many total training steps?
2. With 10% warmup, how many warmup steps?
3. If $\text{LR}_{\max} = 2 \times 10^{-5}$, what is the learning rate at step 0? At step 47 (end of warmup)? At step 469 (midway)? At step 937 (end)?
4. Why is warmup important for fine-tuning pretrained models?
5. Compare: what would happen if you used a constant learning rate of $2 \times 10^{-5}$ throughout?

<details>
<summary>Solution</summary>

**1.**
- Steps per epoch: $\lceil 10{,}000 / 32 \rceil = 313$ steps
- Total steps: $313 \times 3 = 939$ steps

**2.** Warmup steps: $\lfloor 0.10 \times 939 \rfloor = 93$ steps

**3.** Using linear warmup + linear decay:

Warmup phase ($t < 93$):
$\text{LR}(t) = \text{LR}_{\max} \times \frac{t}{93}$

Decay phase ($t \geq 93$):
$\text{LR}(t) = \text{LR}_{\max} \times \frac{939 - t}{939 - 93} = \text{LR}_{\max} \times \frac{939 - t}{846}$

Values:
- Step 0: $2 \times 10^{-5} \times 0/93 = 0$
- Step 47: $2 \times 10^{-5} \times 47/93 = 1.01 \times 10^{-5}$
- Step 469: $2 \times 10^{-5} \times (939-469)/846 = 2 \times 10^{-5} \times 0.556 = 1.11 \times 10^{-5}$
- Step 937: $2 \times 10^{-5} \times (939-937)/846 = 2 \times 10^{-5} \times 0.0024 = 4.7 \times 10^{-8} \approx 0$

**4.** Warmup is important because:
- At initialization, the task head has random weights → gradients from the head are large and noisy
- Large gradients flowing back through the pretrained backbone can disrupt the carefully learned representations
- Warmup starts with a tiny LR, allowing the head to stabilize before the backbone receives significant gradient updates
- Without warmup, the first few batches can cause catastrophic forgetting

**5.** Constant LR of $2 \times 10^{-5}$:
- Training would still work reasonably well (the LR is already small)
- But the initial updates might be slightly destructive (no warmup)
- At the end of training, the model would not converge as tightly (no decay toward zero)
- Linear schedule generally gives 0.5-1.0% better accuracy in practice
- The decay helps the model settle into a good minimum rather than continuing to bounce around

</details>

---

## Exercise 3: Freezing Strategies

You have BERT-base (12 layers) and datasets of different sizes for a binary classification task.

**Tasks**:
1. You have **50 labeled examples**. Describe the best freezing strategy. What is the effective number of trainable parameters?
2. You have **5,000 labeled examples**. Describe the best freezing strategy.
3. You have **500,000 labeled examples**. Describe the best freezing strategy.
4. Explain **gradual unfreezing**: how does it work, and what problem does it solve?
5. What is **discriminative fine-tuning** (different LR per layer)? Write pseudocode for setting up the optimizer with 3 learning rate groups.

<details>
<summary>Solution</summary>

**1.** With 50 examples: **Freeze all BERT layers. Train only the classification head.**
- Trainable parameters: $768 \times 2 + 2 = 1{,}538$
- Rationale: 50 examples are far too few to meaningfully update 110M parameters. The risk of overfitting is extreme. The pretrained representations are already excellent — we just need to find a linear decision boundary in the 768-dim space.
- Consider also: data augmentation, or even using BERT as a fixed feature extractor with a simple SVM/logistic regression on the [CLS] embeddings.

**2.** With 5,000 examples: **Freeze bottom 8 layers. Fine-tune top 4 layers + classification head.**
- Trainable parameters: ~4 layers * ~7.1M per layer + 1,538 ≈ 28.4M
- Rationale: Enough data to adapt the top layers, which capture more task-specific features. Bottom layers encode general linguistic knowledge that transfers well without modification.

**3.** With 500,000 examples: **Fine-tune all layers** with a small learning rate.
- Trainable parameters: ~110M (all BERT parameters) + 1,538 (head)
- Rationale: Abundant data prevents overfitting even with all parameters unfrozen. Full fine-tuning allows the model to adapt its entire representation hierarchy to the task.
- Use learning rate $2 \times 10^{-5}$ with warmup and decay.

**4.** Gradual unfreezing:
```
Epoch 1: Train only the classification head (freeze all BERT layers)
Epoch 2: Unfreeze top 2 BERT layers + head
Epoch 3: Unfreeze top 4 BERT layers + head
Epoch 4: Unfreeze top 8 BERT layers + head
Epoch 5: Unfreeze all layers + head
```
Problem it solves: **catastrophic forgetting**. If all layers are unfrozen at once, the random head produces noisy gradients that can destroy pretrained representations in lower layers. By first stabilizing the head, then gradually unfreezing from top to bottom, each layer adapts incrementally while lower layers retain their general knowledge.

**5.** Discriminative fine-tuning pseudocode:
```python
# Group parameters by layer depth
param_groups = [
    # Bottom layers (0-3): smallest LR
    {"params": [p for n, p in model.bert.named_parameters()
                if any(f"layer.{i}." in n for i in range(4))],
     "lr": 1e-6},

    # Middle layers (4-7): medium LR
    {"params": [p for n, p in model.bert.named_parameters()
                if any(f"layer.{i}." in n for i in range(4, 8))],
     "lr": 5e-6},

    # Top layers (8-11): larger LR
    {"params": [p for n, p in model.bert.named_parameters()
                if any(f"layer.{i}." in n for i in range(8, 12))],
     "lr": 2e-5},

    # Classification head: largest LR
    {"params": model.classifier.parameters(),
     "lr": 1e-4},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

</details>

---

## Exercise 4: Pretraining vs. Fine-Tuning Data

**Tasks** (analytical):
1. BERT was pretrained on ~3.3B tokens (Wikipedia + BooksCorpus). A sentiment dataset has 25,000 labeled reviews. Compare the data scales and explain why fine-tuning works with so much less data.
2. If you pretrained a new BERT from scratch on only the 25,000 sentiment reviews, would it work? Why or why not?
3. What is **domain adaptation**? If you had a medical NLP task, would you fine-tune `bert-base-uncased` directly or first continue pretraining on medical text? Explain.
4. Explain the concept of **negative transfer**. When might fine-tuning a pretrained model perform *worse* than training from scratch?
5. What is the difference between **feature extraction** (frozen BERT) and **fine-tuning** (unfrozen BERT)? When would you prefer each?

<details>
<summary>Solution</summary>

**1.** The data scales differ by 5 orders of magnitude (3.3B vs 25K). Fine-tuning works because:
- Pretraining learns **general language representations** (syntax, semantics, world knowledge) from the 3.3B tokens
- These representations transfer across tasks — understanding what a word means is useful for almost any NLP task
- Fine-tuning only needs to learn the **task-specific mapping** (which representations correspond to positive vs negative sentiment)
- The 25K examples are sufficient to learn this relatively simple mapping on top of already rich features

Analogy: Teaching someone to sort mail is easy if they already know how to read (pretraining). You do not need millions of sorted mail examples.

**2.** No, it would not work well:
- 25K reviews are far too few to learn general language understanding from scratch
- The model would learn surface-level patterns specific to sentiment rather than deep linguistic knowledge
- It would have poor generalization and vocabulary coverage
- BERT's architecture (110M params) would be severely overfit with only 25K examples
- The model would essentially memorize the training data

**3.** Domain adaptation: continuing pretraining on domain-specific text (e.g., medical papers, legal documents) before task-specific fine-tuning.

For medical NLP: **continue pretraining first on medical text** (this is what BioBERT, ClinicalBERT, etc. do):
- `bert-base-uncased` knows general English but lacks medical vocabulary and domain conventions
- Continue MLM pretraining on PubMed abstracts → model learns medical terminology
- Then fine-tune on the specific medical task (e.g., drug interaction detection)
- This two-step process consistently outperforms direct fine-tuning for specialized domains

**4.** Negative transfer occurs when the pretrained model's representations are **harmful** for the target task:
- If the pretraining domain is very different from the target (e.g., using English BERT for code generation)
- If the pretraining data contains biases that conflict with the target task
- If the model is too large for a very small target dataset (overfitting to noise)
- If the tokenizer is a bad fit (e.g., BERT's tokenizer on chemical formulas)

Training from scratch might be better when: the domain is highly specialized, you have sufficient domain data, and pretrained representations do not help.

**5.** Comparison:
- **Feature extraction** (frozen BERT): Use BERT to compute fixed embeddings, then train a simple classifier on top. Advantages: fast, no risk of catastrophic forgetting, works with very small datasets. Disadvantage: BERT representations are not adapted to the task.
- **Fine-tuning** (unfrozen BERT): Update all BERT parameters along with the classifier. Advantages: better accuracy, representations adapt to the task. Disadvantage: slower, risk of overfitting, requires more data.

Prefer feature extraction when: dataset is tiny (<100 examples), compute is limited, or you need embeddings for multiple tasks (compute once, use everywhere).

Prefer fine-tuning when: dataset is sufficient (>1000 examples), accuracy is critical, and the task differs significantly from general language understanding.

</details>

---

## Exercise 5: End-to-End Fine-Tuning Design

You are building a **restaurant review analysis system** that must:
- Classify overall sentiment (positive/neutral/negative)
- Extract dish names mentioned (named entities)
- Identify the aspect being discussed (food/service/ambiance/price)

Given: 10,000 labeled reviews, BERT-base model available.

**Tasks**:
1. Design the model architecture. How many task heads do you need? What are their shapes?
2. Should all three tasks share the same BERT backbone, or should you have separate BERT instances? Justify with a multi-task learning argument.
3. Write the combined loss function. How would you weight the three losses?
4. Describe your training procedure: what to freeze/unfreeze, learning rates, number of epochs.
5. At inference time, how does the model process a single review? What are the output shapes for a review with 50 tokens?

<details>
<summary>Solution</summary>

**1.** Architecture with 3 task heads:

Shared BERT backbone (12 layers, $D = 768$) → three heads:
- **Sentiment head**: Linear(768, 3) on [CLS] → shape $(B, 3)$ → 2,307 params
- **NER head (dish extraction)**: Linear(768, 5) per token (B-DISH, I-DISH, B-ASP, I-ASP, O using BIO) → shape $(B, L, 5)$ → 3,845 params
- **Aspect head**: Linear(768, 4) on [CLS] (or multi-label with sigmoid) → shape $(B, 4)$ → 3,076 params

Total new parameters: ~9,228 on top of 110M BERT.

**2.** **Share one BERT backbone** for all three tasks (multi-task learning):
- Shared representations are more robust — each task acts as regularizer for the others
- Memory-efficient: one BERT instead of three (110M vs 330M params)
- Information sharing: understanding sentiment helps NER (emotional words near dish names), and aspect detection benefits from entity awareness
- Risk: task interference if tasks have conflicting gradient directions. Mitigate with task-specific adapters or gradient manipulation.

**3.** Combined loss:
$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{sentiment}} + \lambda_2 \mathcal{L}_{\text{NER}} + \lambda_3 \mathcal{L}_{\text{aspect}}$$

All three are cross-entropy losses.

Weighting strategy options:
- Start with equal weights: $\lambda_1 = \lambda_2 = \lambda_3 = 1.0$
- Scale by task difficulty: if NER loss is typically 10x sentiment loss, set $\lambda_2 = 0.1$
- Use uncertainty-based weighting (Kendall et al., 2018): learn $\lambda$ values during training
- Practical starting point: normalize each loss to similar magnitudes, then tune on validation set

**4.** Training procedure:
- **Epochs 1-2**: Freeze BERT, train all three heads ($\text{LR} = 10^{-3}$)
- **Epochs 3-5**: Unfreeze top 4 BERT layers + heads ($\text{LR}_{\text{BERT}} = 2 \times 10^{-5}$, $\text{LR}_{\text{heads}} = 10^{-4}$)
- **Epochs 6-8**: Unfreeze all BERT layers ($\text{LR}_{\text{bottom}} = 10^{-6}$, $\text{LR}_{\text{top}} = 10^{-5}$, $\text{LR}_{\text{heads}} = 5 \times 10^{-5}$)
- Warmup: 10% of steps in each phase
- Weight decay: 0.01
- Gradient clipping: max norm 1.0
- Early stopping: monitor average of three validation metrics

**5.** Inference on a 50-token review:
```
Input: "The pasta was incredible but service was slow" (tokenized to 50 tokens)

BERT:      (1, 50) → (1, 50, 768)   hidden states

Sentiment: [CLS] hidden → Linear → (1, 3)
           → softmax → [0.85, 0.10, 0.05] → "positive"

NER:       All hidden states → Linear → (1, 50, 5)
           → argmax per token → "O O B-DISH O O O O O O ..."
           → Extract: ["pasta"]

Aspect:    [CLS] hidden → Linear → (1, 4)
           → sigmoid → [0.9, 0.7, 0.0, 0.0] → ["food", "service"]
           (multi-label: threshold at 0.5)
```

Total forward pass: one BERT encoding + three cheap linear projections.

</details>

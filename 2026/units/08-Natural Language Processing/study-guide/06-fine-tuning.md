# Fine-Tuning Pretrained Models

**Prerequisites**: BERT and GPT architectures, cross-entropy loss, learning rate scheduling, PyTorch training loops
**USAAIO Relevance**: Fine-tuning is how pretrained models are adapted to specific tasks — the practical bridge between "pretrained foundation model" and "working NLP system." IOAI problems test your understanding of when to freeze vs. unfreeze layers, how to add task-specific heads, and the pretraining-fine-tuning paradigm.

---

## Discovery

### Why Pretraining Changed Everything

Before 2018, NLP models were trained **from scratch** for each task. Want a sentiment classifier? Train on labeled sentiment data. Want a named entity recognizer? Train on labeled NER data. Each task required large amounts of **task-specific** labeled data.

The breakthrough insight: **language understanding is transferable**.

A model that has learned to predict masked words (BERT) or next tokens (GPT) has implicitly learned:
- Syntax (grammatical structure)
- Semantics (word meaning)
- World knowledge (facts about entities)
- Reasoning patterns (cause and effect)

Fine-tuning unlocks this knowledge for specific tasks with **very little labeled data**.

```
Traditional approach:
  Task data (10,000 examples) → Train model from scratch → OK accuracy

Pretrained + Fine-tuned:
  Unlabeled text (billions of words) → Pretrain →
  Task data (100 examples) → Fine-tune → Great accuracy
```

> **Socratic question**: If a model pretrained on Wikipedia and books has learned so much about language, why does it still need fine-tuning? Why can it not just solve any NLP task out of the box?

### The Pretraining-Fine-Tuning Paradigm

| Phase | Data | Labels | Duration | Learning Rate |
|---|---|---|---|---|
| **Pretraining** | Massive unlabeled text | Self-supervised (MLM, next token) | Days to weeks on GPU clusters | $10^{-4}$ to $10^{-3}$ |
| **Fine-tuning** | Small task-specific data | Human-annotated | Minutes to hours on one GPU | $2 \times 10^{-5}$ to $5 \times 10^{-5}$ |

---

## Intuition

### How Fine-Tuning Works

Fine-tuning takes a pretrained model and makes small adjustments to adapt it to a specific task:

```
Pretrained BERT                    Fine-tuned BERT (Sentiment)
┌─────────────────────┐            ┌─────────────────────┐
│                     │            │                     │
│  Transformer        │            │  Transformer        │  ← Weights slightly
│  Encoder            │   fine-    │  Encoder            │    adjusted
│  (12 layers)        │   tune     │  (12 layers)        │
│                     │   ───→     │                     │
│  Pretrained weights │            │  Adjusted weights   │
│                     │            │                     │
└────────┬────────────┘            └────────┬────────────┘
         │                                  │
         ↓                                  ↓
   [CLS] hidden                       [CLS] hidden
   (768-dim)                          (768-dim)
                                            │
                                   ┌────────↓────────┐
                                   │  Linear(768, 2) │  ← NEW: task head
                                   │  (random init)  │    (trained from scratch)
                                   └────────┬────────┘
                                            ↓
                                   [positive, negative]
```

### Common Task Heads

```
Task                    Input to Head           Head Architecture
────────────────────    ──────────────────      ─────────────────
Sequence Classification [CLS] hidden state      Linear(D, C)
Sentiment Analysis      [CLS] hidden state      Linear(D, 2)
Token Classification    Each token's hidden      Linear(D, num_labels) per token
Named Entity Recog.     Each token's hidden      Linear(D, num_entity_types)
Question Answering      Each token's hidden      Linear(D, 2) → start, end logits
Text Generation         Last token hidden        Linear(D, |V|) (LM head)
```

### Freezing vs. Unfreezing

```
Strategy 1: Fine-tune everything (most common)
  All layers updated with small learning rate
  → Best accuracy, risk of overfitting on small data

Strategy 2: Freeze backbone, train head only
  Only the task head is updated
  → Fastest, safe from overfitting, lower accuracy

Strategy 3: Gradual unfreezing
  Epoch 1: Only head
  Epoch 2: Head + last 2 layers
  Epoch 3: Head + last 4 layers
  ...
  → Good balance, reduces catastrophic forgetting

Strategy 4: Discriminative learning rates
  Lower layers: very small LR (e.g., 1e-6)
  Upper layers: larger LR (e.g., 1e-5)
  Head: largest LR (e.g., 1e-4)
  → Lower layers preserve general knowledge, upper layers adapt
```

### Learning Rate for Fine-Tuning

Fine-tuning uses a **much smaller** learning rate than pretraining. Why?

```
Pretrained weights are in a good region of the loss landscape.
We want to make small adjustments, not large jumps.

LR too large:   ╱╲     Catastrophic forgetting
                ╱  ╲    (lose pretrained knowledge)
               ╱    ╲

LR just right:  ─╲     Small refinement
                  ╲─   (adapt to task)

LR too small:   ───     No learning
                        (stays at pretrained solution)
```

Typical: $\text{LR} = 2 \times 10^{-5}$ with linear warmup and decay.

### Few-Shot, Zero-Shot, and In-Context Learning

With large enough pretrained models, fine-tuning may not even be necessary:

```
Zero-shot (no examples):
  Prompt: "Classify the sentiment: 'This movie was terrible.' → "
  Model:  "Negative"

One-shot (one example):
  Prompt: "'Great film!' → Positive
           'This movie was terrible.' → "
  Model:  "Negative"

Few-shot (a few examples):
  Prompt: "'Great film!' → Positive
           'Boring and slow.' → Negative
           'Best movie ever!' → Positive
           'This movie was terrible.' → "
  Model:  "Negative"
```

This works because large models (GPT-3+) have learned the **pattern** of following examples from their pretraining data.

---

## Math

### Fine-Tuning Loss

For sequence classification with $C$ classes:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

where $\hat{y}_i = \text{softmax}(W h_{[\text{CLS}]}^{(i)} + b)$ and $W \in \mathbb{R}^{C \times D}$, $b \in \mathbb{R}^C$.

For token classification (NER) with $T$ entity types:

$$\mathcal{L} = -\frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L_i} \sum_{t=1}^{T} y_{ijt} \log \hat{y}_{ijt}$$

where $\hat{y}_{ij} = \text{softmax}(W h_j^{(i)} + b)$ for each token position $j$.

### Learning Rate Scheduling

**Linear warmup + linear decay** (most common for fine-tuning):

$$\text{LR}(t) = \begin{cases} \text{LR}_{\max} \cdot \frac{t}{t_{\text{warmup}}} & \text{if } t < t_{\text{warmup}} \\ \text{LR}_{\max} \cdot \frac{T - t}{T - t_{\text{warmup}}} & \text{otherwise} \end{cases}$$

where $T$ is the total number of training steps and $t_{\text{warmup}}$ is typically 6-10% of $T$.

### Catastrophic Forgetting

When fine-tuning moves weights too far from pretrained values, the model "forgets" general language knowledge. Formally, we want to minimize:

$$\mathcal{L}_{\text{task}}(\theta) \quad \text{subject to} \quad \|\theta - \theta_{\text{pretrained}}\| \text{ staying small}$$

Strategies to prevent this:
1. **Small learning rate**: limits step size
2. **Weight decay**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \|\theta - \theta_{\text{pretrained}}\|^2$
3. **Gradual unfreezing**: adapts top layers first
4. **Early stopping**: stop before overfitting

---

## Code

### Fine-Tuning BERT for Sentiment Classification

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (L,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (L,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)              # (B, num_classes)
        return logits

# Training setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = SentimentClassifier(num_classes=2)

# Separate learning rates for BERT and classifier
optimizer = torch.optim.AdamW([
    {"params": model.bert.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
], weight_decay=0.01)

# Linear warmup + decay
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")
```

### Fine-Tuning for Named Entity Recognition

```python
class NERModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # Per-token classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, 768)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)     # (B, L, num_labels)
        return logits

# NER labels (BIO scheme):
# B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O
# Example:
#   "Barack Obama visited Paris" →
#   [B-PER, I-PER, O, B-LOC]
```

### Using HuggingFace Trainer (Simplified)

```python
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

## Key Takeaways

1. **Pretraining** learns general language knowledge from massive unlabeled text; **fine-tuning** adapts it to specific tasks with small labeled datasets
2. Fine-tuning adds a **task-specific head** (usually a linear layer) on top of the pretrained model
3. Use a **small learning rate** ($2 \times 10^{-5}$) for fine-tuning to avoid catastrophic forgetting
4. **Linear warmup + decay** is the standard learning rate schedule for fine-tuning
5. **Freezing** the backbone (training only the head) is useful when task data is very small
6. **Gradual unfreezing** and **discriminative learning rates** offer a middle ground
7. Large models (GPT-3+) can perform **zero-shot** and **few-shot** learning without fine-tuning via in-context prompting
8. Common fine-tuning tasks: sequence classification (sentiment), token classification (NER), extractive QA (span prediction)
9. On USAAIO exams, understand when to freeze/unfreeze, how to choose learning rates, and how to add task heads to pretrained models

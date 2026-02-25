# Unit 08: Natural Language Processing — Study Guide Overview

**Course**: AI 510-NLP — Natural Language Processing
**USAAIO Relevance**: Core unit for Round 2 — NLP problems appear consistently in IOAI contests

---

## What This Unit Covers

Natural Language Processing bridges the gap between human language and machine understanding. How do you take a sentence — a sequence of characters — and turn it into something a neural network can reason about? This unit covers the full pipeline: from tokenization (converting text to numbers) through embeddings (learning what words mean) to transformers (the architecture that revolutionized NLP).

The central question: **How do we represent language as vectors, and how do we build models that understand context?**

## Roadmap

| # | Topic | Key Ideas |
|---|---|---|
| 01 | Tokenization | Character-level, BPE, WordPiece — text to token IDs |
| 02 | Word Embeddings | Skip-gram, CBOW — learning dense word vectors |
| 03 | GloVe | Co-occurrence matrices, embedding arithmetic |
| 04 | Encoder Transformers | BERT, bidirectional attention, masked language modeling |
| 05 | Decoder Transformers | GPT, causal attention, autoregressive generation |
| 06 | Fine-Tuning | Transfer learning, task heads, practical NLP pipelines |

## Prerequisites

- **Linear algebra** (AI 200): Matrix multiplication, dot products, cosine similarity
- **Calculus & Optimization** (AI 210): Gradients, SGD, cross-entropy loss
- **Deep learning fundamentals** (AI 410): Neural networks, backpropagation, embeddings
- **Transformer basics** (AI 420): Self-attention, multi-head attention, positional encoding
- **PyTorch** (AI 310): `nn.Module`, `nn.Embedding`, training loops, `torch.no_grad()`

## How to Use This Guide

Each topic follows the **D-I-M-C** pattern:

1. **Discovery** — Historical context, key papers, motivating questions
2. **Intuition** — Visual explanations, ASCII diagrams, analogies
3. **Math** — Rigorous derivations of objectives and gradients
4. **Code** — PyTorch implementations from scratch + HuggingFace equivalents

### Study Strategy

1. **First pass**: Read Discovery + Intuition to understand the "why"
2. **Second pass**: Work through Math derivations with pen and paper
3. **Third pass**: Implement Code sections from memory
4. **Practice**: Complete the exercises for each topic
5. **Apply**: Work through the assignment notebooks

## Connections to Other Units

```
AI 200 (Linear Algebra) ──→ Dot products, cosine similarity, matrix factorization
AI 210 (Optimization)   ──→ SGD, cross-entropy, negative sampling
AI 410 (Deep Learning)  ──→ Neural networks, embeddings, backpropagation
AI 420 (Transformers)   ──→ Self-attention, positional encoding, layer norm
         ↓
    ┌────────────┐
    │  AI 510    │ ← YOU ARE HERE
    │  NLP       │
    └────────────┘
         ↓
AI 510-GNN (Graph NNs)  ──→ Message passing on structured data
AI 520 (CV & Gen AI)     ──→ Vision transformers, multimodal models
```

## USAAIO Exam Patterns

On the USAAIO exam, NLP topics typically appear as:

- **Run BPE by hand** on a small corpus (trace merge operations)
- **Compute embedding similarities** using cosine distance
- **Trace attention patterns** through BERT or GPT (identify what each token attends to)
- **Predict masked tokens** given BERT-style context
- **Compare architectures**: When to use encoder vs decoder vs encoder-decoder
- **Fine-tuning pipeline**: Select appropriate model, head, and hyperparameters for a task
- **Implement from scratch** in PyTorch (tokenizer, skip-gram, attention masking)
- **Analyze generation**: Temperature, top-k, and nucleus sampling effects

# Unit 10: Computer Vision & Generative AI — Study Guide Overview

**Course**: AI 520 — Computer Vision 2 and Generative AI
**USAAIO Relevance**: CRITICAL for Round 2. The 2025 Round 2 Problem 3 (100 points) tested CLIP.

---

## What This Unit Covers

This unit covers the advanced computer vision and generative modeling topics that form the core of USAAIO Round 2. You will learn how to detect objects, segment images, generate new images from noise, and connect vision with language. The 2025 Round 2 exam devoted its highest-value problem (100 points) to CLIP, making this unit essential for medal contention.

## Roadmap

| # | Topic | Key Ideas |
|---|---|---|
| 01 | Object Detection | IoU, NMS, anchor boxes, mAP |
| 02 | UNet | Encoder-decoder with skip connections, segmentation |
| 03 | Autoencoders | Encoder-bottleneck-decoder, reconstruction loss |
| 04 | Variational Autoencoder | ELBO, KL divergence, reparameterization trick |
| 05 | GANs | Minimax game, generator/discriminator, mode collapse |
| 06 | Diffusion Models | DDPM forward/reverse, noise prediction, sampling |
| 07 | Stable Diffusion | Latent diffusion, cross-attention conditioning, CFG |
| 08 | CLIP | Dual encoder, InfoNCE, zero-shot classification |
| 09 | Adversarial Attacks | FGSM, PGD, adversarial training |

## Prerequisites

- **Deep Learning** (AI 410/411): CNNs, backpropagation, loss functions, optimizers
- **Transformers** (AI 500): Self-attention, multi-head attention, positional encoding
- **PyTorch** (AI 310): Module subclassing, autograd, training loops
- **Probability** (AI 210): KL divergence, Gaussian distribution, MLE, Bayes' rule
- **Linear Algebra** (AI 200): Matrix operations, cosine similarity, projections

## How to Use This Guide

Each topic follows the **D-I-M-C** pattern:

1. **Discovery** — Historical context, motivating questions, misconception traps
2. **Intuition** — Visual and geometric explanations, ASCII diagrams
3. **Math** — Rigorous derivations with no skipped steps
4. **Code** — PyTorch from-scratch implementations

### Study Strategy

1. **First pass**: Read Discovery + Intuition for all topics (big picture)
2. **Second pass**: Work through VAE, Diffusion, and CLIP math derivations with pen and paper
3. **Third pass**: Implement from scratch — close the guide and code from memory
4. **CLIP deep dive**: Spend extra time on Topic 08 — this is the most exam-relevant topic
5. **Practice**: Complete exercises, then work through all assignment notebooks

## Connections to Other Units

```
AI 410 (DL1: CNNs)      ──→ Conv layers, pooling, feature extraction
AI 411 (DL2: Advanced)   ──→ Residual networks, batch norm, training tricks
AI 500 (Transformers)    ──→ Self-attention, ViT, cross-attention
         ↓
    ┌───────────────────┐
    │  AI 520            │ ← YOU ARE HERE
    │  CV2 & Gen AI      │
    └───────────────────┘
         ↓
AI 530 (Graph NNs) ──→ Non-grid structured data
AI 600 (Grandmaster) ──→ Full competition problems
```

## USAAIO Exam Patterns

On Round 2, CV and generative AI topics typically appear as:

- **Implement a loss function** from its mathematical definition (InfoNCE, ELBO, adversarial loss)
- **Build an architecture** module by module (ViT, UNet, VAE encoder/decoder)
- **Derive a formula** step by step (ELBO decomposition, diffusion forward process)
- **Compute numeric values** from given weights/inputs (cosine similarity, IoU, noise schedule)
- **Trace a forward pass** through a pipeline (CLIP dual encoder, diffusion sampling)
- **Analyze and extend** existing code (modify loss function, add conditioning)

## 2025 Round 2 Problem 3 Analysis

The CLIP problem tested these specific skills:
- Understanding dual-encoder architecture (image encoder + text encoder)
- Implementing InfoNCE / contrastive loss with temperature scaling
- Computing cosine similarity between embedding vectors
- Zero-shot classification pipeline
- Vision Transformer (ViT) as image encoder

**This means**: Topics 08 (CLIP) and the ViT portions of the unit are highest priority. Topics 04 (VAE) and 06 (Diffusion) are the next tier, as they test deep mathematical reasoning that is characteristic of Round 2 problems.

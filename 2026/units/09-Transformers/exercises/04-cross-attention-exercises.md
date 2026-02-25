# Exercises 04 — Cross-Attention & Masked Attention

> 5 competition-level exercises

---

## Exercise 4.1 — Cross-Attention Shapes

An encoder-decoder model has:
- Encoder output: $(B, L_{\text{enc}}, D_{\text{enc}}) = (8, 100, 768)$
- Decoder hidden state: $(B, L_{\text{dec}}, D_{\text{dec}}) = (8, 50, 512)$
- $H = 8, D_{qk} = 64, D_v = 64$

**(a)** What are the shapes of $W^Q, W^K, W^V, W^O$ for the cross-attention layer?

**(b)** What is the shape of the attention matrix?

**(c)** Compute the total parameters for this cross-attention layer.

**(d)** Compare with a self-attention layer in the decoder. Which has more parameters? Why?

---

## Exercise 4.2 — Causal Mask Implementation

**(a)** Write a PyTorch one-liner to create a causal mask of size $L \times L$.

**(b)** Given attention logits of shape $(B, H, L, L)$, write the code to apply the causal mask.

**(c)** Show that after applying the causal mask and softmax, each row $i$ of the attention matrix has non-zero entries only in columns $1, \dots, i$.

**(d)** What is the effective number of non-zero entries in the $L \times L$ attention matrix after causal masking? Express as a fraction of $L^2$.

**(e)** Despite only having $L(L+1)/2$ non-zero entries, standard implementations still compute the full $L \times L$ matrix. Why? (Consider hardware.)

---

## Exercise 4.3 — Three Types of Attention in a Decoder

A transformer decoder block has three attention mechanisms. For each one:

**(a)** Name the attention type (self, cross, or masked self-attention).

**(b)** Where do Q, K, V come from (decoder input, encoder output, or previous layer)?

**(c)** Is a mask applied? If so, what kind?

**(d)** What is the shape of the attention matrix for each, given $L_{\text{enc}} = 100$ and $L_{\text{dec}} = 50$?

The three mechanisms are:
1. First sub-layer of the decoder
2. Second sub-layer of the decoder
3. The encoder's self-attention (for comparison)

---

## Exercise 4.4 — Padding Mask Construction

You have a batch of 3 sequences with lengths [5, 3, 7], padded to max length 7:

```
Sequence 1: [tok, tok, tok, tok, tok, PAD, PAD]
Sequence 2: [tok, tok, tok, PAD, PAD, PAD, PAD]
Sequence 3: [tok, tok, tok, tok, tok, tok, tok]
```

**(a)** Construct the padding mask tensor of shape $(3, 7)$ (1 for real tokens, 0 for padding).

**(b)** Reshape this mask so it can be broadcast with attention logits of shape $(B, H, L, L)$. What shape do you need?

**(c)** Now combine the padding mask with a causal mask. Write the combined mask for sequence 2 (length 3, padded to 7). Show the $7 \times 7$ matrix.

**(d)** After applying this combined mask and softmax, what does the attention matrix look like for sequence 2 at position 2 (the third real token)?

---

## Exercise 4.5 — Bidirectional vs. Causal Attention

**(a)** In BERT (encoder-only), every token attends to every other token. Write the mask matrix for $L = 4$. (Hint: it's trivial.)

**(b)** In GPT (decoder-only), causal masking is used. Write the mask for $L = 4$.

**(c)** In a prefix language model, the first $k$ tokens can attend to each other bidirectionally, and the remaining tokens use causal masking. Write the mask for $L = 6$ with prefix length $k = 3$.

**(d)** Prove that the set of attention patterns achievable with causal masking is a strict subset of those achievable with bidirectional attention.

**(e)** Despite this, GPT-style models can match or exceed BERT on many tasks. Suggest why.

---

*Solutions are intentionally omitted. Discuss with your study group or verify with PyTorch.*

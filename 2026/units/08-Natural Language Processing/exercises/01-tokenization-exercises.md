# Tokenization Exercises

**Topic**: Character-level tokenization, Byte Pair Encoding, WordPiece
**Difficulty**: Increasing (exercises 1-2 foundational, 3-4 intermediate, 5 advanced)

---

## Exercise 1: Character Tokenization

Given the string `"neural nets"`, perform character-level tokenization.

**Tasks**:
1. List all unique characters and assign each an integer ID (alphabetical order, space included).
2. Encode the string as a list of integer IDs.
3. What is the vocabulary size?
4. What is the sequence length?
5. If you had a 500-word document averaging 5 characters per word plus spaces, approximately how long would the token sequence be?

<details>
<summary>Solution</summary>

**1. Vocabulary**:
```
' ' → 0, 'a' → 1, 'e' → 2, 'l' → 3, 'n' → 4, 'r' → 5, 's' → 6, 't' → 7, 'u' → 8
```

**2. Encoding**:
```
"neural nets"
n=4, e=2, u=8, r=5, a=1, l=3, ' '=0, n=4, e=2, t=7, s=6
→ [4, 2, 8, 5, 1, 3, 0, 4, 2, 7, 6]
```

**3. Vocabulary size**: 9 unique characters

**4. Sequence length**: 11 (including space)

**5. Approximate length**: 500 words * 5 chars + 499 spaces = 2999 ≈ 3000 tokens. This is quite long for transformers — attention is O(L^2), so 3000^2 = 9M attention computations per layer.

</details>

---

## Exercise 2: BPE by Hand (5 Merges)

Given the corpus with word frequencies:
```
"low"    : 5
"lower"  : 2
"newest" : 6
"widest" : 3
```

Represent each word as space-separated characters with end-of-word marker `_`.

**Tasks**:
1. Write the initial character-level representation with frequencies.
2. Perform 5 BPE merges. For each merge, show:
   - All pair counts
   - The most frequent pair
   - The corpus after merging
3. What is the final vocabulary?
4. Using the learned merges, tokenize the new word `"newer"`.

<details>
<summary>Solution</summary>

**1. Initial representation**:
```
l o w _         : 5
l o w e r _     : 2
n e w e s t _   : 6
w i d e s t _   : 3
```

**2. Merges**:

**Merge 1**: Count pairs:
- (l, o): 5+2=7, (o, w): 5+2=7, (w, _): 5, (w, e): 2+6=8, (e, r): 2, (r, _): 2, (n, e): 6, (e, s): 6+3=9, (s, t): 6+3=9, (t, _): 6+3=9, (w, i): 3, (i, d): 3, (d, e): 3
- Tie between (e,s), (s,t), (t,_) all at 9. Pick (e, s) → `es`

```
l o w _         : 5
l o w e r _     : 2
n e w es t _    : 6
w i d es t _    : 3
```

**Merge 2**: Count pairs:
- (l, o): 7, (o, w): 7, (w, _): 5, (w, e): 2, (e, r): 2, (r, _): 2, (n, e): 6, (e, w): 6, (w, es): 6, (es, t): 6+3=9, (t, _): 6+3=9, (w, i): 3, (i, d): 3, (d, es): 3
- Tie between (es, t) and (t, _) at 9. Pick (es, t) → `est`

```
l o w _         : 5
l o w e r _     : 2
n e w est _     : 6
w i d est _     : 3
```

**Merge 3**: Count pairs:
- (l, o): 7, (o, w): 7, (w, _): 5, (w, e): 2, (e, r): 2, (r, _): 2, (n, e): 6, (e, w): 6, (w, est): 6, (est, _): 6+3=9, (w, i): 3, (i, d): 3, (d, est): 3
- Most frequent: (est, _) at 9. Merge → `est_`

```
l o w _         : 5
l o w e r _     : 2
n e w est_      : 6
w i d est_      : 3
```

**Merge 4**: Count pairs:
- (l, o): 7, (o, w): 7, (w, _): 5, (w, e): 2, (e, r): 2, (r, _): 2, (n, e): 6, (e, w): 6, (w, est_): 6, (w, i): 3, (i, d): 3, (d, est_): 3
- Most frequent: (l, o) and (o, w) tied at 7. Pick (l, o) → `lo`

```
lo w _          : 5
lo w e r _      : 2
n e w est_      : 6
w i d est_      : 3
```

**Merge 5**: Count pairs:
- (lo, w): 5+2=7, (w, _): 5, (w, e): 2, (e, r): 2, (r, _): 2, (n, e): 6, (e, w): 6, (w, est_): 6, (w, i): 3, (i, d): 3, (d, est_): 3
- Most frequent: (lo, w) at 7. Merge → `low`

```
low _           : 5
low e r _       : 2
n e w est_      : 6
w i d est_      : 3
```

**3. Final vocabulary**: `{_, a, d, e, i, l, n, o, r, s, t, w, es, est, est_, lo, low}`

**4. Tokenize "newer"**:
Start: `n e w e r _`
Apply merges in order:
1. (e, s) → no match
2. (es, t) → no match
3. (est, _) → no match
4. (l, o) → no match
5. (lo, w) → no match

Result: `['n', 'e', 'w', 'e', 'r', '_']` — none of our merges apply to "newer".

</details>

---

## Exercise 3: BPE Vocabulary Size Analysis

You are designing a tokenizer for a multilingual NLP system.

**Tasks**:
1. If your base character set has 256 unique bytes (UTF-8) and you perform 30,000 merges, what is the total vocabulary size?
2. BERT uses a vocabulary of 30,522 tokens. If the base characters take ~1,000 slots, approximately how many merges were performed?
3. GPT-2 uses a vocabulary of 50,257 tokens with a byte-level BPE base of 256 characters. How many merges were performed?
4. Explain the tradeoff: Why not use 1,000,000 merges? Why not use 100?

<details>
<summary>Solution</summary>

**1.** $|V| = 256 + 30{,}000 = 30{,}256$

**2.** $30{,}522 - 1{,}000 = 29{,}522$ merges (approximately). BERT actually uses WordPiece, not pure BPE, but the vocabulary size calculation is similar.

**3.** $50{,}257 - 256 = 50{,}001$ merges

**4. Tradeoff**:
- **Too many merges (1M)**: Vocabulary becomes very large → embedding matrix is $(1{,}000{,}256 \times D)$, consuming massive memory. Many tokens are rare full words with poor embeddings. Approaches word-level tokenization problems (OOV, poor rare word handling).
- **Too few merges (100)**: Vocabulary stays close to character-level → sequences are very long. Attention is $O(L^2)$, so long sequences are computationally expensive. Model must learn to assemble meaning from individual characters.
- **Sweet spot (30K-50K)**: Balances vocabulary size, sequence length, and embedding quality. Common words become single tokens. Rare/new words decompose into meaningful subwords.

</details>

---

## Exercise 4: WordPiece Scoring

Given these token frequencies in a corpus:
```
Token frequencies: "un" = 100, "happy" = 50, "unhappy" = 30
Adjacent pair frequency: ("un", "happy") = 30
```

**Tasks**:
1. Compute the WordPiece merge score for the pair ("un", "happy").
2. Compare: If another pair ("re", "play") has frequencies "re"=80, "play"=60, "replay"=20, what is its score?
3. Which pair would WordPiece merge first? Explain why this makes sense linguistically.
4. How does this scoring differ from standard BPE?

<details>
<summary>Solution</summary>

**1.** WordPiece score:
$$\text{score}(\text{un}, \text{happy}) = \frac{\text{freq}(\text{unhappy})}{\text{freq}(\text{un}) \times \text{freq}(\text{happy})} = \frac{30}{100 \times 50} = \frac{30}{5000} = 0.006$$

**2.**
$$\text{score}(\text{re}, \text{play}) = \frac{\text{freq}(\text{replay})}{\text{freq}(\text{re}) \times \text{freq}(\text{play})} = \frac{20}{80 \times 60} = \frac{20}{4800} \approx 0.00417$$

**3.** WordPiece would merge ("un", "happy") first (score 0.006 > 0.00417). This makes sense because:
- "unhappy" appears relatively often compared to how often "un" and "happy" appear independently
- The merge captures a meaningful linguistic unit (prefix + root)
- The higher score indicates "un" and "happy" co-occur more than expected by chance

**4.** Standard BPE would compare raw adjacent pair counts (30 vs 20), also choosing ("un", "happy"). But in cases where one pair has high absolute frequency but the components are each very common (e.g., "the" + "is"), BPE might merge them while WordPiece would not (because the high individual frequencies lower the score). WordPiece better captures "surprising" co-occurrences.

</details>

---

## Exercise 5: Tokenization in Practice

Consider the sentence: `"ChatGPT is unhelpfully unresponsive"`

**Tasks**:
1. How would a **word-level** tokenizer handle this? What problems arise?
2. How would a **character-level** tokenizer handle this? What is the sequence length?
3. BERT's WordPiece tokenizer would likely produce something like: `["chat", "##gp", "##t", "is", "un", "##help", "##ful", "##ly", "un", "##res", "##pon", "##sive"]`. Explain what the `##` prefix means and why this decomposition is useful.
4. GPT-2's BPE tokenizer might produce: `["Chat", "G", "PT", " is", " unhelpful", "ly", " unrespons", "ive"]`. Notice the leading spaces. Why does GPT-2 attach spaces to the beginning of words?
5. Both tokenizers handle the novel word "ChatGPT" by decomposing it into subwords. Why is this critical for NLP models deployed in the real world?

<details>
<summary>Solution</summary>

**1.** Word-level: `["ChatGPT", "is", "unhelpfully", "unresponsive"]`. Problems:
- "ChatGPT" was not in the training vocabulary → becomes `[UNK]` (unknown token)
- "unhelpfully" may also be OOV if the training corpus did not include this exact form
- OOV tokens lose all semantic information

**2.** Character-level: 35 characters (including spaces). Sequence length = 35. This is manageable for a single sentence but would be very long for documents.

**3.** The `##` prefix means "this token is a continuation of the previous token" (not a word start). This is useful because:
- "un" + "##help" + "##ful" + "##ly" captures morphological structure (prefix + root + suffix + suffix)
- Each subword carries meaning: "un-" = negation, "help" = root meaning, "-ful" = adjective-forming, "-ly" = adverb-forming
- The model can generalize: if it knows "un-" negates, it can understand "un-" + any adjective

**4.** GPT-2 attaches spaces to word beginnings so that tokenization is reversible without ambiguity. If spaces were separate tokens, `"a part"` and `"apart"` might tokenize the same way. By including the space in the token, the model distinguishes word boundaries. This is an encoding choice — GPT-2 uses byte-level BPE where spaces are treated as regular bytes.

**5.** New words (ChatGPT, COVID, NFT, etc.) appear constantly in real-world text. A model deployed in production must handle vocabulary evolution. Subword tokenization ensures:
- No word is truly "unknown" — it can always be decomposed into subwords or characters
- Novel words get at least partial semantic representation from their components
- The model does not need retraining when new terminology appears

</details>

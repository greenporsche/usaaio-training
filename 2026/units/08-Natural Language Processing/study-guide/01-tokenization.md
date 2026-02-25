# Tokenization

**Prerequisites**: Python strings, basic data structures (dictionaries, lists)
**USAAIO Relevance**: Tokenization is the very first step in any NLP pipeline. IOAI problems frequently ask you to implement BPE from scratch or trace merge operations by hand. Understanding tokenization deeply means understanding why transformers have fixed vocabularies and how subword methods handle unseen words.

---

## Discovery

### The Fundamental Problem: Text is Not Numbers

Neural networks operate on tensors of floating-point numbers. Text is a sequence of characters. Before any NLP model can process language, we must answer: **How do we convert "The cat sat on the mat" into a tensor?**

The naive approach — assign each word a unique integer — seems reasonable until you encounter:
- Words you have never seen before ("unfriending", "ChatGPT")
- Morphological variations ("run", "running", "ran", "runner")
- Multilingual text (Chinese has ~50,000 common characters)
- A vocabulary so large that the embedding matrix does not fit in memory

This is the **tokenization problem**, and its solution shaped modern NLP.

> **Socratic question**: If you had to encode every English word as a separate token, how large would your vocabulary be? What happens when someone invents a new word?

### A Brief History

| Year | Method | Key Insight |
|---|---|---|
| Pre-2015 | Word-level | Simple, but massive vocabularies and OOV problems |
| Pre-2015 | Character-level | Tiny vocab, but sequences become very long |
| 2015 | **BPE** (Sennrich et al.) | Merge frequent character pairs — adaptive subword vocab |
| 2016 | **WordPiece** (Schuster & Nakajima) | Like BPE but merge by likelihood — used in BERT |
| 2018 | **Unigram** (Kudo) | Start large, prune — probabilistic tokenization |

The breakthrough: **subword tokenization** — a middle ground between characters and words.

---

## Intuition

### Character-Level Tokenization

The simplest approach: every character is a token.

```
Input:  "hello world"
Tokens: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
IDs:    [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
```

**Vocabulary size**: ~256 (ASCII) or ~65,536 (Unicode BMP)

```
Pros:                          Cons:
┌─────────────────────┐        ┌─────────────────────────────┐
│ ✓ No OOV tokens     │        │ ✗ Very long sequences       │
│ ✓ Tiny vocabulary   │        │ ✗ Each token = 1 character  │
│ ✓ Simple to build   │        │ ✗ Model must learn spelling │
└─────────────────────┘        └─────────────────────────────┘
```

A 500-word document becomes ~2500 characters. Transformer attention is $O(L^2)$, so this gets expensive fast.

### Byte Pair Encoding (BPE)

BPE starts with characters and **iteratively merges the most frequent pair**:

```
Corpus: "low low low lower lower newest newest"

Step 0 (characters):
  l o w _       (3 + 2 = 5 times)
  l o w e r _   (2 times)
  n e w e s t _ (2 times)

Step 1: Most frequent pair = (l, o) → merge into "lo"
  lo w _         (5 times)
  lo w e r _     (2 times)
  n e w e s t _  (2 times)

Step 2: Most frequent pair = (lo, w) → merge into "low"
  low _          (5 times)
  low e r _      (2 times)
  n e w e s t _  (2 times)

Step 3: Most frequent pair = (low, _) → merge into "low_"
  low_           (3 times)
  low e r _      (2 times)
  n e w e s t _  (2 times)

...continue for N merges
```

After training, BPE tokenizes new text by applying learned merges in order:

```
"lowest" → ['low', 'e', 's', 't']  (knows "low" but not "est")
```

### WordPiece

Similar to BPE, but merges are chosen by **likelihood increase** rather than raw frequency:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This favors merging pairs where the combination is more likely than the independent parts — a form of pointwise mutual information.

WordPiece uses `##` prefix for continuation tokens:
```
"unbelievable" → ["un", "##believ", "##able"]
```

### Comparison at a Glance

```
Character:  "playing" → ['p','l','a','y','i','n','g']     7 tokens
BPE:        "playing" → ['play', 'ing']                    2 tokens
Word:       "playing" → ['playing']                         1 token
```

---

## Math

### BPE: Formal Algorithm

**Training**:

Given corpus $C$ as a sequence of words, each represented as a character sequence with end-of-word marker `_`:

1. Initialize vocabulary $V$ = all unique characters in $C$
2. Represent each word as a sequence of characters
3. For $i = 1$ to $N$ (number of merges):
   - Count all adjacent symbol pairs across the corpus
   - Let $(a^*, b^*) = \arg\max_{(a,b)} \text{count}(a, b)$
   - Create new symbol $ab^* = \text{concat}(a^*, b^*)$
   - Add $ab^*$ to $V$
   - Replace all occurrences of $(a^*, b^*)$ with $ab^*$ in the corpus
4. Return $V$ and the ordered list of merges $M = [(a_1, b_1), \dots, (a_N, b_N)]$

**Inference** (tokenizing new text):

1. Split input into characters (with end-of-word markers)
2. For each merge $(a_i, b_i)$ in $M$ (in order learned):
   - Replace all adjacent pairs $(a_i, b_i)$ in the sequence with $a_i b_i$
3. Return the resulting sequence of tokens

**Vocabulary size**: $|V| = |\text{base characters}| + N$

### WordPiece Scoring

For candidate merge of tokens $a$ and $b$:

$$\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a) \cdot \text{count}(b)}$$

This is equivalent to maximizing the log-likelihood of a unigram language model on the training corpus.

### Sequence Length vs. Vocabulary Tradeoff

Let $|V|$ be vocabulary size and $\bar{L}$ be average sequence length.

- Character-level: $|V| \approx 256$, $\bar{L}$ large
- Word-level: $|V| \approx 100{,}000+$, $\bar{L}$ small
- Subword (BPE): $|V| \approx 30{,}000$–$50{,}000$, $\bar{L}$ moderate

The embedding matrix has shape $(|V|, D)$. Larger $|V|$ means more parameters. Longer $\bar{L}$ means more attention computation. BPE balances this tradeoff.

---

## Code

### Character Tokenizer from Scratch

```python
class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join(self.id_to_char[i] for i in ids)
```

### BPE Tokenizer from Scratch

```python
from collections import Counter

class BPETokenizer:
    """Byte Pair Encoding tokenizer."""

    def __init__(self, num_merges: int = 100):
        self.num_merges = num_merges
        self.merges = []  # Ordered list of (a, b) merges
        self.vocab = {}

    def _get_pairs(self, word_freqs: dict) -> Counter:
        """Count all adjacent pairs across the corpus."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        """Replace all occurrences of pair with merged token."""
        new_word_freqs = {}
        a, b = pair
        bigram = f"{a} {b}"
        replacement = a + b
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def train(self, text: str):
        """Learn BPE merges from text."""
        # Tokenize into words and add end-of-word marker
        words = text.strip().split()
        word_counts = Counter(words)

        # Represent each word as space-separated characters + end marker
        word_freqs = {}
        for word, count in word_counts.items():
            key = ' '.join(list(word)) + ' _'
            word_freqs[key] = count

        # Iteratively merge most frequent pairs
        for i in range(self.num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)

        # Build vocabulary
        self.vocab = set()
        for word in word_freqs:
            for token in word.split():
                self.vocab.add(token)

    def encode(self, text: str) -> list[str]:
        """Tokenize text using learned merges."""
        words = text.strip().split()
        all_tokens = []
        for word in words:
            symbols = list(word) + ['_']
            for a, b in self.merges:
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if (i < len(symbols) - 1 and
                        symbols[i] == a and symbols[i + 1] == b):
                        new_symbols.append(a + b)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            all_tokens.extend(symbols)
        return all_tokens
```

### Using HuggingFace Tokenizers

```python
from transformers import AutoTokenizer

# BERT tokenizer (WordPiece)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, how are you?")
# tokens['input_ids'] = [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
# 101 = [CLS], 102 = [SEP]

# GPT-2 tokenizer (BPE)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer("Hello, how are you?")
# tokens['input_ids'] = [15496, 11, 703, 389, 345, 30]
```

---

## Key Takeaways

1. **Character-level** is simple but creates sequences too long for transformers
2. **BPE** is the dominant subword method — merge most frequent pairs iteratively
3. **WordPiece** (BERT) uses likelihood-based merging; adds `##` for continuations
4. **The vocabulary-sequence length tradeoff** is fundamental: larger vocab = shorter sequences but bigger embedding matrix
5. On USAAIO exams, you should be able to **trace BPE merges by hand** on a small corpus

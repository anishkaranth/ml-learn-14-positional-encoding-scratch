# Day 14 — Sinusoidal & Learned Positional Encodings from Scratch

Continues the **AI models & AI builds** phase (Day 13 was attention). Transformers have no recurrence and no convolution — without **positional encoding**, a bag of token embeddings cannot tell *where* each token sat.

Implement the Vaswani et al. **sinusoidal** PE and a **learned** PE table in plain **NumPy**, then prove PE matters on a marker-quarter classification smoke task.

## What you'll learn

- Why self-attention is **permutation-equivariant** without position signals (and why that breaks order-sensitive tasks)
- The **sinusoidal** formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`
- How a **learned** position embedding table `(max_len, d_model)` is an alternative used in BERT-style models
- How to **add** PE to token embeddings and verify geometric structure (heatmap, cosine similarity across positions)
- Empirically: content-gated readout **without PE ≈ chance**; with sinusoidal or learned PE readout, accuracy jumps to ~1.0

## Project layout

```
README.md
requirements.txt
positional_encoding.py       # sinusoidal_PE, LearnedPositionalEncoding, helpers
pe_task.py                   # marker-quarter task utils
pe_smoke_model.py            # content-gated PE training loop
pe_smoke_plots.py            # heatmap / similarity / accuracy plots
run_smoke.py                 # no PE vs sinusoidal vs learned smoke entrypoint
notebooks/positional_encoding_scratch.ipynb
results/
  RESULTS.md
  metrics.json
  JSON.shot
  pe_heatmap.png / pe_heatmap.svg
  position_similarity.png / position_similarity.svg
  accuracy_comparison.png / accuracy_comparison.svg
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_smoke.py
```

Smoke run finishes in well under ~30 seconds on CPU, prints test accuracy for all three PE conditions, and refreshes `results/`.

**Notebook walkthrough:**

```bash
jupyter notebook notebooks/positional_encoding_scratch.ipynb
```

## Task (smoke demo)

Sequences of length `T = 16`, vocab `V = 20`:

1. Every position gets a random token id in `1 … V-1`
2. Exactly one position is overwritten with the **marker** token `id = 0`
3. Label = which of **4 quarters** of the sequence contains the marker (chance = 0.25)

Model for each condition (content-gated PE readout):

1. Soft-select the marker by content: `α = softmax(X w_query)`
2. **No PE:** `h = Σ α_t X_t` (≈ marker embedding — position-blind → ~chance)
3. **With PE:** `h = Σ α_t PE_t` (≈ PE at the marker — position signal → high accuracy)
4. Linear head: `logits = h W + b` over 4 quarters

## Core API

```python
from positional_encoding import (
    sinusoidal_positional_encoding,
    LearnedPositionalEncoding,
    add_positional_encoding,
    position_cosine_similarity,
)

pe = sinusoidal_positional_encoding(seq_len=16, d_model=32)  # (T, D)
X_pe = add_positional_encoding(X, pe)                         # broadcast add

learned = LearnedPositionalEncoding(max_len=16, d_model=32)
X_pe = add_positional_encoding(X, learned.forward())
```

## Math (sinusoidal)

\[
\mathrm{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right), \quad
\mathrm{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
\]

Nearby positions have similar PE vectors (high cosine similarity); far positions diverge — visible in `position_similarity.png`.

## Dependencies

Pinned lightly in `requirements.txt`: **numpy**, **matplotlib**, **jupyter**. CPU-only.

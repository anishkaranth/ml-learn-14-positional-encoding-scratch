# Smoke results — ml-learn-14-positional-encoding-scratch

**Task:** marker-quarter classification (which of 4 sequence quarters holds token id=0).

**Seed:** `42`

## Headline metrics

| Condition | Train acc | Test acc | Marker select |
|-----------|----------:|---------:|--------------:|
| No PE | 0.3271 | 0.2344 | 0.0000 |
| Sinusoidal PE | 1.0000 | 1.0000 | 1.0000 |
| Learned PE | 1.0000 | 1.0000 | 1.0000 |
| Chance | — | 0.2500 | — |

| Metric | Value |
|--------|------:|
| Runtime (s) | 2.11 |
| Unit checks | PASS |

## Plots

Committed SVG plots (re-run `python run_smoke.py` locally for PNG equivalents):

- [`pe_heatmap.svg`](pe_heatmap.svg) — sinusoidal PE matrix
- [`position_similarity.svg`](position_similarity.svg) — cosine similarity between PE rows
- [`accuracy_comparison.svg`](accuracy_comparison.svg) — no PE vs sinusoidal vs learned

## Takeaway

Content attention can find the marker without PE, but the soft-selected content is position-blind (~chance).
Reading out PE at the selected position (sinusoidal or learned) recovers the quarter.

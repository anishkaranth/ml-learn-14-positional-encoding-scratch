"""Marker-quarter task utilities for the positional encoding smoke demo."""
from __future__ import annotations

import numpy as np

SEQ_LEN = 16
VOCAB = 20
MARKER = 0
N_CLASSES = 4


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def quarter_of(pos: np.ndarray, seq_len: int = SEQ_LEN, n_slots: int = N_CLASSES) -> np.ndarray:
    slot = seq_len // n_slots
    return np.clip(pos // slot, 0, n_slots - 1).astype(np.int64)


def make_batch(rng: np.random.Generator, n: int, seq_len: int = SEQ_LEN, vocab: int = VOCAB):
    tokens = rng.integers(1, vocab, size=(n, seq_len), dtype=np.int64)
    marker_pos = rng.integers(0, seq_len, size=n)
    tokens[np.arange(n), marker_pos] = MARKER
    labels = quarter_of(marker_pos, seq_len=seq_len)
    return tokens, labels, marker_pos


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax(logits, axis=-1)
    B = y.shape[0]
    return float(-np.mean(np.log(probs[np.arange(B), y] + 1e-12)))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=-1) == y))

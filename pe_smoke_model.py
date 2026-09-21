"""Training loop for marker-quarter PE comparison."""
from __future__ import annotations

import numpy as np

from pe_task import (
    MARKER,
    N_CLASSES,
    SEQ_LEN,
    VOCAB,
    accuracy,
    cross_entropy,
    make_batch,
    softmax,
)
from positional_encoding import LearnedPositionalEncoding, sinusoidal_positional_encoding

D_MODEL = 32
BATCH = 64
EPOCHS = 80
LR = 0.2
PE_LR = 0.1


def forward(tokens, emb, w_query, W, b, mode, sin_pe, learned):
    X = emb[tokens]
    scores = X @ w_query
    alpha = softmax(scores, axis=-1)
    if mode == "none":
        h = np.einsum("bt,btd->bd", alpha, X)
    else:
        pe = sin_pe if mode == "sinusoidal" else learned.forward(SEQ_LEN)
        h = alpha @ pe
    logits = h @ W + b
    return logits, alpha, h, X


def train_condition(mode, tokens_tr, y_tr, tokens_te, y_te, rng: np.random.Generator):
    emb = rng.normal(0, 0.4, size=(VOCAB, D_MODEL))
    emb[MARKER] = rng.normal(0, 0.4, size=(D_MODEL,)) + 1.5
    w_query = rng.normal(0, 0.3, size=(D_MODEL,))
    W = rng.normal(0, 0.3, size=(D_MODEL, N_CLASSES))
    b = np.zeros(N_CLASSES, dtype=np.float64)
    sin_pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
    learned = (
        LearnedPositionalEncoding(SEQ_LEN, D_MODEL, rng=rng, scale=0.05)
        if mode == "learned"
        else None
    )
    history = {"loss": [], "train_acc": [], "test_acc": []}
    n_train = tokens_tr.shape[0]

    for epoch in range(EPOCHS):
        perm = rng.permutation(n_train)
        losses = []
        lr = LR * (0.96 ** (epoch // 8))
        for start in range(0, n_train, BATCH):
            idx = perm[start : start + BATCH]
            tok, y = tokens_tr[idx], y_tr[idx]
            logits, alpha, h, X = forward(tok, emb, w_query, W, b, mode, sin_pe, learned)
            losses.append(cross_entropy(logits, y))
            Bsz = tok.shape[0]
            probs = softmax(logits, axis=-1)
            dlogits = probs.copy()
            dlogits[np.arange(Bsz), y] -= 1.0
            dlogits /= Bsz
            dW = h.T @ dlogits
            db = dlogits.sum(axis=0)
            dh = dlogits @ W.T
            if mode == "none":
                dX = alpha[:, :, None] * dh[:, None, :]
                dalpha = np.einsum("bd,btd->bt", dh, X)
            else:
                pe = sin_pe if mode == "sinusoidal" else learned.forward(SEQ_LEN)
                dalpha = dh @ pe.T
                dX = np.zeros_like(X)
                if mode == "learned":
                    learned.weight -= PE_LR * (0.96 ** (epoch // 8)) * (alpha.T @ dh)
            sum_da = np.sum(dalpha * alpha, axis=-1, keepdims=True)
            dscores = alpha * (dalpha - sum_da)
            dw_query = np.einsum("bt,btd->d", dscores, X)
            dX = dX + dscores[:, :, None] * w_query[None, None, :]
            demb = np.zeros_like(emb)
            np.add.at(demb, tok, dX)
            W -= lr * dW
            b -= lr * db
            w_query -= lr * dw_query
            emb -= lr * demb

        logits_tr, _, _, _ = forward(tokens_tr, emb, w_query, W, b, mode, sin_pe, learned)
        logits_te, alpha_te, _, _ = forward(tokens_te, emb, w_query, W, b, mode, sin_pe, learned)
        tr_acc, te_acc = accuracy(logits_tr, y_tr), accuracy(logits_te, y_te)
        mean_loss = float(np.mean(losses))
        history["loss"].append(mean_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  [{mode:11s}] epoch {epoch:02d}  loss={mean_loss:.4f}  train={tr_acc:.3f}  test={te_acc:.3f}")

    marker_pos_te = np.argmax(tokens_te == MARKER, axis=1)
    _, alpha_te, _, _ = forward(tokens_te, emb, w_query, W, b, mode, sin_pe, learned)
    sel_acc = float(np.mean(np.argmax(alpha_te, axis=1) == marker_pos_te))
    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "final_train_loss": history["loss"][-1],
        "marker_selection_accuracy": sel_acc,
    }

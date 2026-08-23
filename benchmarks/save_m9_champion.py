#!/usr/bin/env python3
"""Сохранение чемпиона M9: гибрид на DINOv2 ViT-B aug4 (α=0.2, topk16)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression

from zf.data import load_features
from zf.memory.proto import PrototypicalMemory

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "dinov2b_768_aug4.npz"
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]
TOPK, ALPHA = 16, 0.2


def norm_rows(M):
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    return (M - lo) / np.maximum(hi - lo, 1e-8)


def main() -> None:
    data = load_features(CACHE)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    mem = PrototypicalMemory(feature_dim=X_train.shape[1], threshold=0.0,
                             topk=TOPK, never_update=True)
    phase_acc, hist = [], {}
    t0 = time.time()
    for phase_idx, classes in enumerate(PHASES):
        mem.set_phase(phase_idx)
        for c in classes:
            mask = y_train == c
            mem.add_batch(X_train[mask], y_train[mask])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(mem.prototypes[:mem.size], mem.labels[:mem.size])
        seen = sorted(set(mem.labels[:mem.size].tolist()))
        te_mask = np.isin(y_test, seen)
        Xq, yq = X_test[te_mask], y_test[te_mask]
        S_mem, _ = mem.class_scores_batch(Xq, k=TOPK)
        P = clf.predict_proba(Xq)
        col = {c: i for i, c in enumerate(clf.classes_)}
        score = ALPHA * norm_rows(np.stack([P[:, col[c]] for c in seen], 1)) \
            + (1 - ALPHA) * norm_rows(S_mem)
        preds = np.array(seen)[score.argmax(1)]
        for c in seen:
            rows = np.where(yq == c)[0]
            hist.setdefault(c, []).append(float((preds[rows] == c).mean()))
        phase_acc.append(float((preds == yq).mean()))
        print(f"phase {phase_idx}: {phase_acc[-1]*100:.2f}% protos={mem.size}")

    forg = [max(v[:-1]) - v[-1] for v in hist.values() if len(v) >= 2]
    result = {"config": {"cache": "dinov2b_768_aug4", "topk": TOPK,
                         "alpha": ALPHA},
              "p_last": phase_acc[-1],
              "avg_forgetting": float(np.mean(forg)),
              "cls0_forgetting": max(hist[0][:-1]) - hist[0][-1],
              "phase_acc": phase_acc, "elapsed_s": time.time() - t0}
    (ROOT / "results" / "m9_champion.json").write_text(
        json.dumps(result, indent=2))

    out = ROOT / "champions" / "m9_hybrid_dinov2b_aug4_topk16_a02.npz"
    np.savez_compressed(out, prototypes=mem.prototypes[:mem.size],
                        labels=mem.labels[:mem.size],
                        counts=mem.counts[:mem.size])
    print(f"\nЧемпион: {out}")
    print(f"P@last={result['p_last']*100:.2f}% "
          f"forg={result['avg_forgetting']*100:.2f}pp "
          f"cls0={result['cls0_forgetting']*100:.2f}pp "
          f"t={result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()

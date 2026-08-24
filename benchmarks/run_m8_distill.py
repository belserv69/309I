#!/usr/bin/env python3
"""M8: дистилляция гибрида — LwF-lite против дрейфа probe (цель: cls0 ↓).

Механика портирована из CORAL run_trackb_baseline.train_probe:
при переобучении probe на фазе p добавляется KD-член — KL между логитами
старого и нового probe по колонкам СТАРЫХ классов (температура T=2).
Старые классы «якорятся» к своему прежнему поведению → cls0 не дрейфует.

Сетка: lam ∈ {0, 0.5, 1.0, 2.0}, α=0.2, topk16, кэш dinov2 aug4.
Цель: P@last ≥ 90% при cls0-forgetting ≤ 5pp (у M6c без дистилляции: 91.03%/10pp).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from zf.data import load_features

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / "data" / "dinov2_384_aug4.npz")
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]
TOPK, ALPHA, TEMP = 16, 0.2, 2.0
LR, EPOCHS, BATCH = 1e-3, 10, 256


def norm_rows(M: np.ndarray) -> np.ndarray:
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    return (M - lo) / np.maximum(hi - lo, 1e-8)


def train_probe(probe: nn.Linear, feats: np.ndarray, labels: np.ndarray,
                old_probe: nn.Linear | None, old_classes: list[int],
                lam: float) -> float:
    """CE + LwF-lite KD по старым колонкам (порт CORAL)."""
    opt = torch.optim.Adam(probe.parameters(), lr=LR)
    X = torch.from_numpy(feats)
    y = torch.from_numpy(labels.astype(np.int64))
    n = len(X)
    old_cols = None
    if lam > 0 and old_probe is not None and old_classes:
        idx = [c for c in old_classes if c < probe.weight.shape[0]]
        if idx:
            old_cols = torch.tensor(idx, dtype=torch.long)

    last = 0.0
    g = torch.Generator().manual_seed(42)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            logits = probe(X[b])
            loss = F.cross_entropy(logits, y[b])
            if old_cols is not None:
                with torch.no_grad():
                    old_lg = old_probe(X[b])[:, old_cols]
                kd = F.kl_div(
                    F.log_softmax(logits[:, old_cols] / TEMP, dim=1),
                    F.softmax(old_lg / TEMP, dim=1),
                    reduction="batchmean") * (TEMP ** 2)
                loss = loss + lam * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        last = total / max((n + BATCH - 1) // BATCH, 1)
    return last


def run(lam: float, data: dict) -> dict:
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    from zf.memory.proto import PrototypicalMemory

    mem = PrototypicalMemory(feature_dim=X_train.shape[1], threshold=0.0,
                             topk=TOPK, never_update=True)
    dim = X_train.shape[1]
    n_cls_total = int(y_train.max()) + 1
    probe = nn.Linear(dim, n_cls_total)
    phase_acc, hist = [], {}
    t0 = time.time()

    for phase_idx, classes in enumerate(PHASES):
        mem.set_phase(phase_idx)
        for c in classes:
            mask = y_train == c
            mem.add_batch(X_train[mask], y_train[mask])

        seen_before = sorted(set(mem.labels[:mem.size].tolist()) - set(classes))
        old_snapshot = None
        if lam > 0 and seen_before:
            old_snapshot = nn.Linear(dim, n_cls_total)
            old_snapshot.load_state_dict(probe.state_dict())
            old_snapshot.eval()

        feats = mem.prototypes[:mem.size]
        labels = mem.labels[:mem.size]
        ce = train_probe(probe, feats, labels, old_snapshot,
                         seen_before, lam)
        probe.eval()

        seen = sorted(set(labels.tolist()))
        te_mask = np.isin(y_test, seen)
        Xq, yq = X_test[te_mask], y_test[te_mask]

        S_mem, mem_classes = mem.class_scores_batch(Xq, k=TOPK)
        assert mem_classes == seen, "порядок классов памяти разошёлся"
        with torch.no_grad():
            logits = probe(torch.from_numpy(Xq)).numpy()
        # выравнивание колонок probe (100 выходов) под классы памяти
        e = np.exp(logits - logits.max(1, keepdims=True))
        P_full = e / e.sum(1, keepdims=True)
        P = np.stack([P_full[:, c] for c in seen], axis=1)
        score = ALPHA * norm_rows(P) + (1 - ALPHA) * norm_rows(S_mem)
        preds = np.array(seen)[score.argmax(1)]

        for c in seen:
            rows = np.where(yq == c)[0]
            hist.setdefault(c, []).append(float((preds[rows] == c).mean()))
        acc = float((preds == yq).mean())
        phase_acc.append(acc)
        print(f"  lam={lam} phase{phase_idx}: acc={acc*100:.2f}% ce={ce:.3f}",
              flush=True)

    forg = [max(v[:-1]) - v[-1] for v in hist.values() if len(v) >= 2]
    return {
        "lam": lam, "p_last": phase_acc[-1],
        "avg_forgetting": float(np.mean(forg)) if forg else 0.0,
        "cls0_forgetting": max(hist[0][:-1]) - hist[0][-1],
        "elapsed_s": time.time() - t0,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    data = load_features(CACHE)
    print(f"Данные: {data['X_train'].shape}")
    out = []
    for lam in (0.0, 0.5, 1.0, 2.0):
        print(f"--- lam={lam}")
        r = run(lam, data)
        print(f"[lam={lam}] P@last={r['p_last']*100:.2f}% "
              f"forg={r['avg_forgetting']*100:.2f}pp "
              f"cls0={r['cls0_forgetting']*100:.2f}pp t={r['elapsed_s']:.0f}s",
              flush=True)
        out.append(r)
    tag = CACHE.stem
    (RESULTS / f"m8_distill_grid_{tag}.json").write_text(
        json.dumps(out, indent=2))

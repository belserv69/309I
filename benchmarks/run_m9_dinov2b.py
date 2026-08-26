#!/usr/bin/env python3
"""M9: ViT-B/14 (768d) — проверка линии «энкодер важнее тюнинга».

Три конфигурации-чемпиона на новых признаках:
1. Чистая память (M5-стиль): сетка topk {8,16}, sen 0.
2. Гибрид (M6c-стиль): sklearn-probe от памяти, α=0.2, topk16.
3. Дистилляция (M8-стиль): LwF lam=1.0, α=0.2.

Кэши: data/dinov2b_768.npz (base), data/dinov2b_768_aug4.npz (аугментация 4×).
"""
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
from zf.pipeline import run_continual

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE_BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "data" / "dinov2b_768.npz"
CACHE_AUG = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    ROOT / "data" / "dinov2b_768_aug4.npz"
# опционально: кэш с альтернативным X_test (напр. TTA5), train не меняется
TEST_OVERRIDE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]


def load_with_override(path: Path) -> dict:
    d = load_features(path)
    if TEST_OVERRIDE is not None:
        t = np.load(TEST_OVERRIDE)
        assert np.array_equal(t["y_test"], d["y_test"])
        d["X_test"] = t["X_test"]
    return d


def norm_rows(M: np.ndarray) -> np.ndarray:
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    return (M - lo) / np.maximum(hi - lo, 1e-8)


def pure_memory(data: dict, topk: int) -> dict:
    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=topk, never_update=True)
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], PHASES, topk=topk)
    print(f"[pure k{topk}] P@last={stats['p_last']*100:.2f}% "
          f"forg={stats['avg_forgetting']*100:.2f}pp "
          f"cls0={stats['cls0_forgetting']*100:.2f}pp t={stats['elapsed_s']:.1f}s",
          flush=True)
    return {"config": f"pure_k{topk}", "p_last": stats["p_last"],
            "avg_forgetting": stats["avg_forgetting"],
            "cls0_forgetting": stats["cls0_forgetting"]}


def hybrid(data: dict, topk: int = 16, alpha: float = 0.2) -> dict:
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    mem = PrototypicalMemory(feature_dim=X_train.shape[1], threshold=0.0,
                             topk=topk, never_update=True)
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
        S_mem, _ = mem.class_scores_batch(Xq, k=topk)
        P = clf.predict_proba(Xq)
        col = {c: i for i, c in enumerate(clf.classes_)}
        score = alpha * norm_rows(np.stack([P[:, col[c]] for c in seen], 1)) \
            + (1 - alpha) * norm_rows(S_mem)
        preds = np.array(seen)[score.argmax(1)]
        for c in seen:
            rows = np.where(yq == c)[0]
            hist.setdefault(c, []).append(float((preds[rows] == c).mean()))
        phase_acc.append(float((preds == yq).mean()))
    forg = [max(v[:-1]) - v[-1] for v in hist.values() if len(v) >= 2]
    out = {"config": f"hybrid_a{alpha}_k{topk}", "p_last": phase_acc[-1],
           "avg_forgetting": float(np.mean(forg)),
           "cls0_forgetting": max(hist[0][:-1]) - hist[0][-1],
           "elapsed_s": time.time() - t0}
    print(f"[{out['config']}] P@last={out['p_last']*100:.2f}% "
          f"forg={out['avg_forgetting']*100:.2f}pp "
          f"cls0={out['cls0_forgetting']*100:.2f}pp t={out['elapsed_s']:.0f}s",
          flush=True)
    return out


if __name__ == "__main__":
    torch_seed_note = "детерминизм: sklearn LBFGS"
    results = []

    base = load_with_override(CACHE_BASE)
    aug = load_with_override(CACHE_AUG)
    print(f"base {base['X_train'].shape} | aug4 {aug['X_train'].shape}"
          + (f" | test из {TEST_OVERRIDE.name}" if TEST_OVERRIDE else ""))

    for k in (8, 16):
        results.append(pure_memory(base, k))
    results.append(hybrid(aug))
    stem = "m9_dinov2b" if CACHE_BASE.stem.startswith("dinov2b") \
        else "m9_" + CACHE_BASE.stem.replace("_aug4", "")
    if TEST_OVERRIDE:
        stem += "_tta5"
    (RESULTS / f"{stem}.json").write_text(json.dumps(results, indent=2))

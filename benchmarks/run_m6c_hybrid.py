#!/usr/bin/env python3
"""M6c: гибрид «память прототипов + инкрементальный probe» — пивот CONCEPT §10.

Идея: probe обучается ТОЛЬКО на прототипах, накопленных памятью к текущей
фазе (система не выходит за пределы собственного содержимого — честный CL
в терминах ZF-Proto: память и есть хранилище данных).
Скор класса = α·P_probe + (1−α)·score_proto (нормированных).

Ожидание: probe ≈ oracle 90.13%; ошибки памяти (zero forgetting) и probe
комплементарны → фьюжн ≥90%.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features
from zf.memory.proto import PrototypicalMemory

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path(__file__).resolve().parent.parent / "data" / "dinov2_384_aug4.npz"
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_hybrid(data: dict, topk: int, alphas: list[float],
               label_prefix: str = "m6c") -> dict:
    from sklearn.linear_model import LogisticRegression

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    dim = X_train.shape[1]

    mem = PrototypicalMemory(feature_dim=dim, threshold=0.0, topk=topk,
                             never_update=True)
    histories = {a: {"per_class_hist": {}, "phase_acc": []} for a in alphas}
    t0 = time.time()

    for phase_idx, classes in enumerate(PHASES):
        mem.set_phase(phase_idx)
        for c in classes:
            mask = y_train == c
            mem.add_batch(X_train[mask], y_train[mask])

        # probe на содержимом памяти (= все семплы изученных классов)
        feats, labels = mem.prototypes[:mem.size], mem.labels[:mem.size]
        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(feats, labels)

        seen = sorted(set(labels.tolist()))
        te_mask = np.isin(y_test, seen)
        Xq, yq = X_test[te_mask], y_test[te_mask]

        # полная матрица скоров памяти: (N, n_seen)
        S_mem, mem_classes = mem.class_scores_batch(Xq, k=topk)
        assert mem_classes == seen, "порядок классов памяти разошёлся"
        P_probe = clf.predict_proba(Xq)
        # выравнивание колонок probe под порядок классов памяти
        col = {c: i for i, c in enumerate(clf.classes_)}
        P_aligned = np.stack([P_probe[:, col[c]] for c in seen], axis=1)

        # нормировка скоростей в [0,1] по строке для сопоставимой шкалы
        def _norm(M: np.ndarray) -> np.ndarray:
            lo = M.min(axis=1, keepdims=True)
            hi = M.max(axis=1, keepdims=True)
            return (M - lo) / np.maximum(hi - lo, 1e-8)

        S_n, P_n = _norm(S_mem), _norm(P_aligned)

        for a in alphas:
            score_matrix = a * P_n + (1 - a) * S_n
            preds = np.array(seen)[score_matrix.argmax(axis=1)]
            pc_hist = histories[a]["per_class_hist"]
            for c in seen:
                rows = np.where(yq == c)[0]
                acc = float((preds[rows] == c).mean())
                pc_hist.setdefault(c, []).append(acc)
            histories[a]["phase_acc"].append(float((preds == yq).mean()))

    out = {}
    for a in alphas:
        hist = histories[a]["per_class_hist"]
        forg = [max(v[:-1]) - v[-1] for v in hist.values() if len(v) >= 2]
        res = {
            "alpha": a,
            "p_last": histories[a]["phase_acc"][-1],
            "avg_forgetting": float(np.mean(forg)) if forg else 0.0,
            "phase_acc": histories[a]["phase_acc"],
        }
        out[f"alpha{a}"] = res
        print(f"[{label_prefix} a={a}] P@last={res['p_last']*100:.2f}% "
              f"forg={res['avg_forgetting']*100:.2f}pp", flush=True)
    print(f"[{label_prefix}] полное время: {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    data = load_features(CACHE)
    print(f"Данные: train {data['X_train'].shape}")
    results = run_hybrid(data, topk=16, alphas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    (RESULTS / "m6c_hybrid_grid.json").write_text(json.dumps(results, indent=2))

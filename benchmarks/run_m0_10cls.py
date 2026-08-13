#!/usr/bin/env python3
"""M0: smoke-тест порта — RN50 2048d, 10 классов, протокол CORAL.

Цель (PLAN.md M0): P@10cls ≥ 60%, без падений и NaN.
Ориентиры: CORAL proto-only = 64.1% (RN18) / 68.0% (RN50) при 100 классах.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def run(n_classes: int = 10, topk: int = 8, threshold: float = 0.0,
        exemplar: int = 0, label: str = "m0") -> dict:
    data = subset_classes(load_features(), list(range(n_classes)))
    print(f"Данные: train {data['X_train'].shape}, test {data['X_test'].shape}, "
          f"классов {n_classes}")

    assert np.isfinite(data["X_train"]).all(), "NaN в train-признаках!"
    assert np.isfinite(data["X_test"]).all(), "NaN в test-признаках!"

    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=threshold,
        topk=topk,
        never_update=True,
        exemplar_per_class=exemplar,
    )
    phases = [[c] for c in range(n_classes)]
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], phases, topk=topk)

    print(f"\n{'phase':>5} {'cls':>4} {'acc':>7} {'protos':>7}")
    for p in stats["phases"]:
        print(f"{p['phase']:>5} {p['n_classes']:>4} {p['accuracy']*100:>6.2f}% "
              f"{p['protos']:>7}")
    print(f"\nP@last = {stats['p_last']*100:.2f}% | avg forgetting = "
          f"{stats['avg_forgetting']*100:.2f}pp | protos = {stats['total_protos']} "
          f"| время = {stats['elapsed_s']:.1f}s")

    out = {"label": label, "n_classes": n_classes, "topk": topk,
           "threshold": threshold, "exemplar_per_class": exemplar,
           "p_last": stats["p_last"], "avg_forgetting": stats["avg_forgetting"],
           "cls0_forgetting": stats["cls0_forgetting"],
           "total_protos": stats["total_protos"], "elapsed_s": stats["elapsed_s"],
           "phases": [{"phase": p["phase"], "accuracy": p["accuracy"],
                       "protos": p["protos"]} for p in stats["phases"]]}
    path = RESULTS / f"{label}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Сохранено: {path}")
    return out


if __name__ == "__main__":
    run(label="m0_10cls_topk8")

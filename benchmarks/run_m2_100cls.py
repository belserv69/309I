#!/usr/bin/env python3
"""M2: 100 классов — главный эксперимент.

Цель (PLAN.md M2): P@last ≥ 80% при forgetting ≈ 0.
Ориентиры: CORAL hybrid 85.4% (но с forgetting cls0 −13.3pp),
CORAL proto-only 68.0%, oracle 86.2%.

Шаги:
1. Чемпион M1 (topk4, sen0.08) на 100 классах, 10 фаз × 10 классов.
2. Oracle: линейный probe (sklearn LogisticRegression) на всех классах сразу —
   верхняя граница линейной сепарабельности признаков.
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
N_CLASSES = 100


def run_proto(data: dict, topk: int = 4, seniority: float = 0.08,
              label: str = "m2_100cls_champion") -> dict:
    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=0.0,
        topk=topk,
        never_update=True,
        seniority_bonus=seniority,
    )
    # 10 фаз × 10 классов (протокол CORAL)
    phases = [[p * 10 + i for i in range(10)] for p in range(10)]
    t0 = time.time()
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], phases, topk=topk)
    print(f"\n{'phase':>5} {'cls':>4} {'acc':>7} {'protos':>7}")
    for p in stats["phases"]:
        print(f"{p['phase']:>5} {p['n_classes']:>4} {p['accuracy']*100:>6.2f}% "
              f"{p['protos']:>7}")
    print(f"\n[{label}] P@last = {stats['p_last']*100:.2f}% | avg forgetting = "
          f"{stats['avg_forgetting']*100:.2f}pp | cls0 forg = "
          f"{stats['cls0_forgetting']*100:.2f}pp | protos = {stats['total_protos']} "
          f"| t = {time.time()-t0:.1f}s")

    out = {"label": label, "n_classes": N_CLASSES, "topk": topk,
           "seniority": seniority, "p_last": stats["p_last"],
           "avg_forgetting": stats["avg_forgetting"],
           "cls0_forgetting": stats["cls0_forgetting"],
           "per_class_forgetting": stats["per_class_forgetting"],
           "total_protos": stats["total_protos"],
           "phases": [{"phase": p["phase"], "accuracy": p["accuracy"],
                       "protos": p["protos"]} for p in stats["phases"]]}
    (RESULTS / f"{label}.json").write_text(json.dumps(out, indent=2))
    return out


def run_oracle(data: dict) -> dict:
    """Линейный probe на всех классах сразу — верхняя граница."""
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(data["X_train"], data["y_train"])
    acc = float(clf.score(data["X_test"], data["y_test"]))
    print(f"[oracle] linear probe (все {N_CLASSES} классов сразу) = "
          f"{acc*100:.2f}% | t = {time.time()-t0:.1f}s")
    out = {"label": "m2_oracle_linear_probe", "accuracy": acc}
    (RESULTS / "m2_oracle_linear_probe.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    data = subset_classes(load_features(), list(range(N_CLASSES)))
    print(f"Данные: train {data['X_train'].shape}, test {data['X_test'].shape}, "
          f"классов {N_CLASSES}")
    run_proto(data)
    run_oracle(data)

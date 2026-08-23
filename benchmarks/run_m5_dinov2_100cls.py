#!/usr/bin/env python3
"""M5: чемпионская конфигурация ZF-Proto на DINOv2-признаках (384d CLS).

Гипотеза: признаки DINOv2 линейно разделимее RN50 (CORAL probe 90% vs 85%),
значит прототипная память тоже должна подняться выше 80.03%.

Ориентиры CORAL на тех же признаках: champion 90.0% P@last (probe+max-vote+
replay buf20), forgetting cls0 6.3pp.

Шаги:
1. Мини-сетка top-k × seniority на 100 классах (геометрия 384d отличается).
2. Чемпион конфигурации → полный прогон.
3. Oracle: линейный probe — верхняя граница.
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
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]
CACHE = Path(__file__).resolve().parent.parent / "data" / "dinov2_384.npz"


def run_proto(data: dict, topk: int, seniority: float, label: str,
              save: bool = True) -> dict:
    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=0.0,
        topk=topk,
        never_update=True,
        seniority_bonus=seniority,
    )
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], PHASES, topk=topk)
    print(f"[{label}] P@last={stats['p_last']*100:.2f}% "
          f"forg={stats['avg_forgetting']*100:.2f}pp "
          f"cls0={stats['cls0_forgetting']*100:.2f}pp "
          f"protos={stats['total_protos']} t={stats['elapsed_s']:.1f}s")
    out = {"label": label, "topk": topk, "seniority": seniority,
           "p_last": stats["p_last"], "avg_forgetting": stats["avg_forgetting"],
           "cls0_forgetting": stats["cls0_forgetting"],
           "total_protos": stats["total_protos"],
           "phases": [{"phase": p["phase"], "accuracy": p["accuracy"]}
                      for p in stats["phases"]]}
    if save:
        (RESULTS / f"{label}.json").write_text(json.dumps(out, indent=2))
    return out


def run_oracle(data: dict) -> float:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(data["X_train"], data["y_train"])
    acc = float(clf.score(data["X_test"], data["y_test"]))
    print(f"[oracle] linear probe = {acc*100:.2f}%")
    return acc


if __name__ == "__main__":
    data = subset_classes(load_features(CACHE), list(range(N_CLASSES)))
    print(f"Данные: train {data['X_train'].shape}, test {data['X_test'].shape}")

    # Мини-сетка: геометрия 384d может сместить оптимум top-k
    grid = []
    for topk in (1, 2, 4, 8):
        grid.append(run_proto(data, topk, 0.0, f"m5_topk{topk}", save=False))
    best = max(grid, key=lambda r: r["p_last"])
    print(f"\nЛучший по точности: topk{best['topk']} "
          f"({best['p_last']*100:.2f}%)")

    # Seniority на лучшем top-k: trade-off точность/забывание
    sen_grid = [run_proto(data, best["topk"], s, f"m5_topk{best['topk']}_sen{s}",
                          save=False)
                for s in (0.04, 0.08)]

    # Полный прогон чемпиона (конфигурация как у RN50-чемпиона для сравнения)
    champ = run_proto(data, 4, 0.08, "m5_dinov2_champion")
    oracle = run_oracle(data)
    champ["oracle"] = oracle
    (RESULTS / "m5_dinov2_champion.json").write_text(json.dumps(champ, indent=2))

#!/usr/bin/env python3
"""M6: DINOv2 + аугментация 4× — попытка превзойти CORAL champion (90.0%).

Зазор M5 до оракула всего 1.7pp (88.40% при oracle 90.13%). Рычаги:
1. Аугментация 4× → 20000 прототипов вместо 5000 (плотнее покрытие
   confusion-пар; на SOMA это был главный рычаг).
2. Взвешенный top-k ("sim") против простого среднего ("mean").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path(__file__).resolve().parent.parent / "data" / "dinov2_384_aug4.npz"
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]


def run_proto(data: dict, topk: int, seniority: float, weighting: str,
              label: str, save: bool = True) -> dict:
    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=0.0,
        topk=topk,
        never_update=True,
        seniority_bonus=seniority,
        topk_weighting=weighting,
    )
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], PHASES, topk=topk)
    print(f"[{label}] P@last={stats['p_last']*100:.2f}% "
          f"forg={stats['avg_forgetting']*100:.2f}pp "
          f"cls0={stats['cls0_forgetting']*100:.2f}pp "
          f"protos={stats['total_protos']} t={stats['elapsed_s']:.1f}s",
          flush=True)
    out = {"label": label, "topk": topk, "seniority": seniority,
           "topk_weighting": weighting,
           "p_last": stats["p_last"], "avg_forgetting": stats["avg_forgetting"],
           "cls0_forgetting": stats["cls0_forgetting"],
           "total_protos": stats["total_protos"],
           "phases": [{"phase": p["phase"], "accuracy": p["accuracy"]}
                      for p in stats["phases"]]}
    if save:
        (RESULTS / f"{label}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    data = subset_classes(load_features(CACHE), list(range(100)))
    print(f"Данные: train {data['X_train'].shape} (аугментация 4×)")

    grid = []
    for weighting in ("mean", "sim"):
        for topk in (8, 16):
            for sen in (0.0, 0.08):
                grid.append(run_proto(data, topk, sen, weighting,
                                      f"m6_{weighting}_k{topk}_s{sen}",
                                      save=False))

    grid.sort(key=lambda r: r["p_last"], reverse=True)
    print("\n=== Топ-5 ===")
    for r in grid[:5]:
        print(f"{r['label']}: P@last={r['p_last']*100:.2f}% "
              f"forg={r['avg_forgetting']*100:.2f}pp")
    (RESULTS / "m6_grid.json").write_text(json.dumps(grid, indent=2))

#!/usr/bin/env python3
"""M1: 25 классов + абляции top-k / threshold / exemplar.

Цель (PLAN.md M1): P@25cls ≥ 80%.
Ориентир: CORAL RN18 probe — 91.4% P25 / 88.4% P@last.
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
RESULTS.mkdir(exist_ok=True)

N_CLASSES = 25


def run_one(label: str, topk: int = 8, threshold: float = 0.0,
            exemplar: int = 0, data: dict | None = None) -> dict:
    if data is None:
        data = subset_classes(load_features(), list(range(N_CLASSES)))
    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=threshold,
        topk=topk,
        never_update=True,
        exemplar_per_class=exemplar,
    )
    phases = [[c] for c in range(N_CLASSES)]
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], phases, topk=topk)
    row = {
        "label": label, "topk": topk, "threshold": threshold,
        "exemplar_per_class": exemplar,
        "p_last": round(stats["p_last"], 4),
        "avg_forgetting": round(stats["avg_forgetting"], 4),
        "cls0_forgetting": round(stats["cls0_forgetting"], 4),
        "total_protos": stats["total_protos"],
        "elapsed_s": round(stats["elapsed_s"], 2),
    }
    print(f"{label:28s} P@last={row['p_last']*100:6.2f}%  forg={row['avg_forgetting']*100:5.2f}pp  "
          f"cls0={row['cls0_forgetting']*100:5.2f}pp  protos={row['total_protos']}  t={row['elapsed_s']}s")
    return row


if __name__ == "__main__":
    data = subset_classes(load_features(), list(range(N_CLASSES)))
    print(f"Данные: train {data['X_train'].shape}, test {data['X_test'].shape}, "
          f"классов {N_CLASSES}\n")

    rows = []
    # Базовая конфигурация (победитель M0)
    rows.append(run_one("base_topk8_t0", topk=8, threshold=0.0, data=data))
    # Абляция top-k
    rows.append(run_one("topk4", topk=4, threshold=0.0, data=data))
    rows.append(run_one("topk16", topk=16, threshold=0.0, data=data))
    # Абляция threshold
    rows.append(run_one("t0.08", topk=8, threshold=0.08, data=data))
    rows.append(run_one("t0.20", topk=8, threshold=0.20, data=data))
    # Абляция exemplar ring
    rows.append(run_one("exemplar20", topk=8, threshold=0.0, exemplar=20, data=data))
    # Лучшая комбинация — определится после просмотра таблицы

    out = {"milestone": "M1", "n_classes": N_CLASSES, "rows": rows}
    path = RESULTS / "m1_25cls_ablations.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nСохранено: {path}")

    best = max(rows, key=lambda r: r["p_last"])
    print(f"\nЛучший по P@last: {best['label']} → {best['p_last']*100:.2f}%")

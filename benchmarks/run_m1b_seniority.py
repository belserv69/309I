#!/usr/bin/env python3
"""M1b: seniority bonus — снижение забывания без разморозки прототипов.

При never_update прототипы не меняются, но новые классы отбирают предсказания
у старых. SOMA-чемпион C205 решал это seniority_bonus=0.08. Проверяем здесь.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_CLASSES = 25


def run_one(label: str, topk: int, seniority: float, data: dict) -> dict:
    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=0.0,
        topk=topk,
        never_update=True,
        seniority_bonus=seniority,
    )
    phases = [[c] for c in range(N_CLASSES)]
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], phases, topk=topk)
    row = {
        "label": label, "topk": topk, "seniority": seniority,
        "p_last": round(stats["p_last"], 4),
        "avg_forgetting": round(stats["avg_forgetting"], 4),
        "cls0_forgetting": round(stats["cls0_forgetting"], 4),
        "elapsed_s": round(stats["elapsed_s"], 2),
    }
    print(f"{label:18s} P@last={row['p_last']*100:6.2f}%  forg={row['avg_forgetting']*100:5.2f}pp  "
          f"cls0={row['cls0_forgetting']*100:5.2f}pp  t={row['elapsed_s']}s")
    return row


if __name__ == "__main__":
    data = subset_classes(load_features(), list(range(N_CLASSES)))
    print(f"Seniority sweep, 25 классов, topk=4 (победитель M1)\n")

    rows = []
    for sb in [0.0, 0.02, 0.05, 0.08, 0.12]:
        rows.append(run_one(f"sen{sb}", topk=4, seniority=sb, data=data))

    out = {"milestone": "M1b", "n_classes": N_CLASSES, "rows": rows}
    (RESULTS / "m1b_seniority.json").write_text(json.dumps(out, indent=2))
    print(f"\nСохранено: results/m1b_seniority.json")

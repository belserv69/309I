#!/usr/bin/env python3
"""M2b: seniority sweep на 100 классах — торговля точность/забывание."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_CLASSES = 100


def run_one(topk: int, seniority: float, data: dict) -> dict:
    mem = PrototypicalMemory(feature_dim=2048, threshold=0.0, topk=topk,
                             never_update=True, seniority_bonus=seniority)
    phases = [[p * 10 + i for i in range(10)] for p in range(10)]
    s = run_continual(mem, data["X_train"], data["y_train"],
                      data["X_test"], data["y_test"], phases, topk=topk)
    row = {"topk": topk, "seniority": seniority,
           "p_last": round(s["p_last"], 4),
           "avg_forgetting": round(s["avg_forgetting"], 4),
           "cls0_forgetting": round(s["cls0_forgetting"], 4)}
    print(f"topk={topk} sen={seniority:<5} P@last={row['p_last']*100:6.2f}%  "
          f"forg={row['avg_forgetting']*100:5.2f}pp  cls0={row['cls0_forgetting']*100:5.2f}pp")
    return row


if __name__ == "__main__":
    data = subset_classes(load_features(), list(range(N_CLASSES)))
    rows = []
    for sen in [0.0, 0.08, 0.12, 0.15, 0.20]:
        rows.append(run_one(4, sen, data))
    # топ-k тоже может влиять на забывание
    rows.append(run_one(2, 0.15, data))

    (RESULTS / "m2b_seniority_sweep.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("\nСохранено: results/m2b_seniority_sweep.json")

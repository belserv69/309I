#!/usr/bin/env python3
"""M1 финал: высокие threshold + фиксация чемпиона M1."""
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
N_CLASSES = 25


def run_one(label, topk, threshold, seniority, data):
    mem = PrototypicalMemory(feature_dim=2048, threshold=threshold, topk=topk,
                             never_update=True, seniority_bonus=seniority)
    phases = [[c] for c in range(N_CLASSES)]
    s = run_continual(mem, data["X_train"], data["y_train"],
                      data["X_test"], data["y_test"], phases, topk=topk)
    row = {"label": label, "topk": topk, "threshold": threshold,
           "seniority": seniority, "p_last": round(s["p_last"], 4),
           "avg_forgetting": round(s["avg_forgetting"], 4),
           "cls0_forgetting": round(s["cls0_forgetting"], 4)}
    print(f"{label:22s} P@last={row['p_last']*100:6.2f}%  forg={row['avg_forgetting']*100:5.2f}pp")
    return row


data = subset_classes(load_features(), list(range(N_CLASSES)))
rows = []
# победитель M1b
rows.append(run_one("topk4_sen08", 4, 0.0, 0.08, data))
# threshold-свип на высоких значениях
for t in [0.3, 0.4, 0.5]:
    rows.append(run_one(f"topk4_t{t}_sen08", 4, t, 0.08, data))

best = max(rows, key=lambda r: r["p_last"])
print(f"\nЧемпион M1: {best['label']} → {best['p_last']*100:.2f}%, "
      f"forgetting {best['avg_forgetting']*100:.2f}pp")

# Сохранение чемпиона: память как npz (pkl в gitignore) + конфиг в json
mem = PrototypicalMemory(feature_dim=2048, threshold=best["threshold"],
                         topk=best["topk"], never_update=True,
                         seniority_bonus=best["seniority"])
for c in range(N_CLASSES):
    mem.set_phase(c)
    m = data["y_train"] == c
    mem.add_batch(data["X_train"][m], data["y_train"][m])

np.savez_compressed(
    "champions/m1_topk4_sen08.npz",
    prototypes=mem.prototypes[: mem.size],
    labels=mem.labels[: mem.size],
)
(RESULTS / "m1_champion.json").write_text(json.dumps(best, indent=2))
print(f"Сохранено: champions/m1_topk4_sen08.npz ({mem.size} прототипов), "
      f"results/m1_champion.json")

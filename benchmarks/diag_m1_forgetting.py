#!/usr/bin/env python3
"""M1c: диагностика забывания — какие классы забываются и кем перехватываются."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory

N_CLASSES = 25
data = subset_classes(load_features(), list(range(N_CLASSES)))

mem = PrototypicalMemory(feature_dim=2048, threshold=0.0, topk=4, never_update=True)
for c in range(N_CLASSES):
    mem.set_phase(c)
    m = data["y_train"] == c
    mem.add_batch(data["X_train"][m], data["y_train"][m])

preds, confs = mem.query_batch(data["X_test"])
y = data["y_test"]

print(f"P@last = {(preds == y).mean()*100:.2f}%\n")
print(f"{'cls':>4} {'acc':>7} {'top-ошибки':>40}")
for c in range(N_CLASSES):
    mask = y == c
    wrong = preds[mask] != c
    acc = 1 - wrong.mean()
    if wrong.any():
        wrong_preds = preds[mask][wrong]
        vals, cnts = np.unique(wrong_preds, return_counts=True)
        conf_str = ", ".join(f"{v}×{n}" for v, n in sorted(zip(vals, cnts), key=lambda x: -x[1])[:3])
    else:
        conf_str = "-"
    print(f"{c:>4} {acc*100:>6.2f}%   {conf_str:>40}")

#!/usr/bin/env python3
"""Сохранение чемпиона M5 (DINOv2 384d) в champions/ по правилам CONCEPT §9."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "dinov2_384.npz"

# Чемпион сетки m5: лучший P@last = topk8 без seniority
TOPK, SEN = 8, 0.0


def main() -> None:
    data = subset_classes(load_features(CACHE), list(range(100)))
    phases = [[p * 10 + i for i in range(10)] for p in range(10)]

    mem = PrototypicalMemory(
        feature_dim=data["X_train"].shape[1],
        threshold=0.0,
        topk=TOPK,
        never_update=True,
        seniority_bonus=SEN,
    )
    t0 = time.time()
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], phases, topk=TOPK)

    out = ROOT / "champions" / "m5_dinov2_topk8_sen0_100cls.npz"
    np.savez_compressed(
        out,
        prototypes=mem.prototypes[:mem.size],
        labels=mem.labels[:mem.size],
        counts=mem.counts[:mem.size],
    )
    print(f"Сохранено: {out} ({mem.size} прототипов)")
    print(f"P@last={stats['p_last']*100:.2f}% forg={stats['avg_forgetting']*100:.2f}pp "
          f"t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

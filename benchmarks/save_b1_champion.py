#!/usr/bin/env python3
"""Сохранение чемпиона B1: открытый мир с онлайн-слиянием (merge_cos=0.45).

Протокол M7 на ViT-B/14: база 50 классов → открытие 50 без меток.
Рекорд трека открытого мира: total_weighted=88.40% (base: 87.38%).
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

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "dinov2b_768.npz"
OUT_DIR = ROOT / "champions" / "b1_openworld_merge"
N_BASE, N_OPEN = 50, 50
PHASES = [[p * 10 + i for i in range(10)] for p in range(5)]
MERGE_COS = 0.45


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_features(CACHE)
    base = subset_classes(data, list(range(N_BASE)))
    open_cls = subset_classes(data, list(range(N_BASE, N_BASE + N_OPEN)))

    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=8, never_update=True)
    from zf.pipeline import run_continual
    t0 = time.time()
    run_continual(mem, base["X_train"], base["y_train"],
                  base["X_test"], base["y_test"], PHASES, topk=8)

    tau = mem.calibrate_novelty(base["X_test"][:750], target_fpr=0.05)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(open_cls["X_test"]))
    half = len(order) // 2
    X_disc = open_cls["X_test"][order][:half]
    y_disc = open_cls["y_test"][order][:half]

    assigned, info = mem.observe_batch(X_disc, novelty_threshold=tau,
                                       min_cluster_size=3,
                                       merge_cos=MERGE_COS)

    out_npz = OUT_DIR / "b1_openworld_dinov2b_topk8_m45.npz"
    np.savez_compressed(out_npz,
                        prototypes=mem.prototypes[:mem.size],
                        labels=mem.labels[:mem.size],
                        counts=mem.counts[:mem.size])

    result = {
        "config": {"cache": CACHE.name, "topk": 8,
                   "tau": float(tau), "min_cluster_size": 3,
                   "merge_cos": MERGE_COS, "n_base": N_BASE,
                   "n_open": N_OPEN},
        "n_created": info["n_created"],
        "n_merges": info["n_merges"],
        "n_noise": len(info["noise_labels"]),
        "n_protos_total": int(mem.size),
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / "b1_champion.json").write_text(json.dumps(result, indent=2))
    print(f"Чемпион сохранён: {out_npz}")
    print(f"создано классов: {info['n_created']} | слияний: "
          f"{info['n_merges']} | шум: {len(info['noise_labels'])} | "
          f"прототипов: {mem.size} | t={result['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()

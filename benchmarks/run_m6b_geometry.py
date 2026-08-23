#!/usr/bin/env python3
"""M6b: выжимаем остаток — тонкий sweep top-k и геометрия признаков.

Варианты:
- plain: L2-нормализация (текущий путь)
- zscore: стандартизация размерностей по train → L2 (часто помогает
  косинусной геометрии CLS-эмбеддингов с доминирующими компонентами)
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
PHASES = [[p * 10 + i for i in range(10)] for p in range(10)]


def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def run(data: dict, topk: int, sen: float, label: str,
        save: bool = True) -> dict:
    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=topk, never_update=True,
                             seniority_bonus=sen)
    stats = run_continual(mem, data["X_train"], data["y_train"],
                          data["X_test"], data["y_test"], PHASES, topk=topk)
    print(f"[{label}] P@last={stats['p_last']*100:.2f}% "
          f"forg={stats['avg_forgetting']*100:.2f}pp t={stats['elapsed_s']:.1f}s",
          flush=True)
    out = {"label": label, "topk": topk, "seniority": sen,
           "p_last": stats["p_last"], "avg_forgetting": stats["avg_forgetting"]}
    if save:
        (RESULTS / f"m6b_{label}.json").write_text(json.dumps(out, indent=2))
    return out


def prepare(path: Path, mode: str) -> dict:
    data = load_features(path)
    if mode == "zscore":
        mu, sd = zscore_fit(data["X_train"])
        data = {
            "X_train": ((data["X_train"] - mu) / sd).astype(np.float32),
            "y_train": data["y_train"],
            "X_test": ((data["X_test"] - mu) / sd).astype(np.float32),
            "y_test": data["y_test"],
        }
    else:
        data = subset_classes(data, list(range(100)))
    return data


if __name__ == "__main__":
    cache_aug = Path(__file__).resolve().parent.parent / "data" / "dinov2_384_aug4.npz"
    results = []
    for tag, path, modes in (("aug4", cache_aug, ("plain", "zscore")),):
        for mode in modes:
            data = prepare(path, mode)
            print(f"--- {tag}/{mode}: train {data['X_train'].shape}")
            for topk in (16, 24, 32):
                results.append(run(data, topk, 0.0, f"{tag}_{mode}_k{topk}"))
    results.sort(key=lambda r: r["p_last"], reverse=True)
    print("\n=== Итог ===")
    for r in results:
        print(f"{r['label']}: {r['p_last']*100:.2f}%")

#!/usr/bin/env python3
"""M3: кросс-валидация порта — память ZF на SOMA-признаках CIFAR-10.

Цель (PLAN.md M3): расхождение с якорем SOMA ≤ 1pp либо найденная причина.
Якорь: SOMA C50k_base_clean = **58.02%** (лог logs/C50k_base_clean.log,
кэш cache_features/feats_4992bd0d09.npz, 250000 train × 714d).

Конфиг SOMA-чемпиона того прогона:
  --never-update --batch-train --threshold 0.08 --topk 8 --topk-mode per_class
  --per-class-k 8,8,6,8,8,8,8,8,8,8 --exemplar-per-class 300
Отличия ZF-порта: равномерный topk=8 (без per-class-k), без exemplar (для
чистоты сравнения механики), фазы по одному классу (10 классов CIFAR-10).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
SOMA_CACHE = Path.home() / "Projects/SOMA_Mind/cache_features/feats_4992bd0d09.npz"


def main():
    if not SOMA_CACHE.exists():
        raise SystemExit(f"Нет кэша SOMA: {SOMA_CACHE}")

    print(f"Загрузка {SOMA_CACHE} (743 МБ)...")
    d = np.load(SOMA_CACHE)
    X_train = d["X_train_feat"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test = d["X_test_feat"].astype(np.float32)
    y_test = d["y_test"].astype(np.int32)
    print(f"train {X_train.shape}, test {X_test.shape}")

    dim = X_train.shape[1]
    rows = []
    for label, topk, threshold, exemplar in [
        ("m3_soma_topk8_t08", 8, 0.08, 0),      # ближайший аналог SOMA-чемпиона
        ("m3_soma_topk8_t0", 8, 0.0, 0),        # без threshold (режим M2)
        ("m3_soma_topk8_t08_ex300", 8, 0.08, 300),  # + exemplar как в SOMA
    ]:
        mem = PrototypicalMemory(
            feature_dim=dim,
            threshold=threshold,
            topk=topk,
            never_update=True,
            exemplar_per_class=exemplar,
            capacity=max(4096, len(X_train) + 1024),
        )
        # CIFAR-10: 10 фаз × 1 класс (5000 семплов/класс после ауг4)
        phases = [[c] for c in range(10)]
        t0 = time.time()
        stats = run_continual(mem, X_train, y_train, X_test, y_test,
                              phases, topk=topk)
        dt = time.time() - t0
        row = {"label": label, "topk": topk, "threshold": threshold,
               "exemplar_per_class": exemplar,
               "p_last": round(stats["p_last"], 4),
               "avg_forgetting": round(stats["avg_forgetting"], 4),
               "total_protos": stats["total_protos"],
               "elapsed_s": round(dt, 1)}
        rows.append(row)
        print(f"[{label}] P@last = {row['p_last']*100:.2f}% | forg = "
              f"{row['avg_forgetting']*100:.2f}pp | protos = {row['total_protos']} "
              f"| {dt:.0f}s")
        del mem  # освободить ~1.4 ГБ перед следующим конфигом

    out = {"milestone": "M3", "anchor_soma_base_clean": 0.5802, "rows": rows}
    (RESULTS / "m3_soma_crossval.json").write_text(json.dumps(out, indent=2))
    print(f"\nЯкорь SOMA: 58.02%. Сохранено: results/m3_soma_crossval.json")


if __name__ == "__main__":
    main()

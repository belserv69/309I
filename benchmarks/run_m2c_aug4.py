#!/usr/bin/env python3
"""M2c: 100 классов на аугментированных признаках (4× память).

Гипотеза (CONCEPT §10): аугментация 4× даёт плотнее покрытие confusion-пар
и должна снизить структурное забывание (cls0 −13pp) без потери точности.
Ориентир: SOMA на CIFAR-10 аугментация давала стабильный выигрыш; CORAL на
mini-ImageNet hand-crafted аугментация НЕ дала (41.00% = 41.00%) — но там был
потолок 42%, здесь запас до oracle 6.5pp.

Также проверяем:
- память: 20k прототипов вместо 5k (×4);
- переносится ли выигрыш аугментации на тест без TTA.
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
DATA = Path(__file__).resolve().parent.parent / "data"
AUG_CACHE = DATA / "rn50_2048_aug4.npz"
N_CLASSES = 100


def run_proto(X_train, y_train, X_test, y_test, topk=4, seniority=0.08,
              label="m2c_aug4") -> dict:
    mem = PrototypicalMemory(feature_dim=X_train.shape[1], threshold=0.0,
                             topk=topk, never_update=True,
                             seniority_bonus=seniority,
                             capacity=max(4096, len(X_train) + 1024))
    phases = [[p * 10 + i for i in range(10)] for p in range(10)]
    t0 = time.time()
    stats = run_continual(mem, X_train, y_train, X_test, y_test, phases, topk=topk)
    dt = time.time() - t0

    print(f"\n{'phase':>5} {'cls':>4} {'acc':>7} {'protos':>7}")
    for p in stats["phases"]:
        print(f"{p['phase']:>5} {p['n_classes']:>4} {p['accuracy']*100:>6.2f}% "
              f"{p['protos']:>7}")
    print(f"\n[{label}] P@last = {stats['p_last']*100:.2f}% | avg forgetting = "
          f"{stats['avg_forgetting']*100:.2f}pp | cls0 = "
          f"{stats['cls0_forgetting']*100:.2f}pp | protos = {stats['total_protos']} "
          f"| t = {dt:.1f}s")

    out = {"label": label, "topk": topk, "seniority": seniority,
           "p_last": stats["p_last"], "avg_forgetting": stats["avg_forgetting"],
           "cls0_forgetting": stats["cls0_forgetting"],
           "per_class_forgetting": stats["per_class_forgetting"],
           "total_protos": stats["total_protos"], "elapsed_s": dt,
           "phases": [{"phase": p["phase"], "accuracy": p["accuracy"],
                       "protos": p["protos"]} for p in stats["phases"]]}
    (RESULTS / f"{label}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if not AUG_CACHE.exists():
        raise SystemExit(f"Кэш не найден: {AUG_CACHE}. Сначала: "
                         ".venv/bin/python benchmarks/extract_aug4.py")
    d = np.load(AUG_CACHE, allow_pickle=False)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    print(f"Данные aug4: train {X_train.shape} (4× от 5000), test {X_test.shape}")
    print(f"Хэш данных: {d['data_hash']}")

    # базовый чемпион M2 на аугментированных данных
    run_proto(X_train, y_train, X_test, y_test, topk=4, seniority=0.08,
              label="m2c_aug4_topk4_sen08")

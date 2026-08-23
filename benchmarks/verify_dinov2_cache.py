#!/usr/bin/env python3
"""Верификация кэша dinov2_384.npz портированным энкодером zf/encoders/dinov2.py.

Правило CONCEPT §9.7: подозрительный/чужой кэш перепроверяется, прежде чем
цитировать числа. Проверяем: (1) метки совпадают с rn50-кэшем, (2) пере-
извлечение подвыборки даёт косинус ≈ 1.0 с закэшированными векторами.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.encoders.dinov2 import DINOV2Encoder

RAW = Path(__file__).resolve().parent.parent / "data" / "mini100_84.npz"
DINO = Path(__file__).resolve().parent.parent / "data" / "dinov2_384.npz"
RN50 = Path(__file__).resolve().parent.parent / "data" / "rn50_2048.npz"
N_CHECK = 64


def main() -> None:
    dino = np.load(DINO)
    raw = np.load(RAW)
    rn50 = np.load(RN50)

    # 1. Выравнивание меток между кэшами и сырыми изображениями
    assert np.array_equal(raw["y_train"], dino["y_train"]), "y_train расходится!"
    assert np.array_equal(raw["x_train"] if False else raw["y_train"],
                          rn50["y_train"]), "y_train vs rn50 расходится!"
    assert np.array_equal(dino["y_test"], rn50["y_test"]), "y_test расходится!"
    print(f"[ok] метки выровнены: {len(np.unique(dino['y_train']))} классов, "
          f"{dino['X_train'].shape}")

    # 2. Переизвлечение подвыборки портированным энкодером
    enc = DINOV2Encoder()
    idx = np.linspace(0, len(raw["x_test"]) - 1, N_CHECK).astype(int)
    imgs = raw["x_test"][idx]
    feats = enc.encode(imgs)

    cached = dino["X_test"][idx]
    sims = np.sum(
        feats / np.linalg.norm(feats, axis=1, keepdims=True)
        * cached / np.linalg.norm(cached, axis=1, keepdims=True),
        axis=1,
    )
    print(f"[check] cosine(ported, cache) на {N_CHECK} тестовых: "
          f"min={sims.min():.6f} mean={sims.mean():.6f}")
    assert sims.min() > 0.999, "Кэш не соответствует энкодеру — НЕ ИСПОЛЬЗОВАТЬ"

    # 3. Геометрия признаков: нормы до L2-нормализации
    norms = np.linalg.norm(dino["X_train"], axis=1)
    print(f"[info] нормы train: [{norms.min():.3f}, {norms.max():.3f}], "
          f"mean={norms.mean():.3f}")
    print("ВЕРДИКТ: кэш dinov2_384.npz верифицирован, можно бенчмаркать.")


if __name__ == "__main__":
    main()

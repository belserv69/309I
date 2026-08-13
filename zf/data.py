"""Загрузка закэшированных признаков mini-ImageNet (RN50 GAP 2048d).

Кэш создан CORAL: benchmarks/.cache/rn50_2048.npz, скопирован в data/.
Ключи: X_train (5000, 2048) float32, y_train, X_test (3000, 2048), y_test.
100 классов, 50 train / 30 test на класс.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "rn50_2048.npz"


def load_features(path: str | Path | None = None) -> dict:
    """Вернуть dict с X_train, y_train, X_test, y_test."""
    p = Path(path) if path else DEFAULT_CACHE
    if not p.exists():
        raise FileNotFoundError(
            f"Кэш признаков не найден: {p}. Скопируйте из "
            "~/Projects/CORAL/benchmarks/.cache/rn50_2048.npz"
        )
    d = np.load(p)
    return {
        "X_train": d["X_train"],
        "y_train": d["y_train"],
        "X_test": d["X_test"],
        "y_test": d["y_test"],
    }


def subset_classes(data: dict, classes: list[int]) -> dict:
    """Оставить только указанные классы (метки сохраняются исходными)."""
    cls_list = list(classes)
    tr_mask = np.isin(data["y_train"], cls_list)
    te_mask = np.isin(data["y_test"], cls_list)
    return {
        "X_train": data["X_train"][tr_mask],
        "y_train": data["y_train"][tr_mask],
        "X_test": data["X_test"][te_mask],
        "y_test": data["y_test"][te_mask],
    }

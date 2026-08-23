"""Детектор новизны на max-cosine к прототипам — порт CORAL `coral/novelty.py`.

Сигнал: сходство ближайшего прототипа (ниже — «новее» вход).
Порог — квантиль ID-распределения под целевой FPR.
"""
from __future__ import annotations

import numpy as np

__all__ = ["calibrate_threshold", "evaluate_detection"]


def calibrate_threshold(id_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Порог τ: доля ID-сэмплов с score < τ ≈ target_fpr.

    Args:
        id_scores: значения сигнала на in-distribution данных.
        target_fpr: целевая доля ложных тревог (ID, помеченных как новые).

    Returns:
        Порог (квантиль target_fpr распределения id_scores).
    """
    if len(id_scores) == 0:
        raise ValueError("id_scores пуст")
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr={target_fpr}: ожидается в (0, 1)")
    return float(np.quantile(np.asarray(id_scores, dtype=np.float64), target_fpr))


def evaluate_detection(id_scores: np.ndarray, ood_scores: np.ndarray,
                       threshold: float) -> dict[str, float]:
    """Фактические FPR/TPR детектора «score < threshold = новый».

    Returns:
        {"fpr": ..., "tpr": ..., "n_id": ..., "n_ood": ...}
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    return {
        "fpr": float((id_scores < threshold).mean()),
        "tpr": float((ood_scores < threshold).mean()),
        "n_id": float(len(id_scores)),
        "n_ood": float(len(ood_scores)),
    }

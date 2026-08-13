"""CL-харнесс: фазовое обучение + метрики (порт протокола CORAL).

Протокол (10 классов CORAL): фазы [[0],[1],...,[9]] — по одному классу.
После каждой фазы оцениваются ВСЕ изученные классы; accuracy = среднее
per-class accuracy. Forgetting класса = лучший результат за историю −
текущий. P@last = accuracy последней фазы.
"""
from __future__ import annotations

import time

import numpy as np

from .memory.proto import PrototypicalMemory


def run_continual(
    mem: PrototypicalMemory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    phases: list[list[int]],
    topk: int | None = None,
) -> dict:
    """Прогнать память по фазам, вернуть полную статистику.

    Returns:
        dict: phases=[{phase, accuracy, per_class, protos}],
              p_last, avg_forgetting, per_class_forgetting, cls0_forgetting,
              elapsed_s, total_protos
    """
    seen: set[int] = set()
    history: dict[int, list[float]] = {}  # class -> [acc по фазам после появления]
    phase_log = []
    t0 = time.time()

    for phase_idx, classes in enumerate(phases):
        mem.set_phase(phase_idx)
        for c in classes:
            mask = y_train == c
            mem.add_batch(X_train[mask], y_train[mask])
            seen.add(int(c))

        per_class: dict[int, float] = {}
        for c in sorted(seen):
            mask = y_test == c
            preds, _ = mem.query_batch(X_test[mask], k=topk)
            acc = float((preds == y_test[mask]).mean()) if mask.any() else 0.0
            per_class[int(c)] = acc
            history.setdefault(int(c), []).append(acc)

        accs = list(per_class.values())
        phase_log.append(
            {
                "phase": phase_idx,
                "n_classes": len(seen),
                "accuracy": float(np.mean(accs)) if accs else 0.0,
                "per_class": per_class,
                "protos": mem.size,
            }
        )

    # Forgetting: best_so_far − final (для классов, появившихся не в последней фазе)
    per_class_forgetting: dict[int, float] = {}
    for c, accs in history.items():
        if len(accs) >= 2:
            per_class_forgetting[c] = float(max(accs[:-1]) - accs[-1])

    last = phase_log[-1] if phase_log else None
    return {
        "phases": phase_log,
        "p_last": last["accuracy"] if last else 0.0,
        "avg_forgetting": (
            float(np.mean(list(per_class_forgetting.values())))
            if per_class_forgetting
            else 0.0
        ),
        "per_class_forgetting": per_class_forgetting,
        "cls0_forgetting": per_class_forgetting.get(0, 0.0),
        "total_protos": mem.size,
        "elapsed_s": time.time() - t0,
    }

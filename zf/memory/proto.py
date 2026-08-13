"""Прототипная память — порт чемпионного пути SOMA_Mind.

Источник: SOMA_Mind/scripts/test_soma_c205_per_class_exemplar.py, класс
PrototypicalMemory (строки 551-1401). Перенесён только доказавший работу
чемпионный путь:

- never_update: каждый семпл = новый замороженный прототип (0% forgetting);
- add_batch(): векторизованное O(N) добавление (вместо O(N²) по-семплового);
- запрос per-class top-k через FAISS FlatIP с threshold-фильтром;
- exemplar ring (per-class FIFO) — опциональный аддитивный бонус;
- seniority bonus (flat) — опциональный бонус старым классам;
- per_class_cap — лимит прототипов на класс.

НЕ перенесено (измеренные тупики SOMA): LTM-кольцо, mixup, дескрипторы,
merge_damping, conf_ratio, margin-check, centroid_weight-блендинг.

Все векторы хранятся L2-нормированными; сходство = косинус (FlatIP).
"""
from __future__ import annotations

import numpy as np


class PrototypicalMemory:
    """Замороженная прототипная память с per-class top-k запросом."""

    def __init__(
        self,
        feature_dim: int,
        capacity: int = 65536,
        threshold: float = 0.08,
        topk: int = 8,
        never_update: bool = True,
        per_class_cap: int = 0,
        exemplar_per_class: int = 0,
        exemplar_weight: float = 0.05,
        seniority_bonus: float = 0.0,
    ):
        self.feature_dim = feature_dim
        self.capacity = capacity
        self.threshold = threshold
        self.topk = topk
        self.never_update = never_update
        self.per_class_cap = per_class_cap
        self.exemplar_per_class = exemplar_per_class
        self.exemplar_weight = exemplar_weight
        self.seniority_bonus = seniority_bonus

        self.size = 0
        self.prototypes = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.labels = np.zeros(capacity, dtype=np.int32)
        self.counts = np.zeros(capacity, dtype=np.int32)

        # Exemplar ring: per-class FIFO замороженных семплов
        self._exemplar_vecs: np.ndarray | None
        self._exemplar_labels: np.ndarray | None
        self._exemplar_fifo: dict[int, list[int]] | None
        if exemplar_per_class > 0:
            self._exemplar_vecs = np.zeros((0, feature_dim), dtype=np.float32)
            self._exemplar_labels = np.zeros(0, dtype=np.int32)
            self._exemplar_size = 0
            self._exemplar_fifo = {}
        else:
            self._exemplar_vecs = None
            self._exemplar_labels = None
            self._exemplar_size = 0
            self._exemplar_fifo = None

        # Фазы (для seniority bonus)
        self._class_phase: dict[int, int] = {}
        self._current_phase = 0

        # FAISS
        self._faiss_index = None
        self._dirty = False

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #
    def set_phase(self, phase_id: int) -> None:
        self._current_phase = phase_id
        if self.size > 0:
            for c in set(self.labels[:self.size].tolist()):
                if c not in self._class_phase:
                    self._class_phase[c] = phase_id

    def _grow(self) -> None:
        """Авто-расширение ёмкости ×1.5 (порт SOMA _grow)."""
        new_cap = int(self.capacity * 1.5)
        protos = self.prototypes[: self.size].copy()
        lbls = self.labels[: self.size].copy()
        cnts = self.counts[: self.size].copy()
        self.prototypes = np.zeros((new_cap, self.feature_dim), dtype=np.float32)
        self.prototypes[: self.size] = protos
        self.labels = np.zeros(new_cap, dtype=np.int32)
        self.labels[: self.size] = lbls
        self.counts = np.zeros(new_cap, dtype=np.int32)
        self.counts[: self.size] = cnts
        self.capacity = new_cap

    def update(self, vector: np.ndarray, label: int) -> None:
        """По-семпловое добавление (эталон семантики для never_update).

        Для never_update=True идентично созданию нового замороженного
        прототипа на каждый семпл — как в SOMA champion.
        """
        v = np.asarray(vector, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)

        if self.size >= self.capacity:
            self._grow()
        self.prototypes[self.size] = v
        self.labels[self.size] = label
        self.counts[self.size] = 1
        self.size += 1
        self._dirty = True
        self._exemplar_append(int(label), v)

    def add_batch(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Векторизованное добавление. Семантически идентично update() для
        never_update (порт SOMA add_batch, строки 1332-1397)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        labels = np.asarray(labels)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vn = (vectors / norms).astype(np.float32)

        if not self.never_update:
            # merge-режим не используется в чемпионном пути — честный фолбэк
            for i in range(vn.shape[0]):
                self.update(vn[i], int(labels[i]))
            return

        n = vn.shape[0]

        # per-class cap (порт проверки из SOMA add_batch)
        if self.per_class_cap > 0:
            current_counts = {
                int(lab): int(np.sum(self.labels[: self.size] == lab))
                for lab in np.unique(labels)
            }
            keep_mask = np.ones(n, dtype=bool)
            for i in range(n):
                lab = int(labels[i])
                if current_counts.get(lab, 0) >= self.per_class_cap:
                    keep_mask[i] = False
                else:
                    current_counts[lab] += 1
            if not keep_mask.any():
                return
            vn = vn[keep_mask]
            labels = labels[keep_mask]
            n = vn.shape[0]

        while self.size + n > self.capacity:
            self._grow()
        self.prototypes[self.size: self.size + n] = vn
        self.labels[self.size: self.size + n] = labels
        self.counts[self.size: self.size + n] = 1
        self.size += n
        self._dirty = True

        if self._exemplar_vecs is not None:
            for i in range(n):
                self._exemplar_append(int(labels[i]), vn[i])

    def _exemplar_append(self, label: int, v: np.ndarray) -> None:
        """Добавить семпл в exemplar ring (порт SOMA _exemplar_append)."""
        if self._exemplar_vecs is None or self._exemplar_fifo is None:
            return
        fifo = self._exemplar_fifo.setdefault(label, [])
        if len(fifo) < self.exemplar_per_class:
            vec_idx = self._exemplar_size
            if vec_idx >= len(self._exemplar_vecs):
                # рост кольцевого буфера
                new_cap = max(16, len(self._exemplar_vecs) * 2)
                new_vecs = np.zeros((new_cap, self.feature_dim), dtype=np.float32)
                new_lbls = np.zeros(new_cap, dtype=np.int32)
                if self._exemplar_size:
                    new_vecs[: self._exemplar_size] = self._exemplar_vecs
                    new_lbls[: self._exemplar_size] = self._exemplar_labels
                self._exemplar_vecs = new_vecs
                self._exemplar_labels = new_lbls
            assert self._exemplar_labels is not None
            self._exemplar_vecs[vec_idx] = v
            self._exemplar_labels[vec_idx] = label
            self._exemplar_size += 1
            fifo.append(vec_idx)
        else:
            oldest_idx = fifo.pop(0)
            self._exemplar_vecs[oldest_idx] = v
            fifo.append(oldest_idx)

    # ------------------------------------------------------------------ #
    # Запрос
    # ------------------------------------------------------------------ #
    def _rebuild_faiss(self) -> None:
        if self.size == 0:
            self._faiss_index = None
            self._dirty = False
            return
        import faiss

        vecs = np.array(self.prototypes[: self.size], dtype=np.float32, copy=True)
        faiss.normalize_L2(vecs)
        self._faiss_index = faiss.IndexFlatIP(self.feature_dim)
        self._faiss_index.add(vecs)
        self._dirty = False

    def query_batch(self, vectors: np.ndarray, k: int | None = None):
        """Per-class top-k запрос (порт SOMA _query_batch_per_class).

        Returns:
            preds: (N,) int32, -1 если ни один кандидат выше threshold
            confs: (N,) float32 — score лучшего класса
        """
        if self.size == 0:
            return (
                np.full(len(vectors), -1, dtype=np.int32),
                np.zeros(len(vectors), dtype=np.float32),
            )
        if self._dirty:
            self._rebuild_faiss()
        import faiss

        assert self._faiss_index is not None  # rebuilt above when size > 0

        if k is None:
            k = self.topk
        batch = np.asarray(vectors, dtype=np.float32).copy()
        faiss.normalize_L2(batch)

        n_candidates = min(k * 20, self.size)
        D, I = self._faiss_index.search(batch, n_candidates)

        unique_classes = np.unique(self.labels[: self.size])

        # Exemplar-сходства (аддитивный бонус, как в SOMA)
        ex_sims = None
        ex_labels = None
        if self._exemplar_vecs is not None and self._exemplar_size > 0:
            ex_vecs = self._exemplar_vecs[: self._exemplar_size]
            ex_labels = self._exemplar_labels[: self._exemplar_size]
            ex_sims = batch @ ex_vecs.T

        preds = np.empty(len(batch), dtype=np.int32)
        confs = np.empty(len(batch), dtype=np.float32)

        for i in range(len(batch)):
            sims = D[i]
            idxs = I[i]

            # накопление кандидатов по классам с threshold-фильтром
            class_scores: dict[int, list[float]] = {int(c): [] for c in unique_classes}
            for j in range(n_candidates):
                if sims[j] < self.threshold:
                    continue
                lbl = int(self.labels[idxs[j]])
                if lbl in class_scores:
                    class_scores[lbl].append(float(sims[j]))

            class_means = {
                lbl: float(np.mean(sorted(scores, reverse=True)[:k]))
                for lbl, scores in class_scores.items()
                if scores
            }

            # Exemplar-бонус: среднее по всем exemplarам класса × weight
            if ex_sims is not None and ex_labels is not None:
                ex_row = ex_sims[i]
                for cls in unique_classes:
                    cls_int = int(cls)
                    mask = ex_labels == cls_int
                    if mask.any():
                        ex_score = float(ex_row[mask].mean())
                        if cls_int in class_means:
                            class_means[cls_int] += self.exemplar_weight * ex_score
                        else:
                            class_means[cls_int] = self.exemplar_weight * ex_score

            # Seniority bonus (flat-режим SOMA)
            if self.seniority_bonus > 0 and class_means:
                for cls_int in class_means:
                    if self._class_phase.get(cls_int, self._current_phase) < self._current_phase:
                        conf = class_means[cls_int]
                        class_means[cls_int] = conf + self.seniority_bonus * (1.0 - conf ** 2)

            if not class_means:
                preds[i] = -1
                confs[i] = 0.0
            else:
                best_lbl = max(class_means, key=lambda lbl: class_means[lbl])
                preds[i] = best_lbl
                confs[i] = class_means[best_lbl]

        return preds, confs

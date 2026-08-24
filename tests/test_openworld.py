"""Тесты авто-добавления классов и детектора новизны (открытый мир)."""
from __future__ import annotations

import numpy as np
import pytest

from zf.memory.proto import PrototypicalMemory
from zf.novelty import calibrate_threshold, evaluate_detection

rng = np.random.default_rng(123)


def _cluster(center: np.ndarray, n: int, noise: float = 0.05) -> np.ndarray:
    """n векторов вокруг центра (L2-нормированных)."""
    v = center + rng.standard_normal((n, len(center))) * noise
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


class TestNoveltyPort:
    def test_calibrate_threshold_quantile(self):
        scores = np.linspace(0, 1, 101)
        tau = calibrate_threshold(scores, target_fpr=0.05)
        assert abs(tau - 0.05) < 1e-9

    def test_calibrate_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate_threshold(np.array([]))

    def test_evaluate_detection(self):
        res = evaluate_detection(id_scores=np.array([0.9, 0.8]),
                                 ood_scores=np.array([0.1, 0.2]),
                                 threshold=0.5)
        assert res["fpr"] == 0.0 and res["tpr"] == 1.0


class TestObserveBatch:
    def _mem_two_clusters(self) -> tuple[PrototypicalMemory, np.ndarray, np.ndarray]:
        c0 = rng.standard_normal(32).astype(np.float32)
        c1 = rng.standard_normal(32).astype(np.float32)
        mem = PrototypicalMemory(feature_dim=32, threshold=0.0, topk=4)
        mem.add_batch(_cluster(c0, 10), np.zeros(10, dtype=np.int32))
        mem.add_batch(_cluster(c1, 10), np.ones(10, dtype=np.int32))
        return mem, c0 / np.linalg.norm(c0), c1 / np.linalg.norm(c1)

    def test_known_classes_assigned_without_creation(self):
        mem, c0, c1 = self._mem_two_clusters()
        stream = np.concatenate([_cluster(c0, 5), _cluster(c1, 5)])
        assigned, info = mem.observe_batch(stream, novelty_threshold=None)
        assert info["n_created"] == 0
        assert set(assigned.tolist()) <= {0, 1}
        assert (assigned[:5] == assigned[0]).all()
        assert (assigned[5:] == assigned[5]).all()

    def test_repeat_pass_preserves_assignments(self):
        """Повторное обучение на том же потоке не ухудшает назначения
        (концепция проекта): вектор узнаёт себя по замороженному прототипу."""
        mem, c0, c1 = self._mem_two_clusters()
        stream = np.concatenate([_cluster(c0, 5), _cluster(c1, 5)])
        a1, info1 = mem.observe_batch(stream, novelty_threshold=None)
        a2, info2 = mem.observe_batch(stream, novelty_threshold=None)
        assert info1["n_created"] == 0
        assert info2["n_created"] == 0
        assert info2["n_repeats"] == len(stream)
        m = (a1 >= 0) & (a2 >= 0)
        assert (a1[m] == a2[m]).all()

    def test_unknown_cluster_creates_new_class(self):
        mem, c0, _ = self._mem_two_clusters()
        # ортогональный кластер — заведомо «новый»
        c_new = np.zeros(32, dtype=np.float32)
        c_new[16:] = 1.0
        c_new /= np.linalg.norm(c_new)
        stream = np.concatenate([_cluster(c_new, 6), _cluster(c0, 2)])
        scores = mem.novelty_scores_batch(stream)
        tau = float(np.quantile(
            mem.novelty_scores_batch(_cluster(c0, 20)), 0.05))
        assigned, info = mem.observe_batch(stream, novelty_threshold=tau)
        assert info["n_created"] >= 1
        # новые семплы получили метку вне {0, 1}
        new_rows = np.where(~np.isin(assigned, [0, 1]))[0]
        assert len(new_rows) >= 4

    def test_min_cluster_size_marks_noise(self):
        mem, _, _ = self._mem_two_clusters()
        c_noise = np.zeros(32, dtype=np.float32)
        c_noise[:16] = 1.0
        c_noise /= np.linalg.norm(c_noise)
        stream = _cluster(c_noise, 2)  # меньше min_cluster_size
        assigned, info = mem.observe_batch(stream, novelty_threshold=0.999,
                                           min_cluster_size=3)
        assert info["noise_labels"], "одиночный кластер должен стать шумом"
        assert (assigned == -2).all()

    def test_novelty_scores_shape_and_range(self):
        mem, _, _ = self._mem_two_clusters()
        s = mem.novelty_scores_batch(rng.standard_normal((7, 32)))
        assert s.shape == (7,)
        assert ((s >= -1) & (s <= 1)).all()


class TestWeightedTopK:
    def test_sim_weighting_valid_values(self):
        for w in ("mean", "sim"):
            m = PrototypicalMemory(feature_dim=8, topk_weighting=w)
            assert m.topk_weighting == w

    def test_invalid_weighting_raises(self):
        with pytest.raises(ValueError):
            PrototypicalMemory(feature_dim=8, topk_weighting="bogus")

    def test_sim_beats_or_matches_mean_on_tight_clusters(self):
        """На плотных кластерах взвешивание не должно ломать классификацию."""
        centers = [rng.standard_normal(16).astype(np.float32) for _ in range(3)]
        X = np.concatenate([_cluster(c, 15) for c in centers])
        y = np.repeat(np.arange(3), 15).astype(np.int32)
        Xq = np.concatenate([_cluster(c, 10) for c in centers])
        yq = np.repeat(np.arange(3), 10).astype(np.int32)

        accs = []
        for weighting in ("mean", "sim"):
            m = PrototypicalMemory(feature_dim=16, topk=4,
                                   topk_weighting=weighting)
            m.add_batch(X, y)
            preds, _ = m.query_batch(Xq)
            accs.append(float((preds == yq).mean()))
        assert accs[1] >= accs[0] - 0.05

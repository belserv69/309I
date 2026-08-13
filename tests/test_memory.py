"""Юнит-тесты прототипной памяти (порт SOMA)."""
from __future__ import annotations

import numpy as np

from zf.memory.proto import PrototypicalMemory

rng = np.random.default_rng(42)


def _make_mem(dim=64, **kw) -> PrototypicalMemory:
    return PrototypicalMemory(feature_dim=dim, **kw)


class TestAddBatchParity:
    """add_batch ≡ последовательный update для never_update (урок SOMA)."""

    def test_parity_never_update(self):
        X = rng.standard_normal((30, 64)).astype(np.float32)
        y = np.repeat(np.arange(3), 10).astype(np.int32)

        m1 = _make_mem()
        m1.add_batch(X, y)

        m2 = _make_mem()
        for i in range(len(X)):
            m2.update(X[i], int(y[i]))

        assert m1.size == m2.size == 30
        np.testing.assert_allclose(m1.prototypes[: m1.size], m2.prototypes[: m2.size], atol=1e-6)
        np.testing.assert_array_equal(m1.labels[: m1.size], m2.labels[: m2.size])

    def test_prototypes_l2_normalized(self):
        X = rng.standard_normal((20, 32)).astype(np.float32) * 5
        m = _make_mem(dim=32)
        m.add_batch(X, np.zeros(20, dtype=np.int32))
        norms = np.linalg.norm(m.prototypes[: m.size], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestQuery:
    def test_synthetic_two_clusters(self):
        """Два хорошо разделённых кластера → per-class top-k их различает."""
        m = _make_mem(dim=16, threshold=0.0)
        a = np.zeros(16, dtype=np.float32); a[0] = 1.0
        b = np.zeros(16, dtype=np.float32); b[15] = 1.0
        # шум вокруг центров
        xa = a[None] + rng.standard_normal((20, 16)).astype(np.float32) * 0.05
        xb = b[None] + rng.standard_normal((20, 16)).astype(np.float32) * 0.05
        m.add_batch(np.vstack([xa, xb]), np.array([0] * 20 + [1] * 20, dtype=np.int32))

        preds, confs = m.query_batch(np.vstack([xa[:5], xb[:5]]))
        np.testing.assert_array_equal(preds, [0] * 5 + [1] * 5)
        assert (confs > 0.5).all()

    def test_threshold_filter(self):
        """Высокий threshold → все кандидаты отсеяны → pred=-1."""
        m = _make_mem(dim=8, threshold=0.99)
        m.add_batch(np.eye(8, dtype=np.float32), np.arange(8, dtype=np.int32))
        q = np.ones(8, dtype=np.float32) / np.sqrt(8)  # одинаково далёк от всех
        preds, _ = m.query_batch(q[None])
        assert preds[0] == -1

    def test_empty_memory(self):
        m = _make_mem(dim=4)
        preds, confs = m.query_batch(np.ones((3, 4), dtype=np.float32))
        np.testing.assert_array_equal(preds, [-1, -1, -1])
        np.testing.assert_array_equal(confs, [0, 0, 0])


class TestGrow:
    def test_grow_preserves_data(self):
        m = _make_mem(dim=8, capacity=10)
        X = rng.standard_normal((50, 8)).astype(np.float32)
        y = np.arange(50, dtype=np.int32) % 5
        m.add_batch(X, y)
        assert m.size == 50
        assert m.capacity >= 50
        # данные не потеряны: каждый прототип находит себя сам
        preds, _ = m.query_batch(X, k=1)
        np.testing.assert_array_equal(preds, y)


class TestPerClassCap:
    def test_cap_respected(self):
        m = _make_mem(dim=8, per_class_cap=3)
        X = rng.standard_normal((10, 8)).astype(np.float32)
        m.add_batch(X, np.zeros(10, dtype=np.int32))
        assert m.size == 3  # только первые 3 прошли cap


class TestExemplar:
    def test_fifo_eviction(self):
        m = _make_mem(dim=4, exemplar_per_class=2)
        X = np.eye(4, dtype=np.float32)[:4]  # 4 семпла класса 0
        m.add_batch(X, np.zeros(4, dtype=np.int32))
        assert m._exemplar_size == 2
        assert m._exemplar_fifo is not None and m._exemplar_vecs is not None
        fifo = m._exemplar_fifo[0]
        assert len(fifo) == 2
        # в кольце остались два последних семпла
        np.testing.assert_array_equal(m._exemplar_vecs[fifo[0]], X[2])
        np.testing.assert_array_equal(m._exemplar_vecs[fifo[1]], X[3])

    def test_exemplar_bonus_affects_query(self):
        """Exemplar-бонус способен поднять класс, у которого нет proto выше threshold."""
        m = _make_mem(dim=8, threshold=0.95, exemplar_per_class=4, exemplar_weight=1.0)
        e = np.zeros(8, dtype=np.float32); e[0] = 1.0
        m.add_batch(np.stack([e] * 2), np.array([3, 3], dtype=np.int32))
        # запрос похож на exemplar, но threshold для proto-канала высокий:
        # без exemplar-канала предсказания бы не было
        q = e + 0.2 * rng.standard_normal(8).astype(np.float32)
        preds, _ = m.query_batch(q[None])
        assert preds[0] == 3

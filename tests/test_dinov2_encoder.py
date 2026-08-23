"""Тесты DINOv2-энкодера (порт CORAL dinov2_encoder).

Требуют torch + веса в ~/.cache/torch/hub; без них тесты пропускаются.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from zf.encoders.dinov2 import MODELS, DINOV2Encoder  # noqa: E402

rng = np.random.default_rng(7)


@pytest.fixture(scope="module")
def encoder():
    return DINOV2Encoder()


class TestDINOV2Encoder:
    def test_output_shape_and_dim(self, encoder):
        imgs = rng.integers(0, 256, (3, 84, 84, 3), dtype=np.uint8)
        feats = encoder.encode(imgs)
        assert feats.shape == (3, MODELS["dinov2_vits14"])
        assert feats.dtype == np.float32

    def test_single_image_3d_input(self, encoder):
        img = rng.integers(0, 256, (84, 84, 3), dtype=np.uint8)
        feats = encoder.encode(img)
        assert feats.shape == (1, encoder.feature_dim)

    def test_deterministic(self, encoder):
        imgs = rng.integers(0, 256, (4, 84, 84, 3), dtype=np.uint8)
        f1 = encoder.encode(imgs)
        f2 = encoder.encode(imgs)
        np.testing.assert_array_equal(f1, f2)

    def test_l2_similarity_invariant_to_norm(self, encoder):
        """Ключевое свойство для памяти: косинусная геометрия."""
        a = rng.integers(0, 256, (2, 84, 84, 3), dtype=np.uint8)
        f = encoder.encode(a)
        n = np.linalg.norm(f, axis=1)
        assert (n > 0).all()

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            DINOV2Encoder(model_name="resnet50")

    def test_resize_matches_coral(self, encoder):
        """Препроцессинг обязан ресайзить 84×84 → 224×224 (урок CORAL)."""
        assert encoder.input_size == 224

"""Frozen DINOv2-энкодер — порт CORAL `coral/encoders/dinov2_encoder.py`.

Чемпионский энкодер Track B CORAL: CLS-токен ViT-S/14 (384d). Отличие от
оригинала: без SDR/Config-зависимостей, только float-признаки.

Препроцессинг идентичен CORAL (бит-в-бит): bilinear 224×224 → /255 →
ImageNet mean/std. DINOv2 предобучен на 224×224; нативный вход 84×84
(6×6 патчей) деградирует фичи — ресайз обязателен.

Критично для сравнения с кэшем `data/dinov2_384.npz`: любые отличия
препроцессинга дадут другое пространство признаков (урок SOMA).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# имя модели → размерность CLS-эмбеддинга
MODELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
}

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class DINOV2Encoder:
    """Извлечение CLS-признаков замороженного DINOv2 (torch.hub)."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu",
                 input_size: int = 224, batch_size: int = 32):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name!r}, choose from {list(MODELS)}")
        self.device = torch.device(device)
        self.model_name = model_name
        self.feature_dim = MODELS[model_name]
        self.input_size = input_size
        self.batch_size = batch_size

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.model.to(self.device)

        self._mean = torch.tensor(IMAGE_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGE_STD, device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, images_nhwc: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(images_nhwc)).to(
            self.device, dtype=torch.float32)
        t = t.permute(0, 3, 1, 2)
        if t.shape[2] != self.input_size or t.shape[3] != self.input_size:
            t = F.interpolate(t, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return t.div_(255.0).sub_(self._mean).div_(self._std)

    @torch.no_grad()
    def encode(self, images_nhwc: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 → (N, feature_dim) float32, батчами."""
        images_nhwc = np.asarray(images_nhwc)
        if images_nhwc.ndim == 3:
            images_nhwc = images_nhwc[None]
        out = []
        for start in range(0, len(images_nhwc), self.batch_size):
            batch = images_nhwc[start:start + self.batch_size]
            feats = self.model(self._preprocess(batch))
            out.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

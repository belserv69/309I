"""Frozen ResNet-энкодер — порт CORAL `coral/encoders/learned_encoder.py`.

Отличие от оригинала: без SDR/Config-зависимостей, только float-признаки
(GAP после layer4). Препроцессинг идентичен CORAL (бит-в-бит):
bilinear 224×224 → /255 → ImageNet mean/std. Это критично: любые отличия
дадут другое пространство признаков и сломают сравнение с кэшем
`data/rn50_2048.npz` (см. урок SOMA про неверную геометрию wrapper'а).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BACKBONES = {
    "rn18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "rn50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
}


class ResNetEncoder:
    """Извлечение признаков замороженного ResNet (GAP layer4)."""

    def __init__(self, backbone: str = "rn50", device: str = "cpu",
                 batch_size: int = 64):
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}, choose from {list(BACKBONES)}")
        self.device = torch.device(device)
        self.backbone = backbone
        self.feature_dim = BACKBONES[backbone][2]
        self.batch_size = batch_size

        ctor, weights, _ = BACKBONES[backbone]
        self.model = ctor(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        self._features: torch.Tensor | None = None
        self._hook = self.model.layer4.register_forward_hook(self._hook_fn)

        self._image_mean = torch.tensor([0.485, 0.456, 0.406],
                                        device=self.device).view(1, 3, 1, 1)
        self._image_std = torch.tensor([0.229, 0.224, 0.225],
                                       device=self.device).view(1, 3, 1, 1)

    def _hook_fn(self, module: nn.Module, _input, output: torch.Tensor):
        self._features = output

    def _preprocess(self, images_nhwc: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(images_nhwc.copy()).to(self.device, dtype=torch.float32)
        t = t.permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t.div_(255.0)
        t.sub_(self._image_mean).div_(self._image_std)
        return t

    @torch.no_grad()
    def encode(self, images_nhwc: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 → (N, feature_dim) float32, батчами."""
        images_nhwc = np.asarray(images_nhwc)
        if images_nhwc.ndim == 3:
            images_nhwc = images_nhwc[None]
        out = []
        for start in range(0, len(images_nhwc), self.batch_size):
            batch = images_nhwc[start:start + self.batch_size]
            x = self._preprocess(batch)
            _ = self.model(x)
            gap = self._features.mean(dim=[2, 3])
            out.append(gap.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def __del__(self):
        if hasattr(self, "_hook"):
            self._hook.remove()

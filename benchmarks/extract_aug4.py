#!/usr/bin/env python3
"""Извлечение RN50-признаков для аугментированных изображений (M2c).

Аугментация 4× идентична CORAL `_augment_images` (seed=42):
оригинал + h-flip + rotate ±6° + shift ±2px (BORDER_REFLECT).

Кэш сохраняется с хэшем данных в метаданных (урок SOMA: cfg_key обязан
хэшировать данные, а не только параметры).

Верификация порта: признаки оригиналов сравниваются с кэшем CORAL
`data/rn50_2048.npz` — должны совпасть (float32 точность).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from zf.encoders.resnet import ResNetEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PIXELS = DATA / "mini100_84.npz"
OUT = DATA / "rn50_2048_aug4.npz"
CORAL_PIXELS = Path.home() / "Projects/CORAL/benchmarks/.cache/mini100_84.npz"


def augment_images(images: np.ndarray) -> np.ndarray:
    """Порт CORAL _augment_images: N → 4N (seed 42, детерминированно)."""
    n, h, w, c = images.shape
    augmented = [images]
    rng = np.random.RandomState(42)
    for i in range(n):
        img = images[i]
        flip = np.fliplr(img)
        augmented.append(flip[None])
        angle = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        augmented.append(rotated[None])
        dx = rng.randint(-2, 3)
        dy = rng.randint(-2, 3)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented.append(shifted[None])
    return np.concatenate(augmented, axis=0)


def main():
    if not PIXELS.exists():
        if not CORAL_PIXELS.exists():
            raise SystemExit(f"нет ни {PIXELS}, ни {CORAL_PIXELS}")
        import shutil
        shutil.copy(CORAL_PIXELS, PIXELS)
        print(f"Скопировано: {CORAL_PIXELS} → {PIXELS}")

    d = np.load(PIXELS)
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]
    print(f"Пиксели: train {x_train.shape}, test {x_test.shape}")

    data_hash = hashlib.sha256(x_train.tobytes()[:1 << 20]
                               + x_test.tobytes()[:1 << 20]).hexdigest()[:16]
    print(f"Хэш данных: {data_hash}")

    t0 = time.time()
    x_aug = augment_images(x_train)
    print(f"Аугментация: {len(x_train)} → {len(x_aug)} за {time.time()-t0:.1f}s")

    enc = ResNetEncoder(backbone="rn50", batch_size=64)

    # Промежуточный кэш: признаки сохраняются сразу после извлечения,
    # ДО верификации/упаковки — чтобы сбой после 10-минутного кодирования
    # не терял вычисления. Повторный запуск с тем же хэшем переиспользует его.
    raw_cache = DATA / "_aug4_raw.npz"
    if raw_cache.exists():
        rc = np.load(raw_cache)
        if str(rc["data_hash"]) == data_hash:
            feats_train, feats_test = rc["feats_train"], rc["feats_test"]
            print(f"Переиспользую сырой кэш: {raw_cache}")
        else:
            rc.close()
            raw_cache.unlink()
            feats_train = feats_test = None
    else:
        feats_train = feats_test = None

    if feats_train is None:
        t0 = time.time()
        feats_train = enc.encode(x_aug)
        print(f"Признаки train: {feats_train.shape} за {time.time()-t0:.1f}s")

        t0 = time.time()
        feats_test = enc.encode(x_test)
        print(f"Признаки test: {feats_test.shape} за {time.time()-t0:.1f}s")

        np.savez_compressed(raw_cache, feats_train=feats_train,
                            feats_test=feats_test, data_hash=np.array(data_hash))
        print(f"Сырой кэш сохранён: {raw_cache}")

    assert feats_train is not None and feats_test is not None

    # --- верификация порта против кэша CORAL ---
    # ВНИМАНИЕ: порядок CORAL _augment_images — [все N оригиналов, затем
    # перемеженные flip/rot/shift]. Оригиналы занимают индексы [0, N).
    coral_cache = ROOT / "data" / "rn50_2048.npz"
    if coral_cache.exists():
        cc = np.load(coral_cache)
        originals = feats_train[: len(x_train)]
        ref = cc["X_train"].astype(np.float32)
        assert originals.shape == ref.shape, (
            f"форма не совпала: {originals.shape} vs {ref.shape}")
        max_diff = float(np.abs(originals - ref).max())
        print(f"Верификация порта: max|diff| против кэша CORAL = {max_diff:.6f}")
        if max_diff > 1e-3:
            raise SystemExit("ПОРТ ЭНКОДЕРА НЕВЕРЕН — признаки расходятся с кэшем CORAL!")
        print("Порт верен ✓")

    np.savez_compressed(
        OUT,
        X_train=feats_train,
        # порядок CORAL: [оригиналы, flip'ы, rotate'ы, shift'ы] → tile, не repeat
        y_train=np.tile(y_train, 4),
        X_test=feats_test,
        y_test=y_test,
        data_hash=np.array(data_hash),
        augment=np.array("flip+rot6+shift2_seed42_order:orig,flip,rot,shift"),
    )
    print(f"Сохранено: {OUT}")
    print(f"  X_train {feats_train.shape}, y_train {len(np.tile(y_train, 4))}")
    print(f"  X_test {feats_test.shape}")


if __name__ == "__main__":
    main()

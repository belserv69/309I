#!/usr/bin/env python3
"""Извлечение DINOv2-признаков аугментированных изображений (M6).

Аугментация 4× идентична CORAL `_augment_images` (seed=42): оригинал +
h-flip + rotate ±6° + shift ±2px (BORDER_REFLECT) — реиспользует код M2c.
Кэш с хэшем данных + промежуточный raw-кэш против потери вычислений.

Верификация порта: признаки оригиналов сравниваются с кэшем data/dinov2_384.npz.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_aug4 import augment_images  # порт CORAL-аугментаций (seed=42)
from zf.encoders.dinov2 import DINOV2Encoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PIXELS = DATA / "mini100_84.npz"
OUT = DATA / "dinov2_384_aug4.npz"
REF_CACHE = DATA / "dinov2_384.npz"


def main() -> None:
    d = np.load(PIXELS)
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]
    print(f"Пиксели: train {x_train.shape}, test {x_test.shape}", flush=True)

    data_hash = hashlib.sha256(x_train.tobytes()[:1 << 20]
                               + x_test.tobytes()[:1 << 20]).hexdigest()[:16]
    print(f"Хэш данных: {data_hash}")

    t0 = time.time()
    x_aug = augment_images(x_train)
    print(f"Аугментация: {len(x_train)} → {len(x_aug)} за {time.time()-t0:.1f}s",
          flush=True)

    enc = DINOV2Encoder(batch_size=64)

    # Промежуточный raw-кэш (урок M2c: не терять вычисления при сбое)
    raw_cache = DATA / "_dinov2_aug4_raw.npz"
    feats_train = feats_test = None
    if raw_cache.exists():
        rc = np.load(raw_cache)
        if str(rc["data_hash"]) == data_hash:
            feats_train, feats_test = rc["feats_train"], rc["feats_test"]
            print(f"Переиспользую сырой кэш: {raw_cache}")
        else:
            rc.close()
            raw_cache.unlink()
    if feats_train is None:
        t0 = time.time()
        feats_train = enc.encode(x_aug)
        print(f"Признаки train: {feats_train.shape} за {time.time()-t0:.1f}s",
              flush=True)
        t0 = time.time()
        feats_test = enc.encode(x_test)
        print(f"Признаки test: {feats_test.shape} за {time.time()-t0:.1f}s",
              flush=True)
        np.savez(raw_cache, feats_train=feats_train, feats_test=feats_test,
                 data_hash=np.array(data_hash))
        print(f"Сырой кэш сохранён: {raw_cache}")

    assert feats_train is not None and feats_test is not None

    # --- верификация порта: оригиналы [0, N) должны совпасть с кэшем ---
    ref = np.load(REF_CACHE)["X_train"].astype(np.float32)
    originals = feats_train[: len(x_train)]
    assert originals.shape == ref.shape
    max_diff = float(np.abs(originals - ref).max())
    cos = np.sum(originals * ref, axis=1) / (
        np.linalg.norm(originals, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"Верификация: max|diff|={max_diff:.6f}, min cosine={cos.min():.6f}")
    if cos.min() < 0.9999:
        raise SystemExit("ПОРТ DINOv2 НЕВЕРЕН — признаки расходятся с кэшем!")
    print("Порт верен ✓")

    # Порядок augment_images: [N оригиналов][per-sample тройки flip/rot/shift].
    # Метки должны повторять ЭТОТ порядок: y + interleave(y×3), НЕ tile!
    # (баг M6: tile давал рассинхрон меток и обрушал точность до ~7%)
    y_aug = np.concatenate([y_train, np.repeat(y_train, 3)])
    assert len(y_aug) == len(feats_train)

    np.savez_compressed(
        OUT,
        X_train=feats_train,
        y_train=y_aug,
        X_test=feats_test,
        y_test=y_test,
        data_hash=np.array(data_hash),
        augment=np.array("flip+rot6+shift2_seed42_order:orig,(flip,rot,shift)xN"),
        encoder=np.array(enc.model_name),
    )
    print(f"Сохранено: {OUT} | X_train {feats_train.shape}")


def relabel_from_raw() -> None:
    """Пересобрать OUT из raw-кэша с правильными метками (без пере-извлечения)."""
    d = np.load(PIXELS)
    y_train, y_test = d["y_train"], d["y_test"]
    rc = np.load(DATA / "_dinov2_aug4_raw.npz")
    feats_train, feats_test = rc["feats_train"], rc["feats_test"]
    y_aug = np.concatenate([y_train, np.repeat(y_train, 3)])
    assert len(y_aug) == len(feats_train)
    np.savez_compressed(
        OUT,
        X_train=feats_train,
        y_train=y_aug,
        X_test=feats_test,
        y_test=y_test,
        augment=np.array("flip+rot6+shift2_seed42_order:orig,(flip,rot,shift)xN"),
        encoder=np.array("dinov2_vits14"),
    )
    print(f"Пересобрано: {OUT} | X_train {feats_train.shape}, метки исправлены")


if __name__ == "__main__":
    if "--relabel" in sys.argv:
        relabel_from_raw()
    else:
        main()

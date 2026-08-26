#!/usr/bin/env python3
"""Извлечение признаков DINOv2 ViT-L/14 (1024d): base + aug4 кэши.

Продолжение линии энкодеров RN50 → ViT-S (384d) → ViT-B (768d) → ViT-L.
Прогон долгий (~2.5–3ч на CPU), поэтому каждый блок (base_train,
base_test, aug_train, aug_test) кодируется чанками с чекпоинтом:
убийство процесса теряет максимум один чанк (~5–8 мин).

Верификация: оригиналы aug4-блока обязаны совпасть с base-кэшем
(cos ≥ 0.9999) — тот же инвариант, что у extract_dinov2b.py.

Выход: data/vitl_1024.npz, data/vitl_1024_aug4.npz
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from extract_aug4 import augment_images  # CORAL-аугментации, seed=42
from zf.encoders.dinov2 import DINOV2Encoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PIXELS = DATA / "mini100_84.npz"
OUT_BASE = DATA / "vitl_1024.npz"
OUT_AUG = DATA / "vitl_1024_aug4.npz"
CKPT = DATA / "vitl_1024_partial.npz"
CHUNK = 2500   # ~5–8 мин на чанк для ViT-L на CPU


def encode_ckpt(enc: DINOV2Encoder, images: dict[str, np.ndarray],
                blocks: dict[str, np.ndarray]) -> None:
    """Кодирует images[имя] в blocks[имя] (float32), чекпоинт после чанка."""
    if CKPT.exists():
        d = np.load(CKPT)
        for k in list(blocks):
            if f"part_{k}" in d.files:
                blocks[k] = d[f"part_{k}"]
        print("чекпоинт: " + ", ".join(f"{k}={len(blocks[k])}"
                                       for k in blocks), flush=True)
    total = sum(len(v) for v in images.values())
    done_sess = 0
    t0 = time.time()
    for name, imgs in images.items():
        n = len(imgs)
        cur = len(blocks[name])
        if cur >= n:
            print(f"  {name}: готово ({n}/{n})", flush=True)
            continue
        for s in range(cur, n, CHUNK):
            e = min(s + CHUNK, n)
            feats = enc.encode(imgs[s:e]).astype(np.float32)
            blocks[name] = np.concatenate([blocks[name], feats])
            np.savez(CKPT, **{f"part_{k}": v for k, v in blocks.items() if len(v)})
            done_sess += e - s
            left = total - sum(len(blocks[k]) for k in blocks)
            eta = left / max(done_sess / max(time.time() - t0, 1e-9), 1e-9)
            print(f"  {name}: {e}/{n} | ETA {eta/60:.0f} мин", flush=True)


def main() -> None:
    d = np.load(PIXELS)
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]
    data_hash = hashlib.sha256(x_train.tobytes()[:1 << 20]
                               + x_test.tobytes()[:1 << 20]).hexdigest()[:16]

    enc = DINOV2Encoder(model_name="dinov2_vitl14", batch_size=32)
    print(f"Энкодер: {enc.model_name}, dim={enc.feature_dim}", flush=True)

    y_aug = np.concatenate([y_train, np.repeat(y_train, 3)])
    x_aug = augment_images(x_train)
    assert len(y_aug) == len(x_aug)

    blocks = {k: np.empty((0, enc.feature_dim), np.float32)
              for k in ("base_train", "base_test", "aug_train", "aug_test")}
    images = {"base_train": x_train, "base_test": x_test,
              "aug_train": x_aug, "aug_test": x_test}

    encode_ckpt(enc, images, blocks)

    Xb_train, Xb_test = blocks["base_train"], blocks["base_test"]
    Xa_train, Xa_test = blocks["aug_train"], blocks["aug_test"]

    # --- верификация: оригиналы aug4 ≡ base ---
    orig = Xa_train[: len(x_train)]
    cos = np.sum(orig * Xb_train, axis=1) / (
        np.linalg.norm(orig, axis=1) * np.linalg.norm(Xb_train, axis=1))
    print(f"Верификация aug4↔base: min cosine={cos.min():.6f}")
    if cos.min() < 0.9999:
        raise SystemExit("РАСХОЖДЕНИЕ aug4 и base — кэш не годен!")

    np.savez_compressed(OUT_BASE, X_train=Xb_train, y_train=y_train,
                        X_test=Xb_test, y_test=y_test,
                        data_hash=np.array(data_hash),
                        encoder=np.array(enc.model_name))
    np.savez_compressed(OUT_AUG, X_train=Xa_train, y_train=y_aug,
                        X_test=Xa_test, y_test=y_test,
                        data_hash=np.array(data_hash),
                        encoder=np.array(enc.model_name),
                        augment=np.array("flip+rot6+shift2_seed42"))
    CKPT.unlink(missing_ok=True)
    print(f"Сохранено: {OUT_BASE}, {OUT_AUG}")


if __name__ == "__main__":
    main()

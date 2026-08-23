#!/usr/bin/env python3
"""Извлечение признаков DINOv2 ViT-B/14 (768d): base + aug4 кэши.

Верификация: оригиналы aug4-блока обязаны совпасть с base-кэшем бит-в-бит
(внешнего эталона нет — CORAL держал только ViT-S).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from extract_aug4 import augment_images  # CORAL-аугментации, seed=42
from zf.encoders.dinov2 import DINOV2Encoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PIXELS = DATA / "mini100_84.npz"
OUT_BASE = DATA / "dinov2b_768.npz"
OUT_AUG = DATA / "dinov2b_768_aug4.npz"


def main() -> None:
    d = np.load(PIXELS)
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]
    data_hash = hashlib.sha256(x_train.tobytes()[:1 << 20]
                               + x_test.tobytes()[:1 << 20]).hexdigest()[:16]

    enc = DINOV2Encoder(model_name="dinov2_vitb14", batch_size=32)
    print(f"Энкодер: {enc.model_name}, dim={enc.feature_dim}", flush=True)

    # --- base ---
    if OUT_BASE.exists():
        print(f"Base уже есть: {OUT_BASE}")
    else:
        t0 = time.time()
        Xb_train = enc.encode(x_train)
        Xb_test = enc.encode(x_test)
        print(f"base: train {Xb_train.shape} + test {Xb_test.shape} "
              f"за {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(OUT_BASE, X_train=Xb_train, y_train=y_train,
                            X_test=Xb_test, y_test=y_test,
                            data_hash=np.array(data_hash),
                            encoder=np.array(enc.model_name))

    # --- aug4 ---
    if OUT_AUG.exists():
        print(f"aug4 уже есть: {OUT_AUG}")
        return
    t0 = time.time()
    x_aug = augment_images(x_train)
    print(f"аугментация: {len(x_train)} → {len(x_aug)} за {time.time()-t0:.1f}s",
          flush=True)
    t0 = time.time()
    Xa_train = enc.encode(x_aug)
    Xa_test = enc.encode(x_test)
    print(f"aug4: train {Xa_train.shape} за {time.time()-t0:.0f}s", flush=True)

    # --- верификация: оригиналы aug4 ≡ base ---
    ref = np.load(OUT_BASE)["X_train"].astype(np.float32)
    originals = Xa_train[: len(x_train)]
    cos = np.sum(originals * ref, axis=1) / (
        np.linalg.norm(originals, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"Верификация aug4↔base: min cosine={cos.min():.6f}")
    if cos.min() < 0.9999:
        raise SystemExit("РАСХОЖДЕНИЕ aug4 и base — кэш не годен!")

    y_aug = np.concatenate([y_train, np.repeat(y_train, 3)])
    assert len(y_aug) == len(Xa_train)
    np.savez_compressed(OUT_AUG, X_train=Xa_train, y_train=y_aug,
                        X_test=Xa_test, y_test=y_test,
                        data_hash=np.array(data_hash),
                        encoder=np.array(enc.model_name),
                        augment=np.array("flip+rot6+shift2_seed42"))
    print(f"Сохранено: {OUT_BASE}, {OUT_AUG}")


if __name__ == "__main__":
    main()

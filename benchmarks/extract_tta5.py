#!/usr/bin/env python3
"""TTA5: тестовые признаки = среднее 5 аугментированных просмотров.

Виды: оригинал, h-flip, rotate ±6°, shift (+1,+1) — порт идеи TTA5 SOMA
(усреднение признаков повышает устойчивость запроса к искажениям).
Train не трогается: берётся из базового кэша энкодера.

Выход: data/<base>_tta5.npz с усреднённым X_test и исходным y_test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from zf.encoders.dinov2 import DINOV2Encoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PIXELS = DATA / "mini100_84.npz"


def tta_views(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    views = [img, np.fliplr(img)]
    for angle in (6.0, -6.0):
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        views.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT))
    M = np.float32([[1, 0, 1], [0, 1, 1]])
    views.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return np.stack(views)


def main() -> None:
    base_name = sys.argv[1] if len(sys.argv) > 1 else "dinov2b_768.npz"
    model_name = "dinov2_vitb14" if base_name.startswith("dinov2b") \
        else "dinov2_vits14"
    base_cache = DATA / base_name
    out_path = DATA / f"{Path(base_name).stem}_tta5.npz"

    d = np.load(base_cache)
    raw = np.load(PIXELS)
    x_test = raw["x_test"]
    assert np.array_equal(d["y_test"], raw["y_test"])

    enc = DINOV2Encoder(model_name=model_name, batch_size=64)
    n = len(x_test)
    acc = np.zeros((n, enc.feature_dim), dtype=np.float64)
    t0 = time.time()
    for v in range(5):
        views = np.concatenate([tta_views(x_test[i])[[v]] for i in range(n)])
        feats = enc.encode(views).astype(np.float64)
        acc += feats
        print(f"view {v}: done ({time.time()-t0:.0f}s)", flush=True)
    X_test_avg = (acc / 5.0).astype(np.float32)

    np.savez_compressed(out_path,
                        X_train=d["X_train"], y_train=d["y_train"],
                        X_test=X_test_avg, y_test=d["y_test"],
                        encoder=np.array(enc.model_name),
                        tta=np.array("orig,flip,rot6,rot-6,shift11"))
    print(f"Сохранено: {out_path} | X_test {X_test_avg.shape}")


if __name__ == "__main__":
    main()

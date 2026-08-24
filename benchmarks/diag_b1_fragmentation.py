#!/usr/bin/env python3
"""B1: диагностика перефрагментации открытий + абляция онлайн-слияния.

Вопросы: какие истинные классы разбились на несколько кластеров; косинусы
центроидов фрагментов; помогает ли онлайн-слияние по порогу merge_cos и
где проходит граница между «склеил фрагменты» и «склеил разные классы».
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory

ROOT = Path(__file__).resolve().parent.parent
N_BASE, N_OPEN = 50, 50


def build_base_memory(data) -> tuple[PrototypicalMemory, dict]:
    """База 50 классов + калиброванный порог новизны."""
    base = subset_classes(data, list(range(N_BASE)))
    open_cls = subset_classes(data, list(range(N_BASE, N_BASE + N_OPEN)))

    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=8, never_update=True)
    phases = [[p * 10 + i for i in range(10)] for p in range(5)]
    from zf.pipeline import run_continual
    run_continual(mem, base["X_train"], base["y_train"],
                  base["X_test"], base["y_test"], phases, topk=8)

    tau = mem.calibrate_novelty(base["X_test"][:750], target_fpr=0.05)
    return mem, {"open_cls": open_cls, "tau": tau}


def discovery_batch(open_cls: dict, seed: int = 42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(open_cls["X_test"]))
    X_disc = open_cls["X_test"][order][:375]
    y_disc = open_cls["y_test"][order][:375]
    return X_disc, y_disc


def cluster_metrics(y_disc: np.ndarray, assigned: np.ndarray) -> dict:
    mask = assigned >= 0
    clusters = sorted(set(assigned[mask].tolist()) - set(range(N_BASE)))

    cls_clusters: defaultdict[int, set[int]] = defaultdict(set)
    purity_num = purity_den = 0
    mixed = 0
    for c in clusters:
        rows = np.where(assigned == c)[0]
        cnt = Counter(y_disc[rows].tolist())
        dominant, dcnt = cnt.most_common(1)[0]
        if len(cnt) > 1:
            mixed += 1
        purity_num += dcnt
        purity_den += len(rows)
        cls_clusters[int(dominant)].add(c)

    split = {t: v for t, v in cls_clusters.items() if len(v) > 1}
    return {
        "n_clusters": len(clusters),
        "split_classes": len(split),
        "mixed_clusters": mixed,
        "purity": purity_num / max(purity_den, 1),
    }


def centroid_stats(X_disc: np.ndarray, y_disc: np.ndarray,
                   assigned: np.ndarray) -> None:
    """Косинусы центроидов фрагментов внутри класса и между классами."""
    mask = assigned >= 0
    clusters = sorted(set(assigned[mask].tolist()) - set(range(N_BASE)))
    cents = {}
    doms = {}
    for c in clusters:
        rows = np.where(assigned == c)[0]
        v = X_disc[rows].mean(axis=0)
        cents[c] = v / (np.linalg.norm(v) + 1e-12)
        doms[c] = Counter(y_disc[rows].tolist()).most_common(1)[0][0]

    near = [float(cents[a] @ cents[b])
            for i, a in enumerate(clusters) for b in clusters[i + 1:]
            if doms[a] == doms[b]]
    cross = [float(cents[a] @ cents[b])
             for i, a in enumerate(clusters) for b in clusters[i + 1:]
             if doms[a] != doms[b]]
    if near:
        print(f"  внутриклассовые cos центроидов: min={min(near):.3f} "
              f"max={max(near):.3f} mean={np.mean(near):.3f}")
    if cross:
        print(f"  межклассовые cos центроидов: max={max(cross):.3f} "
              f"p95={np.quantile(cross, 0.95):.3f}")


def diagnose(data) -> None:
    print("\n=== базовая диагностика ===")
    mem, ctx = build_base_memory(data)
    open_cls, tau = ctx["open_cls"], ctx["tau"]
    X_disc, y_disc = discovery_batch(open_cls)

    assigned, info = mem.observe_batch(X_disc, novelty_threshold=tau,
                                       min_cluster_size=3)
    m = cluster_metrics(y_disc, assigned)
    print(f"кластеров: {m['n_clusters']} на {N_OPEN} классов | "
          f"расколото классов: {m['split_classes']} | "
          f"смешанных кластеров: {m['mixed_clusters']} | "
          f"чистота: {m['purity']:.3f}")

    cls_clusters: defaultdict[int, list] = defaultdict(list)
    for c in sorted(set(assigned[assigned >= 0].tolist()) - set(range(N_BASE))):
        rows = np.where(assigned == c)[0]
        dominant, cnt = Counter(y_disc[rows].tolist()).most_common(1)[0]
        cls_clusters[dominant].append((c, len(rows), round(cnt / len(rows), 2)))
    for t in sorted(cls_clusters):
        if len(cls_clusters[t]) > 1:
            frags = ", ".join(f"#{c}(n={n},p={p})"
                              for c, n, p in cls_clusters[t])
            print(f"  класс {t}: {frags}")
    centroid_stats(X_disc, y_disc, assigned)


def ablate_merge(data, thresholds=(0.40, 0.45, 0.50, 0.55, 0.60)) -> None:
    print("\n=== абляция онлайн-слияния (merge_cos свип) ===")
    _, ctx = build_base_memory(data)
    open_cls, tau = ctx["open_cls"], ctx["tau"]
    X_disc, y_disc = discovery_batch(open_cls)

    header = (f"{'merge_cos':>9} {'кластеров':>9} {'слияний':>8} "
              f"{'расколото':>9} {'смешанных':>9} {'чистота':>7}")
    print(header)
    print("-" * len(header))

    for mc in (None, *thresholds):
        mem, _ = build_base_memory(data)   # свежая память на каждое значение
        assigned, info = mem.observe_batch(
            X_disc, novelty_threshold=tau, min_cluster_size=3, merge_cos=mc)
        m = cluster_metrics(y_disc, assigned)
        tag = "base" if mc is None else f"{mc:.2f}"
        print(f"{tag:>9} {m['n_clusters']:>9} {info['n_merges']:>8} "
              f"{m['split_classes']:>9} {m['mixed_clusters']:>9} "
              f"{m['purity']:>7.3f}")


if __name__ == "__main__":
    for name in ("dinov2_384.npz", "dinov2b_768.npz"):
        p = ROOT / "data" / name
        if p.exists():
            data = load_features(p)
            print(f"\n########## {name} ##########")
            diagnose(data)
            ablate_merge(data)

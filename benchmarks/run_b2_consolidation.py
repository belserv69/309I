#!/usr/bin/env python3
"""B2: replay-консолидация открытого мира повторными проходами (repeat_cos).

Развитие B1b (FINAL_REPORT §6): после первого прохода потока открытий база
консолидируется повторным показом тех же данных через observe_batch().

Механика (B1b): каждый семпл узнаёт собственный замороженный прототип
(cos >= repeat_cos=0.999) — новые классы не создаются (created=0), назначения
стабильны (a1==a2 100%), а шумовые кластеры первого прохода восстанавливаются:
фильтр min_cluster_size применяется только к кластерам, созданным в ТЕКУЩЕМ
проходе. Повторное обучение монотонно улучшает NCM (B1b: 56.5% → 88.7%).

Метрики после каждого прохода:
- покрытие и Hungarian на полном покрытии;
- Hungarian на стабильном подмножестве (строки с валидной меткой прохода 1);
- согласованность назначений с предыдущим проходом;
- NCM-обобщение центроидов на невиданную половину теста.

Использование:
    .venv/bin/python benchmarks/run_b2_consolidation.py [cache.npz] [merge_cos] [passes]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / "data" / "dinov2b_768.npz")
MERGE_COS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
PASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N_BASE = 50
N_OPEN = 50
PHASES = [[p * 10 + i for i in range(10)] for p in range(5)]


def hungarian_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    t_true, t_pred = np.unique(true), np.unique(pred)
    cont = np.zeros((len(t_true), len(t_pred)), dtype=np.int64)
    ti = {c: i for i, c in enumerate(t_true)}
    pi = {c: j for j, c in enumerate(t_pred)}
    for t, p in zip(true, pred):
        cont[ti[t], pi[p]] += 1
    row, col = linear_sum_assignment(-cont)
    return cont[row, col].sum() / len(true)


def ncm_centroid_acc(X_disc: np.ndarray, y_disc: np.ndarray,
                     assigned: np.ndarray, X_ncm: np.ndarray,
                     y_ncm: np.ndarray) -> tuple[float, int]:
    """NCM невиданного сплита по центроидам открытых кластеров."""
    from collections import defaultdict

    centroids: dict[int, list[np.ndarray]] = defaultdict(list)
    for lbl in set(assigned[assigned >= 0].tolist()):
        rows = np.where(assigned == lbl)[0]
        centroids[lbl].append(X_disc[rows].mean(axis=0))
    if not centroids:
        return 0.0, 0
    cl_ids = sorted(centroids)
    C = np.stack([np.mean(centroids[c], axis=0) for c in cl_ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    Q = X_ncm / (np.linalg.norm(X_ncm, axis=1, keepdims=True) + 1e-8)
    sims = Q @ C.T
    pred_cl = np.array(cl_ids)[sims.argmax(axis=1)]
    return hungarian_accuracy(y_ncm, pred_cl), len(cl_ids)


def main() -> None:
    data = load_features(CACHE)
    base = subset_classes(data, list(range(N_BASE)))
    open_cls = subset_classes(data, list(range(N_BASE, N_BASE + N_OPEN)))

    # 1. База с метками (как в M7)
    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=8, never_update=True)
    stats = run_continual(mem, base["X_train"], base["y_train"],
                          base["X_test"], base["y_test"], PHASES, topk=8)
    print(f"[base] P@{N_BASE}cls = {stats['p_last']*100:.2f}%", flush=True)

    # 2. Порог новизны на невиданных векторах базы (как в M7: seed 7)
    rng0 = np.random.default_rng(7)
    base_order = rng0.permutation(len(base["X_test"]))
    half_base = len(base_order) // 2
    cal_rows = base["X_test"][base_order[:half_base]]
    eval_rows = base["X_test"][base_order[half_base:]]
    tau = mem.calibrate_novelty(cal_rows, target_fpr=0.05)

    # 3. Поток открытий (как в M7: первая половина теста, seed 42)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(open_cls["X_test"]))
    X_all, y_all = open_cls["X_test"][order], open_cls["y_test"][order]
    half = len(X_all) // 2
    X_disc, y_disc = X_all[:half], y_all[:half]
    X_ncm, y_ncm = X_all[half:], y_all[half:]

    # 4. Консолидация: PASSES проходов одного и того же потока
    print(f"\n[consolidation] passes={PASSES} merge_cos={MERGE_COS} "
          f"| поток {len(X_disc)} семплов")
    header = (f"{'pass':>4} {'created':>7} {'merges':>6} {'repeats':>7} "
              f"{'cov%':>6} {'H-full':>7} {'H-stable':>8} {'agree':>6} "
              f"{'clusters':>8} {'NCM':>7}")
    print(header)
    print("-" * len(header))

    rows_out: list[dict] = []
    stable_rows: np.ndarray | None = None   # валидные строки прохода 1
    prev_assigned: np.ndarray | None = None

    for p in range(1, PASSES + 1):
        assigned, info = mem.observe_batch(
            X_disc, novelty_threshold=tau, min_cluster_size=3,
            merge_cos=MERGE_COS)
        mask = assigned >= 0
        cov = float(mask.mean())
        acc_full = hungarian_accuracy(y_disc[mask], assigned[mask])
        if p == 1:
            stable_rows = mask.copy()
            acc_stable = acc_full
            agree = 1.0
        else:
            assert prev_assigned is not None
            acc_stable = hungarian_accuracy(
                y_disc[stable_rows], assigned[stable_rows])
            both = stable_rows & mask
            agree = float((assigned[both] == prev_assigned[both]).mean())
        acc_ncm, n_cl = ncm_centroid_acc(
            X_disc, y_disc, assigned, X_ncm, y_ncm)

        print(f"{p:>4} {info['n_created']:>7} {info['n_merges']:>6} "
              f"{info['n_repeats']:>7} {cov*100:>6.1f} {acc_full*100:>7.2f} "
              f"{acc_stable*100:>8.2f} {agree*100:>6.1f} {n_cl:>8} "
              f"{acc_ncm*100:>7.2f}")
        rows_out.append({
            "pass": p,
            "n_created": info["n_created"],
            "n_merges": info["n_merges"],
            "n_repeats": info["n_repeats"],
            "coverage": cov,
            "hungarian_full": acc_full,
            "hungarian_stable": acc_stable,
            "agreement_with_prev": agree,
            "n_clusters": n_cl,
            "ncm_centroid_acc": acc_ncm,
        })
        prev_assigned = assigned.copy()

    # 5. Итог по формуле CORAL (лучший проход по NCM)
    best = max(rows_out, key=lambda r: r["ncm_centroid_acc"])
    total = (stats["p_last"] * len(base["X_test"])
             + best["hungarian_full"] * len(X_disc)) / (
        len(base["X_test"]) + len(X_disc))
    print(f"\n[total@pass{best['pass']}] база {stats['p_last']*100:.1f}% + "
          f"открытия {best['hungarian_full']*100:.1f}% → взвешенно "
          f"{total*100:.1f}% | NCM {best['ncm_centroid_acc']*100:.2f}%")

    tag = CACHE.stem
    out = {
        "protocol": f"B2 consolidation: {N_BASE} labeled base → "
                    f"{N_OPEN} unlabeled, {PASSES} passes",
        "cache": str(CACHE.name),
        "merge_cos": MERGE_COS,
        "passes": PASSES,
        "base_p_last": stats["p_last"],
        "tau": tau,
        "per_pass": rows_out,
        "best_pass": best["pass"],
        "total_weighted_best_pass": float(total),
    }
    (RESULTS / f"b2_consolidation_{tag}_m{int(MERGE_COS*100)}.json").write_text(
        json.dumps(out, indent=2))
    print(f"[saved] results/b2_consolidation_{tag}_"
          f"m{int(MERGE_COS*100)}.json")


if __name__ == "__main__":
    main()

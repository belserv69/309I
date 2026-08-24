#!/usr/bin/env python3
"""M7: открытый мир — авто-открытие классов без меток (протокол CORAL).

Протокол повторяет последний цикл CORAL (50/50):
1. База: классы 0–49 обучаются С МЕТКАМИ (5 фаз × 10).
2. Открытия: тестовые векторы классов 50–99 (1500 шт, перемешаны)
   подаются БЕЗ ЕДИНОЙ МЕТКИ через observe_batch() — полная конкуренция
   колонок, без жёсткого гейта известное/новое (инсайт абляции CORAL:
   hard gate 69.5% < soft 76.8% < no gate 78.0%).
3. Метрики как в CORAL:
   - база: P@last с метками;
   - открытия: Hungarian accuracy кластеров;
   - NCM-диагностика: вторая половина теста невиданных классов проверяет,
     что открытые кластеры обобщаются на невиданный сплит;
   - детектор новизны: AUROC ID vs OOD.

Сравнение с CORAL: база 83.7–85.3%, открытия 70.4–70.6%, итого 77–78%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zf.data import load_features, subset_classes
from zf.memory.proto import PrototypicalMemory
from zf.novelty import evaluate_detection
from zf.pipeline import run_continual

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / "data" / "dinov2_384.npz")
MERGE_COS = float(sys.argv[2]) if len(sys.argv) > 2 else None  # онлайн-слияние
N_BASE = 50          # классы 0..49 — база с метками
N_OPEN = 50          # классы 50..99 — открытия без меток
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


def main() -> None:
    data = load_features(CACHE)
    base = subset_classes(data, list(range(N_BASE)))
    open_cls = subset_classes(data, list(range(N_BASE, N_BASE + N_OPEN)))

    # 1. База с метками
    mem = PrototypicalMemory(feature_dim=data["X_train"].shape[1],
                             threshold=0.0, topk=8, never_update=True)
    stats = run_continual(mem, base["X_train"], base["y_train"],
                          base["X_test"], base["y_test"], PHASES, topk=8)
    print(f"[base] P@{N_BASE}cls = {stats['p_last']*100:.2f}%", flush=True)

    # 2. Порог новизны: калибровка ТОЛЬКО на невиданных векторах.
    # Train нельзя — каждый семпл лежит в памяти прототипом, его max-cosine
    # = 1.0 и порог вырождается (урок первого прогона M7).
    rng0 = np.random.default_rng(7)
    base_order = rng0.permutation(len(base["X_test"]))
    half_base = len(base_order) // 2
    cal_rows = base["X_test"][base_order[:half_base]]
    eval_rows = base["X_test"][base_order[half_base:]]

    tau = mem.calibrate_novelty(cal_rows, target_fpr=0.05)
    id_scores = mem.novelty_scores_batch(eval_rows)
    ood_scores = mem.novelty_scores_batch(open_cls["X_test"])
    det = evaluate_detection(id_scores, ood_scores, tau)
    from sklearn.metrics import roc_auc_score
    # сигнал «ниже = новее» → для AUROC (OOD=1) инвертируем знак
    auroc = roc_auc_score(
        np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)]),
        np.concatenate([-id_scores, -ood_scores]))
    print(f"[novelty] tau={tau:.4f} | AUROC={auroc:.3f} "
          f"| FPR={det['fpr']:.3f} TPR={det['tpr']:.3f}")

    # 3. Открытие: первая половина теста открытых классов, без меток
    rng = np.random.default_rng(42)
    order = rng.permutation(len(open_cls["X_test"]))
    X_all, y_all = open_cls["X_test"][order], open_cls["y_test"][order]
    half = len(X_all) // 2
    X_disc, y_disc = X_all[:half], y_all[:half]      # поток открытий
    X_ncm, y_ncm = X_all[half:], y_all[half:]        # NCM-проверка

    assigned, info = mem.observe_batch(X_disc, novelty_threshold=tau,
                                       min_cluster_size=3,
                                       merge_cos=MERGE_COS)
    mask = assigned >= 0
    n_disc = len(set(assigned[mask].tolist()))
    acc_disc = hungarian_accuracy(y_disc[mask], assigned[mask])
    print(f"[discovery] классов создано: {info['n_created']} "
          f"(шум: {len(info['noise_labels'])}); выжило ≥3 семплов: {n_disc} / {N_OPEN}")
    print(f"[discovery] Hungarian accuracy: {acc_disc*100:.2f}% "
          f"(покрытие {mask.mean()*100:.1f}%)")

    # 4. NCM-диагностика (как CORAL): центроиды открытых кластеров
    #    должны правильно классифицировать НЕвиданную половину теста
    from collections import defaultdict
    centroids: dict[int, np.ndarray] = defaultdict(list)
    for lbl in set(assigned[mask].tolist()):
        rows = np.where(assigned == lbl)[0]
        centroids[lbl].append(X_disc[rows].mean(axis=0))
    cl_ids = sorted(centroids)
    C = np.stack([np.mean(centroids[c], axis=0) for c in cl_ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8

    Q = X_ncm / (np.linalg.norm(X_ncm, axis=1, keepdims=True) + 1e-8)
    sims = Q @ C.T                                   # (n_ncm, n_clusters)
    pred_cl = np.array(cl_ids)[sims.argmax(axis=1)]

    # Hungarian между предсказанными кластерами и истинными метками NCM
    acc_ncm = hungarian_accuracy(y_ncm, pred_cl)

    # вариант «через память»: NCM-векторы классифицируются всей памятью
    preds_mem, _ = mem.query_batch(X_ncm)
    known_mask = preds_mem < N_BASE                  # ушли в старые классы
    acc_mem_new = hungarian_accuracy(y_ncm[~known_mask], preds_mem[~known_mask])
    print(f"[ncm] обобщение на невиданный сплит: centroid={acc_ncm*100:.2f}% | "
          f"memory={acc_mem_new*100:.2f}% (ушло в базу {known_mask.mean()*100:.1f}%)")

    # 5. Пост-слияние фрагментов: жадное объединение кластеров-новинок
    #    с косинусом центроидов > merge_cos (лечит перефрагментацию)
    def greedy_merge(assigned_arr: np.ndarray, merge_cos: float) -> np.ndarray:
        arr = assigned_arr.copy()
        new_ids = sorted(set(arr[arr >= 0].tolist()) - set(range(N_BASE)))
        cents = {c: X_disc[arr == c].mean(axis=0) for c in new_ids
                 if (arr == c).sum() > 0}
        for c in list(cents):
            cents[c] /= np.linalg.norm(cents[c]) + 1e-8
        parent = {c: c for c in new_ids}

        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        ids = list(cents)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = find(ids[i]), find(ids[j])
                if a == b:
                    continue
                if float(cents[a] @ cents[b]) > merge_cos:
                    lo, hi = min(a, b), max(a, b)
                    parent[hi] = lo
        remap = {}
        out = arr.copy()
        for c in new_ids:
            remap[c] = find(c)
        for c, r in remap.items():
            out[arr == c] = r
        return out

    merged = greedy_merge(assigned, merge_cos=0.85)
    m_mask = merged >= 0
    n_merged_cls = len(set(merged[m_mask].tolist()) - set(range(N_BASE)))
    acc_merged = hungarian_accuracy(y_disc[m_mask], merged[m_mask])
    print(f"[merge@0.85] кластеров после слияния: {n_merged_cls} / {N_OPEN} | "
          f"Hungarian: {acc_merged*100:.2f}%")

    # 6. Итог по формуле CORAL: база + открытия
    total = (stats["p_last"] * len(base["X_test"]) + acc_disc * mask.sum()) / (
        len(base["X_test"]) + len(X_disc))
    print(f"\n[total] база {stats['p_last']*100:.1f}% + открытия "
          f"{acc_disc*100:.1f}% → взвешенно {total*100:.1f}% "
          f"(CORAL: 77–78%, supervised ref 90.0%)")

    # 7. Итог по формуле CORAL: база + открытия (после слияния)
    total = (stats["p_last"] * len(base["X_test"]) + acc_merged * m_mask.sum()) / (
        len(base["X_test"]) + len(X_disc))
    print(f"\n[total] база {stats['p_last']*100:.1f}% + открытия "
          f"{acc_merged*100:.1f}% → взвешенно {total*100:.1f}% "
          f"(CORAL: 77–78%, supervised ref 90.0%)")

    tag = CACHE.stem  # dinov2_384 | dinov2b_768
    mtag = "base" if MERGE_COS is None else f"m{int(MERGE_COS*100)}"
    out = {
        "protocol": f"train {N_BASE} labeled → discover {N_OPEN} unlabeled",
        "cache": str(CACHE.name),
        "merge_cos": MERGE_COS,
        "base_p_last": stats["p_last"],
        "tau": tau, "auroc": auroc, **det,
        "n_created": info["n_created"],
        "n_noise": len(info["noise_labels"]),
        "n_discovered_raw": int(n_disc),
        "discovery_hungarian_acc_raw": acc_disc,
        "n_clusters_after_merge": int(n_merged_cls),
        "discovery_hungarian_acc_merged": acc_merged,
        "ncm_centroid_acc": acc_ncm,
        "ncm_memory_acc": acc_mem_new,
        "total_weighted": float(total),
    }
    (RESULTS / f"m7_openworld_{tag}_{mtag}.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

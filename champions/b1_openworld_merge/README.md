# Чемпион B1: открытый мир с онлайн-слиянием (ViT-B/14, merge_cos=0.45)

**Дата:** 2026-08-24
**Результат:** total_weighted = **88.40%** — рекорд трека открытого мира
(предыдущий: 87.38% без слияния на ViT-B; 81.42% ViT-S в коммите M6+M7).
Hungarian accuracy открытий: **85.89%** (+3.8pp к base).
**Артефакт:** `b1_openworld_dinov2b_topk8_m45.npz` (prototypes + labels +
counts; состояние памяти после базы и потока открытий).
Тяжёлые npz в git не входят (`*.npz` игнорируется) — воспроизводятся командой ниже.

## Команда воспроизведения

```bash
cd ~/Projects/309I
.venv/bin/python benchmarks/save_b1_champion.py        # модель + json
.venv/bin/python benchmarks/run_m7_openworld.py data/dinov2b_768.npz 0.45   # метрики
```

## Конфиг

```python
PrototypicalMemory(
    feature_dim=768,           # DINOv2 ViT-B/14 CLS, кэш data/dinov2b_768.npz
    threshold=0.0,
    topk=8,
    never_update=True,
)
# observe_batch(..., novelty_threshold=tau, min_cluster_size=3,
#               merge_cos=0.45)   # tau = квантиль FPR=5% по базе ≈ 0.364
```

## Механика онлайн-слияния

Когда новый кластер вырастает до `min_cluster_size`, его центроид сравнивается
с центроидами других выросших новых кластеров; при косинусе выше `merge_cos`
донор перевешивается на метку акцептора (прототипы не удаляются).

## Результаты (протокол M7: база 50 классов → открытие 50 без меток)

| Метрика | base | merge@0.45 |
|---|---:|---:|
| Hungarian accuracy открытий | 82.13% | **85.89%** |
| NCM centroid acc | 85.20% | **87.73%** |
| NCM memory acc | 76.45% | **79.68%** |
| total_weighted | 87.38% | **88.40%** |

Абляция порога (свип 0.40–0.60) — окно 0.40–0.50 стабильно лучше base,
чувствительность ±0.05 минимальна. Детали и свип по обоим энкодерам:
`BENCHMARK_LOG.md` §B1, `results/m7_openworld_*_base/_m45.json`.

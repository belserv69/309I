# Продолжение проекта 309I (после переноса на новый диск)

Этот файл лежит внутри папки проекта. Папку нужно скопировать **целиком**
(включая `.venv` и `data/*.npz` — они в `.gitignore`, в git их нет).

## Что это за проект

309I — онлайн (streaming) классификация изображений в открытом мире
(open-world), **без градиентного обучения**. Память-на-основе прототипов
поверх признаков DINOv2-B (768-d). Ключевые результаты:

| Этап | Метрика | Значение |
|---|---|---|
| M8 (distill, single-pass) | top-1 | **92.15%** |
| M9 (TTA, 5 видов, single-pass) | top-1 | **93.06%** |
| B1 (open-world, merge@0.45) | Hungarian / NCM | **88.40% / 92.15%** |
| B1b (repeat_cos, повторный показ) | NCM | **56.51% → 88.65%** (монотонно) |

Подробности — в `FINAL_REPORT.md` и `BENCHMARK_LOG.md` (последние коммиты
`0940e32`, `6930b63`, запушены в `belserv69/309I`).

## Что скопировать (важно!)

`git clone` **не хватает** — в репозитории нет тяжёлых артефактов:

1. **Весь каталог 309I** (код + этот файл).
2. **`data/`** (351 МБ, gitignored) — кэши признаков:
   `dinov2b_768.npz`, `dinov2b_768_aug4.npz`, `dinov2_384*.npz`,
   `mini100_84.npz`, `rn50_2048.npz`. Без них переизвлечение займёт ~55 мин.
3. **`.venv/`** (Python 3.11.15, зависимости numpy/faiss/scipy).
4. **`champions/`** — сохранённый чемпион B1 в
   `champions/b1_openworld_merge/` (`b1_champion.json` + `.npz` + README).

## Запуск после переноса

Если на новом диске домашний путь тот же (`/home/bulbach/...`), venv
заработает сразу (shebang абсолютный). Если путь другой — venv надо
пересоздать:

```bash
cd /путь/к/309I
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy faiss-cpu scipy
# (torch/timm нужны ТОЛЬКО для повторного extract; кэши уже готовы)
```

Проверка окружения:

```bash
.venv/bin/python -m pytest          # ожидается 26 passed
.venv/bin/python -c "import numpy, faiss, scipy; print('ok')"
```

## Ключевые команды

```bash
# Тесты
.venv/bin/python -m pytest

# Open-world (B1), merge@0.45
.venv/bin/python benchmarks/run_m7_openworld.py data/dinov2b_768.npz 0.45

# Distill (M8/M9)
.venv/bin/python benchmarks/run_m9_dinov2b.py

# TTA5: среднее по 5 видам (пишет data/dinov2b_768_tta5.npz, ~27 мин)
.venv/bin/python benchmarks/extract_tta5.py dinov2b_768.npz

# Чемпион B1 уже сохранён:
ls champions/b1_openworld_merge/
```

## Где что в коде

- `zf/memory/proto.py` — `PrototypicalMemory`: `merge_cos`, `repeat_cos`,
  `observe_batch` (параметры чемпиона B1: `threshold=0.0, topk=8,
  never_update=True`, `novelty_threshold≈0.364, min_cluster_size=3,
  merge_cos=0.45, repeat_cos=0.999`).
- `zf/memory/kmeans.py` — KMeans-кластеризация.
- `benchmarks/extract_tta5.py` — извлечение признаков + TTA-усреднение.
- `benchmarks/run_m7_openworld.py`, `run_m8_distill.py`, `run_m9_dinovb.py`
  принимают аргументы командной строки (см. argv в начале файлов).
- `save_b1_champion.py` — сохранение чемпиона.

## Что НЕ завершено (следующие шаги)

1. **TTA5 open-world** — прогон `extract_tta5.py` был прерван (готовы только
   виды 0–1 из 5), `data/dinov2b_768_tta5.npz` НЕ создан. Перезапустить
   фоном:
   ```bash
   nohup .venv/bin/python benchmarks/extract_tta5.py dinov2b_768.npz \
     > logs/extract_tta5_run2.log 2>&1 &
   # ~27 мин, затем:
   .venv/bin/python benchmarks/run_m9_dinov2b.py data/dinov2b_768_tta5.npz
   .venv/bin/python benchmarks/run_m7_openworld.py data/dinov2b_768_tta5.npz 0.45
   ```
2. **Replay/consolidation** — сейчас replay случайный; заменить на
   приоритетный (uncertainty / novelty).
3. **ViT-L энкодер** — сравнить DINOv2-B (768-d) vs ViT-L (1024-d) по NCM.
4. **Fine-tune энкодера** на in-domain (опционально, через базу знаний B1).

## Git / пуш

Репозиторий: `git@github.com:belserv69/309I.git` (SSH, ключ `id_rsa`,
перенесён в составе агента). Тяжёлые `*.npz` и `.venv` в `.gitignore` —
не коммитить.

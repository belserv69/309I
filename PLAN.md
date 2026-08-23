# ZF-Proto — пошаговый план

Выполнение отмечается чекбоксами; вердикт каждого milestone фиксируется в `BENCHMARK_LOG.md`
(русский вывод + таблица). Концепция и обоснования — в `CONCEPT.md`.

## Фаза 0 — каркас

- [x] Каталог `~/Projects/309I`, структура (`zf/`, `benchmarks/`, `champions/`, `best_model/`, `logs/`, `tests/`)
- [x] `git init` (ветка master)
- [x] `.venv` + зависимости: numpy, faiss-cpu, scikit-learn, pytest
- [x] Документ концепции `CONCEPT.md`
- [x] Этот план `PLAN.md`
- [x] `.gitignore` (venv, кэши данных, pkl, pycache)
- [x] Первоначальный коммит (`c87cba8`)
- [x] GitHub: репозиторий создан, remote + push (`https://github.com/belserv69/309I`, токен от OmniCore-EMS)

## M0 — порт ядра и smoke-тест (10 классов)

Цель: доказать корректность порта. Точность прототипов заметно выше случайной (≥ 60%),
без падений и NaN.

- [x] `data/rn50_2048.npz` — скопировать кэш RN50-признаков из CORAL
- [x] `zf/data.py` — загрузчик признаков по списку классов
- [x] `zf/memory/proto.py` — порт SOMA `PrototypicalMemory`:
  - [x] `never_update` (каждый семпл = замороженный прототип)
  - [x] `add_batch()` — O(N) добавление
  - [x] запрос per-class top-k через FAISS FlatIP + threshold-фильтр
  - [x] `_grow()` (авто-расширение ×1.5)
  - [x] exemplar ring (per-class FIFO)
- [x] `zf/pipeline.py` — CL-харнесс: фазы, train→eval, метрики (accuracy, per-class, forgetting)
- [x] `tests/test_memory.py`:
  - [x] `add_batch` ≡ последовательный `update` (режим never_update)
  - [x] прототипы L2-нормированы
  - [x] per-class top-k возвращает верный класс на синтетике
  - [x] `_grow` не теряет данные
- [x] `benchmarks/run_m0_10cls.py` — RN50, 10 классов, top-k=8, без аугментаций
- [x] Прогон + вердикт в `BENCHMARK_LOG.md` — **P@10cls = 95.67%** (цель ≥60%), 0.6 с, тесты 9/9
- [x] Git-коммит

**Definition of done:** P@10cls ≥ 60% (ориентир: CORAL proto-only = 64.1% на RN18 / 68.0% на RN50
при 100 классах; на 10 классах должно быть выше), тесты зелёные, коммит есть.

## M1 — 25 классов

Цель: проверка масштабирования перед сотней классов.

- [x] `benchmarks/run_m1_25cls.py`
- [x] Базовый прогон: конфигурация-победитель M0 на 25 классах — 90.53%
- [x] Абляция: top-k ∈ {4, 8, 16} — topk4 лучший (91.20%)
- [x] Абляция: threshold ∈ {0.0, 0.08, 0.2} — нет эффекта (косинусы RN50 в 0.4-0.6)
- [x] Абляция: exemplar ring вкл/выкл — +0.14pp, цена 5x время, отклонён
- [x] Абляция seniority bonus (M1b): sen0.08 → forgetting 4.58→3.19pp при +0.4pp точности
  (по пути найден и исправлен off-by-one в регистрации фаз классов в `add_batch`/`update`)
- [x] Диагностика забывания: confusion-пары (cls0→1/22, cls1→8, cls13→15) — структурное, не механическое
- [x] Лучшую конфигурацию → `champions/` — **topk4_sen08: P@25cls = 91.60%**, forgetting 3.19pp
      (`champions/m1_topk4_sen08.npz` + `.log`)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** P@25cls ≥ 80% (ориентир: CORAL RN18 probe — 91.4% P25 / 88.4% P@last),
чемпион сохранён.

## M2 — 100 классов (главная цель)

Цель: P@last ≥ 80% при forgetting ≈ 0.

- [x] `benchmarks/run_m2_100cls.py` — 10 фаз × 10 классов
- [x] Полный прогон чемпионной конфигурации M1 — **P@100cls = 80.03%** (цель ≥80% ✓)
- [x] Измерить: P@last, фазовая таблица, forgetting по классам, память, время — 7.5 с, 5000 прототипов
- [x] Oracle: линейный probe sklearn = 86.50% → зазор инкрементальности 6.5pp
- [x] M2b: seniority sweep — trade-off 81.10%/6.41pp ↔ 77.20%/0.37pp; cls0 не лечится seniority
- [x] Чемпион → `champions/m2_topk4_sen08_100cls` (.log + npz)
- [ ] Аугментация обучения 4×:
  - [x] torch CPU в `.venv` (2.13.0+cpu, зависимости починены)
  - [ ] `zf/encoders/resnet.py` — порт энкодера CORAL (RN50 GAP 2048d)
  - [ ] извлечение признаков аугментированных изображений → `data/rn50_2048_aug4.npz`
        (кэш с хэшем данных!)
  - [ ] M2c: прогон на 4× памяти, сравнение с чемпионом M2
- [ ] Опционально — TTA5 на тесте (усреднение по flip/shift-вариантам)
- [ ] Победителя → `champions/` + `best_model/`
- [ ] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** P@last ≥ 80% И forgetting ≤ 1pp — либо честный фиксированный вывод
с направлением пивота (CONCEPT §10).

## M3 — кросс-валидация порта (территория SOMA)

Цель: доказать, что код памяти faithful оригиналу SOMA.

- [x] Взять SOMA-признаки CIFAR-10 (кэш `feats_4992bd0d09.npz`, 250k×714d)
- [x] Прогнать память ZF-Proto в конфигурациях SOMA-чемпиона
      (`benchmarks/run_m3_soma_crossval.py`)
- [x] **Результат: 58.26% против якоря 58.02% — расхождение +0.24pp ≤ 1pp ✓**
      (t08=t0 — threshold не влияет, как на современных признаках;
       exemplar300 +0.11pp — согласуется с абляциями M1)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** расхождение с SOMA-якорем ≤ 1pp ✓ (+0.24pp)

## M4 — итоги и решение о будущем

- [x] `FINAL_REPORT.md`: сводная таблица против обоих родительских проектов
- [x] Решение: разрыв с probe закрыт (гибрид 91.03% ≥ probe-oracle) →
      фиксация двух production-конфигураций:
      **M6c** (максимальная точность, α=0.2) / **M5** (гарантированное
      удержание старых классов, cls0=0)
- [x] Финальный коммит

## M5 — энкодер CORAL-чемпиона: DINOv2 вместо RN50

Цель: поднять точность той же механикой памяти на более разделимых признаках
(CORAL probe на DINOv2 90.0% против 85% на RN50).

- [x] Порт энкодера: `zf/encoders/dinov2.py` (CLS ViT-S/14, бит-в-бит с CORAL)
- [x] Кэш `data/dinov2_384.npz` из CORAL + верификация портом
      (`benchmarks/verify_dinov2_cache.py`): cosine = 1.000000
- [x] Сетка top-k × seniority (`benchmarks/run_m5_dinov2_100cls.py`) —
      оптимум сместился к topk8 (в отличие от RN50 topk4)
- [x] Чемпион → `champions/m5_dinov2_topk8_sen0_100cls` (.log + npz) —
      **P@100cls = 88.40%**, cls0 forgetting = 0.00pp, зазор до oracle 1.7pp
- [x] Тесты энкодера (`tests/test_dinov2_encoder.py`, 6 шт., skip без torch)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** P@last > 80.03% (RN50-чемпион), верифицированный кэш,
тесты зелёные, чемпион сохранён.

## M6 — аугментации, геометрия и гибрид с probe: CORAL превзойдён

Цель: закрыть разрыв до CORAL-чемпиона (90.0%) и превзойти его.

- [x] Извлечение aug4-признаков DINOv2 (`benchmarks/extract_dinov2_aug4.py`),
      порт верифицирован (cosine=1.0); **багфикс:** метки aug-блока —
      `concat(y, repeat(y,3))`, не `tile` (рассинхрон обрушал точность до ~7%)
- [x] Сетка M6 (`run_m6_aug4_dinov2.py`): лучший 88.60% (mean, k16) —
      аугментации дают лишь +0.2pp, взвешенный top-k ("sim") хуже среднего
- [x] Геометрия M6b (`run_m6b_geometry.py`): z-score нормировка +0.2pp → 88.83%;
      плато чистой памяти ~88.8% при oracle ~91%
- [x] Гибрид M6c (`run_m6c_hybrid.py`, пивот CONCEPT §10): инкрементальный
      probe обучается ТОЛЬКО на прототипах памяти; фьюжн скор-матриц
      α·P_probe+(1−α)·S_mem → **91.03% при α=0.2** — CORAL (90.0%) превзойдён
- [x] `class_scores_batch()` в памяти (полная матрица скор классов)
- [x] Чемпион → `champions/m6c_hybrid_dinov2aug4_topk16_a02` (.log + npz)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** P@last ≥ CORAL 90.0% ✓ (91.03%)

## M7 — открытый мир: авто-добавление классов без меток

Цель: повторить и превзойти сквозной протокол CORAL 50/50
(база 83.7–85.3% + открытия 70.4–70.6% → итого 77–78%).

- [x] Порт `zf/novelty.py` (calibrate_threshold, evaluate_detection)
- [x] `observe_batch()` — unlabeled-поток: max-cosine < τ → новый класс;
      абсорбция (каждый вектор замораживается прототипом своего класса)
- [x] **Багфиксы калибровки:** τ только по невиданным векторам (train лежит
      в памяти → max-cosine=1.0, порог вырождается); AUROC при сигнале
      «ниже = новее» считается по −score
- [x] `run_m7_openworld.py`: база 50 классов 93.07% → открытия 50 классов
      без единой метки: Hungarian 71.59% (покрытие 81%), AUROC 0.906,
      NCM-обобщение 76.00%, пост-слияние фрагментов 93→58 кластеров
- [x] **Итог 81.4% против 77–78% у CORAL** ✓
- [x] Тесты открытого мира (`tests/test_openworld.py`, 10 шт.)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** total ≥ 77% ✓ (81.4%)

## M8 — дистилляция гибрида (LwF-lite)

- [x] Порт train_probe из CORAL (CE + KL по старым колонкам, T=2)
- [x] Багфикс выравнивания колонок probe под классы памяти
- [x] Сетка lam: lam≥1 полностью снимает cls0-дрейф (10pp→0pp);
      lam=1.0 — лучший баланс (89.20%, avg forg 0.37pp)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

## M9 — энкодер ViT-B/14 (768d)

- [x] Извлечение base + aug4 (`extract_dinov2b.py`), верификация cosine=1.0
- [x] Чистая память k8: **92.23%** (+3.8pp к ViT-S)
- [x] Гибрид α=0.2: **94.23%** (+3.2pp к ViT-S) — отрыв от CORAL +4.2pp
- [x] Чемпион → `champions/m9_hybrid_dinov2b_aug4_topk16_a02` (.log + npz)
- [x] Вердикт в `BENCHMARK_LOG.md`, коммит

**Definition of done:** P@last > 91.03% (ViT-S-чемпион) ✓ (94.23%)

## Правила на каждом шаге

1. Перед новым экспериментом сохранить текущего лидера в `champions/` (pkl на диске + log в git).
2. Подозрительно высокий результат → перепрогон на чистом кэше до цитирования.
3. Логи только в `logs/`, никогда в `/tmp`.
4. Коммит после каждого завершённого блока чекбоксов.

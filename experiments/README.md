# Эксперименты

Модели прогноза `target_1h` и их проверка на исторических срезах. Таблица результатов — [results.md](results.md), сырые логи — `logs/`.

## Протокол проверки

- Срез — 10:30 дня, прогноз на 8 шагов 11:00–14:30, как в test.
- **DEV** — 14 срезов с 11.10 по 24.10 2025. На них выбирается всё: признаки, лосс, число деревьев, веса ансамбля.
- **HOLDOUT** — 7 срезов с 25.10 по 31.10. Смотрится один раз на итоговом варианте. До сих пор не открывался.
- Метрика — WAPE + |Relative Bias| по 8000 строкам одного среза, затем среднее по срезам.
- Шум: изменения признаков меньше ~0.003, весов ансамбля меньше ~0.005 и между семействами моделей меньше ~0.011 неотличимы от случайности.

Утечка будущего проверена отдельно: если заменить данные после среза случайными числами, признаки не меняются.

## Окружение

Проверено на Python 3.13. Версии закреплены в `requirements.txt` этой папки.

```bash
python3.13 -m venv .venv-exp
```

```bash
.venv-exp/bin/pip install -r experiments/requirements.txt
```

## Запуск

Все команды из корня репозитория. Предсказания по срезам сохраняются в `experiments/preds/<tag>.npy` (в git не попадают).

Бустинг на DEV (конфиг — JSON, значения по умолчанию в начале `exp_gbm.py`):

```bash
.venv-exp/bin/python experiments/exp_gbm.py data/train_solo_track.parquet '{"tag":"V3_tw","feats":"v3","clip":true,"route_cat":false,"objective":"tweedie"}'
```

Сабмит лучшей конфигурации (обучение на всей истории):

```bash
.venv-exp/bin/python experiments/exp_gbm.py data/train_solo_track.parquet '{"feats":"v3","clip":true,"route_cat":false,"objective":"tweedie","rounds":[400],"final_rounds":400}' data/test_solo_track.parquet submissions/02_lgbm_v3_tweedie.csv
```

N-HiTS на DEV и HOLDOUT (21 срез; на Mac использует видеокарту, `accelerator='mps'` в коде — на другой машине заменить на `'cpu'` или `'gpu'`):

```bash
.venv-exp/bin/python experiments/exp_nhits.py data/train_solo_track.parquet cv '{"tag":"N8_huber_xh","loss":"huber","seed":1,"xh":true}'
```

Ансамбль по сохранённым предсказаниям (`тег:строка`, строка = номер столбца деревьев из `rounds`, 1 — первый):

```bash
.venv-exp/bin/python experiments/ensemble.py V3_tw:3 V3_xgb:1 V3_cat:1
```

Таблица результатов из логов: `python3 experiments/collect_results.py`.

## Файлы

| Файл | Что |
|---|---|
| `exp_gbm.py` | LightGBM / XGBoost / CatBoost, одна модель на 8 шагов, таргет — доля от недельного уровня маршрута, вес — уровень |
| `exp_nhits.py` | N-HiTS (neuralforecast), режимы `cv` и `final` |
| `ensemble.py` | веса и множитель суммы подбираются на DEV, затем один раз применяются к HOLDOUT |
| `collect_results.py` | сводка логов в `results.md` |
| `xh_check.py`, `decomp_check.py` | проверка разложения `target_1h` на получасовые потоки |
| `baseline_grid.py` | простые прогнозы «уровень × суточный профиль» |

## Наборы признаков в `exp_gbm.py`

- `v1` — лаги таргета, скользящие средние, статусы на момент среза, тот же слот вчера и неделю назад, общий объём по всем маршрутам.
- `v2` — плюс длинные лаги статусов, отношения статусов к своим средним, «долг» по статусам, субботний профиль маршрута, медиана слота.
- `v3` — плюс восстановленный получасовой поток `x` (`y_t = x_t + x_{t-1}`, границы из знакочередующихся сумм по окну 16), известная половина первого шага (`known`), глобальные аномалии статусов.

# RWB Shipment Forecasting

Решение кейса RWB «Отгрузки без простоев» для кейс-чемпионата Т1, 2026.

[Соревнование WB Space](https://wbspace.wb.ru/competitions/kejs-chempionat) · [Техническое задание](docs/task.md) · [Данные и исходные материалы](data/README.md)

## Задача

По истории 1000 маршрутов спрогнозировать `target_1h` на следующие четыре часа: восемь точек с шагом 30 минут. `target_1h` — объём отгрузок за час, заканчивающийся в соответствующий момент времени.

Метрика — **WAPE + |Relative Bias|**, меньше — лучше. Формат ответа — CSV `id,y_pred`, 8000 строк. Прикладная цель — поддержать решение о времени подачи машины на склад.

## Материалы

| Путь | Содержимое |
|---|---|
| `data/raw/` | Оригинальные train/test в Parquet |
| `data/reference/baseline.ipynb` | Неизменённый notebook организаторов |
| `data/reference/sample_submission.csv` | Пример формата результата |
| `data/manifest.json` | Исходные имена, размеры и SHA-256 |
| `docs/task.md` | Условия, схема данных, метрика и формат посылки |
| `docs/source/` | Конспект общих правил организаторов |
| `scripts/` | Проверка данных и построение обзорного графика |
| `notebooks/solution_v1.ipynb` | Решение с пояснениями и результатами: от данных до посылки `ensemble_v1.csv` (лидерборд 0.324) |
| `submissions/` | Генерируемые CSV для загрузки на платформу, исключены из Git |
| `models/` | Генерируемые модели, исключены из Git |

Train содержит 4 630 000 строк за 28.07.2025–01.11.2025 10:30. Test — 8000 строк за 01.11.2025 11:00–14:30. Времена не содержат часового пояса. Исходные данные включены в Git.

## Установка

Проверенное окружение — Python 3.14.6. Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/profile_data.py
```

Одна установка включает зависимости проекта, тестов и запуска notebook через JupyterLab. Аудит не изменяет исходники; его отчёты сохраняются локально в `.local/reports/data_audit/`.

## Решение

Код — в `src/rwb/`, запуск — из `scripts/`. Разбор с результатами — в `notebooks/solution_v1.ipynb` (тонкий ноутбук поверх этих модулей; тяжёлые шаги берут сохранённые предсказания из `models/oof/`, `RETRAIN = True` пересчитывает всё заново). Все признаки строятся только из данных до точки прогноза; это проверяет `tests/test_features.py`, подменяя будущее мусором.

- `data.py` — данные как матрицы «маршрут × время», метрика, сборка посылки.
- `features.py` — признаки: лаги, окна, тот же слот в прошлые дни/недели, статусы, время. Плюс восстановленные получасовые отгрузки `x`: `target_1h[t] = x[t] + x[t-1]`, поэтому `x` получается знакопеременной суммой с одной константой на маршрут, которую ограничивает `x ≥ 0`.
- `models.py` — бустинги (XGBoost GPU / LightGBM / CatBoost), прямая стратегия: модель на каждый горизонт учит поправку к сезонной базе в долях недельного уровня маршрута. Взвешенный L1 по этой поправке совпадает с числителем WAPE. Варианты цели `--target`: `x`/`x1` — получасовые отгрузки со склейкой в часы, `ratio` — неотрицательная доля уровня для Tweedie.
- `nn.py` — MLP или GRU (`--arch gru`) на GPU, батчи собираются прямо из тензоров панели.
- `backtest.py` — валидация как в тесте: точки прогноза в 10:30 за 14 прошлых дней (2 фолда по неделе), обучение только на метках до первой точки фолда.

```bash
python scripts/experiment.py --naive                                   # опорные прогнозы
python scripts/experiment.py --name xgb --lib xgb --min_child 500 --leaves 31 --final
python scripts/experiment.py --name huber --lib lgbm --objective huber --leaves 31 --min_child 1000 --rounds 400 --extra '{"alpha":1.0}' --final
python scripts/ensemble.py xgb huber --final --out ensemble          # веса и множитель подбираются перекрёстно по фолдам
```

Текущая посылка `ensemble_v2.csv` (лидерборд 0.323; v1 с тремя моделями — 0.324). Модели обучаются с `--final`, затем seed'ы усредняются и собирается ансамбль:

```bash
for s in 0 1 2; do python scripts/experiment.py --name xgb_lr_s$s --seed $s --lib xgb --days 28 --stride 2 --rounds 900 --lr 0.02 --leaves 63 --min_child 200 --final; done
python scripts/experiment.py --name lgbm_huber --lib lgbm --days 28 --stride 4 --rounds 400 --leaves 31 --min_child 1000 --objective huber --extra '{"alpha":1.0}' --final
python scripts/experiment.py --name lgbm_tweedie --lib lgbm --days 28 --stride 4 --rounds 600 --leaves 31 --min_child 1000 --objective tweedie --target ratio --extra '{"tweedie_variance_power":1.5}' --final
for s in 0 1 2; do python scripts/experiment_nn.py --name mlp_big_s$s --seed $s --epochs 12 --days 84 --hidden 1024 --dropout 0.2 --recent 192 --final; done
python scripts/bag_oof.py xgb_lr_s0 xgb_lr_s1 xgb_lr_s2 --out xgb_lr_bag3
python scripts/bag_oof.py mlp_big_s0 mlp_big_s1 mlp_big_s2 --out mlp_bag3
python scripts/ensemble.py xgb_lr_bag3 lgbm_huber mlp_bag3 lgbm_tweedie --final --out ensemble_v2
python scripts/organizer_baseline.py                                  # baseline организаторов на тех же фолдах
```

Результаты прогонов пишутся в `models/results.jsonl`, предсказания фолдов — в `models/oof/`, посылки — в `submissions/`.

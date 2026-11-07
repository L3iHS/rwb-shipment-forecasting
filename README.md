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
| `notebooks/solution_v2.ipynb` | Версия 2: усреднение seed'ов и Tweedie, посылка `ensemble_v2.csv` (лидерборд 0.323) |
| `notebooks/solution_v3.ipynb` | Версия 3: сглаженные профили маршрутов, посылка `ensemble_v3.csv` (валидация 0.3249, лидерборд 0.339) |
| `notebooks/solution_service.ipynb` | Рабочее решение для бизнеса: прогноз с диапазонами, когда вызывать машину, контроль качества |
| `outputs/dispatch_board.html` | Пульт диспетчера на 01.11.2025 10:30, открывается в браузере без интернета |
| `reports/service/` | Результаты проверки сервиса на истории |
| `submissions/` | Генерируемые CSV для загрузки на платформу; в Git — только итоговые посылки версий |
| `models/` | Генерируемые модели; в Git — сохранённые предсказания моделей версий и пакет сервиса (см. ниже) |

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

Код — в `src/rwb/`, запуск — из `scripts/`. Разбор с результатами — в `notebooks/solution_v1.ipynb`, `solution_v2.ipynb` и `solution_v3.ipynb` (тонкий ноутбук поверх этих модулей; тяжёлые шаги берут сохранённые предсказания из `models/oof/`, `RETRAIN = True` пересчитывает всё заново). Все признаки строятся только из данных до точки прогноза; это проверяет `tests/test_features.py`, подменяя будущее мусором.

- `data.py` — данные как матрицы «маршрут × время», метрика, сборка посылки.
- `features.py` — признаки: лаги, окна, тот же слот в прошлые дни/недели, статусы, время. Плюс восстановленные получасовые отгрузки `x`: `target_1h[t] = x[t] + x[t-1]`, поэтому `x` получается знакопеременной суммой с одной константой на маршрут, которую ограничивает `x ≥ 0`. Дополнительные наборы признаков `--feat_sets`: `lowrank12` — профиль дня маршрута за 28 дней и его сглаживание SVD ранга 12 по всем маршрутам (главный прирост v3); `res`, `ratio`, `sdiff` — проверены, прироста не дали.
- `models.py` — бустинги (XGBoost GPU / LightGBM / CatBoost), прямая стратегия: модель на каждый горизонт учит поправку к сезонной базе в долях недельного уровня маршрута. Взвешенный L1 по этой поправке совпадает с числителем WAPE. Варианты цели `--target`: `x`/`x1` — получасовые отгрузки со склейкой в часы, `ratio` — неотрицательная доля уровня для Tweedie. `--base lowrank` — в сезонной базе 7-дневное среднее заменено сглаженным 28-дневным профилем.
- `nn.py` — MLP или GRU (`--arch gru`) на GPU, батчи собираются прямо из тензоров панели.
- `backtest.py` — валидация как в тесте: точки прогноза в 10:30 за 14 прошлых дней (2 фолда по неделе), обучение только на метках до первой точки фолда.

```bash
python scripts/experiment.py --naive                                   # опорные прогнозы
python scripts/experiment.py --name xgb --lib xgb --min_child 500 --leaves 31 --final
python scripts/experiment.py --name huber --lib lgbm --objective huber --leaves 31 --min_child 1000 --rounds 400 --extra '{"alpha":1.0}' --final
python scripts/ensemble.py xgb huber --final --out ensemble          # веса и множитель подбираются перекрёстно по фолдам
```

Финальная посылка соревнования — `ensemble_v2.csv` (лидерборд 0.323). У `ensemble_v3.csv` валидация лучше — 0.3249 (по 14 дням и перекрёстно) против 0.3268 / 0.3288 у v2, — но на лидерборде 0.339. По точкам версии почти одинаковы; сумма прогноза v3 на 1.6% ниже, а каждый 1% промаха по сумме дня добавляет к метрике 0.01. Большая часть разницы — подобранный множитель (1.0075 у v3 против ≈1.02 у v2). Симуляция деления строк на публичную и приватную части показывает, что такой разрыв почти наверняка сохранится и на приватной части (см. `solution_service.ipynb`).

Сборка v3. Профиль маршрута и в признаках, и в базе; доли фиксированы 0.8 / 0.1 / 0.1, подбирается только множитель:

```bash
XGB="--lib xgb --days 28 --stride 2 --rounds 900 --lr 0.02 --leaves 63 --min_child 200 --feat_sets lowrank12 --base lowrank --final"
for s in 0 1 2; do python scripts/experiment.py --name xgb_lr_blr_s$s --seed $s $XGB; done
python scripts/experiment.py --name lgbm_huber_blr --lib lgbm --days 28 --stride 4 --rounds 400 --leaves 31 --min_child 1000 --objective huber --feat_sets lowrank12 --base lowrank --extra '{"alpha":1.0}' --final
python scripts/experiment.py --name lgbm_tweedie_lr12 --lib lgbm --days 28 --stride 4 --rounds 600 --leaves 31 --min_child 1000 --objective tweedie --target ratio --feat_sets lowrank12 --extra '{"tweedie_variance_power":1.5}' --final
python scripts/bag_oof.py xgb_lr_blr_s0 xgb_lr_blr_s1 xgb_lr_blr_s2 --out xgb_blr_bag3
python scripts/ensemble.py xgb_blr_bag3 lgbm_huber_blr lgbm_tweedie_lr12 --shares 0.8,0.1,0.1 --final --out ensemble_v3
```

Посылка v2 (лидерборд 0.323; v1 с тремя моделями — 0.324):

```bash
XGB="--lib xgb --days 28 --stride 2 --rounds 900 --lr 0.02 --leaves 63 --min_child 200 --final"
python scripts/experiment.py --name xgb_lr $XGB                       # seed 0
for s in 1 2; do python scripts/experiment.py --name xgb_lr_s$s --seed $s $XGB; done
python scripts/experiment.py --name lgbm_huber --lib lgbm --days 28 --stride 4 --rounds 400 --leaves 31 --min_child 1000 --objective huber --extra '{"alpha":1.0}' --final
python scripts/experiment.py --name lgbm_tweedie --lib lgbm --days 28 --stride 4 --rounds 600 --leaves 31 --min_child 1000 --objective tweedie --target ratio --extra '{"tweedie_variance_power":1.5}' --final
MLP="--epochs 12 --days 84 --hidden 1024 --dropout 0.2 --recent 192 --final"
python scripts/experiment_nn.py --name mlp_big $MLP                   # seed 0
for s in 1 2; do python scripts/experiment_nn.py --name mlp_big_s$s --seed $s $MLP; done
python scripts/bag_oof.py xgb_lr xgb_lr_s1 xgb_lr_s2 --out xgb_lr_bag3
python scripts/bag_oof.py mlp_big mlp_big_s1 mlp_big_s2 --out mlp_bag3
python scripts/ensemble.py xgb_lr_bag3 lgbm_huber mlp_bag3 lgbm_tweedie --final --out ensemble_v2
python scripts/organizer_baseline.py                                  # baseline организаторов на тех же фолдах
```

Результаты прогонов пишутся в `models/results.jsonl`, предсказания фолдов — в `models/oof/`, посылки — в `submissions/`.

## Сервис для диспетчера

Цель кейса — решить, когда вызывать машину. Рабочее решение поэтому не ансамбль, а сервис из одной модели и трёх слоёв над ней. Разбор с графиками — в `notebooks/solution_service.ipynb`.

- `forecaster.py` — одна LightGBM (Huber) с профилями маршрутов: 8 моделей по шагам, только CPU, обучение ≈4 минуты, прогноз 1000 маршрутов за 2 секунды. Подбираемого множителя нет: на трёх неделях проверки оптимальный множитель для неё 1.00, средний балл 0.3284.
- `intervals.py` — диапазоны прогноза конформным методом: квантили относительной ошибки по шагу и размеру маршрута. На отложенной неделе в 80%-диапазон попало 79.9% фактов, в 90% — 91.1%.
- `dispatch.py` — правило вызова: следующая машина вызывается, когда груз у ворот плюс ожидаемый приход за время подачи покрывает ещё один кузов сверх уже вызванных. Симулятор проигрывает реальные отгрузки и считает простой машин, ожидание груза и недогруз.
- `monitor.py` — дневные метрики отдельно для точности и перекоса суммы, тревоги и запасной прогноз для маршрутов с пропусками данных.
- `service.py` — пакет сервиса: модель, таблицы диапазонов и параметры машин в одной папке.

Проверка на неделе 25–31 октября: модель обучена до её начала, прогноз из каждого получаса, все 1000 маршрутов. Потери на одну машину (простой + ожидание груза), минут:

| Подача машины | Вызов, когда кузов полон | Темп последних 2 ч | Как неделю назад | **Модель** |
|---|---|---|---|---|
| 30 мин | 30 | 19 | 18 | **16** |
| 1 ч | 60 | 28 | 23 | **20** |
| 2 ч | 120 | 41 | 30 | **26** |

Недогруженных машин при подаче 1 час: 6 в день у модели против 31–62 у простых правил. Вместимость машины (2 часа среднего потока маршрута), время подачи и максимальное ожидание (2 часа) в данных отсутствуют и заданы параметрами; модель лучше простых правил при всех проверенных сочетаниях.

```bash
python scripts/rolling.py --folds 2 --final   # прогнозы из каждого получаса двух недель и модель на всех данных
python scripts/service_backtest.py            # диапазоны, симуляция вызова машин -> reports/service/lean.json
python scripts/pipeline.py train              # пакет сервиса models/prod/service
python scripts/pipeline.py forecast --at "2025-11-01 10:30"   # outputs/forecast_*.csv (p10/p50/p90), trucks_*.csv
python scripts/pipeline.py monitor --daily    # метрики и тревоги
python scripts/pipeline.py report             # пульт outputs/dispatch_board.html
```

### Сохранённые предсказания

Обучение всех моделей одной версии занимает 1.5–2.5 часа на RTX 4060 Laptop, поэтому их результаты лежат в репозитории (остальное содержимое `models/` и `submissions/` по-прежнему исключено из Git):

- `models/oof/<имя>.npz` — прогнозы на 14 днях валидации (`pred_fold0/1`) и истинные значения (`truth_fold0/1`):
  - v1/v2: `xgb_lr`, `xgb_lr_s1`, `xgb_lr_s2`, `lgbm_huber`, `lgbm_tweedie`, `mlp_big`, `mlp_big_s1`, `mlp_big_s2` и усреднённые `xgb_lr_bag3`, `mlp_bag3`;
  - v3: `xgb_lr_blr_s0..s2`, усреднённый `xgb_blr_bag3`, `lgbm_huber_blr`, `lgbm_tweedie_lr12`;
  - проверка на третьей неделе (`pred_fold2`): `xgb_base3`, `xgb_lowrank3`, `xgb_blr3`, `lgbm_huber_blr3`;
- `models/oof/<имя>_test.npy` — их прогнозы на тест, массив маршрут × 8 шагов;
- `models/rolling/lean_fold0/1.npz` — прогнозы сервисной модели из каждого получаса двух недель (`origins`, `pred`);
- `models/prod/service/` — пакет сервиса: 8 моделей LightGBM, таблицы диапазонов, параметры машин;
- `submissions/ensemble_v1/v2/v3.*` — посылки и их коэффициенты;
- `models/results.jsonl` — журнал всех прогонов с конфигурациями и результатами.

С ними ноутбуки `solution_v1/v2/v3.ipynb` и `solution_service.ipynb` выполняются за минуту без обучения, а `pipeline.py forecast/monitor/report` работают сразу после клонирования. Чтобы проверить новую модель на тех же днях, достаточно обучить только её и смешать с готовыми:

```bash
python scripts/experiment.py --name my_model ... --final
python scripts/ensemble.py xgb_blr_bag3 lgbm_huber_blr lgbm_tweedie_lr12 my_model
```

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
| `scripts/` | Проверка данных, сравнение моделей, графики и создание CSV |
| `src/rwb_forecasting/` | Метрика, признаки, временные срезы, простые прогнозы, Ridge и бустинг |
| `tests/` | Проверка данных, метрики и защиты от будущих наблюдений |
| `results/baselines/` | Агрегированные воспроизводимые результаты сравнения |
| `results/ridge_features/` | Сравнение Ridge с последовательным добавлением признаков |
| `configs/comparison.json` | Зафиксированные конфигурации и даты сравнения моделей |
| `results/optimization/` | Результаты исследования и отдельной контрольной проверки |
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

## Временная валидация

```bash
python scripts/run_backtest.py --origin-count 7 --origin-time 10:30 --train-days 14 --alpha 4.0
python scripts/plot_backtest.py
python -m pytest -q
```

Сравниваются четыре простых прогноза и Ridge на семи исторических четырёхчасовых окнах без доступа моделей к будущим наблюдениям. Подробные выходные файлы сохраняются локально в `.local/runs/temporal_baselines/`.

Средний score исходного Ridge — **0,3558**, лучшего простого ориентира — **0,4424**. Это локальная проверка, не оценка платформы. [Исходный протокол и результаты](docs/validation.md).

## Ridge с дополнительными признаками

```bash
python scripts/run_backtest.py --ridge-features status lags rolling calendar --output .local/runs/ridge_features
python scripts/plot_backtest.py --run-dir .local/runs/ridge_features
```

На тех же семи окнах добавление лагов таргета, скользящих средних и календаря снизило средний score до **0,3389** — на 4,75% относительно исходного Ridge. [Состав признаков и сравнение](docs/ridge-features.md).

## Готовый CSV для платформы

```bash
python scripts/make_candidate_submission.py
```

Результат — локальный `submissions/hgb_direct_norm_advanced.csv` с колонками `id,y_pred`. CSV и файл с его параметрами исключены из Git. [Правила каталога submissions](submissions/README.md). Оценка реального теста до загрузки неизвестна; команда ничего не отправляет на платформу.

## Сравнение дополнительных моделей

После двух раундов и 12 проверенных конфигураций выбран нормализованный бустинг с отдельной моделью для каждого горизонта. Средний score снизился с **0,338864 до 0,334538** на исследовательских окнах. На семи дополнительных контрольных окнах результат — **0,335532 против 0,338221** у прежнего Ridge. Цель среднего score ниже 0,32 пока не достигнута.

```bash
python scripts/run_comparison.py --split research --output .local/reproduce/research
python scripts/run_comparison.py --split control --output .local/reproduce/control
```

Контроль не использовался для выбора модели. Это дополнительные исторические окна, расположенные раньше исследовательского периода; они не заменяют проверку на скрытом тесте платформы. [Протокол, признаки и результаты](docs/optimization.md).

Предыдущий CSV Ridge по-прежнему можно создать командой `python scripts/make_submission.py --feature-set calendar`.

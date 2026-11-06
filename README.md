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
| `src/rwb_forecasting/` | Метрика, временные срезы, простые прогнозы и Ridge |
| `tests/` | Проверка данных, метрики и защиты от будущих наблюдений |
| `results/baselines/` | Агрегированные воспроизводимые результаты сравнения |
| `results/ridge_features/` | Сравнение Ridge с последовательным добавлением признаков |
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
python scripts/make_submission.py --feature-set calendar
```

Результат — локальный `submissions/ridge_calendar.csv` с колонками `id,y_pred`; он подходит по формату для ручной загрузки на платформу. CSV и файл с его параметрами исключены из Git. [Правила каталога submissions](submissions/README.md). Оценка реального теста до загрузки неизвестна.

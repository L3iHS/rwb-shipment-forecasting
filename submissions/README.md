# Готовые файлы для платформы

В этот каталог сохраняются наши CSV с предсказаниями реального test. В лидерборд загружается **только CSV** с двумя колонками `id,y_pred` и ровно одной строкой на каждый `id` теста — 8000 строк для исходного набора.

Новый экспериментальный кандидат — **`v3_tweedie_h1_600.csv`**: локальный средний score **0,327937** на 14 окнах, выигрыш у `mae_share_mass25` на 11 датах. Новый контроль ещё не открыт, публичный результат неизвестен. Это воспроизведение модели Владо. [Описание и проверка](../docs/v3_tweedie_h1_600.md).

```bash
python scripts/run_v3.py submit --config configs/v3_tweedie_h1_600.json \
  --train data/raw/train_solo_track.parquet --test data/raw/test_solo_track.parquet \
  --output submissions/v3_tweedie_h1_600_reproduced.csv
```

Предыдущий экспериментальный кандидат — **`mae_share_mass25.csv`**: локальный средний score **0,332149** на 14 окнах, контроль на семи окнах — **0,343478** против **0,343437** у `latent_mass_blend50`. Улучшение на контроле не подтверждено, публичный результат неизвестен. [Результаты и ограничения](../docs/mae-shares.md).

```bash
python scripts/make_candidate_submission.py --config configs/mae_share_mass25.json
```

Предыдущий экспериментальный вариант — **`objective_scale1_mass50.csv`**, средний локальный score **0,332322**; публичный результат неизвестен. [Описание](../docs/weighted-mass.md).

```bash
python scripts/make_candidate_submission.py --config configs/objective_scale1_mass50.json
```

Лучший подтверждённый публичный результат — **`latent_mass_blend50.csv`**, score **0,330** по скриншоту участника от 07.11.2026. Локальный средний score — **0,332570** на 14 окнах; на семи дополнительных контрольных окнах — **0,343437** против **0,343906** у исходного гибрида. [Устройство модели, ограничения и результаты](../docs/mass-blend.md).

```bash
python scripts/make_candidate_submission.py --config configs/latent_mass_blend50.json
```

Предыдущий гибрид с публичным score **0,331** сохраняется отдельно:

```bash
python scripts/make_candidate_submission.py --config configs/round3.json
```

Команда создаёт `latent_ridge_h1_hybrid.csv` и `latent_ridge_h1_hybrid.metadata.json`. Принятый гибрид использует Ridge с восстановлением часового объёма из получасовых для первого горизонта и прежний бустинг для горизонтов 2–8. Обе модели обучаются заново на известных данных до последней точки train с параметрами из `configs/round3.json`. Значения test не используются для обучения; сопоставление прогнозов с `id` выполняется по маршруту и времени. [Третий раунд и контрольная проверка](../docs/round3.md).

На 14 исследовательских окнах средний локальный score — **0,333307**, на семи новых контрольных — **0,337542** против 0,339590 у прежнего бустинга. Это не оценка платформы; цель ниже 0,32 не достигнута.

Параметр `--config configs/round3.json` обязателен для принятой модели: CLI по умолчанию использует старый конфиг. Старый бустинг создаётся командой `python scripts/make_candidate_submission.py --config configs/comparison.json`: файлы `hgb_direct_norm_advanced.csv` и `.metadata.json`. [История первых двух раундов](../docs/optimization.md).

Предыдущий вариант доступен через `python scripts/make_submission.py --feature-set calendar`: файлы `ridge_calendar.csv` и `.metadata.json`, обучающее окно источников 14 дней.

CSV и метаданные исключены из Git. JSON содержит параметры, границы обучения и контрольные суммы; на платформу его загружать не нужно. Для нового файла используйте `--output`: существующие файлы команда не перезаписывает.

Файлы `predictions.csv` из локальных backtest-запусков относятся к историческим проверочным окнам. Они не подходят для отправки на платформу. Пример организаторов хранится отдельно в `data/reference/sample_submission.csv`.

Создание CSV само по себе не отправляет его на платформу и не даёт значения score реального теста.

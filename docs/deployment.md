# rwb.ochev.net: выпуск приложения

`main` — релизная ветка. Push или merge запускает [.github/workflows/web.yml](../.github/workflows/web.yml): проверки, сборку двух образов в GitHub Actions и выпуск фронтенда, FastAPI и сохранённой модели LightGBM. Обучение модели и конкурсные отправки в CI/CD не выполняются.

## Устройство

`HTTPS → существующий edge Caddy → rwb-web:8080 → /api/* → rwb-api:8000`.

- `rwb-web`: Caddy проекта, React/TypeScript/Vite, снимок `board.json`, шрифты и демо-CSV.
- `rwb-api`: FastAPI, модель `models/prod/service/`, история и служебные массивы, подготовленные при сборке образа. `/api/forecast` действительно пересчитывает прогноз по CSV/Parquet. `/api/agent` вызывает DeepSeek через RouterAI и функции над снимком пульта.
- Общий edge Caddy, его сертификаты, порты 80/443 и другие проекты при выпуске не меняются.

API соединён только с внутренней сетью приложения, без портов хоста. Он доверяет forwarded-заголовкам от проектного Caddy в этой сети; Caddy получает IP посетителя от edge. Внешний доступ API к RouterAI остаётся разрешён.

Оба контейнера работают от UID 10001, с read-only корнем, без capabilities и с `no-new-privileges`. У API один worker, лимит 640 МиБ, без swap, не больше одного расчёта файла одновременно. Рабочие файлы живут в отдельном Docker volume `api-scratch`, удаляются после запроса; остатки прерванных расчётов очищаются при старте.

Для сервера с 2 ГБ RAM массивы истории и накопленных сумм читаются через memory mapping. Суммы для загрузки считаются небольшими партиями теми же функциями проекта; равенство исходным массивам проверяет тест. Веса модели не менялись. Формат пакета приведён к `Service` schema 2 существующим `scripts/prepare_service_v2.py`, по архивным прогнозам, без обучения; происхождение и хеши — [provenance.json](../models/prod/service/provenance.json).

## Ключ RouterAI

Только на сервере: `/srv/rwb-shipment-forecasting/backend.env`, root:root, права 600:

```env
ROUTERAI_API_KEY=ключ_проекта
MODEL=deepseek/deepseek-v4.1-flash-1017
RATE_PER_10MIN=15
RATE_PER_HOUR=100
AGENT_CONCURRENCY=2
UPLOADS_PER_10MIN=10
UPLOAD_CONCURRENCY=1
```

Ключ не попадает в Git, образ, браузер или GitHub Secrets. Адрес API фиксирован: `https://routerai.ru/api/v1/chat/completions`. После изменения файла нужен новый выпуск или пересоздание только `api` из уже скачанного образа с актуальной Compose-конфигурацией. Обычный `docker restart` новые переменные не подхватывает.

Без ключа прогноз по файлу работает, а `/api/agent` отдаёт 503; интерфейс использует локальные ответы. `/api/health` показывает `model_ready`, `agent_configured` и SHA выпуска, без секретов и внутренних ошибок.

## Короткий CI/CD

Сборки API и web идут параллельно, каждая с лимитом пять минут; затем короткий job деплоя с лимитом две минуты:

1. Sparse checkout: фронтенд, бэкенд, `src`, рабочий пакет модели, `data/raw`, тесты и `deploy`; без ноутбуков, экспериментов и результатов обучения.
2. Две проверки ограниченной SSH-команды и синтаксиса скриптов.
3. Сборка тестового образа API. 12 проверок: ввод файла, равенство массивов, загрузка реальной модели, запрос прогноза и протокол RouterAI с подставным ответом, без платных вызовов. Контейнер проверяется с тем же лимитом памяти, что на сервере.
4. Три frontend-теста, TypeScript и Vite внутри сборки web-образа.
5. Для `main` — публикация приватных образов в GHCR с тегом точного SHA и проверка приватности. Используются кеши слоёв Docker.
6. SSH-команда `deploy <SHA>` скачивает образы, ждёт готовности API с загруженной моделью, обновляет web и проверяет публичный HTTPS, ассеты и реальный расчёт по демо-CSV.

Pull request проверяется без деплоя и серверных секретов. Ручной повтор: **Actions → App checks and deploy → Run workflow → main**. На сервере обычных команд больше не требуется.

Деплои сериализованы workflow и серверным `flock`. Устаревший SHA пропускается, чужие местные изменения в клоне останавливают выпуск. Ошибка скачивания оставляет старые контейнеры; ошибка запуска или проверки возвращает предыдущие образы API и web. Первый переход с одного web-контейнера при откате останавливает новый API и возвращает прежний web. Предыдущие образы сохраняются локально.

До подключения API первый frontend-выпуск занял 1:45, повторный 0:42 с очередью GitHub. Это исторические замеры прежнего workflow; полное приложение измеряется отдельно по Actions.

## Два SSH-ключа

- **Сервер → GitHub:** прежний read-only deploy key `rwb-ochev-net`, `/home/rwb/.ssh/rwb-ochev-net`. SSH alias `github-rwb`, origin `git@github-rwb:L3iHS/rwb-shipment-forecasting.git`.
- **GitHub Actions → сервер:** отдельный ключ `RWB_DEPLOY_SSH_KEY`, учётная запись `rwb-cd`. `authorized_keys` содержит `restrict,command="/usr/local/libexec/rwb-deploy-entry"`; shell, PTY и forwarding запрещены. Root-owned `/usr/local/sbin/rwb-deploy-main` принимает только один полный SHA, сверяет его с `origin/main` и делает fast-forward.

Variables GitHub: `RWB_DEPLOY_HOST=rwb.ochev.net`, `RWB_DEPLOY_ENABLED=true` после установки. Host key сервера закреплён в [known_hosts](../deploy/known_hosts), проверка обязательна.

Для скачивания приватных образов runner передаёт краткоживущий `GITHUB_TOKEN` по защищённому SSH-вводу. Docker credentials создаются в root-only временном каталоге `/run` и удаляются после выпуска; постоянный PAT на сервере не нужен. Root-owned wrapper обновляется администратором при изменении самого протокола, а не произвольной командой через CI-ключ.

## Серверные файлы

- Клон: `/srv/rwb-shipment-forecasting/app`, владелец `rwb`, ветка `main`, чистое рабочее дерево. Для pull достаточно sparse checkout `frontend deploy docs`; модель и данные уже в образе.
- `/srv/rwb-shipment-forecasting/deploy.env`: `RWB_EDGE_NETWORK=remnawave-network`, опционально `RWB_BACKEND_ENV`; права 600, вне Git.
- `/srv/rwb-shipment-forecasting/deployment.json`: SHA и предыдущие образы последнего успешно проверенного выпуска; права 600.

Сборку и проверки выполняет GitHub, сервер запускает только готовые образы. Для локального воспроизведения API: `docker build -f deploy/backend.Dockerfile --target test -t rwb-api-test .`, затем `docker run --rm --memory 640m --memory-swap 640m rwb-api-test`. Тесты требуют writable `/tmp` и `/var/cache/rwb`; production-ограничения задаёт [compose.yml](../deploy/compose.yml).

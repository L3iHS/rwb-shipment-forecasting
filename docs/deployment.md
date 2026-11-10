# Демо на rwb.ochev.net

Публикуется согласованный пульт «Ночная смена»: React, TypeScript, Vite и Motion, сохранённый `board.json`, локальные ответы ассистента и журнал вызовов в браузере. Новый ML snapshot, backend и внешняя LLM не подключены. `/api` и `/api/*` возвращают 404.

## Автодеплой из main

После push или merge в `main` запускается [.github/workflows/web.yml](../.github/workflows/web.yml). Для pull request в `main` выполняются те же проверки, без доступа к серверу.

Один job на Ubuntu с Node 22:

1. Получить только `frontend/`, `deploy/` и `.github/`, без моделей и конкурсных данных.
2. Установить зависимости из lock-файла с npm cache.
3. Выполнить пять тестов расчётов и две короткие проверки протокола деплоя.
4. Проверить TypeScript и собрать Vite.
5. Для `main` вызвать ограниченную серверную команду с SHA именно проверенного коммита.
6. Дождаться готовности контейнера и сверить публичный HTTPS, снимок и ассеты.

ML-тесты, обучение, матрица версий и установка Python ML-зависимостей не запускаются. Цель — меньше пяти минут от старта job; фактическое время нужно смотреть в первом выполненном workflow. Лимит job — пять минут. Повторные запуски используют npm cache и слои Docker.

Одновременные деплои исключены: workflow `main` не отменяет уже начавшийся выпуск, сервер дополнительно использует `flock`. Если проверенный SHA уже уступил новому коммиту `main`, старый выпуск пропускается. При ошибке сборки сайт продолжает работать на старом образе; при ошибке запуска или проверки возвращается предыдущий образ. Обновляется только `rwb-web`.

## Доступ GitHub к серверу

Отдельная учётная запись `rwb-cd` не входит в группу Docker. SSH-ключ разрешает только `deploy <SHA>` через root-owned [ssh-entry.sh](../deploy/ssh-entry.sh) и [main-release.sh](../deploy/main-release.sh). Обычный shell, PTY и forwarding запрещены. Привилегированная команда сверяет SHA с актуальной `origin/main`, проверяет чистоту клона и делает только fast-forward. Серверный ключ для чтения GitHub остаётся прежним, `rwb-ochev-net`.

В GitHub настроены:

- Variable `RWB_DEPLOY_HOST`: адрес существующего сервера.
- Variable `RWB_DEPLOY_ENABLED`: `true` после активации доступа; до этого выполняются только проверки.
- Secret `RWB_DEPLOY_SSH_KEY`: отдельный приватный ключ CI, доступный только шагу CD, не pull request.

Проверенный публичный host key сервера закреплён в [known_hosts](../deploy/known_hosts), с обязательным `StrictHostKeyChecking`. Он прочитан через существующий административный сеанс, не принят на доверие из сетевого ответа.

Приватный ключ не хранится в Git, образе или логах. На runner он создаётся во временном каталоге и удаляется после шага. На сервере root-owned копии entry points находятся в `/usr/local/libexec/rwb-deploy-entry` и `/usr/local/sbin/rwb-deploy-main`; sudo разрешает `rwb-cd` только вторую команду. Первичная установка и активация нового ключа выполняются администратором отдельно.

## Ручной запуск и восстановление

Обычные изменения: push в `main`, без команд на сервере. Повторить выпуск можно через **Actions → Web checks and deploy → Run workflow**, выбрав `main`.

При недоступности Actions администратор может выполнить:

```sh
cd /srv/rwb-shipment-forecasting/app && runuser -u rwb -- git pull --ff-only
bash deploy/update.sh
```

[update.sh](../deploy/update.sh) проверяет ветку `main` и отсутствие местных изменений, собирает образ с тегом текущего SHA, обновляет только `web` и ждёт его готовности. Не запускать ручной выпуск параллельно с Actions. Настройки находятся в `/srv/rwb-shipment-forecasting/deploy.env`, подтверждённый SHA и предыдущий релиз — в `/srv/rwb-shipment-forecasting/deployment.json`.

## Репозиторий и доступ

Источник — `L3iHS/rwb-shipment-forecasting`, релизная ветка `main`; разработка остаётся в `develop`. На Ubuntu используется отдельный пользователь `rwb` с домашним каталогом `/home/rwb` и shell `/usr/sbin/nologin`, без входящего SSH-доступа. Репозиторий размещается в `/srv/rwb-shipment-forecasting/app`; Docker-команды выполняет администратор, отдельное членство `rwb` в группе Docker не требуется.

Ключ Ed25519 генерируется **только на сервере** в `/home/rwb/.ssh/rwb-ochev-net`. В GitHub добавляется только `.pub` как deploy key с названием `rwb-ochev-net`, без права записи. Приватная часть остаётся на сервере и не попадает в вывод команд, Git или образ. Каталог `.ssh` — 700, ключ и config — 600, владелец `rwb`.

Настройка `/home/rwb/.ssh/config`:

```sshconfig
Host github-rwb
    HostName github.com
    User git
    IdentityFile ~/.ssh/rwb-ochev-net
    IdentitiesOnly yes
```

Ключ хоста GitHub сверяется с [официальными отпечатками](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints). URL клонирования для пользователя `rwb`: `git@github-rwb:L3iHS/rwb-shipment-forecasting.git`.

Для этого демо достаточно неглубокого sparse-клона: `git clone --depth 1 --filter=blob:none --sparse --branch main URL /srv/rwb-shipment-forecasting/app`, затем `git -C /srv/rwb-shipment-forecasting/app sparse-checkout set frontend deploy docs`. Обе команды выполняются от `rwb`; модели и конкурсные данные для сборки не нужны.

## Архитектура и контейнер приложения

`HTTPS → общий edge Caddy (TLS, 80/443) → rwb-web:8080 (Caddy проекта) → собранный React`.

Проектный контейнер на официальном `caddy:2-alpine` раздаёт статику из `/srv` по HTTP на 8080; admin API и сохранение конфигурации отключены. TLS остаётся у отдельного общего edge Caddy.

Сборка выполняется из корня репозитория. `.dockerignore` должен разрешать содержимое `frontend/` и `deploy/`, исключая `node_modules`, `dist`, `.env` и локальные рабочие файлы.

Перед запуском определить фактическую Docker-сеть, доступную существующему Caddy, либо создать отдельную согласованную сеть и подключить к ней Caddy без пересоздания контейнера. Название нельзя подставлять по догадке. Скопировать `deploy/.env.example` в `/srv/rwb-shipment-forecasting/deploy.env` вне клона (права 600), заполнить `RWB_EDGE_NETWORK` и указать SHA развёртываемого коммита в `RWB_IMAGE_TAG`.

Из `/srv/rwb-shipment-forecasting/app`:

```sh
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml config --quiet
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml up -d --build --wait web
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml ps
```

Проект Compose — `rwb-shipment`, контейнер — `rwb-web`. Он слушает 8080 только в Docker-сети, без опубликованных портов хоста. Caddy проекта работает как UID/GID 10001, корневая файловая система read-only, рабочие данные и конфигурация направлены в непостоянный `/tmp`; лимит памяти 128 MiB. `/healthz` возвращает `200 ok`.

## Общий edge Caddy

Существующий Caddy уже обслуживает 80/443 и сертификат домена. Его правило сохраняется без изменений:

```caddyfile
rwb.ochev.net {
    reverse_proxy rwb-web:8080
}
```

При обновлении проекта не менять и не пересоздавать edge Caddy, его сеть, сертификаты или другие сервисы. `RWB_EDGE_NETWORK` указывает фактическую общую сеть. Новый Caddy внутри `rwb-web` не публикует порты хоста и не получает доступ к сертификатам edge.

## Проверка и обновления

После запуска проверить HTTPS `https://rwb.ochev.net/healthz`, загрузку интерфейса и ассетов. У корня, `index.html` и `board.json` — `Cache-Control: no-cache`, у существующих `/assets/*` — `immutable`; отсутствующий ассет и `/api/*` возвращают 404. Обычные клиентские маршруты получают SPA HTML. Порядок `route`/`handle` в [Caddyfile](../deploy/Caddyfile) изолирует 404 от SPA fallback; это соответствует [правилам Caddy](https://caddyserver.com/docs/caddyfile/directives/route).

Для обычных обновлений используется push в `main` и автоматический workflow. При необходимости ручного отката администратор задаёт `RWB_IMAGE_TAG` предыдущего сохранённого образа и выполняет Compose `up -d --no-build --no-deps --wait web`. Скрипт не удаляет старые образы. Данные и остальные сервисы этот демо-контейнер не изменяет.

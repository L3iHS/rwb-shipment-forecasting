# Демо на rwb.ochev.net

Публикуется только фронтенд выбранного направления A: React, TypeScript, Vite и Motion. Данные и ответы ассистента условные; ML snapshot, backend и внешняя LLM не подключены. `/api` возвращает 404. Старый `board.json` не включается в сборку (`publicDir: false`).

## Репозиторий и доступ

Источник — `L3iHS/rwb-shipment-forecasting`, ветка `develop`. На Ubuntu используется отдельный пользователь `rwb` с домашним каталогом `/home/rwb` и shell `/usr/sbin/nologin`, без входящего SSH-доступа. Репозиторий размещается в `/srv/rwb-shipment-forecasting/app`; Docker-команды выполняет администратор, отдельное членство `rwb` в группе Docker не требуется.

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

Для этого демо достаточно неглубокого sparse-клона: `git clone --depth 1 --filter=blob:none --sparse --branch develop URL /srv/rwb-shipment-forecasting/app`, затем `git -C /srv/rwb-shipment-forecasting/app sparse-checkout set frontend deploy docs`. Обе команды выполняются от `rwb`; модели и конкурсные данные для сборки не нужны.

## Контейнер приложения

Сборка выполняется из корня репозитория. `.dockerignore` должен разрешать содержимое `frontend/` и `deploy/`, исключая `node_modules`, `dist`, `.env` и локальные рабочие файлы.

Перед запуском определить фактическую Docker-сеть, доступную существующему Caddy, либо создать отдельную согласованную сеть и подключить к ней Caddy без пересоздания контейнера. Название нельзя подставлять по догадке. Скопировать `deploy/.env.example` в `/srv/rwb-shipment-forecasting/deploy.env` вне клона (права 600), заполнить `RWB_EDGE_NETWORK` и указать SHA развёртываемого коммита в `RWB_IMAGE_TAG`.

Из `/srv/rwb-shipment-forecasting/app`:

```sh
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml config --quiet
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml up -d --build --wait web
docker compose --env-file /srv/rwb-shipment-forecasting/deploy.env -f deploy/compose.yml ps
```

Проект Compose — `rwb-shipment`, контейнер — `rwb-web`. Он слушает 8080 только в Docker-сети, без опубликованных портов хоста. Nginx работает как `nginx`, корневая файловая система read-only, запись разрешена в `/tmp`; лимит памяти 128 MiB. `/healthz` возвращает `200 ok`.

## Существующий Caddy

Публичные 80/443 уже заняты Caddy 2.9. Определить его имя, путь смонтированного конфига и Docker-сети через inspect. Остальные сервисы и их Compose не перезапускать.

Подключить работающий Caddy к сети `RWB_EDGE_NETWORK`, если он ещё не подключён. Конфиг подключён к контейнеру через single-file bind mount: после резервного копирования редактировать содержимое существующего файла на месте. Не заменять файл атомарным rename, иначе контейнер продолжит видеть прежний inode. В фактический Caddyfile добавить отдельный блок, сохранив остальные:

```caddyfile
rwb.ochev.net {
    reverse_proxy rwb-web:8080
}
```

DNS домена должен указывать на этот сервер. Перед применением сохранить предыдущий Caddyfile, выполнить `caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile` внутри существующего контейнера, затем `caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile`. Это [плавное обновление конфигурации](https://caddyserver.com/docs/command-line), без остановки или пересоздания общего Caddy.

## Проверка и обновления

После запуска проверить HTTPS `https://rwb.ochev.net/healthz`, загрузку интерфейса и ассетов. У `index.html` — `Cache-Control: no-cache`, у существующих `/assets/*` — `immutable`; отсутствующий ассет и `/api/*` должны возвращать 404, а не SPA HTML.

Обновление: получить согласованный коммит `develop` от пользователя `rwb`, повторить Compose build/up только для `web`, проверить health и HTTPS. Для отката собрать ранее зафиксированный коммит и повторить тот же запуск; данные и остальные сервисы этот демо-контейнер не изменяет.

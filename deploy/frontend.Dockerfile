FROM node:22-alpine AS build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM caddy:2-alpine AS runtime

RUN addgroup -S -g 10001 rwb && adduser -S -D -H -u 10001 -G rwb rwb
ENV XDG_CONFIG_HOME=/tmp/config \
    XDG_DATA_HOME=/tmp/data

COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY --from=build --chown=10001:10001 /app/frontend/dist/ /srv/

USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["caddy"]
CMD ["run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]

# Изменения ДЗ5 относительно предоставленного архива 0.3.0

Коммит и push не выполнялись. Ни один файл не добавлен в staging: это рабочая
копия из ZIP без `.git`. Список ниже — состав изменений для переноса в Git-репозиторий.

## Добавлены (26)

- `docs/CHANGES-DZ5.md`
- `docs/DZ5.md`
- `examples/example.jsonl`
- `observability/compose.yaml`
- `observability/grafana/dashboards/pipeline.json`
- `observability/grafana/provisioning/alerting/alerts.yaml`
- `observability/grafana/provisioning/dashboards/dashboards.yaml`
- `observability/grafana/provisioning/datasources/datasources.yaml`
- `observability/log-agent/config.alloy`
- `observability/loki.yaml`
- `observability/otel-collector.yaml`
- `observability/probe/Dockerfile`
- `observability/prometheus.yaml`
- `observability/tempo.yaml`
- `rabbitmq/enabled_plugins`
- `requirements-pipeline.txt`
- `scripts/load.py`
- `src/mlops/api/v1/tasks.py`
- `src/mlops/broker.py`
- `src/mlops/processing.py`
- `src/mlops/source.py`
- `src/mlops/tasks.py`
- `src/mlops/telemetry.py`
- `src/mlops/worker.py`
- `tests/test_pipeline.py`
- `tests/test_tasks.py`

## Изменены (13)

- `.dockerignore`
- `.env.example`
- `.github/workflows/ci.yml`
- `.gitignore`
- `Dockerfile`
- `README.md`
- `docker-compose.yml`
- `src/mlops/__main__.py`
- `src/mlops/api/process.py`
- `src/mlops/api/v1/__init__.py`
- `src/mlops/config.py`
- `src/mlops/logging.py`
- `src/mlops/main.py`

## Проверка

8 unit-тестов обработки/контекста/логирования прошли. Проверены синтаксис Python,
YAML/JSON и структура конфигурации, наличие bind-файлов, профили и лимиты памяти.
API-тесты добавлены, но полный pytest, Ruff, сборка Docker, живой RabbitMQ, Grafana
и замеры docker stats не запускались: Docker отсутствует, новые пакеты недоступны
из-за DNS. На Windows default asyncio loop также упирается в недоступный loopback;
для этих unit-тестов использован временный timer-only loop, транспорт/БД замоканы.
Это не проверка сетевого стека.

`uv.lock` и `pyproject.toml` сохранены: дополнения установлены отдельным
`requirements-pipeline.txt`, новый общий lockfile пока не разрешён.

В коммит не входят `.env`, `.uv-cache`, `__pycache__`, большие сгенерированные
JSONL-файлы нагрузки. Старая папка `../MLOps2026-main` не изменена.

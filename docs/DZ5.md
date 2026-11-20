# ДЗ5. Наблюдаемость конвейера

В исходном архиве был синхронный сервис ДЗ3. Добавлена необходимая для ДЗ5 основа
ДЗ4: потоковый JSONL-источник, ограниченная очередь RabbitMQ, FastStream-воркер,
API постановки задач и таблица результатов PostgreSQL. Старый `/process` сохранён.

## Запуск

Нужны Docker Engine и Compose **2.24.4+** (используется `include`). Из корня проекта:

```bash
cp .env.example .env
docker compose up -d --build
```

Задайте свои `POSTGRES_PASSWORD`, `RABBITMQ_PASSWORD` и `GRAFANA_PASSWORD` в `.env`.
Для пароля RabbitMQ используйте буквы и цифры: он подставляется в AMQP URL.
Доступы стенда локальные; это не схема для публикации сервисов в интернете.

Обычный запуск поднимает PostgreSQL, RabbitMQ, MLflow, обучение, API, воркер и источник.
Обучение завершается до старта API/воркера. Источник отправляет одну демонстрационную
строку и завершается; большие файлы запускаются отдельным `compose run`.
При повторном `up` завершённый train может выполниться снова и создать новые версии
модели. Для чистого сравнения масштабируйте воркер с `--no-deps`.

По умолчанию `OTEL_SDK_DISABLED=true`: приложение работает без Collector.
Для трейсов/метрик установите `OTEL_SDK_DISABLED=false` в `.env`, затем:

```bash
docker compose --profile logs up -d --build
docker compose --profile traces up -d --build --force-recreate app worker
docker compose --profile metrics up -d --build --force-recreate app worker
docker compose --profile observability up -d --build
```

Если приложение уже запущено с отключённым SDK, после изменения `.env` выполните
`docker compose up -d --force-recreate app worker`. Профили включают только свои
сигналы; для Grafana и всего стенда используется `observability`. При `traces` без
`metrics` Collector принимает метрики, но Prometheus их не сохраняет; при `metrics`
без `traces` экспорт трейсов в отсутствующий Tempo ограничен очередью и таймаутом.

| Интерфейс | Адрес |
| --- | --- |
| API | http://127.0.0.1:8000/docs |
| MLflow | http://127.0.0.1:5000 |
| RabbitMQ | http://127.0.0.1:15672 |
| Grafana | http://127.0.0.1:3000/d/pipeline |

Grafana: пользователь `admin`, пароль из `GRAFANA_PASSWORD`.
Datasources, дашборд «Конвейер» и два правила алертов provisioned из репозитория.
Они восстанавливаются после удаления томов. `down -v` удаляет также БД и модели;
это команда только для чистого учебного стенда:

```bash
docker compose --profile observability down -v
docker compose --profile observability up -d --build
```

## Контракт задач и доставка

`POST /api/v1/tasks` принимает тот же JSON, что `/process`:

```json
{"samples":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}
```

Ответ `202` содержит `task_id`. `GET /api/v1/tasks/{task_id}` возвращает результат;
`404` означает, что результата ещё нет. При переполнении/недоступности брокера API
отвечает `503` с `Retry-After: 1`. В логах этой попытки есть task_id и trace_id.

Очередь `tasks` — durable quorum queue, `x-max-length=QUEUE_LIMIT`,
`x-overflow=reject-publish`. Publisher confirms включены. Источник читает по строке,
повторяет публикацию с тем же task_id после nack и ждёт `SOURCE_RETRY_DELAY`.
Очередь ограничивает ready-сообщения, unacked считаются отдельно. У quorum queue
возможен небольшой выход за лимит из-за публикаций в полёте; это не строгий семафор.

У воркера prefetch=1. Модель загружается по alias один раз при старте; в логах есть
alias и разрешённая версия. `INFERENCE_DELAY=0.1` делает противодавление заметным
на маленьком Iris; само предсказание выполняется в `asyncio.to_thread`.

Успех подтверждается только после коммита результата. `task_id` — первичный ключ,
`ON CONFLICT DO NOTHING` делает повторную запись безопасной. При ошибке — nack
с requeue. Номер попытки берётся из `x-delivery-count` RabbitMQ 4.0, а не из памяти
процесса. На `MAX_ATTEMPTS` исходное сообщение публикуется в `tasks.dlq` с confirm,
после этого подтверждается оригинал. При сбое публикации DLQ оригинал не ack-ается.
Дополнительный broker delivery limit и at-least-once dead-lettering защищают от
бесконечных циклов при падениях процесса. Доставка at-least-once: повторный inference
возможен, дубли результатов в БД не создаются.

FastStream при SIGTERM прекращает приём и ждёт обработчики до 60 секунд;
Compose даёт 75 секунд до SIGKILL. События начала/конца shutdown пишутся в stdout.
Если обработка не успела завершиться, unacked-сообщение доставится снова.

## Логи и трейсы

Использован существующий stdlib `logging`: он уже есть в проекте, не требует ещё
одной библиотеки и приводит библиотечные логи к тому же JSON. Формат содержит время,
уровень, logger, service, env; внутри задачи автоматически добавляется task_id через
`ContextVar`, внутри спана — trace_id/span_id. Пароли в URL редактируются. Сообщения
исключений обработчика заменяются на имя класса с сохранением traceback: Pydantic
ValidationError иначе раскрывает входную запись. Полные строки файла не логируются.
Успешные health-check-и пишутся на DEBUG, ошибки — WARNING.

Alloy обнаруживает только контейнеры `com.docker.compose.project=mlops2026`
(либо значение `COMPOSE_PROJECT_NAME`), получает docker.sock с `:ro` и отправляет
stdout в Loki. `:ro` относится к монтированию; сам Docker API остаётся чувствительным
интерфейсом. Alloy не делает управляющих запросов. Лейблы потоков: только service,
level, env. task_id/trace_id — поля JSON, а не отдельные потоки на каждую задачу.

OpenTelemetry SDK создаёт серверный span API, publish и consumer process вручную.
Контекст W3C `traceparent` передаётся в заголовках RabbitMQ, task_id — в correlation_id
и заголовке. Все попытки обработки остаются в том же trace_id. Дочерние спаны:
inference, db.write_result, source.wait. `asyncio.to_thread` явно копирует contextvars
в поток; тест проверяет родителя inference. process хранит время ожидания с момента
публикации (`messaging.queue.wait_seconds`), попытку, события retry/DLQ; ошибки имеют
статус ERROR и exception event. Для 503 серверный span тоже получает ERROR.

Маршрут метрик — шаблон FastAPI (`/tasks/{identifier}`), а не уникальный URL.
Ресурсы задаются `OTEL_SERVICE_NAME`, `OTEL_SERVICE_VERSION`,
`OTEL_DEPLOYMENT_ENVIRONMENT`, `OTEL_SERVICE_INSTANCE_ID`, `OTEL_RESOURCE_ATTRIBUTES`.
При отсутствии instance ID используется hostname контейнера: два воркера различаются.
Адрес/таймаут экспорта задаются `OTEL_EXPORTER_OTLP_*`, sampler — `OTEL_TRACES_SAMPLER`.
Для стенда `parentbased_always_on`: сохраняем все самостоятельные задачи, а входящий
sampling flag уважаем. Для продакшена: parentbased_traceidratio + tail sampling на
Collector для ошибок/медленных запросов; head sampling не позволяет вернуть уже
отброшенные спаны, поэтому при обязательном сохранении всех ошибок лучше отправлять
в Collector все спаны и выбирать там, учитывая память и трафик.

SDK экспортирует в фоновых потоках с ограниченной очередью (1024 спана) и таймаутом
1 с. Приложение не зависит от Collector через depends_on. При его остановке задача
продолжает обрабатываться, а часть телеметрии после заполнения очередей теряется.
Надёжную доставку телеметрии здесь не обещаем; для повседневной разработки SDK
полностью отключается. В Collector memory_limiter стоит до batch: сначала ограничить
память, потом накапливать пакет. Очередь экспорта и время повторов тоже ограничены.

В Grafana поле TraceID записи Loki открывает Tempo; в Tempo настроен обратный поиск
логов по trace_id. В Explore можно искать `{env="dev"} | json | task_id="..."`.
Exemplars не добавлены: это дополнительный пункт, переходы лог ↔ трейс реализованы.

## Метрики и кардинальность

API: `pipeline_api_requests_total{route,status}` и гистограмма ответа.
Источник: `pipeline_source_published_total`, `pipeline_source_rejected_total`,
`pipeline_source_wait_seconds_total`. Воркер: `pipeline_worker_processed_total{result}`,
`pipeline_worker_inflight`, `pipeline_worker_redeliveries_total`, гистограммы inference,
ожидания в очереди и полного времени успешной задачи от публикации до ack
(`pipeline_task_duration_seconds`). Runtime RSS — `pipeline_process_memory_bytes` из psutil.

Приложения **push**-ят OTLP: короткоживущий источник делает финальный flush, поэтому
не обязан доживать до scrape. Collector сохраняет последние cumulative-значения до
24 часов, а Prometheus их читает. Для файла из одной строки rate может быть нулевым:
он успел закончиться до первого scrape; итоговый counter всё равно виден. Для rate
нужен нагрузочный файл. Детальные размеры `tasks` и DLQ, unacked берутся прямо из
`rabbitmq_prometheus`, `/metrics/detailed?family=queue_coarse_metrics`; приложение
не пытается вычислить длину очереди самостоятельно.

Бакеты inference/API зависят от `INFERENCE_DELAY`: 0.5×, 1×, 2×, 5×, 10× плюс
0.005, 0.01, 0.025, 0.5 и 30 секунд. При 0.1 с — 9 конечных границ. Значение 0.5 с
зафиксировано для SLO; хвосты выше последней границы попадают в +Inf.
Для B конечных границ одна комбинация лейблов даёт B+3 ряда гистограммы.
Для 2 воркеров inference: 2 × (9+3) = 24 рядов; счётчик исходов: 2 × 3 = 6.
Для API с 10 маршрутами и 6 кодами: максимум 10 × 6 = 60 рядов счётчика,
гистограмма без status: 10 × 12 = 120. Это оценки верхних границ, не замер.
Instance ID добавляет множитель числа экземпляров; перезапуски тоже создают новые
ряды, которые удаляются по retention. task_id добавил бы ещё множитель 6000 на прогон.

Примеры PromQL (совпадают с дашбордом):

```promql
sum(rate(pipeline_source_published_total[1m]))
sum by(service_instance_id)(rate(pipeline_worker_processed_total{result="ok"}[1m]))
histogram_quantile(0.95, sum by(le)(rate(pipeline_inference_duration_seconds_bucket[1m])))
sum(rate(pipeline_api_requests_total{status="503"}[1m])) / clamp_min(sum(rate(pipeline_api_requests_total[1m])), 0.000001)
```

Перцентили вычисляются после суммирования бакетов, не усредняются между воркерами.
Панель SLO считает долю задач, успешно завершённых ≤15 с от публикации до ack,
за последние 5 минут: цель 99%, error budget 1%. В знаменателе все терминальные
исходы ok/DLQ; DLQ и медленные задачи тратят бюджет. Незавершённые задачи появятся
в знаменателе после завершения, поэтому это SLO по завершённым задачам, не контроль
возраста всего backlog. Для гистограммы полного времени отдельно заданы границы
с учётом `INFERENCE_DELAY × QUEUE_LIMIT`, плюс фиксированный порог 15 с.

Алерты: заполнение >90% в течение 1 минуты — устойчивое противодавление;
DLQ >0 в течение 15 секунд — любое необработанное сообщение, с небольшой защитой
от краткого колебания. Достаточно Firing в Grafana, внешние уведомления не настроены.

## Эксперимент и защита

```bash
python scripts/load.py generate --count 6000
docker compose run --rm source python -m mlops.source /data/load.jsonl
```

Источник работает примерно 10 минут при одном воркере и задержке 0.1 с; реальные
скорости зависят от CPU/БД/модели. На дашборде ready достигает лимита, скорость
источника сближается с обработкой воркера. Во время нагрузки в другом терминале:

```bash
docker compose up -d --no-deps --scale worker=2 worker
docker compose stop -t 75 worker
```

Для второго сценария сначала снова запустите воркер. При остановке активные задачи
должны закончиться, это видно по task completed и shutdown в логах. Второй воркер
должен увеличить пропускную способность, но удвоение нужно подтвердить измерением.
Не меняйте QUEUE_LIMIT у уже объявленной очереди: RabbitMQ отвергнет несовпадающие
аргументы. Для нового размера используйте другое QUEUE_NAME или чистый учебный том.

Ядовитое сообщение и нагрузка API:

```bash
python scripts/load.py generate --count 1 --poison --path examples/poison.jsonl
docker compose run --rm source python -m mlops.source /data/poison.jsonl
python scripts/load.py api --count 1000 --concurrency 20
```

По task_id ядовитого сообщения видны три process-спана, ошибки и DLQ, затем Firing.
Для полного пути API → publish → process отправьте POST через Swagger, сохраните
task_id из 202 и найдите его в панели «История задачи». Его результат можно получить
через GET или SQL:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "select task_id from task_results order by created_at desc limit 1"'
docker compose --profile observability stop otel-collector
```

При отключённом Collector новые задачи должны продолжать появляться в БД.
Alloy/Loki можно также остановить: stdout контейнера остаётся независимым от агента.
Если память источника растёт с размером файла, это ошибка: он хранит лишь одну строку.

## Бюджет и проверка

| Сервис наблюдаемости | Лимит MiB |
| --- | ---: |
| Loki | 256 |
| Alloy | 128 |
| Tempo | 256 |
| Collector | 192 |
| Prometheus | 192 |
| Grafana | 256 |
| **Всего** | **1280 (1.25 GiB)** |

Это лимиты контейнеров, не фактическое потребление; память приложения считается
отдельно. Loki/Tempo/Prometheus хранят данные 48 часов; позиции Alloy и состояние
Grafana — именованные тома. Grafana хранит конфигурацию/состояние, не телеметрию;
retention к ней не применяется. Версии всех базовых образов закреплены. Для distroless
Tempo/Collector добавлен только busybox-пробник, чтобы healthcheck работал в контейнере.

Обязательный замер на машине с Docker под нагрузкой:

```bash
docker stats --no-stream
```

**Фактические docker stats, Firing алертов, конфиги валидаторами самих сервисов и
живой end-to-end здесь пока не проверены:** на рабочем компьютере не найден Docker,
скачивание пакетов блокируется DNS. Замеры не выдуманы. Перед защитой нужно выполнить
команды выше и сохранить реальные значения/скриншоты.

Дополнительные зависимости отделены в `requirements-pipeline.txt`, версии прямых
зависимостей закреплены. Исходный `uv.lock` не изменён без доступного resolver;
он фиксирует только исходный сервис, не транзитивные зависимости ДЗ5. Для локального
окружения нужен Python 3.12.2+, затем:

```bash
uv sync --locked
uv pip install -r requirements-pipeline.txt
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
docker compose config --quiet
```

`uv run` без `--no-sync` может удалить дополнительные пакеты. Docker и CI сначала
синхронизируют исходное окружение, затем устанавливают дополнение. Следующий шаг
для полностью воспроизводимого общего lockfile — включить эти зависимости в
pyproject и выполнить `uv lock` на машине с доступом к registry.

Unit-тесты `tests/test_pipeline.py` покрывают ack после записи, retry, DLQ с publisher
confirm, сбой DLQ, изоляцию task context, родителя inference в потоке и редактирование
секретов. Они не заменяют проверку настоящего RabbitMQ и сетевых протоколов.

Официальные описания: [RabbitMQ quorum](https://www.rabbitmq.com/docs/4.0/quorum-queues),
[OTel Python](https://opentelemetry.io/docs/languages/python/),
[Grafana provisioning](https://grafana.com/docs/grafana/v11.4/administration/provisioning/).

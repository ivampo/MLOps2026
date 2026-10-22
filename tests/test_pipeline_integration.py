import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from mlops.config import Config
from mlops.db import get_engine
from mlops.main import app
from mlops.messaging import create_broker, declare_queues, get_task_queue, publish_with_retry
from mlops.schemas import IrisFeatures, TaskMessage
from mlops.storage import get_result, init_storage, next_attempt, save_result, task_results
from mlops.worker import create_worker, predict

pytestmark = pytest.mark.integration
TEST_PASSWORD = "test"
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture(scope="session")
def services():
    with (
        RabbitMqContainer(
            "rabbitmq:4-management", username="test", password=TEST_PASSWORD, init=True
        ).with_env("RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS", "+S 2:2") as rabbit,
        PostgresContainer(
            "postgres:16-alpine",
            username="test",
            password=TEST_PASSWORD,
            dbname="test",
            driver="psycopg",
        ) as postgres,
    ):
        yield rabbit, postgres


@pytest.fixture
async def pipeline_config(services, monkeypatch, loaded_model):
    monkeypatch.setattr("mlops.main.load_model", lambda config: loaded_model)
    rabbit, postgres = services
    config = Config.model_validate(
        {
            "_env_file": None,
            "postgres_host": postgres.get_container_host_ip(),
            "postgres_port": int(postgres.get_exposed_port(5432)),
            "postgres_user": "test",
            "postgres_password": TEST_PASSWORD,
            "postgres_db": "test",
            "rabbitmq_host": rabbit.get_container_host_ip(),
            "rabbitmq_port": int(rabbit.get_exposed_port(5672)),
            "rabbitmq_user": "test",
            "rabbitmq_password": TEST_PASSWORD,
            "rabbitmq_queue": f"test-{uuid4().hex}",
            "rabbitmq_queue_max_length": 2,
            "inference_delay": 0.05,
            "retry_delay": 0.02,
            "publish_timeout": 0.5,
        }
    )
    engine = get_engine(config)
    await init_storage(engine)
    try:
        yield config, engine
    finally:
        await engine.dispose()


def task():
    return TaskMessage(
        task_id=uuid4(),
        submitted_at=datetime.now(UTC),
        samples=[IrisFeatures.model_validate(SAMPLE)],
    )


async def wait_result(engine, task_id):
    async with asyncio.timeout(10):
        while (result := await get_result(engine, task_id)) is None:
            await asyncio.sleep(0.02)
        return result


async def test_real_queue_overflow_backpressure_and_api_503(pipeline_config, monkeypatch):
    config, engine = pipeline_config
    monkeypatch.setattr("mlops.main.get_config", lambda: config)
    monkeypatch.setattr("mlops.api.v1.tasks.get_config", lambda: config)
    messages = [task() for _ in range(3)]
    async with app.router.lifespan_context(app):
        broker = app.state.broker
        for message in messages[:2]:
            await broker.publish(
                message.model_dump(mode="json"), queue=config.rabbitmq_queue, persist=True
            )
        with pytest.raises(DeliveryError):
            await broker.publish(messages[2].model_dump(mode="json"), queue=config.rabbitmq_queue)
        rejected_id = uuid4()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            started = time.perf_counter()
            response = await http.post(
                "/api/v1/tasks",
                json={"samples": [SAMPLE]},
                headers={"Idempotency-Key": str(rejected_id)},
            )
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"
            assert time.perf_counter() - started < 1
        assert await get_result(engine, rejected_id) is None
        blocked = asyncio.create_task(
            publish_with_retry(
                broker,
                config,
                messages[2].model_dump(mode="json"),
                queue=config.rabbitmq_queue,
                message_id=str(messages[2].task_id),
            )
        )
        await asyncio.sleep(0.12)
        assert not blocked.done()
        queue = await broker.declare_queue(get_task_queue(config))
        first = await queue.get(timeout=1)
        assert TaskMessage.model_validate_json(first.body).task_id == messages[0].task_id
        await first.ack()
        await asyncio.wait_for(blocked, 2)
        received = []
        for _ in range(2):
            message = await queue.get(timeout=1)
            received.append(TaskMessage.model_validate_json(message.body).task_id)
            await message.ack()
        assert received == [item.task_id for item in messages[1:]]


async def test_real_end_to_end_duplicate_and_graceful_shutdown(
    pipeline_config, monkeypatch, loaded_model
):
    config, engine = pipeline_config
    config.inference_delay = 0.3
    monkeypatch.setattr("mlops.worker.load_model", Mock(return_value=loaded_model))
    monkeypatch.setattr("mlops.main.get_config", lambda: config)
    monkeypatch.setattr("mlops.api.v1.tasks.get_config", lambda: config)
    worker = create_worker(config)
    task_id = uuid4()
    async with app.router.lifespan_context(app):
        async with (
            worker.start_lifespan_context(),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
        ):
            for _ in range(2):
                response = await http.post(
                    "/api/v1/tasks",
                    json={"samples": [SAMPLE]},
                    headers={"Idempotency-Key": str(task_id)},
                )
                assert response.status_code == 202
            async with AsyncClient(
                transport=ASGITransport(app=worker), base_url="http://worker"
            ) as health:
                assert (await health.get("/healthz")).status_code == 204
            await asyncio.sleep(0.1)
            assert await get_result(engine, task_id) is None
        result = await wait_result(engine, task_id)
        assert result.predictions[0].class_name == "setosa"
        async with engine.connect() as connection:
            count = await connection.scalar(
                select(func.count())
                .select_from(task_results)
                .where(task_results.c.task_id == task_id)
            )
        assert count == 1


async def test_real_poison_dlq_and_persistent_attempts(pipeline_config, monkeypatch, loaded_model):
    config, engine = pipeline_config
    bad = task()
    assert await next_attempt(engine, bad.task_id) == 1
    predictor = Mock(side_effect=RuntimeError("always fails"))
    monkeypatch.setattr(loaded_model, "predict", predictor)
    monkeypatch.setattr("mlops.worker.load_model", lambda cfg: loaded_model)
    worker = create_worker(config)
    async with create_broker(config) as broker:
        await declare_queues(broker, config)
        async with worker.start_lifespan_context():
            await broker.publish(
                bad.model_dump(mode="json"), queue=config.rabbitmq_queue, persist=True
            )
            from faststream.rabbit import RabbitQueue

            dlq = await broker.declare_queue(
                RabbitQueue(f"{config.rabbitmq_queue}.dlq", durable=True)
            )
            async with asyncio.timeout(10):
                while True:
                    message = await dlq.get(fail=False)
                    if message is not None:
                        break
                    await asyncio.sleep(0.05)
            assert TaskMessage.model_validate_json(message.body).task_id == bad.task_id
            assert message.headers["failure"] == "always fails"
            await message.ack()
    assert predictor.call_count == config.max_attempts - 1
    assert await get_result(engine, bad.task_id) is None


async def test_database_unique_key_under_concurrent_inserts(pipeline_config, loaded_model):
    _, engine = pipeline_config
    message = task()
    result = predict(message, loaded_model, 0)
    inserted = await asyncio.gather(
        *(save_result(engine, message.task_id, result) for _ in range(12))
    )
    assert sum(inserted) == 1
    stored = await get_result(engine, message.task_id)
    assert stored is not None
    assert stored.model == loaded_model.info

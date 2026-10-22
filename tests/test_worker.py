import asyncio
import threading
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from sqlalchemy.exc import OperationalError

from mlops.config import get_config
from mlops.schemas import IrisFeatures, TaskMessage
from mlops.worker import create_worker

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture
async def worker(monkeypatch, loaded_model):
    config = get_config().model_copy(update={"inference_delay": 0.01, "retry_delay": 0.001})
    engine = AsyncMock()
    results, attempts = {}, {}

    async def get_result(engine, task_id):
        return results.get(task_id)

    async def next_attempt(engine, task_id):
        attempts[task_id] = attempts.get(task_id, 0) + 1
        return attempts[task_id]

    async def save_result(engine, task_id, response):
        results.setdefault(task_id, response)
        return True

    monkeypatch.setattr("mlops.worker.get_engine", lambda cfg: engine)
    loader = Mock(return_value=loaded_model)
    monkeypatch.setattr("mlops.worker.load_model", loader)
    monkeypatch.setattr("mlops.worker.init_storage", AsyncMock())
    monkeypatch.setattr("mlops.worker.get_result", get_result)
    monkeypatch.setattr("mlops.worker.next_attempt", next_attempt)
    monkeypatch.setattr("mlops.worker.save_result", save_result)
    app = create_worker(config)
    broker = app.brokers[0]
    assert isinstance(broker, RabbitBroker)
    dead = []

    @broker.subscriber(f"{config.rabbitmq_queue}.dlq", decoder=lambda msg: msg.body)
    async def dlq(body: bytes):
        dead.append(body)

    async with TestRabbitBroker(broker, connect_only=False), app.lifespan_context():
        yield broker, config, results, attempts, dead, loader
    engine.dispose.assert_awaited_once()


def task():
    return TaskMessage(
        task_id=uuid4(),
        submitted_at=datetime.now(UTC),
        samples=[IrisFeatures.model_validate(SAMPLE)],
    )


async def test_worker_inference_duplicate_and_single_load(worker, loaded_model, monkeypatch):
    broker, config, results, attempts, dead, loader = worker
    original = loaded_model.predict
    threads = []

    def predict(samples):
        threads.append(threading.get_ident())
        return original(samples)

    monkeypatch.setattr(loaded_model, "predict", predict)
    message = task()
    for _ in range(2):
        await broker.publish(message.model_dump(mode="json"), queue=config.rabbitmq_queue)
    assert len(results) == 1
    assert attempts[message.task_id] == 1
    assert threads == [threads[0]]
    assert threads[0] != threading.get_ident()
    assert not dead
    loader.assert_called_once()


async def test_poison_task_retries_then_dlq(worker, loaded_model, monkeypatch):
    broker, config, results, attempts, dead, _ = worker
    predictor = Mock(side_effect=RuntimeError("poison"))
    monkeypatch.setattr(loaded_model, "predict", predictor)
    message = task()
    await broker.publish(message.model_dump(mode="json"), queue=config.rabbitmq_queue)
    assert predictor.call_count == config.max_attempts
    assert attempts[message.task_id] == config.max_attempts
    assert not results
    assert len(dead) == 1
    assert TaskMessage.model_validate_json(dead[0]).task_id == message.task_id


async def test_invalid_json_goes_to_dlq(worker):
    broker, config, results, attempts, dead, _ = worker
    await broker.publish(b"not json", queue=config.rabbitmq_queue)
    assert dead == [b"not json"]
    assert not results
    assert not attempts


async def test_attempt_budget_survives_redelivery(worker, loaded_model, monkeypatch):
    broker, config, _results, attempts, dead, _ = worker
    predictor = Mock()
    monkeypatch.setattr(loaded_model, "predict", predictor)
    message = task()
    attempts[message.task_id] = config.max_attempts
    await broker.publish(message.model_dump(mode="json"), queue=config.rabbitmq_queue)
    predictor.assert_not_called()
    assert len(dead) == 1


async def test_storage_outage_holds_message(worker, monkeypatch):
    broker, config, results, _attempts, dead, _ = worker
    get = AsyncMock(side_effect=[OperationalError("select", None, Exception()), None])
    monkeypatch.setattr("mlops.worker.get_result", get)
    message = task()
    await broker.publish(message.model_dump(mode="json"), queue=config.rabbitmq_queue)
    assert message.task_id in results
    assert not dead


async def test_worker_event_loop_stays_responsive(worker):
    broker, config, *_ = worker
    config.inference_delay = 0.15
    publication = asyncio.create_task(
        broker.publish(task().model_dump(mode="json"), queue=config.rabbitmq_queue)
    )
    await asyncio.sleep(0.03)
    assert not publication.done()
    await publication


async def test_worker_startup_failure_cleanup(monkeypatch):
    engine = AsyncMock()
    monkeypatch.setattr("mlops.worker.get_engine", lambda cfg: engine)
    monkeypatch.setattr("mlops.worker.init_storage", AsyncMock())
    monkeypatch.setattr("mlops.worker.load_model", Mock(side_effect=RuntimeError("alias missing")))
    app = create_worker(get_config())
    with pytest.raises(RuntimeError, match="alias missing"):
        async with app.lifespan_context():
            pass
    engine.dispose.assert_awaited_once()


async def test_database_write_retry_does_not_repeat_inference(worker, loaded_model, monkeypatch):
    broker, config, _results, attempts, dead, _ = worker
    predictor = Mock(wraps=loaded_model.predict)
    monkeypatch.setattr(loaded_model, "predict", predictor)
    save = AsyncMock(side_effect=[OperationalError("insert", None, Exception()), True])
    monkeypatch.setattr("mlops.worker.save_result", save)
    message = task()
    await broker.publish(message.model_dump(mode="json"), queue=config.rabbitmq_queue)
    assert predictor.call_count == 1
    assert attempts[message.task_id] == 1
    assert save.await_count == 2
    assert not dead

from unittest.mock import AsyncMock
from uuid import uuid4

from aio_pika.exceptions import DeliveryError
from pamqp.commands import Basic

from mlops.config import get_config
from mlops.messaging import get_task_queue, publish_with_retry
from mlops.source import read_samples


async def test_publisher_waits_and_reuses_same_task(monkeypatch):
    broker = AsyncMock()
    broker.publish.side_effect = [DeliveryError(None, Basic.Nack()), TimeoutError(), None]
    sleep = AsyncMock()
    monkeypatch.setattr("mlops.messaging.asyncio.sleep", sleep)
    body = {"task_id": str(uuid4())}
    await publish_with_retry(broker, get_config(), body, queue="test", message_id=body["task_id"])
    assert broker.publish.await_count == 3
    assert sleep.await_count == 2
    assert all(call.args[0] > 0 for call in sleep.call_args_list)
    assert all(call.args[0] is body for call in broker.publish.call_args_list)
    assert all(
        call.kwargs["message_id"] == body["task_id"] for call in broker.publish.call_args_list
    )


def test_queue_modes():
    config = get_config()
    arguments = get_task_queue(config).arguments
    assert arguments is not None
    assert arguments["x-overflow"] == "reject-publish"
    unlimited = config.model_copy(update={"backpressure_enabled": False})
    arguments = get_task_queue(unlimited).arguments
    assert arguments is not None
    assert "x-max-length" not in arguments


def test_csv_source_streams_and_validates(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2\n6,3,4,1\n")
    samples = read_samples(path, 1)
    assert iter(samples) is samples
    assert len(list(samples)) == 1
    assert len(list(read_samples(None, 2))) == 2

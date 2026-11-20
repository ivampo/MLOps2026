import asyncio
import logging
import time
from uuid import uuid4

from opentelemetry import propagate, trace

from mlops.schemas import ProcessRequest, ProcessResponse
from mlops.tasks import save_result
from mlops.telemetry import instruments, measured_inference, task_context, tracer

logger = logging.getLogger(__name__)


async def handle_task(message, model, engine, broker, config) -> None:
    identifier = message.correlation_id or message.headers.get("task_id") or str(uuid4())
    attempt = int(message.headers.get("x-delivery-count", 0)) + 1
    parent = propagate.extract(message.headers)
    with task_context(identifier), tracer.start_as_current_span(
        "process", context=parent, kind=trace.SpanKind.CONSUMER,
        attributes={"task.id": identifier, "delivery.attempt": attempt},
    ) as span:
        if attempt > 1 or message.raw_message.redelivered:
            instruments.redeliveries.add(1)
        published = message.headers.get("published_at_ns")
        if published:
            elapsed = max(0, (time.time_ns() - int(published)) / 1e9)
            span.set_attribute("messaging.queue.wait_seconds", elapsed)
            instruments.queue_wait.record(elapsed)
        instruments.inflight.add(1)
        logger.info("task started", extra={"fields": {"attempt": attempt}})
        try:
            payload = ProcessRequest.model_validate_json(message.body)
            # asyncio.to_thread copies contextvars, including the active OTel span.
            result = await asyncio.to_thread(predict, model, payload, config.inference_delay)
            await save_result(engine, identifier, result.model_dump())
        except Exception as exc:
            span.record_exception(safe_exception(exc)[1].with_traceback(exc.__traceback__))
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(trace.StatusCode.ERROR)
            instruments.processed.add(1, {"result": "error"})
            # ValidationError contains input values: never log it or its traceback.
            logger.error(
                "task processing failed", exc_info=safe_exception(exc),
                extra={"fields": {"attempt": attempt, "error_type": type(exc).__name__}},
            )
            if attempt >= config.max_attempts:
                try:
                    headers = dict(message.headers)
                    propagate.inject(headers)
                    await broker.publish(
                        message.body, queue=f"{config.queue_name}.dlq", headers=headers,
                        correlation_id=identifier, persist=True, timeout=config.publish_timeout,
                    )
                except Exception:
                    logger.exception("DLQ publish failed; message requeued")
                    await message.nack(requeue=True)
                    raise
                instruments.processed.add(1, {"result": "dlq"})
                span.add_event("dlq", {"delivery.attempt": attempt})
                logger.error("task moved to DLQ", extra={"fields": {"attempt": attempt}})
                await message.ack()
            else:
                span.add_event("retry", {"delivery.attempt": attempt})
                await asyncio.sleep(config.source_retry_delay)
                await message.nack(requeue=True)
        else:
            await message.ack()
            instruments.processed.add(1, {"result": "ok"})
            if published:
                instruments.task_duration.record(max(0, (time.time_ns() - int(published)) / 1e9))
            logger.info("task completed", extra={"fields": {"attempt": attempt}})
        finally:
            instruments.inflight.add(-1)


def safe_exception(exc: Exception) -> tuple:
    # Keep traceback frames and exception class, discard arbitrary message/input text.
    return RuntimeError, RuntimeError(f"{type(exc).__name__}: details redacted"), exc.__traceback__


def predict(model, payload: ProcessRequest, delay: float) -> ProcessResponse:
    with measured_inference(model):
        time.sleep(delay)
        return ProcessResponse(model=model.info, predictions=model.predict(payload.samples))



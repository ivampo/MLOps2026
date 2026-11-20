import asyncio
import json
import logging
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mlops.logging import Formatter
from mlops.processing import handle_task
from mlops.schemas import ModelInfo, Prediction
from mlops.telemetry import histogram_boundaries, task_context, task_id

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
EXPORTER = InMemorySpanExporter()
PROVIDER = TracerProvider()
PROVIDER.add_span_processor(SimpleSpanProcessor(EXPORTER))
TRACER = PROVIDER.get_tracer(__name__)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        EXPORTER.clear()
        self.model = SimpleNamespace(
            info=ModelInfo(name="iris", version="2", alias="champion", run_id="run"),
            predict=Mock(
                return_value=[Prediction(class_name="setosa", probabilities={"setosa": 1})]
            ),
        )
        self.config = SimpleNamespace(
            max_attempts=3, queue_name="tasks", publish_timeout=1,
            source_retry_delay=0.001, inference_delay=0,
        )
        self.message = SimpleNamespace(
            body=json.dumps({"samples": [SAMPLE]}).encode(),
            correlation_id="test-task",
            headers={"published_at_ns": time.time_ns()},
            raw_message=SimpleNamespace(redelivered=False),
            ack=AsyncMock(), nack=AsyncMock(),
        )
        self.broker = SimpleNamespace(publish=AsyncMock())
        self.save = AsyncMock()
        self.patcher = patch("mlops.processing.save_result", self.save)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.traces = patch("mlops.processing.tracer", TRACER)
        self.traces.start()
        self.addCleanup(self.traces.stop)
        self.inference_traces = patch("mlops.telemetry.tracer", TRACER)
        self.inference_traces.start()
        self.addCleanup(self.inference_traces.stop)

    async def process(self):
        await handle_task(self.message, self.model, "engine", self.broker, self.config)

    async def test_ack_only_after_successful_database_write(self):
        async def save(*args):
            self.message.ack.assert_not_awaited()

        self.save.side_effect = save
        await self.process()
        self.save.assert_awaited_once()
        self.message.ack.assert_awaited_once()
        self.message.nack.assert_not_awaited()
        self.assertIsNone(task_id.get())

    async def test_failed_database_write_requeues_original_message(self):
        self.save.side_effect = ConnectionError("database unavailable")
        with self.assertLogs("mlops.processing", level="ERROR"):
            await self.process()
        self.message.nack.assert_awaited_once_with(requeue=True)
        self.message.ack.assert_not_awaited()
        self.broker.publish.assert_not_awaited()
        self.assertTrue(EXPORTER.get_finished_spans()[-1].status.is_ok is False)

    async def test_poison_message_retries_then_moves_to_dlq(self):
        self.message.body = b'{"private_record":"must-not-appear-in-logs"}'
        with self.assertLogs("mlops.processing", level="ERROR") as logs:
            await self.process()
            self.message.headers["x-delivery-count"] = 1
            await self.process()
            self.message.headers["x-delivery-count"] = 2

            async def publish(*args, **kwargs):
                self.message.ack.assert_not_awaited()

            self.broker.publish.side_effect = publish
            await self.process()
        self.assertEqual(self.message.nack.await_count, 2)
        self.message.ack.assert_awaited_once()
        self.assertEqual(self.broker.publish.call_args.kwargs["queue"], "tasks.dlq")
        self.assertNotIn("must-not-appear-in-logs", "\n".join(logs.output))
        spans = [span for span in EXPORTER.get_finished_spans() if span.name == "process"]
        self.assertEqual([span.attributes["delivery.attempt"] for span in spans], [1, 2, 3])
        self.assertEqual(spans[-1].events[-1].name, "dlq")

    async def test_dlq_publish_failure_does_not_ack(self):
        self.message.body = b"invalid json"
        self.message.headers["x-delivery-count"] = 2
        self.broker.publish.side_effect = ConnectionError("broker unavailable")
        with self.assertLogs("mlops.processing", level="ERROR"):
            with self.assertRaises(ConnectionError):
                await self.process()
        self.message.ack.assert_not_awaited()
        self.message.nack.assert_awaited_once_with(requeue=True)

    async def test_thread_inference_is_child_of_process_and_keeps_trace(self):
        with TRACER.start_as_current_span("publish") as parent:
            propagate.inject(self.message.headers)
            identifier = parent.get_span_context().trace_id
        await self.process()
        spans = {span.name: span for span in EXPORTER.get_finished_spans()}
        self.assertEqual(spans["inference"].context.trace_id, identifier)
        self.assertEqual(spans["inference"].parent.span_id, spans["process"].context.span_id)
        self.assertEqual(spans["process"].parent.span_id, spans["publish"].context.span_id)
        self.assertEqual(spans["inference"].attributes["model.version"], "2")

    async def test_task_context_is_isolated_between_concurrent_tasks(self):
        async def task(identifier):
            with task_context(identifier):
                await asyncio.sleep(0)
                return task_id.get()

        self.assertEqual(await asyncio.gather(task("one"), task("two")), ["one", "two"])
        self.assertIsNone(task_id.get())


class LoggingTests(unittest.TestCase):
    def record(self, message):
        return logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None)

    def test_json_redacts_credentials_and_binds_task(self):
        with task_context("task-123"):
            record = Formatter().format(self.record("failed amqp://user:password@rabbitmq/"))
        payload = json.loads(record)
        self.assertNotIn("password", record)
        self.assertEqual(payload["task_id"], "task-123")
        self.assertIsNone(task_id.get())

    def test_histogram_covers_configured_latency_and_slo(self):
        for delay in (0, 0.1, 1, 30):
            boundaries = histogram_boundaries(delay)
            self.assertEqual(boundaries, sorted(set(boundaries)))
            self.assertIn(0.5, boundaries)
            self.assertGreaterEqual(boundaries[-1], delay)


if __name__ == "__main__":
    unittest.main()

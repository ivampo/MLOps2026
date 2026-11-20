import os
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

import psutil
from opentelemetry import metrics, trace
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

task_id: ContextVar[str | None] = ContextVar("task_id", default=None)
tracer = trace.get_tracer("mlops")


def setup_telemetry(service: str, inference_delay: float = 0.1) -> None:
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
        return
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", service),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "0.3.0"),
            "deployment.environment.name": os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "dev"),
            "service.instance.id": os.getenv("OTEL_SERVICE_INSTANCE_ID") or socket.gethostname(),
        }
    )
    # Exporters use OTEL_EXPORTER_OTLP_*; retries never block a request or a worker.
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    boundaries = histogram_boundaries(inference_delay)
    queue_time = max(inference_delay, 0.01) * int(os.getenv("QUEUE_LIMIT", "100"))
    metrics.set_meter_provider(
        MeterProvider(
            resource=resource,
            metric_readers=[reader],
            views=[
                View(
                    instrument_name=name,
                    aggregation=ExplicitBucketHistogramAggregation(boundaries),
                )
                for name in ("pipeline_inference_duration", "pipeline_api_duration")
            ] + [
                View(
                    instrument_name="pipeline_task_duration",
                    aggregation=ExplicitBucketHistogramAggregation(
                        sorted({0.1, 0.5, 1, 5, 10, 15, queue_time, queue_time * 2, 60})
                    ),
                )
            ],
        )
    )
    process = psutil.Process()
    metrics.get_meter("mlops").create_observable_gauge(
        "pipeline_process_memory",
        callbacks=[lambda options: [Observation(process.memory_info().rss)]],
        unit="By",
    )


def histogram_boundaries(inference_delay: float) -> list[float]:
    duration = max(inference_delay, 0.01)
    return sorted({0.005, 0.01, 0.025, 0.5, *[duration * n for n in (0.5, 1, 2, 5, 10)], 30})


def shutdown_telemetry() -> None:
    for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=2000)
        if hasattr(provider, "shutdown"):
            if isinstance(provider, MeterProvider):
                provider.shutdown(timeout_millis=2000)
            else:
                provider.shutdown()


class Instruments:
    def __init__(self) -> None:
        meter = metrics.get_meter("mlops")
        self.api_requests = meter.create_counter("pipeline_api_requests")
        self.api_duration = meter.create_histogram("pipeline_api_duration", unit="s")
        self.published = meter.create_counter("pipeline_source_published")
        self.rejected = meter.create_counter("pipeline_source_rejected")
        self.wait = meter.create_counter("pipeline_source_wait", unit="s")
        self.processed = meter.create_counter("pipeline_worker_processed")
        self.inference = meter.create_histogram("pipeline_inference_duration", unit="s")
        self.inflight = meter.create_up_down_counter("pipeline_worker_inflight")
        self.redeliveries = meter.create_counter("pipeline_worker_redeliveries")
        self.queue_wait = meter.create_histogram("pipeline_queue_wait", unit="s")
        self.task_duration = meter.create_histogram("pipeline_task_duration", unit="s")
        self.queue_limit = meter.create_observable_gauge(
            "pipeline_queue_limit",
            callbacks=[lambda options: [Observation(int(os.getenv("QUEUE_LIMIT", "100")))]],
        )


instruments = Instruments()


@contextmanager
def task_context(identifier: str):
    token = task_id.set(identifier)
    try:
        yield
    finally:
        task_id.reset(token)


@contextmanager
def measured_inference(model):
    with tracer.start_as_current_span(
        "inference",
        attributes={"model.alias": model.info.alias or "", "model.version": model.info.version},
    ):
        started = perf_counter()
        try:
            yield
        finally:
            instruments.inference.record(perf_counter() - started)

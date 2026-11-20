import json
import logging
import os
import re
import sys

from opentelemetry import trace

from mlops.telemetry import task_id


class Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.getenv("OTEL_SERVICE_NAME", "api"),
            "env": os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "dev"),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if task_id.get() is not None:
            payload["task_id"] = task_id.get()
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            payload["trace_id"] = format(context.trace_id, "032x")
            payload["span_id"] = format(context.span_id, "016x")
        # Includes exception text from libraries, which can contain connection URLs.
        return re.sub(
            r"(\w+://)[^\s\"@]+@", r"\1[redacted]@", json.dumps(payload, ensure_ascii=False)
        )


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(Formatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("uvicorn", "uvicorn.error", "faststream", "aio_pika", "aiormq"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

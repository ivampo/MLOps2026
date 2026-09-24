import json
import logging
import sys

from mlops.logging import Formatter, setup_logging


def _record(**kwargs: object) -> logging.LogRecord:
    defaults = {
        "name": "mlops.db",
        "level": logging.INFO,
        "pathname": __file__,
        "lineno": 10,
        "msg": "postgres ok in %s ms",
        "args": (3.2,),
        "exc_info": None,
    }
    return logging.LogRecord(**{**defaults, **kwargs})


def test_formatter() -> None:
    payload = json.loads(Formatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "mlops.db"
    assert payload["message"] == "postgres ok in 3.2 ms"
    assert "timestamp" in payload


def test_traceback_in_log() -> None:
    assert "exception" not in json.loads(Formatter().format(_record()))

    try:
        raise ConnectionRefusedError("error")
    except ConnectionRefusedError:
        record = _record(level=logging.ERROR, exc_info=sys.exc_info(), msg="error", args=())

    payload = json.loads(Formatter().format(record))
    assert "ConnectionRefusedError: error" in payload["exception"]
    assert payload["level"] == "ERROR"


def test_setup_logging() -> None:
    setup_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(h.formatter, Formatter) for h in root.handlers)

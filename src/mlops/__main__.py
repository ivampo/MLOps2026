import uvicorn

from mlops.config import get_config
from mlops.logging import setup_logging
from mlops.telemetry import shutdown_telemetry


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)
    try:
        uvicorn.run(
            "mlops.main:app", host=config.host, port=config.port, log_config=None, access_log=False
        )
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()

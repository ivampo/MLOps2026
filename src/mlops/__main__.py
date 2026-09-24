import uvicorn

from mlops.config import get_config
from mlops.logging import setup_logging


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)
    uvicorn.run("mlops.main:app", host=config.host, port=config.port, log_config=None)


if __name__ == "__main__":
    main()

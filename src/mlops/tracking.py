from mlflow.cli import server

from mlops.config import get_config


def main() -> None:
    config = get_config()
    backend = config.postgres_dsn.set(drivername="postgresql+psycopg")
    server(
        args=[
            "--backend-store-uri",
            backend.render_as_string(hide_password=False),
            "--serve-artifacts",
            "--artifacts-destination",
            "/mlartifacts",
            "--host",
            "0.0.0.0",  # noqa: S104 - доступ между контейнерами
            "--port",
            "5000",
            "--workers",
            "1",
            "--allowed-hosts",
            "mlflow:5000,localhost:*,127.0.0.1:*",
        ]
    )

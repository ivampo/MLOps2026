from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr
    postgres_db: str = "mlops"

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    db_conn_timeout: int = 10
    health_timeout: int = 5

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "iris-classification"
    mlflow_model_name: str = "iris-classifier"
    mlflow_model_alias: str = "champion"

    @property
    def postgres_dsn(self) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_config() -> Config:
    return Config()

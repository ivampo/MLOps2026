from functools import lru_cache

from pydantic import Field, SecretStr
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
    model_cache_size: int = Field(default=3, ge=1, le=10)
    rabbitmq_url: SecretStr = SecretStr("amqp://guest:guest@localhost:5672/")
    queue_name: str = "tasks"
    queue_limit: int = Field(default=100, ge=1)
    max_attempts: int = Field(default=3, ge=1, le=20)
    publish_timeout: float = Field(default=2, gt=0)
    source_retry_delay: float = Field(default=0.2, gt=0)
    inference_delay: float = Field(default=0.1, ge=0, le=30)
    pipeline_enabled: bool = False

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

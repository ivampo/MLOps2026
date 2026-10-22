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

    host: str = "0.0.0.0"
    port: int = 8000

    db_conn_timeout: int = 10
    health_timeout: int = 5

    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_experiment_name: str = "iris-classification"
    mlflow_model_name: str = "iris-classifier"
    mlflow_model_alias: str = "champion"
    model_cache_size: int = Field(default=3, ge=1, le=10)

    rabbitmq_host: str = "127.0.0.1"
    rabbitmq_port: int = Field(default=5672, ge=1, le=65535)
    rabbitmq_user: str = "mlops"
    rabbitmq_password: SecretStr

    rabbitmq_queue: str = "predict.tasks"
    rabbitmq_queue_max_length: int = Field(default=10, ge=1)
    backpressure_enabled: bool = True
    worker_concurrency: int = Field(default=1, ge=1, le=32)
    worker_prefetch_count: int = Field(default=1, ge=0)
    inference_delay: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    publish_timeout: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    api_max_inflight: int = Field(default=32, ge=1)

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
    return Config.model_validate({})

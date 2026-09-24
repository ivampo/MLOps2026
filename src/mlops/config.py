from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

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

    @property
    def postgres_dsn(self):
        password = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


@lru_cache
def get_config() -> Config:
    return Config()

from unittest.mock import Mock

from sqlalchemy import make_url

from mlops.config import Config
from mlops.tracking import main


def test_postgres_password_with_special_characters():
    special_characters = "p@ss:/?#%"
    config = Config(postgres_password=special_characters, _env_file=None)
    url = config.postgres_dsn
    assert url.password == special_characters
    assert make_url(url.render_as_string(hide_password=False)).password == url.password


def test_tracking_server_configuration(monkeypatch):
    server = Mock()
    monkeypatch.setattr("mlops.tracking.server", server)
    main()
    args = server.call_args.kwargs["args"]
    backend = make_url(args[args.index("--backend-store-uri") + 1])
    assert backend.drivername == "postgresql+psycopg"
    assert "--serve-artifacts" in args
    assert args[args.index("--artifacts-destination") + 1] == "/mlartifacts"

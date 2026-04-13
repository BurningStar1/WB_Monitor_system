from config import get_settings


def test_sqlalchemy_url_created() -> None:
    settings = get_settings()
    assert settings.sqlalchemy_url.startswith("postgresql+psycopg2://")

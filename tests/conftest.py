import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    monkeypatch.setattr(settings, "embed_api_key", "test")
    monkeypatch.setattr(settings, "llm_api_key", "test")

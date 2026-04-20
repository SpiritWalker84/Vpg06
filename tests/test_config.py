"""Проверки загрузки конфигурации."""

from __future__ import annotations

from vpg05.config import Settings


def test_openai_base_fallback(monkeypatch) -> None:
    """Если OPENAI_API_BASE не задан, используется OPENAI_BASE_URL."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("WEAVIATE_URL", "https://w.example")
    monkeypatch.setenv("WEAVIATE_API_KEY", "x")

    settings = Settings.from_env()

    assert settings.openai_api_base == "https://proxy.example/v1"

"""Загрузка конфигурации из окружения и .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Параметры приложения."""

    telegram_bot_token: str
    log_level: str
    openai_api_key: str
    openai_api_base: str
    openai_chat_model: str
    openai_embedding_model: str
    embedding_dimension: int
    weaviate_url: str
    weaviate_api_key: str
    weaviate_collection_name: str
    similarity_threshold: float

    @staticmethod
    def from_env() -> "Settings":
        """Читает переменные окружения."""
        base_raw = (
            os.environ.get("OPENAI_API_BASE", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
        )
        base_url = (base_raw or "https://api.proxyapi.ru/openai/v1").rstrip("/")
        return Settings(
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            openai_api_base=base_url,
            openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
            openai_embedding_model=os.environ.get(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
            weaviate_url=os.environ.get("WEAVIATE_URL", "").strip().rstrip("/"),
            weaviate_api_key=os.environ.get("WEAVIATE_API_KEY", "").strip(),
            weaviate_collection_name=os.environ.get(
                "WEAVIATE_COLLECTION_NAME", "Vpg05Memory"
            ).strip(),
            similarity_threshold=float(os.environ.get("SIMILARITY_THRESHOLD", "0.78")),
        )

    def require_bot(self) -> None:
        """Проверяет токен Telegram."""
        if not self.telegram_bot_token:
            raise RuntimeError("Задайте TELEGRAM_BOT_TOKEN в .env")

    def require_openai(self) -> None:
        """Проверяет OpenAI-совместимые настройки."""
        if not self.openai_api_key:
            raise RuntimeError("Задайте OPENAI_API_KEY в .env")

    def require_weaviate(self) -> None:
        """Проверяет настройки Weaviate."""
        if not self.weaviate_url or not self.weaviate_api_key:
            raise RuntimeError("Задайте WEAVIATE_URL и WEAVIATE_API_KEY в .env")


def load_settings(env_file: str | None = ".env") -> Settings:
    """Загружает .env и возвращает настройки."""
    if env_file:
        load_dotenv(env_file)
    return Settings.from_env()

"""Telegram-бот с долговременной векторной памятью."""

from __future__ import annotations

import logging
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from vpg05.config import load_settings
from vpg05.embeddings import OpenAIClient
from vpg05.pinecone_manager import PineconeManager


class TelegramMemoryBot:
    """Бот-помощник с хранением пользовательских сообщений в Weaviate."""

    def __init__(self) -> None:
        settings = load_settings()
        settings.require_bot()
        settings.require_openai()
        settings.require_weaviate()

        self._settings = settings
        self._memory = PineconeManager()
        self._llm = OpenAIClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            embedding_model=settings.openai_embedding_model,
            chat_model=settings.openai_chat_model,
        )

    async def on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Стартовая команда."""
        del context
        if not update.message:
            return
        await update.message.reply_text(
            "Привет! Я запоминаю ваши сообщения в векторной памяти и отвечаю с учетом контекста."
        )

    async def on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Справка по командам."""
        del context
        if not update.message:
            return
        await update.message.reply_text(
            "/start — запуск\n/help — помощь\n"
            "Просто пишите сообщения, а я буду хранить пользовательские факты."
        )

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает обычное сообщение пользователя."""
        del context
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        user_message = (update.message.text or "").strip()
        if not user_message:
            return

        metadata = self._build_metadata(user_id=user.id, user_data=user.to_dict(), text=user_message)
        memory_context = self._build_memory_context(user_message=user_message, user_id=user.id)
        answer = self._llm.chat(
            system_prompt=(
                "Ты дружелюбный Telegram-помощник. "
                "Если есть релевантная память пользователя, учитывай ее при ответе."
            ),
            user_prompt=f"Контекст памяти:\n{memory_context}\n\nСообщение пользователя:\n{user_message}",
        )

        result = self._memory.upsert_document(
            text=user_message,
            metadata=metadata,
            user_id=user.id,
            check_similarity=True,
        )
        logging.getLogger(__name__).info("Memory upsert result: %s", result)
        await update.message.reply_text(answer)

    def _build_memory_context(self, *, user_message: str, user_id: int) -> str:
        hits = self._memory.query_by_text(user_message, top_k=3, user_id=user_id)
        if not hits:
            return "Релевантная память не найдена."
        lines: list[str] = []
        for item in hits:
            text = str(item.get("metadata", {}).get("text", "")).strip()
            score = float(item.get("score", 0.0))
            if text:
                lines.append(f"- ({score:.3f}) {text}")
        return "\n".join(lines) if lines else "Релевантная память не найдена."

    @staticmethod
    def _build_metadata(*, user_id: int, user_data: dict[str, Any], text: str) -> dict[str, Any]:
        """
        Формирует безопасные метаданные без обязательности полей Telegram.

        В память сохраняется только исходный текст пользователя и его доступные атрибуты.
        """
        username = user_data.get("username")
        first_name = user_data.get("first_name")
        last_name = user_data.get("last_name")
        return {
            "text": text,
            "user_id": int(user_id),
            "username": str(username) if username else "",
            "first_name": str(first_name) if first_name else "",
            "last_name": str(last_name) if last_name else "",
            "scope": "user",
        }

    def build_application(self) -> Application:
        """Собирает Telegram-приложение и регистрирует хендлеры."""
        app = Application.builder().token(self._settings.telegram_bot_token).build()
        app.add_handler(CommandHandler("start", self.on_start))
        app.add_handler(CommandHandler("help", self.on_help))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.on_text))
        return app

    def close(self) -> None:
        """Освобождает ресурсы."""
        self._memory.close()


def run() -> None:
    """Точка входа для запуска бота."""
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bot = TelegramMemoryBot()
    app = bot.build_application()
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        bot.close()


if __name__ == "__main__":
    run()

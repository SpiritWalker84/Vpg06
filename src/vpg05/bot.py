"""Telegram-бот: pyTelegramBotAPI + Haystack-агент и Weaviate-память."""

from __future__ import annotations

import logging
import telebot

from vpg05.config import load_settings
from vpg05.haystack_assistant import HaystackPersonalAssistant

_TELEGRAM_MAX = 4096


def _chunk_text(text: str, limit: int = _TELEGRAM_MAX) -> list[str]:
    t = text.strip()
    if not t:
        return ["Пустой ответ."]
    return [t[i : i + limit] for i in range(0, len(t), limit)]


class TelegramAgentBot:
    """Long polling бот с персональным агентом."""

    def __init__(self) -> None:
        settings = load_settings()
        settings.require_bot()
        settings.require_openai()
        settings.require_weaviate()
        self._settings = settings
        self._assistant = HaystackPersonalAssistant(settings)
        self._bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)

    def _display_name(self, message: telebot.types.Message) -> str:
        u = message.from_user
        if not u:
            return ""
        parts = [u.first_name or "", u.last_name or ""]
        name = " ".join(p for p in parts if p).strip()
        if u.username:
            return f"{name} (@{u.username})".strip()
        return name or str(u.id)

    def register_handlers(self) -> None:
        @self._bot.message_handler(commands=["start"])
        def on_start(message: telebot.types.Message) -> None:
            self._bot.reply_to(
                message,
                "Привет! Я персональный помощник на Haystack: помню контекст в Weaviate и веду диалог. "
                "Спросите что угодно; для развлечения можно попросить факт о кошках или описание случайной собаки с фото.",
            )

        @self._bot.message_handler(commands=["help"])
        def on_help(message: telebot.types.Message) -> None:
            self._bot.reply_to(
                message,
                "/start — приветствие\n/help — эта справка\n\n"
                "Пишите сообщения как в обычном чате. Память по смыслу хранится в Weaviate; "
                "недавние реплики учитываются в истории сессии.",
            )

        @self._bot.message_handler(content_types=["text"])
        def on_text(message: telebot.types.Message) -> None:
            if not message.from_user:
                return
            raw = (message.text or "").strip()
            if not raw:
                return
            user_id = int(message.from_user.id)
            name = self._display_name(message)
            try:
                reply = self._assistant.reply(user_id=user_id, user_text=raw, display_name=name)
            except Exception:
                logging.getLogger(__name__).exception("assistant.reply failed")
                self._bot.reply_to(message, "Произошла ошибка при обработке. Попробуйте ещё раз позже.")
                return
            chat_id = message.chat.id
            reply_to = message.message_id
            for i, photo_url in enumerate(reply.photo_urls):
                try:
                    if i == 0:
                        self._bot.send_photo(chat_id, photo_url, reply_to_message_id=reply_to)
                    else:
                        self._bot.send_photo(chat_id, photo_url)
                except Exception:
                    logging.getLogger(__name__).exception("send_photo failed for %s", photo_url)
            body = (reply.text or "").strip()
            if body:
                for chunk in _chunk_text(body):
                    if chunk.strip():
                        self._bot.reply_to(message, chunk)

    def run(self) -> None:
        self._assistant.warm_up()
        self.register_handlers()
        self._bot.infinity_polling(skip_pending=True, interval=0, timeout=60)

    def close(self) -> None:
        self._assistant.close()


def run() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    bot = TelegramAgentBot()
    try:
        bot.run()
    finally:
        bot.close()


if __name__ == "__main__":
    run()

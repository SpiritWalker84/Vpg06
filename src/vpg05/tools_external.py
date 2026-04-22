"""Примеры инструментов агента: внешние API (кошки, собаки + vision)."""

from __future__ import annotations

import logging

import requests
from haystack.tools import Tool, create_tool_from_function
from openai import OpenAI

logger = logging.getLogger(__name__)

# Первая строка ответа инструмента — для Telegram: бот отправляет фото по этому URL пользователю.
DOG_IMAGE_URL_LINE_PREFIX = "DOG_IMAGE_URL:"

_CAT_FACT_URL = "https://catfact.ninja/fact"
_DOG_RANDOM_API = "https://dog.ceo/api/breeds/image/random"


def build_external_tools(
    *,
    openai_api_key: str,
    openai_base_url: str,
    vision_model: str,
) -> list[Tool]:
    """Собирает инструменты с замыканием на OpenAI-клиент для vision."""
    oa_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    def fetch_random_cat_fact() -> str:
        """
        Возвращает случайный короткий факт о кошках с бесплатного API catfact.ninja.
        Вызывай, когда пользователю интересны факты о кошках или нужна лёгкая развлекательная информация.
        """
        try:
            r = requests.get(_CAT_FACT_URL, timeout=15)
            r.raise_for_status()
            data = r.json()
            fact = data.get("fact", "").strip()
            return fact or "Не удалось получить факт."
        except Exception as exc:  # noqa: BLE001
            logger.warning("cat fact failed: %s", exc)
            return f"Не удалось получить факт о кошках: {exc}"

    def describe_random_dog_from_photo() -> str:
        """
        Загружает случайное изображение собаки (dog.ceo), отправляет его в vision-модель OpenAI
        и возвращает: какая порода на снимке, краткая характеристика и очень краткая предыстория породы
        (откуда произошла, для чего выводили). Вызывай по запросам про собак, породы или случайное фото собаки.
        Первая строка ответа служебная (URL фото) — её нужно сохранить в результате инструмента как есть.
        """
        try:
            r = requests.get(_DOG_RANDOM_API, timeout=15)
            r.raise_for_status()
            data = r.json()
            image_url = data.get("message", "").strip()
            if not image_url:
                return "API не вернул ссылку на изображение."

            prompt = (
                "По этому изображению собаки ответь по-русски, кратко и по делу:\n"
                "1) Какая это порода (или ближайший guess, если метис).\n"
                "2) Один–два предложения: типичный характер/назначение породы.\n"
                "3) Один–два предложения: откуда порода произошла или как сформировалась "
                "(исторический контекст без выдумывания деталей; если не уверен — так и скажи).\n"
                "Не начинай с приветствий."
            )
            completion = oa_client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_completion_tokens=500,
            )
            content = completion.choices[0].message.content
            body = ((content or "").strip() or "Пустой ответ модели.")
            return f"{DOG_IMAGE_URL_LINE_PREFIX}{image_url}\n\n{body}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("dog vision failed: %s", exc)
            return f"Не удалось описать собаку: {exc}"

    return [
        create_tool_from_function(fetch_random_cat_fact),
        create_tool_from_function(describe_random_dog_from_photo),
    ]

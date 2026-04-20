"""OpenAI-совместимый клиент для эмбеддингов и чата."""

from __future__ import annotations

from openai import OpenAI


class OpenAIClient:
    """Обертка вокруг OpenAI SDK."""

    def __init__(self, *, api_key: str, base_url: str, embedding_model: str, chat_model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._embedding_model = embedding_model
        self._chat_model = chat_model

    def create_embedding(self, text: str) -> list[float]:
        """Возвращает вектор эмбеддинга для текста."""
        response = self._client.embeddings.create(model=self._embedding_model, input=text)
        return list(response.data[0].embedding)

    def chat(self, *, system_prompt: str, user_prompt: str) -> str:
        """Генерирует ответ чат-модели."""
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Не удалось сформировать ответ."

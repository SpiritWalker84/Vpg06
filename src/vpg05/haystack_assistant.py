"""Персональный агент на Haystack: Weaviate (cosine) + OpenAI + инструменты."""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from haystack import Document
from haystack.components.agents import Agent
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)

from vpg05.config import Settings
from vpg05.tools_external import DOG_IMAGE_URL_LINE_PREFIX, build_external_tools

logger = logging.getLogger(__name__)

_EXTRA_WEAVIATE_PROPS = [
    {"name": "user_id", "dataType": ["int"]},
    {"name": "role", "dataType": ["text"]},
    {"name": "chat_ts", "dataType": ["text"]},
]


def _collection_settings(class_name: str) -> dict:
    return {
        "class": class_name[0].upper() + class_name[1:] if class_name else "Default",
        "invertedIndexConfig": {"indexNullState": True},
        "properties": list(DOCUMENT_COLLECTION_PROPERTIES) + _EXTRA_WEAVIATE_PROPS,
    }


def _format_memory_block(documents: list[Document]) -> str:
    if not documents:
        return "Записей долговременной памяти по этому запросу не найдено."
    lines: list[str] = []
    for doc in documents:
        score = doc.score
        score_s = f"{score:.3f}" if isinstance(score, float) else "n/a"
        role = (doc.meta or {}).get("role", "")
        prefix = f"[{role}] " if role else ""
        text = (doc.content or "").strip()
        if text:
            lines.append(f"- ({score_s}) {prefix}{text}")
    return "\n".join(lines) if lines else "Память пуста."


def _strip_system(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [m for m in messages if not m.is_from(ChatRole.SYSTEM)]


def _tool_result_as_strings(result: object) -> list[str]:
    if isinstance(result, str):
        return [result]
    if isinstance(result, list):
        out: list[str] = []
        for item in result:
            if isinstance(item, str):
                out.append(item)
            else:
                t = getattr(item, "text", None)
                if isinstance(t, str) and t:
                    out.append(t)
        return out
    return []


def _text_blobs_from_message(msg: ChatMessage) -> list[str]:
    blobs = list(msg.texts)
    for tcr in msg.tool_call_results:
        blobs.extend(_tool_result_as_strings(tcr.result))
    return blobs


def _extract_dog_photo_urls(messages: list[ChatMessage]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    prefix = DOG_IMAGE_URL_LINE_PREFIX
    for msg in messages:
        for blob in _text_blobs_from_message(msg):
            for line in blob.splitlines():
                line = line.strip()
                if line.startswith(prefix):
                    url = line[len(prefix) :].strip()
                    if url and url not in seen:
                        seen.add(url)
                        ordered.append(url)
    return tuple(ordered)


def _extract_dog_photo_urls_for_current_turn(
    messages: list[ChatMessage], user_text: str
) -> tuple[str, ...]:
    """
    Берёт URL только из ответов на текущую реплику пользователя.
    История чата содержит прошлые tool-сообщения с DOG_IMAGE_URL — без среза они
    накапливались бы и send_photo слал бы все старые фото снова.
    """
    ut = user_text.strip()
    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.is_from(ChatRole.USER) and (m.text or "").strip() == ut:
            last_user_idx = i
            break
    if last_user_idx is None:
        return _extract_dog_photo_urls(messages)
    return _extract_dog_photo_urls(messages[last_user_idx + 1 :])


def _strip_sent_photo_markdown(text: str, photo_urls: tuple[str, ...]) -> str:
    """
    Убирает из ответа модели markdown-картинки и ссылки на те же URL,
    которые бот уже отправляет через send_photo (иначе дублирование).
    """
    if not text.strip() or not photo_urls:
        return text
    out = text
    for url in photo_urls:
        out = re.sub(r"!\[[^\]]*\]\(\s*" + re.escape(url) + r"\s*\)\s*", "", out)
    # На случай чуть другого URL в markdown, чем в служебной строке инструмента
    out = re.sub(r"!\[[^\]]*\]\(\s*https://images\.dog\.ceo/[^)\s]+\s*\)\s*", "", out)
    lines: list[str] = []
    for line in out.splitlines():
        s = line.strip()
        if s in photo_urls:
            continue
        if re.fullmatch(r"https://images\.dog\.ceo/\S+", s):
            continue
        lines.append(line)
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()
    return collapsed


@dataclass(frozen=True)
class AssistantReply:
    """Ответ ассистента для Telegram: текст и URL фото (dog.ceo), если вызывался vision-инструмент."""

    text: str
    photo_urls: tuple[str, ...] = ()


class HaystackPersonalAssistant:
    """Агент с долговременной памятью в Weaviate (Haystack) и краткой историей чата."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._histories: dict[int, list[ChatMessage]] = defaultdict(list)

        class_name = settings.weaviate_collection_name.strip()
        if not class_name:
            class_name = "Vpg06HaystackMemory"
        self._document_store = WeaviateDocumentStore(
            url=settings.weaviate_url,
            auth_client_secret=AuthApiKey(api_key=Secret.from_token(settings.weaviate_api_key)),
            collection_settings=_collection_settings(class_name),
        )

        api_key_secret = Secret.from_token(settings.openai_api_key)
        base = settings.openai_api_base or None

        self._text_embedder = OpenAITextEmbedder(
            api_key=api_key_secret,
            model=settings.openai_embedding_model,
            api_base_url=base,
            dimensions=settings.embedding_dimension,
        )
        self._doc_embedder = OpenAIDocumentEmbedder(
            api_key=api_key_secret,
            model=settings.openai_embedding_model,
            api_base_url=base,
            dimensions=settings.embedding_dimension,
        )

        self._retriever = WeaviateEmbeddingRetriever(
            document_store=self._document_store,
            top_k=settings.memory_top_k,
            filters={},
        )

        tools = build_external_tools(
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_api_base,
            vision_model=settings.openai_vision_model,
        )

        self._agent = Agent(
            chat_generator=OpenAIChatGenerator(
                api_key=api_key_secret,
                model=settings.openai_chat_model,
                api_base_url=base,
                generation_kwargs={"temperature": 0.7},
            ),
            tools=tools,
            system_prompt=None,
            max_agent_steps=settings.max_agent_steps,
            exit_conditions=["text"],
        )

    def warm_up(self) -> None:
        self._document_store.client
        self._agent.warm_up()

    def close(self) -> None:
        self._document_store.close()

    def _memory_filter(self, user_id: int) -> dict:
        """Только сообщения пользователя — в Weaviate не попадают ответы ассистента."""
        return {
            "operator": "AND",
            "conditions": [
                {"field": "user_id", "operator": "==", "value": int(user_id)},
                {"field": "role", "operator": "==", "value": "user"},
            ],
        }

    def _retrieve(self, *, user_id: int, query_text: str) -> list[Document]:
        emb = self._text_embedder.run(text=query_text)["embedding"]
        out = self._retriever.run(
            query_embedding=emb,
            filters=self._memory_filter(user_id),
            top_k=self._settings.memory_top_k,
        )
        return list(out.get("documents") or [])

    def _persist_user_message(self, *, user_id: int, user_text: str) -> None:
        """В Weaviate пишется только текст сообщения пользователя (требование к ДЗ)."""
        ts = datetime.now(timezone.utc).isoformat()
        docs = [
            Document(
                id=str(uuid.uuid4()),
                content=user_text.strip(),
                meta={"user_id": int(user_id), "role": "user", "chat_ts": ts},
            ),
        ]
        with_embeddings = self._doc_embedder.run(documents=docs)["documents"]
        n = self._document_store.write_documents(with_embeddings, policy=DuplicatePolicy.NONE)
        logger.info("Weaviate memory write (user only): %s documents", n)

    def _trim_history(self, user_id: int) -> None:
        max_m = self._settings.chat_history_max_messages
        hist = self._histories[user_id]
        if len(hist) > max_m:
            self._histories[user_id] = hist[-max_m:]

    def _build_system_prompt(self, *, memory_block: str, display_name: str) -> str:
        who = display_name or "пользователь"
        return (
            f"Ты умный персональный помощник в Telegram. Обращайся естественно, помни контекст разговора.\n"
            f"Собеседник: {who}.\n"
            "Ниже релевантные фрагменты долговременной памяти (семантический поиск, косинусная близость). "
            "Используй их, если уместно; не выдумывай факты, которых нет в памяти и переписке.\n"
            "Инструменты: зови только когда пользователь явно интересуется фактами о кошках, собаках, породах "
            "или согласен на развлекательный контент. Не вызывай инструменты на каждое сообщение.\n"
            "Если вызывал инструмент со случайной собакой: само фото пользователь получит отдельным сообщением в Telegram. "
            "В своём ответе не вставляй markdown-картинки вида ![...](url), не дублируй ссылку на изображение — "
            "только обычный текст с описанием породы.\n\n"
            f"Память:\n{memory_block}"
        )

    def reply(self, *, user_id: int, user_text: str, display_name: str) -> AssistantReply:
        memory_docs = self._retrieve(user_id=user_id, query_text=user_text)
        memory_block = _format_memory_block(memory_docs)
        system_prompt = self._build_system_prompt(memory_block=memory_block, display_name=display_name)

        prior = self._histories[user_id]
        messages_in = prior + [ChatMessage.from_user(user_text)]

        result = self._agent.run(messages=messages_in, system_prompt=system_prompt)
        out_messages = list(result.get("messages") or [])
        self._histories[user_id] = _strip_system(out_messages)
        self._trim_history(user_id)

        last = result.get("last_message")
        assistant_text = (last.text if last else "").strip()
        photo_urls = _extract_dog_photo_urls_for_current_turn(out_messages, user_text)
        if photo_urls:
            assistant_text = _strip_sent_photo_markdown(assistant_text, photo_urls).strip()
        if not assistant_text and not photo_urls:
            assistant_text = "Не удалось сформулировать ответ."

        self._persist_user_message(user_id=user_id, user_text=user_text)
        return AssistantReply(text=assistant_text, photo_urls=photo_urls)

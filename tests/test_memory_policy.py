"""Политика памяти: в Weaviate только текст пользователя."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage

from haystack.dataclasses import ChatMessage, ToolCall, ToolCallResult

from vpg05.config import load_settings
from vpg05.haystack_assistant import (
    HaystackPersonalAssistant,
    _extract_dog_photo_urls,
    _extract_dog_photo_urls_for_current_turn,
    _strip_sent_photo_markdown,
)
from vpg05.tools_external import DOG_IMAGE_URL_LINE_PREFIX


def test_strip_markdown_when_photo_sent_separately() -> None:
    url = "https://images.dog.ceo/breeds/terrier-toy/n02087046_4235.jpg"
    raw = (
        "Вот случайное фото собаки:\n\n"
        f"![Собака]({url})\n\n"
        "Это терьер. Они активные."
    )
    clean = _strip_sent_photo_markdown(raw, (url,))
    assert "![" not in clean and url not in clean
    assert "терьер" in clean.lower()


def test_dog_urls_only_from_latest_user_turn() -> None:
    """Повтор той же фразы не должен подтягивать URL из прошлых tool-сообщений."""
    origin = ToolCall(tool_name="describe_random_dog_from_photo", arguments={}, id="1")
    u1 = ChatMessage.from_user("покажи собаку")
    t1 = ChatMessage.from_tool(
        tool_result=f"{DOG_IMAGE_URL_LINE_PREFIX}https://images.dog.ceo/a.jpg\n\nold",
        origin=origin,
    )
    a1 = ChatMessage.from_assistant("ответ 1")
    u2 = ChatMessage.from_user("покажи собаку")
    t2 = ChatMessage.from_tool(
        tool_result=f"{DOG_IMAGE_URL_LINE_PREFIX}https://images.dog.ceo/b.jpg\n\nnew",
        origin=origin,
    )
    a2 = ChatMessage.from_assistant("ответ 2")
    history = [u1, t1, a1, u2, t2, a2]
    assert _extract_dog_photo_urls(history) == (
        "https://images.dog.ceo/a.jpg",
        "https://images.dog.ceo/b.jpg",
    )
    assert _extract_dog_photo_urls_for_current_turn(history, "покажи собаку") == (
        "https://images.dog.ceo/b.jpg",
    )


def test_extract_dog_photo_url_from_tool_message() -> None:
    origin = ToolCall(tool_name="describe_random_dog_from_photo", arguments={}, id="1")
    payload = f"{DOG_IMAGE_URL_LINE_PREFIX}https://images.dog.ceo/breeds/test.jpg\n\nОписание породы."
    tool_msg = ChatMessage.from_tool(tool_result=payload, origin=origin)
    assert _extract_dog_photo_urls([tool_msg]) == ("https://images.dog.ceo/breeds/test.jpg",)


@pytest.fixture
def env_for_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("WEAVIATE_URL", "https://test.weaviate.network")
    monkeypatch.setenv("WEAVIATE_API_KEY", "w")


@patch("vpg05.haystack_assistant.build_external_tools", return_value=[])
@patch("vpg05.haystack_assistant.Agent")
@patch("vpg05.haystack_assistant.OpenAIChatGenerator")
@patch("vpg05.haystack_assistant.WeaviateEmbeddingRetriever")
@patch("vpg05.haystack_assistant.OpenAIDocumentEmbedder")
@patch("vpg05.haystack_assistant.OpenAITextEmbedder")
@patch("vpg05.haystack_assistant.WeaviateDocumentStore")
def test_weaviate_receives_only_user_message(
    mock_store_cls: MagicMock,
    mock_te_cls: MagicMock,
    mock_de_cls: MagicMock,
    mock_ret_cls: MagicMock,
    mock_cg_cls: MagicMock,
    mock_agent_cls: MagicMock,
    _mock_tools: MagicMock,
    env_for_assistant: None,
) -> None:
    store = MagicMock()
    mock_store_cls.return_value = store

    te = MagicMock()
    mock_te_cls.return_value = te
    te.run.return_value = {"embedding": [0.0, 1.0, 0.0]}

    de = MagicMock()
    mock_de_cls.return_value = de

    def embed_docs(documents: list) -> dict:
        return {"documents": documents}

    de.run.side_effect = embed_docs

    ret = MagicMock()
    mock_ret_cls.return_value = ret
    ret.run.return_value = {"documents": []}

    agent = MagicMock()
    mock_agent_cls.return_value = agent
    agent.run.return_value = {
        "messages": [
            ChatMessage.from_user("Меня зовут Аня"),
            ChatMessage.from_assistant("Приятно познакомиться!"),
        ],
        "last_message": ChatMessage.from_assistant("Приятно познакомиться!"),
    }

    settings = load_settings()
    assistant = HaystackPersonalAssistant(settings)
    out = assistant.reply(user_id=42, user_text="Меня зовут Аня", display_name="Test")

    assert out.text == "Приятно познакомиться!"
    assert out.photo_urls == ()

    store.write_documents.assert_called_once()
    written = store.write_documents.call_args[0][0]
    assert len(written) == 1
    assert written[0].content == "Меня зовут Аня"
    assert written[0].meta.get("role") == "user"


@patch("vpg05.haystack_assistant.build_external_tools", return_value=[])
@patch("vpg05.haystack_assistant.Agent")
@patch("vpg05.haystack_assistant.OpenAIChatGenerator")
@patch("vpg05.haystack_assistant.WeaviateEmbeddingRetriever")
@patch("vpg05.haystack_assistant.OpenAIDocumentEmbedder")
@patch("vpg05.haystack_assistant.OpenAITextEmbedder")
@patch("vpg05.haystack_assistant.WeaviateDocumentStore")
def test_retrieval_filters_by_role_user(
    mock_store_cls: MagicMock,
    mock_te_cls: MagicMock,
    mock_de_cls: MagicMock,
    mock_ret_cls: MagicMock,
    mock_cg_cls: MagicMock,
    mock_agent_cls: MagicMock,
    _mock_tools: MagicMock,
    env_for_assistant: None,
) -> None:
    mock_store_cls.return_value = MagicMock()
    te = MagicMock()
    mock_te_cls.return_value = te
    te.run.return_value = {"embedding": [1.0, 0.0]}
    mock_de_cls.return_value = MagicMock()
    ret = MagicMock()
    mock_ret_cls.return_value = ret
    ret.run.return_value = {"documents": []}
    agent = MagicMock()
    mock_agent_cls.return_value = agent
    agent.run.return_value = {
        "messages": [ChatMessage.from_user("q"), ChatMessage.from_assistant("a")],
        "last_message": ChatMessage.from_assistant("a"),
    }

    assistant = HaystackPersonalAssistant(load_settings())
    assistant.reply(user_id=7, user_text="q", display_name="U")

    filters = ret.run.call_args[1]["filters"]
    conditions = filters["conditions"]
    assert any(c.get("field") == "role" and c.get("value") == "user" for c in conditions)

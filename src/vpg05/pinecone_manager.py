"""Менеджер векторной памяти с интерфейсом PineconeManager, но на Weaviate."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from vpg05.config import load_settings
from vpg05.embeddings import OpenAIClient
from vpg05.weaviate_store import SearchHit, WeaviateMemoryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarityDecision:
    """Результат проверки похожести перед записью."""

    action: str
    similarity_score: float | None
    existing_id: str | None


class PineconeManager:
    """
    Совместимый по интерфейсу менеджер для векторной памяти.

    Историческое имя файла/класса сохранено для совместимости с учебным ТЗ.
    Внутреннее хранилище реализовано через Weaviate Cloud.
    """

    def __init__(
        self,
        *,
        weaviate_url: str | None = None,
        weaviate_api_key: str | None = None,
        index_name: str | None = None,
        openai_api_key: str | None = None,
        openai_api_base: str | None = None,
        openai_model: str | None = None,
        embedding_dimension: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = load_settings()

        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

        base_url = openai_api_base or settings.openai_api_base
        emb_model = openai_model or settings.openai_embedding_model
        api_key = openai_api_key or settings.openai_api_key
        vector_size = embedding_dimension or settings.embedding_dimension
        weaviate_cluster_url = weaviate_url or settings.weaviate_url
        weaviate_key = weaviate_api_key or settings.weaviate_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY не задан для менеджера памяти")
        if not weaviate_cluster_url or not weaviate_key:
            raise RuntimeError("WEAVIATE_URL и WEAVIATE_API_KEY обязательны для менеджера памяти")

        self._llm = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            embedding_model=emb_model,
            chat_model=settings.openai_chat_model,
        )

        self._store = WeaviateMemoryStore(
            cluster_url=weaviate_cluster_url,
            api_key=weaviate_key,
            collection_name=index_name or settings.weaviate_collection_name,
            vector_size=vector_size,
        )
        self._store.ensure_collection_exists()

    def close(self) -> None:
        """Закрывает соединение с хранилищем."""
        self._store.close()

    def create_embedding(self, text: str) -> list[float]:
        """Создает эмбеддинг текста."""
        return self._llm.create_embedding(text=text)

    def upsert_vector(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        *,
        check_similarity: bool = True,
        similarity_threshold: float | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Записывает вектор с проверкой похожести для дедупликации."""
        payload = metadata or {}
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )
        decision = SimilarityDecision(
            action="inserted",
            similarity_score=None,
            existing_id=None,
        )
        target_id = vector_id

        if check_similarity:
            similar = self._check_similarity(
                vector=vector,
                top_k=1,
                user_id=user_id,
                exclude_vector_id=vector_id,
            )
            if similar:
                best = similar[0]
                if best.score >= threshold:
                    incoming_text = str(payload.get("text", "")).strip().lower()
                    stored_text = str(best.metadata.get("text", "")).strip().lower()
                    if incoming_text and stored_text and incoming_text == stored_text:
                        return {
                            "action": "skipped",
                            "similarity_score": best.score,
                            "existing_id": best.point_id,
                            "vector_id": best.point_id,
                        }
                    decision = SimilarityDecision(
                        action="updated",
                        similarity_score=best.score,
                        existing_id=best.point_id,
                    )
                    target_id = best.point_id
                else:
                    decision = SimilarityDecision(
                        action="inserted",
                        similarity_score=best.score,
                        existing_id=best.point_id,
                    )

        self._store.upsert_vector(vector_id=target_id, vector=vector, metadata=payload)
        return {
            "action": decision.action,
            "similarity_score": decision.similarity_score,
            "existing_id": decision.existing_id,
            "vector_id": target_id,
        }

    def upsert_document(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None = None,
        vector_id: str | None = None,
        check_similarity: bool = True,
        similarity_threshold: float | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Преобразует текст в эмбеддинг и записывает запись."""
        vector = self.create_embedding(text)
        current_id = vector_id or str(uuid.uuid4())
        payload = {"text": text, "scope": "user"}
        if metadata:
            payload.update(metadata)
        return self.upsert_vector(
            vector_id=current_id,
            vector=vector,
            metadata=payload,
            check_similarity=check_similarity,
            similarity_threshold=similarity_threshold,
            user_id=user_id,
        )

    def upsert_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        check_similarity: bool = True,
        similarity_threshold: float | None = None,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Записывает список документов."""
        results: list[dict[str, Any]] = []
        for item in documents:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            metadata = item.get("metadata")
            vector_id = item.get("id")
            results.append(
                self.upsert_document(
                    text=text,
                    metadata=metadata if isinstance(metadata, dict) else None,
                    vector_id=str(vector_id) if vector_id else None,
                    check_similarity=check_similarity,
                    similarity_threshold=similarity_threshold,
                    user_id=user_id,
                )
            )
        return results

    def query_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет похожие записи по вектору."""
        hits = self._store.query_by_vector(vector=vector, top_k=top_k, user_id=user_id)
        return [self._hit_to_dict(hit) for hit in hits]

    def query_by_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Ищет похожие записи по тексту."""
        vector = self.create_embedding(text)
        return self.query_by_vector(vector, top_k=top_k, user_id=user_id)

    def fetch_vectors(self, vector_ids: list[str]) -> list[dict[str, Any]]:
        """Получает записи по списку идентификаторов."""
        return self._store.fetch_vectors(vector_ids)

    def describe_index_stats(self) -> dict[str, Any]:
        """Возвращает статистику хранилища."""
        return self._store.describe_index_stats()

    def _check_similarity(
        self,
        *,
        vector: list[float],
        top_k: int = 1,
        user_id: int | None = None,
        exclude_vector_id: str | None = None,
    ) -> list[SearchHit]:
        exclude: set[str] | None = {exclude_vector_id} if exclude_vector_id else None
        return self._store.query_by_vector(
            vector=vector,
            top_k=top_k,
            user_id=user_id,
            exclude_ids=exclude,
        )

    @staticmethod
    def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
        return {"id": hit.point_id, "score": hit.score, "metadata": hit.metadata}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_settings()
    cfg.require_openai()
    cfg.require_weaviate()

    manager = PineconeManager()
    try:
        logger.info("Проверка подключения: %s", manager.describe_index_stats())
        first = manager.upsert_document(
            text="Я хочу полететь на Марс",
            metadata={"user_id": 1, "scope": "user"},
            user_id=1,
        )
        second = manager.upsert_document(
            text="Я бы полетел на Марс",
            metadata={"user_id": 1, "scope": "user"},
            user_id=1,
        )
        logger.info("Первый результат записи: %s", first)
        logger.info("Второй результат записи: %s", second)
        logger.info("Поиск по тексту: %s", manager.query_by_text("Марс", top_k=3, user_id=1))
    finally:
        manager.close()

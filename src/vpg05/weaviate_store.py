"""Weaviate Cloud хранилище с self-provided векторами."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

_DUP_PATTERN = re.compile(r"already\s+exists|409|duplicate|unique|conflict", re.IGNORECASE)


@dataclass(frozen=True)
class SearchHit:
    """Результат семантического поиска."""

    point_id: str
    score: float
    metadata: dict[str, Any]


class WeaviateMemoryStore:
    """CRUD-операции поверх коллекции Weaviate."""

    def __init__(
        self,
        *,
        cluster_url: str,
        api_key: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._url = cluster_url.rstrip("/")
        self._api_key = api_key
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client: Any | None = None

    def _connect(self) -> Any:
        if self._client is None:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._url,
                auth_credentials=Auth.api_key(self._api_key),
                skip_init_checks=True,
            )
        return self._client

    def close(self) -> None:
        """Закрывает клиент Weaviate."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def ensure_collection_exists(self) -> None:
        """Создает коллекцию при первом запуске."""
        client = self._connect()
        if client.collections.exists(self._collection_name):
            return
        client.collections.create(
            name=self._collection_name,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.INT),
                Property(name="username", data_type=DataType.TEXT),
                Property(name="first_name", data_type=DataType.TEXT),
                Property(name="last_name", data_type=DataType.TEXT),
                Property(name="scope", data_type=DataType.TEXT),
            ],
        )

    def describe_index_stats(self) -> dict[str, Any]:
        """Возвращает базовую статистику коллекции."""
        client = self._connect()
        collection = client.collections.get(self._collection_name)
        aggregate = collection.aggregate.over_all(total_count=True)
        total_count = getattr(aggregate, "total_count", 0)
        return {
            "collection_name": self._collection_name,
            "vector_size": self._vector_size,
            "total_count": int(total_count or 0),
        }

    def upsert_vector(self, *, vector_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Добавляет или обновляет запись по ID."""
        if len(vector) != self._vector_size:
            raise ValueError(f"Размерность вектора {len(vector)} != {self._vector_size}")
        client = self._connect()
        collection = client.collections.get(self._collection_name)
        self._insert_or_replace(collection, vector_id, metadata, vector)

    @staticmethod
    def _insert_or_replace(collection: Any, uid: str, metadata: dict[str, Any], vector: list[float]) -> None:
        data_api = collection.data
        try:
            data_api.insert(uuid=uid, properties=metadata, vector=vector)
        except Exception as exc:  # noqa: BLE001
            if not _DUP_PATTERN.search(str(exc)):
                raise
            if hasattr(data_api, "replace"):
                data_api.replace(uuid=uid, properties=metadata, vector=vector)
            elif hasattr(data_api, "delete_by_id"):
                data_api.delete_by_id(uid)
                data_api.insert(uuid=uid, properties=metadata, vector=vector)
            else:
                raise

    def fetch_vectors(self, vector_ids: list[str]) -> list[dict[str, Any]]:
        """Получает записи по списку ID."""
        client = self._connect()
        collection = client.collections.get(self._collection_name)
        result: list[dict[str, Any]] = []
        for vector_id in vector_ids:
            obj = collection.query.fetch_object_by_id(uuid=vector_id)
            if obj is None:
                continue
            props = dict(obj.properties or {})
            result.append({"id": str(obj.uuid), "metadata": props})
        return result

    def query_by_vector(
        self,
        *,
        vector: list[float],
        top_k: int = 5,
        user_id: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        """Поиск похожих записей по вектору."""
        client = self._connect()
        collection = client.collections.get(self._collection_name)
        filters = None
        if user_id is not None:
            filters = Filter.by_property("user_id").equal(user_id)
        exclude = exclude_ids or set()
        fetch_limit = max(top_k * 5, top_k + 10, 25)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=fetch_limit,
            filters=filters,
            return_metadata=MetadataQuery(certainty=True, distance=True),
            return_properties=["text", "user_id", "username", "first_name", "last_name", "scope"],
        )
        hits: list[SearchHit] = []
        for obj in response.objects:
            point_id = str(obj.uuid)
            if point_id in exclude:
                continue
            metadata = dict(obj.properties or {})
            hits.append(
                SearchHit(
                    point_id=point_id,
                    score=self._score_from_metadata(obj.metadata),
                    metadata=metadata,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    @staticmethod
    def _score_from_metadata(metadata: Any) -> float:
        """
        Выше score = релевантнее.

        Для cosine-метрики в Weaviate обычно доступен ``distance``; для нормализованных векторов
        это ``1 - cosine_similarity``. Тогда ``similarity = 1 - distance`` стабильнее, чем
        полагаться на ``certainty`` во всех случаях.
        """
        if metadata is None:
            return 0.0
        distance = getattr(metadata, "distance", None)
        if distance is not None:
            similarity = 1.0 - float(distance)
            if similarity < 0.0:
                return 0.0
            if similarity > 1.0:
                return 1.0
            return similarity
        certainty = getattr(metadata, "certainty", None)
        if certainty is not None:
            return float(certainty)
        return 0.0

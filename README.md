# VPg05: Telegram-бот + Weaviate + эмбеддинги (ProxyAPI)

## Краткое описание

**Что делает.** Учебный Telegram-бот с **долговременной векторной памятью** сообщений пользователя: эмбеддинги через **OpenAI-совместимый API** ([ProxyAPI](https://proxyapi.ru/docs/overview)), хранение и поиск в **[Weaviate Cloud](https://console.weaviate.cloud/)**. Перед записью выполняется проверка близости (cosine similarity) с порогом `SIMILARITY_THRESHOLD`, чтобы уменьшить дубли (в логах это видно как `inserted` / `updated` / `skipped`).

Историческое имя **`PineconeManager`** в `src/vpg05/pinecone_manager.py` оставлено для совместимости с формулировками ДЗ, но внутри используется **Weaviate**, не Pinecone.

**Как запускать.** `cp .env.example .env`, задать `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY` (см. комментарии в примере). Затем `docker compose up -d --build`. Логи: `docker compose logs -f bot`. Остановка: `docker compose down`.

Исходящие запросы к **Telegram Bot API**, к ProxyAPI и к **Weaviate** идут по **HTTPS**. Входящий HTTP/TLS-сервер приложение **не поднимает** — бот работает через **long polling**.

Структура кода согласована с [Guide](https://github.com/SpiritWalker84/Guide) (`docs/conventions/`).

## Стек

Python **3.12**, **weaviate-client** v4, **OpenAI SDK**, **python-telegram-bot**, **python-dotenv**, Docker / Compose.

## Структура

| Путь | Назначение |
|------|------------|
| `src/vpg05/config.py` | Загрузка настроек из `.env` |
| `src/vpg05/embeddings.py` | OpenAI-совместимый клиент (эмбеддинги + чат) |
| `src/vpg05/weaviate_store.py` | Низкоуровневое хранилище Weaviate |
| `src/vpg05/pinecone_manager.py` | Менеджер памяти (интерфейс ДЗ) + дедупликация |
| `src/vpg05/bot.py` | Telegram-бот (handlers) |
| `main.py` | Точка входа для контейнера/локального запуска |
| `.env.example` | Переменные окружения |
| `docker-compose.yml`, `Dockerfile` | Запуск бота в контейнере |
| `tests/` | Pytest |

## Запуск

### Docker Compose (основной способ)

```bash
cp .env.example .env
# WEAVIATE_URL, WEAVIATE_API_KEY, TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

docker compose up -d --build
docker compose logs -f bot
```

Контейнер: `python main.py`.

### Локально

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 main.py
```

Ручная проверка памяти без Telegram (нужны ключи в `.env`):

```bash
PYTHONPATH=src python3 -m vpg05.pinecone_manager
```

## Команды бота

| Команда | Действие |
|---------|----------|
| `/start` | Короткое приветствие |
| `/help` | Справка |
| Текст (не команда) | Ответ LLM + запись **только текста пользователя** в память |

## Конфигурация

См. **`.env.example`**. `EMBEDDING_DIMENSION` должен совпадать с размерностью векторов в классе Weaviate (по умолчанию **1536** для `text-embedding-3-small`).

| Переменная | Обязательность | Описание |
|------------|----------------|----------|
| `OPENAI_API_KEY` | да | Ключ ProxyAPI / совместимого API |
| `OPENAI_API_BASE` | нет | По умолчанию `https://api.proxyapi.ru/openai/v1`; запасной ключ `OPENAI_BASE_URL` |
| `OPENAI_CHAT_MODEL` | нет | Модель для ответов в чате |
| `OPENAI_EMBEDDING_MODEL` | нет | Модель для эмбеддингов |
| `EMBEDDING_DIMENSION` | нет | Размерность векторов (**1536** для `text-embedding-3-small`) |
| `WEAVIATE_URL` | да | REST endpoint кластера |
| `WEAVIATE_API_KEY` | да | API key Weaviate Cloud |
| `WEAVIATE_COLLECTION_NAME` | нет | Имя класса в Weaviate (по умолчанию `Vpg05Memory`) |
| `TELEGRAM_BOT_TOKEN` | да | Токен от [@BotFather](https://t.me/BotFather) |
| `SIMILARITY_THRESHOLD` | нет | Порог похожести (0..1). Слишком низкий порог часто даёт ложные совпадения |
| `LOG_LEVEL` | нет | Уровень логирования (по умолчанию `INFO`) |

Секреты в git не коммитить.

### Где смотреть результат дедупликации

В обычном режиме Telegram это **не показывает пользователю** — смотрите строки логов вида `Memory upsert result: ...` (`docker compose logs -f bot`).

## Тесты

```bash
PYTHONPATH=src pytest
```

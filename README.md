# RAG + MCP Homework

# RAG-система с MCP и GigaChat для поиска по документам

## Описание проекта

В рамках проекта реализована простая Retrieval-Augmented Generation (RAG) система для поиска информации по локальному корпусу документов.

Система использует векторную базу данных Qdrant для хранения эмбеддингов текстовых фрагментов, локальный MCP-сервер для доступа к базе знаний и языковую модель GigaChat в качестве агента.

В качестве корпуса используется документ с информацией о Южной Корее. Документ автоматически разбивается на чанки, индексируется в Qdrant и становится доступным для семантического поиска.

Основные возможности системы:

* автоматическая загрузка и индексация документов;
* разбиение текста на чанки;
* построение эмбеддингов с помощью FastEmbed;
* хранение данных в локальном Qdrant;
* поиск релевантных фрагментов по смыслу запроса;
* доступ к базе знаний через MCP-инструмент;
* использование агента на основе GigaChat для выполнения поисковых запросов.

Архитектура решения:

```text
Документ
    ↓
Chunking
    ↓
Embeddings (FastEmbed)
    ↓
Qdrant
    ↓
MCP Server
    ↓
GigaChat Agent
```

В ходе работы была проведена оценка качества поиска на наборе из 20 тестовых запросов. Для каждого запроса проверялось наличие релевантного чанка среди первых трёх результатов поиска (Top-3 Accuracy).

## Структура проекта

```text
agent_rag/
│
├── docs/
│   └── south_korea.md
│
├── qdrant_data/
│
├── ingest.py
├── search_demo.py
├── local_mcp_server.py
├── agent_demo.py
├── evaluate.py
├── eval_queries.csv
│
├── evaluation_report1.xslx
├── requirements.txt
└── README.md
```

## Стек

- Qdrant local vector database
- FastEmbed embeddings: `BAAI/bge-small-en-v1.5`
- Official Qdrant MCP server
- LangChain / LangGraph agent
- Python
- Gigachat

## Корпус

Корпус лежит в `docs/`.

## Установка

- Создайте и активируйте виртуальное окружение

- Установите необходимые библиотеки

- Настройте переменный окружения .env

## Разбиение текста на чанки и индексация

Код для разбиение на чанки и создания векторной базы данных лежит в файле ingest.py 

## Запуск MCP-сервера

В проекте используется локальный MCP-сервер (инструмент qdrant_find):

local_mcp_server.py

Параметры инструмента:

query — поисковый запрос
k — количество возвращаемых результатов

Инструмент возвращает:

document_id
chunk_id
source
score
text

MCP-сервер запускается автоматически при запуске агента.

## Работа с запросами

Запуск агента: python agent_demo.py

Агент использует модель GigaChat и инструмент MCP qdrant_find.

Пример запроса:

Найди top-3 фрагмента про universities in Korea.

Пример результата:

document_id=south_korea
chunk_id=south_korea_chunk_004
source=docs/south_korea.md
score=0.7715

Leading universities include:
- Seoul National University
- Korea University
- Yonsei University
- KAIST
- POSTECH

## Валидация на 15–20 запросах

Набор запросов хранится в файле: eval.csv

Файл для запуска валидации: evaluate.py


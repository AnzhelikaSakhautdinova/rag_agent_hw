# RAG + MCP Homework

## Goal

This project demonstrates a small RAG knowledge base connected to an AI agent through an MCP server.

Pipeline:

documents → chunks → metadata → Qdrant index → MCP server → agent → top-k search results

## Stack

- Qdrant local vector database
- FastEmbed embeddings: `BAAI/bge-small-en-v1.5`
- Official Qdrant MCP server
- LangChain / LangGraph agent
- Python

## Corpus

The corpus is located in `docs/`.

It contains 5 markdown documents:

- `ai_risks.md`
- `rag_intro.md`
- `agents.md`
- `mcp.md`
- `eval.md`

Each document is split into chunks. Each chunk has metadata:

- `document_id`
- `chunk_id`
- `source`
- `title`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install mcp-server-qdrant
# ⚖️ Legal Research Assistant

A RAG-based chatbot that searches and summarizes **case law** and **statutes** from public legal databases — [CourtListener](https://www.courtlistener.com/) and [Congress.gov](https://congress.gov/).

Built with **FastAPI · LangChain · ChromaDB · GPT-4o**.

---

## Features

- **Dual-source ingestion** — pull opinions from CourtListener and bills/summaries from Congress.gov into a persistent vector store with one API call
- **RAG-powered chat** — GPT-4o answers questions grounded in the retrieved legal texts, with cited sources
- **Streaming responses** — answers stream token-by-token via SSE
- **Conversation history** — multi-turn chat with rolling context window
- **Source filtering** — restrict answers to case law, statutes, or both
- **Clean chat UI** — self-contained HTML/CSS/JS frontend served by FastAPI

---

## Project Structure

```
legal-research-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & middleware
│   │   ├── api/
│   │   │   └── routes.py        # /health  /chat  /chat/stream  /search
│   │   ├── core/
│   │   │   ├── config.py        # Pydantic settings (env vars)
│   │   │   └── embeddings.py    # OpenAI embeddings + ChromaDB client
│   │   ├── services/
│   │   │   ├── courtlistener.py # CourtListener API client
│   │   │   ├── congress.py      # Congress.gov API client
│   │   │   └── rag.py           # LangChain RAG pipeline
│   │   └── models/
│   │       └── schemas.py       # Pydantic request/response models
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html               # Chat UI
│   ├── style.css
│   └── app.js
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12+ **or** Docker + Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/legal-research-assistant.git
cd legal-research-assistant

cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally the other keys)
```

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) — the chat UI is served from there.

### 3b. Run locally (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser, or navigate to [http://localhost:8000](http://localhost:8000).

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### `GET /health`

Returns API status and the number of indexed document chunks.

### `POST /chat`

Ask a legal question (non-streaming).

```json
{
  "question": "What did the Supreme Court say about Miranda rights?",
  "history": [],
  "source_filter": "all"
}
```

### `POST /chat/stream`

Same payload — returns a **Server-Sent Events** stream. Event types:

| Type | Data |
|------|------|
| `sources` | Array of source document objects |
| `token` | A string token from GPT-4o |
| `done` | Empty — stream complete |
| `error` | Error message string |

### `POST /search`

Search one or both databases and ingest results into ChromaDB.

```json
{
  "query": "Fourth Amendment search and seizure",
  "source": "both",
  "max_results": 10,
  "ingest": true
}
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `COURTLISTENER_API_TOKEN` | — | Optional — higher rate limits |
| `CONGRESS_API_KEY` | — | Optional — required for full access |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data |

---

## Getting API Keys

- **OpenAI** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **CourtListener** — Create a free account at [courtlistener.com](https://www.courtlistener.com/sign-in/), then generate a token under your profile. Unauthenticated access also works but is rate-limited.
- **Congress.gov** — Request a free key at [api.congress.gov/sign-up](https://api.congress.gov/sign-up/).

---

## How It Works

```
User question
     │
     ▼
ChromaDB vector search  ──►  Top-K relevant chunks (case law + statutes)
     │
     ▼
LangChain RAG chain  ──►  Context + history + question ──►  GPT-4o
     │
     ▼
Streamed answer with cited sources
```

The `/search` endpoint lets you populate the vector store on demand — search for any legal topic, ingest the results, then chat about them immediately.

---

## Disclaimer

This tool is for **research and informational purposes only**. It does not constitute legal advice. Always consult a qualified attorney for legal matters.

---

## License

MIT

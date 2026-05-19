# codon-rag

A fully local Retrieval-Augmented Generation (RAG) system supporting multiple independent knowledge bases. Users can ask natural-language questions against any configured KB and receive grounded answers with source citations — no cloud APIs, no API keys required.

---

## How It Works

```
data/kb_registry.json (KB config + build manifests)
data/corpus/<kb_id>/*.md  →  ingest  →  ChromaDB (chroma_db/<collection>)
                                              ↓
User question  →  embed (MiniLM)  →  similarity search (top-K)  →  threshold filter
                                              ↓
                               src/rag.py  or  src/rag_lc.py
                                              ↓
                          Ollama (ministral-3:3b @ localhost:11434)
                                              ↓
                                   Answer + source citations
                                              ↓
              ┌──────────────────────┬─────────────────────────────┐
              │  app.py (Streamlit)  │  workbench/ (FastAPI + SPA) │
              │  localhost:8501      │  localhost:8000              │
              └──────────────────────┴─────────────────────────────┘
```

1. **Ingest** — Markdown files are chunked, embedded with a local sentence-transformer model, and stored in ChromaDB. Ingest is incremental: unchanged files (identified by `file_hash`) are skipped; deleted files can be removed surgically.
2. **Retrieve** — A user question is embedded the same way and the nearest chunks are pulled from ChromaDB using cosine similarity, then filtered by `RELEVANCE_THRESHOLD`.
3. **Generate** — The filtered chunks are inserted into a prompt and sent to a locally running Ollama LLM, which produces a grounded answer with source citations.
4. **UI** — Two parallel frontends share the same `src/` modules, `chroma_db/`, and `data/` directories. Run one or both simultaneously.

### Stack

| Layer | Technology |
|---|---|
| Corpus | Multiple independent knowledge bases configured in `data/kb_registry.json` |
| Chunking | Paragraph-split (default) or fixed-character with configurable overlap |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (persistent, cosine distance, metadata: `kb_id`, `doc_id`, `file_hash`) |
| Ingestion | Incremental — skips unchanged files; supports surgical file/KB deletion |
| LLM | Ollama — `ministral-3:3b` running locally at `http://127.0.0.1:11434` |
| RAG (raw) | `src/rag.py` — plain Python + `urllib`, no framework |
| RAG (LangChain) | `src/rag_lc.py` — LangChain LCEL chain |
| Streamlit UI | `app.py` + `pages/` — classic multipage app for KB management and chat |
| Workbench UI | `workbench/` — FastAPI backend + vanilla-JS SPA with SSE streaming |
| Tunneling | `ngrok.exe` — expose Streamlit to the internet (optional) |

---

## Project Layout

```
codon-rag/
├── app.py                       # Streamlit entry point (Landing Page)
├── pages/
│   ├── 1_Knowledge_Bases.py     # Streamlit KB management UI
│   └── 2_Ask.py                 # Streamlit chat interface
│
├── workbench/                   # FastAPI workbench — runs alongside Streamlit
│   ├── __init__.py
│   ├── main.py                  # All API routes (KB CRUD, build SSE, ask SSE)
│   ├── requirements.txt         # fastapi + uvicorn + python-multipart
│   └── static/
│       └── index.html           # Single-page app (Build KB + Ask tabs)
│
├── requirements.txt             # Core Python dependencies
├── ngrok.exe                    # ngrok binary (optional, Windows)
│
├── src/
│   ├── config.py                # All tunable parameters — edit here first
│   ├── ingest.py                # Chunk → embed → write to ChromaDB (incremental)
│   ├── query.py                 # CLI retrieval debugger (no LLM)
│   ├── rag.py                   # Full RAG CLI (raw HTTP to Ollama)
│   └── rag_lc.py                # Full RAG CLI using LangChain
│
├── data/
│   ├── kb_registry.json         # Registry of KB configs + build manifests
│   ├── corpus/                  # Source documents (one subdir per KB)
│   └── test_questions.txt       # Canonical eval questions
│
├── chroma_db/                   # Persisted ChromaDB vector store (git-ignored)
└── notebooks/
    └── scratch.ipynb
```

---

## Prerequisites

### 1. Python 3.12+

```bash
python --version
```

### 2. Ollama

**macOS**
```bash
brew install ollama
ollama pull ministral-3:3b
ollama serve
```

**Windows / Linux** — download from [ollama.com/download](https://ollama.com/download), then:
```bash
ollama pull ministral-3:3b
```

Verify the daemon is running:
```bash
curl http://127.0.0.1:11434/api/tags
```

### 3. Python dependencies

```bash
# Core deps (embeddings, ChromaDB, LangChain, Streamlit)
pip install -r requirements.txt

# Workbench deps (FastAPI, uvicorn, multipart)
pip install -r workbench/requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd codon-rag
```

**Windows**
```bat
.\setup.bat
.venv\Scripts\activate
```

**macOS / Linux**
```bash
bash setup.sh
source .venv/bin/activate
```

Both scripts create `.venv/`, install `requirements.txt` and `workbench/requirements.txt` into it, and print the next-step commands. Paths are resolved relative to the repo root via `pathlib` — no manual path editing required.

> If you prefer to manage your own environment, install manually:
> ```bash
> python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
> pip install -r requirements.txt -r workbench/requirements.txt
> ```

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
# → http://localhost:8501
```

Three pages: Landing · Knowledge Bases · Ask.

### Workbench UI (FastAPI + SPA)

```bash
uvicorn workbench.main:app --port 8000 --reload
# → http://localhost:8000
```

Both UIs can run simultaneously — they share `src/`, `chroma_db/`, and `data/` without conflict.

The workbench has two tabs:

**Build KB** — select an existing KB or create a new one, upload `.md`/`.txt` corpus files, configure chunking strategy and retrieval index parameters, then click "Build Knowledge Base." Progress streams live via SSE. On success, a build manifest is stamped into `kb_registry.json` recording exactly what the index was built with.

**Ask** — pick a KB from the sidebar (showing its build manifest), adjust top-K and relevance threshold, and chat. The LLM response streams token-by-token. The right panel shows every retrieved chunk with its cosine distance and a visual bar — chunks filtered by the threshold appear dimmed with dashed borders.

Additional workbench features: light/dark mode toggle, Ollama online/offline status pill, live chunk count per KB.

### CLI — Ingest

```bash
python src/ingest.py --kb codon                    # incremental (skips unchanged files)
python src/ingest.py --kb codon --delete-file f.md # drop one file's chunks
python src/ingest.py --kb codon --delete-kb        # drop entire KB + registry entry
```

### CLI — Retrieval debugger (no LLM)

```bash
python src/query.py --kb codon "What industries does Codon work in?"
python src/query.py --kb codon   # interactive
```

Prints ranked chunks with cosine distances. Use this to tune `TOP_K`, `RELEVANCE_THRESHOLD`, and `CHUNK_STRATEGY` without waiting for an LLM response.

### CLI — Full RAG

```bash
# Raw Python
python src/rag.py --kb codon "What makes Codon different from other AI consultancies?"
python src/rag.py --kb codon   # interactive

# LangChain
python src/rag_lc.py --kb codon "What AI services does Codon offer?"
```

### Public tunnel (optional)

```bash
./ngrok http 8501    # expose Streamlit
./ngrok http 8000    # expose workbench
```

---

## Knowledge Base Registry

`data/kb_registry.json` is the single source of truth for all KB metadata. Each entry maps a `kb_id` to a ChromaDB collection, corpus directory, and (after first build) a frozen build manifest:

```json
{
  "codon": {
    "kb_id": "codon",
    "collection": "codon_docs",
    "corpus_dir": "data/corpus",
    "description": "Codon Consulting public website content",
    "build": {
      "embed_model":    "all-MiniLM-L6-v2",
      "chunk_strategy": "paragraph",
      "chunk_size":     null,
      "chunk_step":     null,
      "distance":       "cosine",
      "top_k":          5,
      "threshold":      0.65,
      "built_at":       "2026-05-19T10:22:00+00:00",
      "total_chunks":   312,
      "sources":        8
    }
  }
}
```

The `build` key is written by the workbench after a successful ingest. It records exactly what parameters were used so the Ask page always queries with the right embedding model. New KBs have no `build` key until their first ingest run.

---

## Configuration (`src/config.py`)

All tunable knobs for the CLI and Streamlit UI live here. The workbench accepts these as request parameters per-build, so you can configure each KB independently without touching this file.

```python
EMBED_MODEL         = "all-MiniLM-L6-v2"   # 384-dim, ~80 MB, no API key
OLLAMA_MODEL        = "ministral-3:3b"
CHUNK_STRATEGY      = "paragraph"           # "paragraph" | "fixed"
FIXED_CHUNK_SIZE    = 200                   # chars
FIXED_CHUNK_STEP    = 150                   # overlap step (fixed only)
TOP_K               = 5
RELEVANCE_THRESHOLD = 0.65                  # cosine distance cutoff
DISTANCE            = "cosine"              # ChromaDB hnsw:space
```

---

## Workbench API Reference

All routes are prefixed `/api/`. The SPA at `/` consumes them.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/kbs` | List all KBs with live chunk counts and build manifests |
| `POST` | `/api/kbs` | Create a new KB entry (no ingest) |
| `DELETE` | `/api/kbs/{kb_id}` | Drop collection + remove from registry |
| `GET` | `/api/kbs/{kb_id}/files` | List corpus files `[{name, size}]` |
| `POST` | `/api/kbs/{kb_id}/files` | Upload a `.md`/`.txt` file to the corpus |
| `DELETE` | `/api/kbs/{kb_id}/files/{filename}` | Remove a file from the corpus |
| `POST` | `/api/kbs/{kb_id}/build` | **SSE** — run ingest with given params; streams progress |
| `GET` | `/api/ollama/models` | Query local Ollama for available models |
| `POST` | `/api/ask` | **SSE** — embed question → retrieve chunks → stream LLM answer |

**Build request body** (`POST /api/kbs/{kb_id}/build`):
```json
{
  "embed_model":    "all-MiniLM-L6-v2",
  "chunk_strategy": "paragraph",
  "chunk_size":     200,
  "chunk_step":     50,
  "distance":       "cosine",
  "top_k":          5,
  "threshold":      0.65
}
```

**Ask SSE event sequence** (`POST /api/ask`):
```
data: {"type": "chunks",  "chunks": [...kept...], "all_chunks": [...all...]}
data: {"type": "token",   "content": "Hello "}
data: {"type": "token",   "content": "world"}
data: {"type": "done",    "elapsed_ms": 1843}
```

---

## Two RAG Implementations

| | `rag.py` | `rag_lc.py` / `pages/2_Ask.py` |
|---|---|---|
| Framework | Raw Python (`urllib`, `chromadb`) | LangChain LCEL |
| Ollama | Direct HTTP POST to `/api/chat` | `ChatOllama` |
| Embeddings | `SentenceTransformer` directly | `HuggingFaceEmbeddings` |
| ChromaDB | `chromadb.PersistentClient` | `langchain_chroma.Chroma` |
| Relevance filter | Distance check in list comprehension | `RunnableLambda` |

The workbench (`workbench/main.py`) follows the `rag.py` pattern — raw Python, no LangChain — and adds SSE streaming on top.

---

## Design Notes

**Fully local stack.** No OpenAI, no Anthropic, no cloud embeddings. The only external network call in normal operation is `http://127.0.0.1:11434` (Ollama).

**Build manifests as frozen records.** When the workbench builds a KB it stamps the full parameter set into `kb_registry.json`. The Ask page reads the manifest's `embed_model` and uses it for query embedding — ensuring the query vector lives in the same space as the index vectors. To experiment with different params, rebuild (which overwrites the manifest) or create a new KB ID.

**Incremental ingest (CLI only).** `src/ingest.py` skips files whose `file_hash` hasn't changed. The workbench always does a full rebuild (drops and recreates the collection) to guarantee consistency with the new params.

**SSE over polling.** Both the build progress and LLM streaming use SSE via FastAPI's `StreamingResponse`. The frontend reads them with the Fetch Streams API (`resp.body.getReader()`), avoiding the need for WebSockets or a client-side SSE library. A `keepalive` SSE comment is emitted during long embeds to prevent proxy timeouts.

**ChromaDB + threading.** ChromaDB's SQLite layer is not thread-safe across OS threads. Any code that creates a `PersistentClient` calls `chromadb.api.client.SharedSystemClient.clear_system_cache()` first to get a fresh, thread-local connection. See CLAUDE.md for the full explanation.

**SentenceTransformer cache.** The workbench caches loaded models in a module-level dict (`_model_cache`). Model loading takes 1–3 s; subsequent requests pay zero load time. Models are thread-safe for inference.

**Config hot-reload (Streamlit only).** The Streamlit UI calls `importlib.reload(cfg)` on startup so that edits to `config.py` take effect on the next rerun without restarting the server. The workbench reads `config.py` once at import time; restart uvicorn to pick up config changes.

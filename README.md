# codon-rag

A fully local Retrieval-Augmented Generation (RAG) system supporting multiple independent knowledge bases. Users can ask natural-language questions against any configured KB and receive grounded answers with source citations — no cloud APIs, no API keys required.

---

## How It Works

```
data/kb_registry.json (KB config)
data/corpus/<kb_id>/*.md  →  src/ingest.py (incremental)  →  ChromaDB (chroma_db/<collection>)
                                              ↓
User question  →  embed (MiniLM)  →  similarity search (top-K)
                                              ↓
                               src/rag.py  or  src/rag_lc.py
                                              ↓
                          Ollama (ministral-3:3b @ localhost:11434)
                                              ↓
                                   Answer + source citations
                                              ↓
                       app.py  (Streamlit multipage UI — Landing / KBs / Ask)
```

1. **Ingest** — Markdown files are chunked, embedded with a local sentence-transformer model, and stored in ChromaDB. Ingest is incremental: unchanged files (identified by `file_hash`) are skipped; deleted files can be removed surgically.
2. **Retrieve** — A user question is embedded the same way and the nearest chunks are pulled from ChromaDB using cosine similarity, then filtered by `RELEVANCE_THRESHOLD`.
3. **Generate** — The filtered chunks are inserted into a prompt and sent to a locally running Ollama LLM, which produces a grounded answer with source citations.
4. **UI** — A three-page Streamlit app ties everything together: a landing page, a KB management page, and a chat interface.

### Stack

| Layer | Technology |
|---|---|
| Corpus | Multiple independent knowledge bases configured in `data/kb_registry.json` |
| Chunking | Paragraph-split (default) or fixed-character with overlap |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (persistent, cosine distance, metadata: `kb_id`, `doc_id`, `file_hash`) |
| Ingestion | Incremental — skips unchanged files; supports surgical file/KB deletion |
| LLM | Ollama — `ministral-3:3b` running locally at `http://127.0.0.1:11434` |
| RAG (raw) | `src/rag.py` — plain Python + `urllib`, no framework |
| RAG (LangChain) | `src/rag_lc.py` — LangChain LCEL chain |
| UI | Streamlit multipage app (`app.py` + `pages/1_Knowledge_Bases.py` + `pages/2_Ask.py`) |
| Tunneling | `ngrok.exe` — expose Streamlit to the internet (optional) |

---

## Project Layout

```
codon-rag/
├── app.py                       # Streamlit entry point (Landing Page)
├── pages/
│   ├── 1_Knowledge_Bases.py     # KB management UI (create, upload, delete, re-ingest)
│   └── 2_Ask.py                 # Chat interface — per-KB RAG with retrieved chunk viewer
├── requirements.txt             # Python dependencies
├── ngrok.exe                    # ngrok binary for public tunneling (optional)
├── src/
│   ├── config.py                # All tunable parameters — edit here first
│   ├── ingest.py                # Chunk → embed → write to ChromaDB (incremental)
│   ├── query.py                 # CLI retrieval debugger (no LLM)
│   ├── rag.py                   # Full RAG CLI (raw HTTP to Ollama)
│   └── rag_lc.py                # Full RAG CLI using LangChain
├── data/
│   ├── kb_registry.json         # Registry of knowledge base configs
│   ├── corpus/                  # Source documents, one subdirectory per KB
│   └── test_questions.txt       # Canonical eval questions
├── chroma_db/                   # Persisted ChromaDB vector store (git-ignored)
└── notebooks/
    └── scratch.ipynb            # Scratch notebook for experimentation
```

---

## Prerequisites

### 1. Python 3.12+

```bash
python --version
```

### 2. Ollama

Ollama runs the LLM locally.

**macOS**
```bash
brew install ollama
```
Or download from [ollama.com/download](https://ollama.com/download).

**Windows** — download and run the installer from [ollama.com/download](https://ollama.com/download/windows).

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the model and start the server:
```bash
ollama pull ministral-3:3b
ollama serve       # macOS/Windows: the desktop app does this automatically
```

Verify:
```bash
curl http://127.0.0.1:11434/api/tags
```

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd codon-rag
pip install -r requirements.txt
```

Paths are resolved relative to the repo root via `pathlib` — no manual path editing required.

Ingest a knowledge base to build its vector store:
```bash
python src/ingest.py --kb codon
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Three pages:

- **Landing** — project overview.
- **Knowledge Bases** — create KBs, upload/delete documents, trigger re-ingest, view chunk counts and last-ingested timestamps.
- **Ask** — select a KB from the sidebar, then ask questions. Each answer shows an expandable "Retrieved chunks" section with source file and cosine distance for every chunk used.

### CLI — Ingest

```bash
python src/ingest.py --kb codon                    # incremental ingest (skips unchanged files)
python src/ingest.py --kb codon --delete-file f.md # surgically remove one file's chunks
python src/ingest.py --kb codon --delete-kb        # drop entire KB and remove from registry
```

### CLI — Retrieval debugger (no LLM)

```bash
python src/query.py --kb codon "What industries does Codon work in?"
python src/query.py --kb codon   # interactive prompt
```

Prints ranked chunks with cosine distances. Use this to tune `TOP_K`, `RELEVANCE_THRESHOLD`, and `CHUNK_STRATEGY` without waiting for an LLM response.

### CLI — Full RAG

```bash
# Raw Python
python src/rag.py --kb codon "What makes Codon different from other AI consultancies?"
python src/rag.py --kb codon   # interactive prompt

# LangChain
python src/rag_lc.py --kb codon "What AI services does Codon offer?"
```

### Public tunnel (optional)

```bash
./ngrok http 8501
```

Requires a free ngrok account and a configured authtoken (`ngrok config add-authtoken <token>`).

---

## Knowledge Base Registry

KBs are defined in `data/kb_registry.json`. Each entry maps a `kb_id` to a ChromaDB collection and a corpus directory:

```json
{
  "codon": {
    "kb_id": "codon",
    "collection": "codon_docs",
    "corpus_dir": "data/corpus",
    "description": "Codon Consulting public website content"
  }
}
```

New KBs can be created from the UI or by editing this file directly and running `ingest.py --kb <id>`. The corpus directory path is relative to the repo root.

---

## Configuration (`src/config.py`)

All tunable knobs live here. **Edit this file, not the individual scripts.**

```python
# Paths — resolved relative to repo root, no manual editing needed
BASE_DIR   = pathlib.Path(__file__).parent.parent
CORPUS_DIR = str(BASE_DIR / "data" / "corpus")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
KB_REGISTRY_PATH = str(BASE_DIR / "data" / "kb_registry.json")

# ChromaDB
COLLECTION = "codon_docs"        # default collection name
DISTANCE   = "cosine"            # "cosine" | "l2" | "ip"

# Embedding
EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, local, no API key

# Ollama
OLLAMA_MODEL = "ministral-3:3b"

# Chunking
CHUNK_STRATEGY   = "paragraph"   # "paragraph" | "fixed"
FIXED_CHUNK_SIZE = 200           # chars — used only when CHUNK_STRATEGY == "fixed"
FIXED_CHUNK_STEP = 150           # overlap step for fixed chunking

# Retrieval
TOP_K               = 5
RELEVANCE_THRESHOLD = 0.65       # cosine distance cutoff — chunks above this are dropped
```

---

## Two RAG Implementations

| | `rag.py` | `rag_lc.py` / `pages/2_Ask.py` |
|---|---|---|
| Framework | Raw Python (`urllib`, `chromadb`, `sentence-transformers`) | LangChain LCEL |
| Ollama | Direct HTTP POST to `/api/chat` | `ChatOllama` |
| Embeddings | `SentenceTransformer` directly | `HuggingFaceEmbeddings` |
| ChromaDB | `chromadb.PersistentClient` | `langchain_chroma.Chroma` |
| Relevance filter | `RunnableLambda` — drops chunks with `dist > RELEVANCE_THRESHOLD` | Same — `RunnableLambda` filter applied in both CLI and UI |
| Primary use | Learning / debugging / minimal deps | Production UI |

---

## Corpus Documents (built-in `codon` KB)

Eight Markdown files scraped from `codon.se` (April 2026):

| File | Content |
|---|---|
| `01_company_overview.md` | Identity, mission, core team (Erik Fredlund, Founder & MD), founding 2019, Stockholm HQ |
| `02_services.md` | Analytics, custom ML models, AI-driven operations, NLP, computer vision, data strategy |
| `03_industries.md` | Manufacturing, process industries, life sciences, property management |
| `04_industrial_edge.md` | OT/IT integration, ISA-95 vertical strategy, Siemens Industrial Edge Partnership (Jan 2026) |
| `05_case_studies.md` | 14 client projects across IoT, agriculture, pharma, life sciences, NLP, retail, media |
| `06_blog_and_resources.md` | Blog posts, Data Science Friday events, ebook |
| `07_contact_and_careers.md` | Contact info (info@codon.se), careers/culture, role types |
| `08_sitemap_and_links.md` | Full URL inventory of codon.se |

`data/test_questions.txt` contains five canonical eval questions for validating retrieval quality after ingestion or config changes.

---

## Design Notes

**Fully local stack.** No OpenAI, no Anthropic, no cloud embeddings. The only network call in normal operation is to `http://127.0.0.1:11434` (Ollama).

**Incremental ingest.** Each chunk is stored with a `file_hash`. Re-running `ingest.py` skips files whose hash hasn't changed, making it safe to run after adding a single document without re-embedding the whole corpus.

**Relevance threshold applied everywhere.** Both CLI (`rag.py`, `rag_lc.py`) and the Streamlit UI filter out chunks with cosine distance above `RELEVANCE_THRESHOLD` via a `RunnableLambda`. If the UI returns "I couldn't find any relevant information," use `query.py` to inspect raw distances and tune the threshold.

**ChromaDB + Streamlit threading.** Streamlit's multi-threaded model can cause SQLite cross-thread violations if the ChromaDB client is cached with `@st.cache_resource`. The UI avoids this by calling `chromadb.api.client.SharedSystemClient.clear_system_cache()` before each connection, and by caching only the embedding model (which is thread-safe).

**Config hot-reload.** The UI explicitly calls `importlib.reload(cfg)` on startup so that changes to `config.py` (e.g., tuning `RELEVANCE_THRESHOLD`) take effect on the next Streamlit rerun without restarting the server.

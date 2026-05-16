# codon-rag

A fully local Retrieval-Augmented Generation (RAG) system that powers a **Codon Sales Assistant** chatbot. Users can ask natural-language questions about [Codon Consulting AB](https://www.codon.se) and receive grounded answers with source citations — no cloud APIs, no API keys required.

---

## How It Works

```
data/corpus/*.md  →  src/ingest.py  →  ChromaDB (chroma_db/)
                                              ↓
User question  →  embed (MiniLM)  →  similarity search (top-K)
                                              ↓
                               src/rag.py  or  src/rag_lc.py
                                              ↓
                          Ollama (ministral-3:3b @ localhost:11434)
                                              ↓
                                   Answer + source citations
                                              ↓
                              app.py  (Streamlit chat UI)
```

1. **Ingest** — Markdown files from `data/corpus/` are split into chunks, embedded with a local sentence-transformer model, and stored in a ChromaDB vector store on disk.
2. **Retrieve** — A user question is embedded the same way, and the nearest chunks are pulled from ChromaDB using cosine similarity.
3. **Generate** — The retrieved chunks are inserted into a prompt and sent to a locally running Ollama LLM, which produces a grounded answer with source citations.
4. **UI** — A Streamlit chat interface ties everything together for interactive use.

### Stack

| Layer | Technology |
|---|---|
| Corpus | 8 Markdown files in `data/corpus/` |
| Chunking | Paragraph-split (default) or fixed-character |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (persistent, cosine distance) |
| LLM | Ollama — `ministral-3:3b` running locally at `http://127.0.0.1:11434` |
| RAG (raw) | `src/rag.py` — plain Python + `urllib`, no framework |
| RAG (LangChain) | `src/rag_lc.py` — LangChain LCEL chain |
| UI | `app.py` — Streamlit chat interface |
| Tunneling | `ngrok.exe` — expose Streamlit to the internet (optional) |

---

## Project Layout

```
codon-rag/
├── app.py                  # Streamlit chat UI (uses rag_lc-style chain)
├── requirements.txt        # Python dependencies
├── ngrok.exe               # ngrok binary for public tunneling (optional)
├── src/
│   ├── config.py           # All tunable parameters — edit here first
│   ├── ingest.py           # Chunk → embed → write to ChromaDB
│   ├── query.py            # CLI retrieval debugger (no LLM)
│   ├── rag.py              # Full RAG CLI (raw HTTP to Ollama)
│   └── rag_lc.py           # Full RAG CLI using LangChain
├── data/
│   ├── corpus/             # Source documents (8 .md files)
│   │   ├── 01_company_overview.md
│   │   ├── 02_services.md
│   │   ├── 03_industries.md
│   │   ├── 04_industrial_edge.md
│   │   ├── 05_case_studies.md
│   │   ├── 06_blog_and_resources.md
│   │   ├── 07_contact_and_careers.md
│   │   └── 08_sitemap_and_links.md
│   └── test_questions.txt  # 5 canonical eval questions
├── chroma_db/              # Persisted ChromaDB vector store (git-ignored)
└── notebooks/
    └── scratch.ipynb       # Scratch notebook for experimentation
```

---

## Prerequisites

### 1. Python 3.12+

Download from [python.org](https://www.python.org/downloads/) and verify with:

```bash
python --version
```

### 2. Ollama

Ollama runs the LLM locally. Install it for your platform:

**macOS**

```bash
brew install ollama
```

Or download the app from [ollama.com/download](https://ollama.com/download).

**Windows**

Download and run the installer from [ollama.com/download](https://ollama.com/download/windows). The installer adds `ollama` to your PATH.

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installing, pull the model this project uses and start the server:

```bash
ollama pull ministral-3:3b
ollama serve           # starts the API at http://127.0.0.1:11434
```

> On macOS and Windows the Ollama desktop app starts the server automatically. On Linux, run `ollama serve` explicitly (or set it up as a systemd service).

Verify Ollama is running:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should see a JSON list that includes `ministral-3:3b`.

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

---

## Setup

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd codon-rag
pip install -r requirements.txt
```

If you're running on a different machine, update the two hardcoded paths in `src/config.py`:

```python
CORPUS_DIR = r"C:\path\to\your\codon-rag\data\corpus"
CHROMA_DIR = r"C:\path\to\your\codon-rag\chroma_db"
```

Then ingest the corpus to build the vector store:

```bash
python src/ingest.py
```

---

## Usage

### 1. Ingest (required before first use, and after any corpus changes)

```bash
python src/ingest.py
```

Reads all `.md` files from `CORPUS_DIR`, chunks them, embeds with `EMBED_MODEL`, drops and recreates the ChromaDB collection, and writes everything to `CHROMA_DIR`. Re-run whenever you add or edit corpus files, or change `CHUNK_STRATEGY` or `EMBED_MODEL`.

### 2. Retrieval debugger (no LLM)

```bash
python src/query.py "What industries does Codon work in?"
python src/query.py   # interactive prompt
```

Embeds the question, queries ChromaDB, and prints ranked chunks with cosine distances. Use this to tune `TOP_K`, `DISTANCE`, and `CHUNK_STRATEGY` without waiting for an LLM response.

### 3. Full RAG — raw Python (CLI)

```bash
python src/rag.py "What makes Codon different from other AI consultancies?"
python src/rag.py   # interactive prompt
```

Retrieves top-K chunks, applies the `RELEVANCE_THRESHOLD` filter (drops chunks with `dist > 0.45`), builds a prompt, calls Ollama via raw HTTP, and prints the answer together with retrieved chunk previews. If it returns "No relevant chunks found," either the question is outside the corpus or the threshold is too tight — inspect distances with `query.py` first.

### 4. Full RAG — LangChain (CLI)

```bash
python src/rag_lc.py "What AI services does Codon offer?"
```

Same pipeline as `rag.py` but implemented as a LangChain LCEL chain. Does **not** apply the relevance threshold filter.

### 5. Streamlit chat UI

```bash
streamlit run app.py
```

Launches the Codon Sales Assistant at `http://localhost:8501`. Shows an expandable "Retrieved chunks" section below each answer. The chain is cached with `@st.cache_resource` so the embedding model and vector store load only once per session.

### 6. Public tunnel (optional)

```bash
./ngrok http 8501
```

Exposes the local Streamlit server publicly via an ngrok tunnel. Requires a free ngrok account and a configured authtoken (`ngrok config add-authtoken <token>`).

---

## Configuration (`src/config.py`)

All tunable knobs live here. **Edit this file, not the individual scripts.**

```python
# Paths (update for your machine)
CORPUS_DIR = r"C:\...\data\corpus"
CHROMA_DIR = r"C:\...\chroma_db"

# ChromaDB
COLLECTION = "codon_docs"
DISTANCE   = "cosine"          # "cosine" | "l2" | "ip"

# Embedding
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, local, no API key

# Chunking
CHUNK_STRATEGY   = "paragraph"   # "paragraph" | "fixed"
FIXED_CHUNK_SIZE = 200           # chars — used only when CHUNK_STRATEGY == "fixed"

# Retrieval
TOP_K               = 5
RELEVANCE_THRESHOLD = 0.45       # cosine distance cutoff — only applied in rag.py
```

---

## Two RAG Implementations

The project maintains two parallel RAG implementations that produce equivalent results but differ in approach:

| | `rag.py` | `rag_lc.py` / `app.py` |
|---|---|---|
| Framework | Raw Python (`urllib`, `chromadb`, `sentence-transformers`) | LangChain LCEL |
| Ollama | Direct HTTP POST to `/api/chat` | `ChatOllama` |
| Embeddings | `SentenceTransformer` directly | `HuggingFaceEmbeddings` |
| ChromaDB | `chromadb.PersistentClient` | `langchain_chroma.Chroma` |
| Relevance filter | Yes — drops chunks with `dist > RELEVANCE_THRESHOLD` | No |
| Primary use | Learning / debugging / minimal deps | Production UI (`app.py`) |

---

## Corpus Documents

The knowledge base is 8 Markdown files scraped from `codon.se` (April 2026):

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

### Test Questions

`data/test_questions.txt` contains five canonical questions for validating retrieval quality after ingestion or config changes:

```
What industries does Codon work in?
What makes Codon different from other AI consultancies?
How do I contact Codon?
What is Codon's Forward Deployed Engineering model?
What AI services does Codon offer?
```

---

## Design Decisions & Gotchas

**Fully local stack.** No OpenAI, no Anthropic, no cloud embeddings. The only network call in normal operation is to `http://127.0.0.1:11434` (Ollama).

**Ingest is destructive.** `ingest.py` drops and recreates the ChromaDB collection on every run. This ensures idempotency but means re-ingesting from scratch — there is no incremental update path.

**Hardcoded Windows paths in `config.py`.** Update `CORPUS_DIR` and `CHROMA_DIR` when running on a new machine.

**LLM model hardcoded in two places.** `ministral-3:3b` appears in both `rag.py` and `app.py`/`rag_lc.py`. Centralizing it in `config.py` as `OLLAMA_MODEL` would be a clean improvement.

**Relevance threshold only in `rag.py`.** The LangChain chain in `rag_lc.py` and `app.py` does not filter by relevance distance. To add it, wrap the retriever in a `RunnableLambda` that filters by score.

**No incremental ingest.** Adding a single document requires re-embedding the entire corpus. This is acceptable at the current scale (8 files) but becomes expensive as the corpus grows.

---

## Extending the Project

**Add corpus documents:** Drop new `.md` files in `data/corpus/` and re-run `python src/ingest.py`.

**Tune retrieval:** Adjust `TOP_K`, `RELEVANCE_THRESHOLD`, `CHUNK_STRATEGY`, or `DISTANCE` in `config.py`, re-ingest, then test with `query.py`.

**Swap the LLM:** Change `OLLAMA_MODEL` in `rag.py` and the `ChatOllama(model=...)` call in `app.py` to any model available in your Ollama instance. List available models with `ollama list`.

**Swap the embedding model:** Update `EMBED_MODEL` in `config.py` and re-ingest. The new model will produce different-dimensional embeddings, so the existing vector store must be rebuilt.

**Add chunk overlap:** `chunk_fixed()` in `ingest.py` currently has no overlap. Add a `step` parameter smaller than `size` to create sliding windows, which improves retrieval for queries that span chunk boundaries.

**Add relevance filtering to the LangChain chain:** Wrap the retriever in a `RunnableLambda` that filters by distance score, mirroring the logic in `rag.py`.

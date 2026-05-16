# codon-rag

A fully local Retrieval-Augmented Generation (RAG) system that powers a **Codon Sales Assistant** chatbot. Users can ask natural-language questions about [Codon Consulting AB](https://www.codon.se) and receive grounded answers with source citations — no cloud APIs, no API keys.

---

## Architecture Overview

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

### Stack

| Layer | Technology |
|-------|-----------|
| Corpus | 8 Markdown files in `data/corpus/` |
| Chunking | Paragraph-split (default) or fixed-character |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` / `langchain-huggingface` |
| Vector store | ChromaDB (persistent, cosine distance) |
| LLM | Ollama — `ministral-3:3b` running locally at `http://127.0.0.1:11434` |
| RAG (raw) | `src/rag.py` — plain Python + `urllib`, no framework |
| RAG (LangChain) | `src/rag_lc.py` — LangChain LCEL chain |
| UI | `app.py` — Streamlit chat interface |
| Tunneling | `ngrok.exe` — expose Streamlit to the internet |

---

## Project Layout

```
codon-rag/
├── app.py                  # Streamlit chat UI (uses rag_lc-style chain)
├── requirements.txt        # Python dependencies (currently empty — see Prerequisites)
├── ngrok.exe               # ngrok binary for public tunneling
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

## Configuration (`src/config.py`)

All tunable knobs live here. **Change config here, not in individual scripts.**

```python
# Paths
CORPUS_DIR   # Absolute path to data/corpus/
CHROMA_DIR   # Absolute path to chroma_db/

# ChromaDB
COLLECTION   = "codon_docs"
DISTANCE     = "cosine"          # "cosine" | "l2" | "ip"

# Embedding
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, runs fully locally

# Chunking
CHUNK_STRATEGY   = "paragraph"   # "paragraph" | "fixed"
FIXED_CHUNK_SIZE = 200           # chars — only used when CHUNK_STRATEGY == "fixed"

# Retrieval
TOP_K                = 5
RELEVANCE_THRESHOLD  = 0.45      # cosine distance cutoff — only applied in rag.py
```

> **Note:** `CORPUS_DIR` and `CHROMA_DIR` are hardcoded absolute Windows paths. Update them when running on a different machine.

---

## Commands

### 1. Ingest (required before first use, and after any corpus/config changes)

```bash
python src/ingest.py
```

- Reads all `.md` files from `CORPUS_DIR`
- Chunks them using `CHUNK_STRATEGY`
- Embeds chunks with `EMBED_MODEL`
- Drops the existing ChromaDB collection (idempotent) and recreates it
- Writes everything to `CHROMA_DIR`

Re-run whenever you: add/edit corpus files, change `CHUNK_STRATEGY`, or change `EMBED_MODEL`.

### 2. Retrieval debug (no LLM)

```bash
python src/query.py "What industries does Codon work in?"
python src/query.py   # interactive prompt
```

Embeds the question, queries ChromaDB, and prints ranked chunks with distances. Use this to tune `TOP_K`, `DISTANCE`, and `CHUNK_STRATEGY` without waiting for an LLM.

### 3. Full RAG — raw (CLI)

```bash
python src/rag.py "What makes Codon different from other AI consultancies?"
python src/rag.py   # interactive prompt
```

Retrieves top-K chunks, filters by `RELEVANCE_THRESHOLD` (drops chunks with `dist > 0.45`), builds a prompt, calls Ollama via raw HTTP, and prints the answer + retrieved chunk previews.

### 4. Full RAG — LangChain (CLI)

```bash
python src/rag_lc.py "What AI services does Codon offer?"
```

Same pipeline as `rag.py` but implemented as a LangChain LCEL chain. Does **not** apply the relevance threshold filter.

### 5. Streamlit UI

```bash
streamlit run app.py
```

Launches the Codon Sales Assistant chat interface at `http://localhost:8501`. Shows retrieved chunks in an expander below each answer. The chain is cached with `@st.cache_resource` so the model loads once per session.

### 6. Public tunnel (optional)

```bash
./ngrok http 8501
```

Exposes the local Streamlit server publicly. Requires an ngrok account and authtoken configured.

---

## Prerequisites

- **Python 3.12+**
- **Ollama** running locally with the `ministral-3:3b` model pulled:
  ```bash
  ollama serve               # starts Ollama at http://127.0.0.1:11434
  ollama pull ministral-3:3b
  ```
- **Python dependencies** (install manually — `requirements.txt` is currently empty):
  ```bash
  pip install sentence-transformers chromadb \
              langchain-chroma langchain-ollama \
              langchain-huggingface langchain-core \
              streamlit
  ```

---

## Corpus Documents

The knowledge base is 8 Markdown files scraped from `codon.se` (April 2026):

| File | Content |
|------|---------|
| `01_company_overview.md` | Identity, mission, core team (Erik Fredlund, Founder & MD), founding 2019, Stockholm HQ |
| `02_services.md` | Analytics, custom ML models, AI-driven operations, NLP, computer vision, data strategy |
| `03_industries.md` | Manufacturing, process industries, life sciences, property management |
| `04_industrial_edge.md` | OT/IT integration, ISA-95 vertical strategy, Siemens Industrial Edge Partnership (Jan 2026) |
| `05_case_studies.md` | 14 client projects across IoT, agriculture, pharma, life sciences, NLP, retail, media |
| `06_blog_and_resources.md` | Blog posts, Data Science Friday events, ebook |
| `07_contact_and_careers.md` | Contact info (info@codon.se), careers/culture, role types |
| `08_sitemap_and_links.md` | Full URL inventory of codon.se |

### Test Questions (`data/test_questions.txt`)

```
What industries does Codon work in?
What makes Codon different from other AI consultancies?
How do I contact Codon?
What is Codon's Forward Deployed Engineering model?
What AI services does Codon offer?
```

Use these to validate retrieval quality after any ingestion or config change.

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
| Use case | Learning / debugging / minimal deps | Production UI (`app.py`) |

---

## Design Decisions & Gotchas

**Fully local stack.** No OpenAI, no Anthropic, no cloud embeddings. The only network call in normal operation is to `http://127.0.0.1:11434` (Ollama).

**Ingest is destructive.** `ingest.py` drops and recreates the ChromaDB collection on every run. This ensures idempotency but means re-ingesting from scratch — there is no incremental update path.

**Hardcoded Windows paths in `config.py`.** `CORPUS_DIR` and `CHROMA_DIR` use `r"C:\Users\DmitriApassov\..."`. Update these when cloning to a new machine or moving the project.

**Relevance threshold only in `rag.py`.** If `rag.py` returns "No relevant chunks found," the question is likely outside the corpus scope or the threshold (`0.45`) is too tight. Inspect raw distances with `query.py` first.

**LLM model hardcoded in two places.** `ministral-3:3b` appears in both `rag.py` (`OLLAMA_MODEL` constant) and `app.py`/`rag_lc.py` (`ChatOllama(model=...)`). Centralizing it in `config.py` would be a clean improvement.

**`requirements.txt` is empty.** Pin working versions with `pip freeze > requirements.txt` (filtered to project deps) when the environment is stable.

---

## Extending the Project

- **Add corpus documents:** Drop new `.md` files in `data/corpus/` and re-run `python src/ingest.py`.
- **Tune retrieval:** Adjust `TOP_K`, `RELEVANCE_THRESHOLD`, `CHUNK_STRATEGY`, or `DISTANCE` in `config.py`, re-ingest, then test with `query.py`.
- **Swap the LLM:** Change `OLLAMA_MODEL` in `rag.py` and the `ChatOllama(model=...)` call in `app.py` to any model pulled in your Ollama instance.
- **Swap the embedding model:** Update `EMBED_MODEL` in `config.py` and re-ingest (the existing ChromaDB embeddings will be stale).
- **Add chunk overlap:** `chunk_fixed()` in `ingest.py` currently has no overlap. Add a `step` parameter smaller than `size` to create overlapping windows.
- **Add relevance filtering to the LangChain chain:** Wrap the retriever in a custom `RunnableLambda` that filters by distance score, mirroring the logic in `rag.py`.

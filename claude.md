<!-- Section from: c:\Users\DmitriApassov\Documents\codon\codon-rag\CLAUDE.md | Lines: 7-38 -->

## Architecture Overview

```
data/kb_registry.json (KB config)
data/corpus/<kb_id>/*.md  →  src/ingest.py (Incremental)  →  ChromaDB (chroma_db/<collection_name>)
                                               ↓
User question  →  embed (MiniLM)  →  similarity search (top-K)
                                               ↓
                              workbench/main.py (FastAPI + SSE)
                                               ↓
                          Ollama (ministral-3:3b @ localhost:11434)
                                               ↓
                                   Answer + source citations
                                               ↓
                              workbench/static/index.html  (SPA)
```

### Active UI: Workbench (FastAPI)

> **All active development happens in `workbench/`.  Do NOT modify Streamlit files
> (`app.py`, `pages/`) — they are legacy and may be removed later.**

### Stack

| Layer | Technology |
|---|---|
| Corpus | Multiple independent knowledge bases configured in `data/kb_registry.json` |
| Chunking | Paragraph-split (default) or fixed-character |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector store | ChromaDB (persistent, cosine distance, with rich metadata: `kb_id`, `doc_id`, `file_hash`) |
| Ingestion | Incremental ingestion natively skipping unchanged files, with surgical file/KB deletion |
| LLM | Ollama — `ministral-3:3b` running locally at `http://127.0.0.1:11434` |
| RAG (raw) | `src/rag.py` — plain Python + `urllib`, no framework |
| RAG (LangChain) | `src/rag_lc.py` — LangChain LCEL chain |
| UI (active) | **Workbench** — FastAPI backend (`workbench/main.py`) + vanilla JS SPA (`workbench/static/index.html`) at `http://localhost:8000` |
| UI (legacy) | Streamlit multipage app (`app.py` + `pages/`) — not actively maintained, do not modify |
| Tunneling | `ngrok.exe` — expose app to the internet |

---
<!-- Section from: c:\Users\DmitriApassov\Documents\codon\codon-rag\CLAUDE.md | Lines: 39-61 -->

## Project Layout

```
codon-rag/
├── workbench/              # ★ ACTIVE UI — all new work goes here
│   ├── main.py             # FastAPI backend (KB mgmt, ingest SSE, ask SSE, eval SSE)
│   ├── static/index.html   # Vanilla JS single-page app
│   └── requirements.txt    # Workbench-specific deps (fastapi, uvicorn, etc.)
├── app.py                  # [LEGACY] Streamlit entry point — do not modify
├── pages/                  # [LEGACY] Streamlit pages — do not modify
│   ├── 1_Knowledge_Bases.py
│   └── 2_Ask.py
├── requirements.txt        # Python dependencies
├── ngrok.exe               # ngrok binary for public tunneling
├── src/
│   ├── config.py           # All tunable parameters — edit here first
│   ├── ingest.py           # Chunk → embed → write to ChromaDB (incremental)
│   ├── query.py            # CLI retrieval debugger (no LLM)
│   ├── rag.py              # Full RAG CLI (raw HTTP to Ollama)
│   └── rag_lc.py           # Full RAG CLI using LangChain
├── data/
│   ├── kb_registry.json    # Registry containing Knowledge Base configs
│   ├── corpus/             # Source documents for each KB
│   └── test_questions.txt  # 5 canonical eval questions
├── chroma_db/              # Persisted ChromaDB vector store (git-ignored)
└── notebooks/
    └── scratch.ipynb       # Scratch notebook for experimentation
```

---
<!-- Section from: c:\Users\DmitriApassov\Documents\codon\codon-rag\CLAUDE.md | Lines: 62-73 -->

## Quick Commands

```bash
# ── Workbench (active UI) ────────────────────────────────────────────────────
uvicorn workbench.main:app --reload               # open Workbench at localhost:8000

# ── CLI tools ────────────────────────────────────────────────────────────────
python src/ingest.py --kb codon                   # build / rebuild the vector store incrementally
python src/ingest.py --kb codon --delete-file f.md # drop a single file's chunks
python src/ingest.py --kb codon --delete-kb       # drop entire KB and remove from registry
python src/query.py --kb codon "question"         # retrieval debug (no LLM)
python src/rag.py --kb codon "question"           # full RAG, raw Python
python src/rag_lc.py --kb codon "question"        # full RAG, LangChain
```

---
<!-- Section from: c:\Users\DmitriApassov\Documents\codon\codon-rag\CLAUDE.md | Lines: 205-224 -->

## Stretch: Hybrid Retrieval (BM25 + Vector + Knowledge Graph)

The current retrieval pipeline is pure cosine similarity over dense embeddings. This works well at small corpus sizes but has a known weakness: vocabulary mismatch. A query like "database performance optimization" will miss a chunk that says "fixed the N+1 query problem" unless the embedding model happens to place them close in vector space.

[agentmemory](https://github.com/rohitg00/agentmemory) (a persistent memory layer for AI coding agents) uses a three-stream retrieval architecture worth borrowing from if retrieval quality becomes a concern — for example, as the corpus grows with multi-KB support:

**Stream 1 — BM25 (keyword).** Stemmed full-text search with domain synonym expansion (e.g. "db" ↔ "database", "perf" ↔ "performance"). Catches exact-match cases that vector search misses. Add via `rank_bm25` (pure Python, no new infrastructure).

**Stream 2 — Vector (semantic).** Existing cosine similarity over `all-MiniLM-L6-v2` embeddings. Catches paraphrase and concept-level matches that keyword search misses.

**Stream 3 — Knowledge graph (structural).** Entity extraction over the corpus (people, products, concepts, file names) stored as nodes and edges. Queries traverse the graph via BFS to pull in structurally related chunks even when neither keyword nor vector similarity is high. More complex — requires an extraction pass at ingest time and a graph store (NetworkX is sufficient at this scale).

**Fusion.** All three streams are merged using **Reciprocal Rank Fusion** (RRF, k=60): each document gets a score of `1 / (k + rank)` from each stream, scores are summed, and the top-K results are returned. RRF is robust to streams with incompatible score scales and requires no tuning beyond k.

agentmemory's benchmarks on 240 real-world observations show BM25-only and vector-only each achieve ~56% Recall@10, while the fused hybrid reaches **64% Recall@10** with perfect MRR — using 92% fewer tokens than dumping everything into context. The relative gain should transfer to a document RAG setting.

This is not on the immediate roadmap (the corpus is currently 8 files and pure vector retrieval is adequate), but it is the right next retrieval upgrade once the corpus scales or multi-KB queries are introduced.

---
<!-- Section from: c:\Users\DmitriApassov\Documents\codon\codon-rag\CLAUDE.md | Lines: 225-231 -->

## Known Issues & Technical Debt

All previously known technical debt issues have been addressed:
- ~~`OLLAMA_MODEL` (`ministral-3:3b`) is hardcoded in both `rag.py` and `app.py`/`rag_lc.py`. Centralize it in `config.py`.~~ (Resolved: Centralized `OLLAMA_MODEL` in `config.py` and updated all scripts to reference `cfg.OLLAMA_MODEL`)
- ~~`CORPUS_DIR` and `CHROMA_DIR` are absolute Windows paths. Replace with `pathlib.Path(__file__).parent.parent / "..."` for cross-platform portability.~~ (Resolved: Updated paths in `config.py` using `pathlib.Path(__file__).parent.parent`)
- ~~`chunk_fixed()` in `ingest.py` has no overlap. Add a `step` parameter smaller than `size` to improve retrieval across chunk boundaries.~~ (Resolved: Added `step` argument to `chunk_fixed` in `ingest.py` and `FIXED_CHUNK_STEP` config to `config.py`)
- ~~The LangChain chain in `rag_lc.py` and `app.py` does not apply `RELEVANCE_THRESHOLD`. Add a `RunnableLambda` filter to match `rag.py` behavior.~~ (Resolved: Implemented `RunnableLambda` to filter retrieved chunks by `cfg.RELEVANCE_THRESHOLD` via `similarity_search_with_score` in both `app.py` and `src/rag_lc.py`)
## ChromaDB Threading Gotchas (learned from Streamlit, still applies to Workbench)

> The Streamlit UI is legacy, but these lessons carry over to the Workbench's
> FastAPI backend, which also runs multi-threaded (uvicorn default thread pool).
> The `_chroma_client()` helper in `workbench/main.py` already applies the fix.

### 1. SQLite Thread-Sharing Crashes
**Problem:** Streamlit runs on a multi-threaded execution model. By default, `chromadb.PersistentClient` maintains a singleton cache of the SQLite connection in the background. If Streamlit caches the `chromadb` client using `@st.cache_resource`, subsequent user interactions may be routed to a different background thread. When the new thread attempts to query the database, SQLite throws a strict cross-thread violation: `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`.
**Solution:**
- Do not cache `Chroma` or `chromadb.PersistentClient` initialization with `@st.cache_resource`. Let Langchain or the script instantiate the connection anew on every Streamlit rerun.
- Before creating the `Chroma` vectorstore, explicitly clear the Chroma system cache: `chromadb.api.client.SharedSystemClient.clear_system_cache()`. This forces Chroma to create a fresh, thread-safe SQLite connection tied to the current execution thread.

### 2. Silent Failures with Exception Swallowing
**Problem:** When implementing error handling to prevent the app from crashing on empty collections (e.g., using `try...except Exception:` around `similarity_search_with_score`), it can inadvertently swallow critical database errors like the SQLite thread-sharing crash mentioned above. This leads to the app silently assuming zero chunks were retrieved, causing the LLM to hallucinate or the app to trigger early-exit handlers inappropriately.
**Solution:** Always log or print the traceback explicitly within broad `except` blocks (`import traceback; traceback.print_exc()`) to ensure underlying database connection issues are visible in the Streamlit terminal.

### 3. Module Hot-Reloading Stagnation
**Problem:** When updating configuration files (like `src/config.py`) to tune hyperparameters such as `RELEVANCE_THRESHOLD`, Streamlit detects the file change and reruns the UI script. However, standard Python aggressive module caching (`sys.modules`) means that the UI script continues to use the stale values from the initial import, leading to confusing retrieval behavior.
**Solution:** Explicitly force Python to reload the configuration module at the top of the Streamlit script:
```python
import config as cfg
import importlib
importlib.reload(cfg)
```

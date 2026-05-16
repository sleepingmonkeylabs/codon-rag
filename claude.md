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
|---|---|
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
├── requirements.txt        # Python dependencies
├── ngrok.exe               # ngrok binary for public tunneling
├── src/
│   ├── config.py           # All tunable parameters — edit here first
│   ├── ingest.py           # Chunk → embed → write to ChromaDB
│   ├── query.py            # CLI retrieval debugger (no LLM)
│   ├── rag.py              # Full RAG CLI (raw HTTP to Ollama)
│   └── rag_lc.py           # Full RAG CLI using LangChain
├── data/
│   ├── corpus/             # Source documents (8 .md files)
│   └── test_questions.txt  # 5 canonical eval questions
├── chroma_db/              # Persisted ChromaDB vector store (git-ignored)
└── notebooks/
    └── scratch.ipynb       # Scratch notebook for experimentation
```

---

## Quick Commands

```bash
python src/ingest.py                    # build / rebuild the vector store
python src/query.py "question"         # retrieval debug (no LLM)
python src/rag.py "question"           # full RAG, raw Python
python src/rag_lc.py "question"       # full RAG, LangChain
streamlit run app.py                   # chat UI at localhost:8501
```

---

## Next Steps

The items below are the agreed-upon improvements for the next development phase, ordered roughly by dependency. Complete them in sequence — each one unblocks the next.

---

### 1. Better ChromaDB Indexing

**Current state.** Each chunk is stored with only two metadata fields — `source` (filename) and `chunk_index` (integer). There is no stable identifier linking a chunk to the knowledge base it belongs to, and no way to filter or delete by file without a full re-ingest.

**Goal.** Store richer, queryable metadata on every chunk so that retrieval can be scoped and individual documents or entire KBs can be managed surgically.

**Fields to add to every chunk's metadata in `ingest.py`:**

| Field | Type | Description |
|---|---|---|
| `kb_id` | `str` | Stable identifier for the knowledge base (e.g. `"codon_kb"`). Enables per-KB filtering and deletion. |
| `source_file` | `str` | Filename (rename from `source` for clarity). |
| `source_path` | `str` | Repo-relative path to the source file. |
| `doc_id` | `str` | Stable hash of `kb_id + filename`. Groups all chunks from one file — essential for file-level deletion. |
| `chunk_index` | `int` | Position of this chunk within the file (already stored). |
| `total_chunks` | `int` | Total chunks produced from this file. Useful for completeness checks. |
| `ingested_at` | `str` | ISO-8601 timestamp of ingestion. |

**Implementation notes.** ChromaDB already supports metadata filtering and deletion — no schema changes are needed. The only change is populating the metadata dict in `ingest.py`. Update `query.py` to print the new fields for visibility during debugging.

---

### 2. Multiple Knowledge Bases

**Current state.** There is a single ChromaDB collection (`codon_docs`) and a single corpus directory. Every ingest wipes and recreates this one collection.

**Goal.** Support multiple independent knowledge bases (e.g. one per client, product, or domain) that do not interfere with each other. Users should be able to select which KB to query.

**Proposed design.** Introduce a KB registry at `data/kb_registry.json`:

```json
{
  "codon": {
    "kb_id": "codon",
    "collection": "codon_docs",
    "corpus_dir": "data/corpus/codon",
    "description": "Codon Consulting public website content"
  },
  "client_x": {
    "kb_id": "client_x",
    "collection": "client_x_docs",
    "corpus_dir": "data/corpus/client_x",
    "description": "Client X product documentation"
  }
}
```

**Changes required across the codebase:**

- `config.py` — add `KB_REGISTRY_PATH`; remove the single hardcoded `COLLECTION` and `CORPUS_DIR` (or keep them as defaults for backward compatibility).
- `ingest.py` — accept `--kb <kb_id>` argument; look up corpus dir and collection name from the registry; write `kb_id` into chunk metadata.
- `rag.py` / `rag_lc.py` — accept `--kb <kb_id>` to set the target collection.
- `app.py` — add a sidebar `st.selectbox` listing all registered KBs; pass the selected collection to the chain.

**Cross-KB search (stretch goal).** When the user selects "All", query each collection separately, merge results, re-rank by distance, de-duplicate, then send to the LLM.

---

### 3. Deletion Strategy

**Current state.** There is no deletion path. The only way to remove content is a full destructive re-ingest via `ingest.py`.

**Goal.** Support three levels of targeted deletion so content can be removed or updated without rebuilding everything.

**File-level deletion** — remove all chunks from a single source file:

```python
# ChromaDB supports this natively once doc_id metadata is in place (see §1)
collection.delete(where={"doc_id": "<hash>"})
```

Expose this as: `python src/ingest.py --delete-file <filename> [--kb <kb_id>]`

**KB-level deletion** — drop an entire collection:

```python
client.delete_collection(collection_name)
# then remove from kb_registry.json
```

Expose this as: `python src/ingest.py --delete-kb <kb_id>`

**Incremental add** — add new files without a full re-ingest:

Before embedding, check whether a `doc_id` already exists in the collection (`collection.get(where={"doc_id": "..."})`). If found and the file is unchanged (compare a `file_hash` stored in metadata), skip it. If changed, delete the old chunks and re-embed. This removes the need for destructive re-ingest on routine corpus updates.

---

### 4. Two-Page Streamlit Frontend

**Current state.** `app.py` is a single-page chat interface with no UI for managing knowledge bases or documents.

**Goal.** Split the UI into two focused pages using Streamlit's multipage app structure:

```
app.py                         # landing / entry point
pages/
├── 1_Knowledge_Bases.py       # KB management
└── 2_Ask.py                   # chat interface
```

**Page 1 — Knowledge Bases (`pages/1_Knowledge_Bases.py`)**

This page is for managing what the assistant knows.

- **KB overview table** — list all KBs from `kb_registry.json` with name, description, chunk count (from `collection.count()`), and last ingested timestamp.
- **Create KB** — a form with KB name, description, and corpus directory; writes to `kb_registry.json` and triggers ingestion.
- **Add documents** — `st.file_uploader` accepting `.md` or `.txt` files; saves to the KB's corpus directory and runs incremental ingest on just the new files.
- **Delete document** — selectbox listing files in the active KB; calls file-level deletion (§3).
- **Delete KB** — confirmation dialog followed by collection drop and registry removal.
- **Re-ingest KB** — force a full destructive re-ingest of the selected KB's corpus directory.

**Page 2 — Ask (`pages/2_Ask.py`)**

This page is the existing chat interface, extended with KB awareness.

- **KB selector** — `st.sidebar.selectbox` listing all registered KBs; the chain is rebuilt (or retrieved from cache) when the selection changes.
- **Chat history** — existing `st.chat_message` loop, unchanged.
- **Retrieved chunks expander** — existing expandable section showing source, distance, and chunk preview.
- **Clear chat button** — `st.sidebar.button("Clear conversation")` resetting `st.session_state.messages`.

**Caching note.** Use `st.cache_resource` keyed by `kb_id` so switching KBs loads the correct vector store without reloading the embedding model. Cache the embedding model once at the top level since it is KB-agnostic.

---

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

## Known Issues & Technical Debt

- `OLLAMA_MODEL` (`ministral-3:3b`) is hardcoded in both `rag.py` and `app.py`/`rag_lc.py`. Centralize it in `config.py`.
- `CORPUS_DIR` and `CHROMA_DIR` are absolute Windows paths. Replace with `pathlib.Path(__file__).parent.parent / "..."` for cross-platform portability.
- `chunk_fixed()` in `ingest.py` has no overlap. Add a `step` parameter smaller than `size` to improve retrieval across chunk boundaries.
- The LangChain chain in `rag_lc.py` and `app.py` does not apply `RELEVANCE_THRESHOLD`. Add a `RunnableLambda` filter to match `rag.py` behavior.

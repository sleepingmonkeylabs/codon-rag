# codon-rag Project Specification

Date: 2026-05-23

This document describes the current repository as implemented. It is intended
to be the factual project spec: what the system is, how the pieces fit
together, what contracts the code exposes, and what operational constraints are
already visible in the codebase.

Speculative improvements live in `EXTENSIONS_ROADMAP.md`.

## 1. Executive Summary

`codon-rag` is a fully local Retrieval-Augmented Generation system. It lets a
user build one or more knowledge bases from Markdown or text files, embed those
files into ChromaDB, ask natural-language questions, and receive answers from a
locally running Ollama model with citations from retrieved source chunks.

The active user interface is the FastAPI workbench:

```text
http://localhost:8000
```

The older Streamlit app still exists and works as a legacy/development surface,
but current active development should target the workbench:

```text
workbench/main.py
workbench/static/index.html
```

Current major capabilities:

- Multi-KB registry in `data/kb_registry.json`
- Persistent local ChromaDB vector store in `chroma_db/`
- Local embeddings through SentenceTransformers
- Local generation through Ollama
- Workbench tabs for Build, Ask, and Evaluate
- SSE streaming for KB builds, RAG answers, and evaluation runs
- CLI tools for ingesting, retrieval debugging, raw RAG, LangChain RAG, and RAGAS evaluation

## 2. One-Screen Architecture

```text
                       +-----------------------------+
                       | data/kb_registry.json       |
                       | - KB ids                    |
                       | - collection names          |
                       | - corpus directories        |
                       | - frozen build manifests    |
                       +-------------+---------------+
                                     |
                                     v
+-------------------+       +-------+--------+       +------------------+
| data/corpus/...   | ----> | ingest/chunk   | ----> | chroma_db/       |
| .md and .txt      |       | embed/write    |       | ChromaDB         |
+-------------------+       +-------+--------+       +--------+---------+
                                     |                         ^
                                     |                         |
                                     v                         |
                             +-------+-------------------------+------+
                             | workbench/main.py                     |
                             | FastAPI routes + SSE streams          |
                             +-------+-------------------------+------+
                                     |                         |
                                     v                         v
                         +-----------+---------+       +-------+-------+
                         | SPA browser UI      |       | Ollama        |
                         | Build / Ask / Eval  |       | local LLM     |
                         +---------------------+       +---------------+
```

ELI5 version:

```text
1. Put documents into labeled folders.
2. Cut each document into small labeled cards.
3. Turn each card into a number fingerprint.
4. Store the fingerprints in a searchable box.
5. When someone asks a question, fingerprint the question too.
6. Pull out the closest cards.
7. Give only those cards to the local LLM and ask it to answer.
```

## 3. Project Layout

```text
codon-rag/
|-- app.py                         Legacy Streamlit landing page
|-- pages/
|   |-- 1_Knowledge_Bases.py        Legacy Streamlit KB management
|   `-- 2_Ask.py                    Legacy Streamlit chat UI
|
|-- workbench/
|   |-- main.py                     Active FastAPI backend
|   |-- requirements.txt            FastAPI-specific dependencies
|   |-- __init__.py
|   `-- static/
|       `-- index.html              Active vanilla JS SPA
|
|-- src/
|   |-- config.py                   Shared paths, model names, defaults, prompt
|   |-- ingest.py                   CLI ingest plus importable build_kb()
|   |-- query.py                    Retrieval debugger, no LLM
|   |-- rag.py                      Raw Python RAG path using urllib + Ollama
|   |-- rag_lc.py                   LangChain LCEL RAG path
|   |-- eval.py                     Two-phase RAGAS evaluation
|   `-- eval_grid.py                Parameter grid evaluation
|
|-- data/
|   |-- kb_registry.json            Source of truth for configured KBs
|   |-- test_questions.txt          Older 5-question plain-text test set
|   |-- test_questions.json         Current 15-question eval set
|   |-- eval_log.jsonl              RAGAS run history
|   |-- eval_responses_*.jsonl      Saved eval checkpoints
|   `-- corpus/
|       |-- 01_company_overview.md
|       |-- ...
|       |-- 08_sitemap_and_links.md
|       `-- buddhism/
|           |-- class1_script_en.md
|           |-- ...
|           `-- vimalakirti_upaya_extractions.md
|
|-- chroma_db/                      Persistent ChromaDB state
|-- docs/                           Prior assessment and AI map docs
|-- notebooks/scratch.ipynb         Empty scratch notebook
|-- setup.bat                       Windows bootstrap
|-- setup.sh                        macOS/Linux bootstrap
|-- requirements.txt                Core dependencies
|-- README.md                       Existing user-facing overview
|-- CLAUDE.md                       Existing agent/developer notes
`-- ngrok.exe                       Optional local tunnel binary
```

## 4. Core Concepts

Knowledge base, or KB:

- A named corpus plus a ChromaDB collection.
- Defined in `data/kb_registry.json`.
- Example ids in the current registry: `codon`, `buddhism`.

Corpus:

- A folder of `.md` or `.txt` files.
- The `codon` KB points to `data/corpus` and ingests only top-level files.
- The `buddhism` KB points to `data/corpus/buddhism`.

Chunk:

- A piece of a source file.
- Current chunking strategies are `paragraph` and `fixed`.
- Each chunk gets metadata such as source file, chunk index, file hash, and KB id.

Embedding:

- A vector representation of text.
- Default model is `all-MiniLM-L6-v2`.
- Stored in ChromaDB and used for similarity search.

Collection:

- A ChromaDB collection containing chunks for a KB.
- Current collection names: `codon_docs`, `buddhism_docs`.

Build manifest:

- A frozen record written to the KB registry after a workbench build.
- It records the embedding model, chunk strategy, distance metric, default top-k,
  threshold, source count, total chunk count, and build time.

Relevance threshold:

- A cosine-distance cutoff.
- Lower distance means more similar.
- A chunk is kept when `distance <= threshold`.

System prompt:

- Default prompt is defined in `src/config.py`.
- `POST /api/ask` already supports a request-level `system_prompt` override.
- The current workbench Ask UI does not expose that override yet.
- The current eval endpoint does not accept prompt overrides.

## 5. Current Runtime Stack

Core dependencies from `requirements.txt`:

- `sentence-transformers`: local embeddings
- `chromadb`: persistent vector store
- `langchain-core`, `langchain-chroma`, `langchain-ollama`, `langchain-huggingface`: LangChain path
- `streamlit`: legacy UI
- `ragas`, `datasets`: evaluation

Workbench dependencies from `workbench/requirements.txt`:

- `fastapi`: API framework
- `uvicorn[standard]`: ASGI server
- `python-multipart`: file upload support

External local service:

- Ollama at `http://127.0.0.1:11434`
- Default generation model: `ministral-3:3b`
- Evaluation judge model: `gemma4:31b-cloud`

## 6. Configuration

Main shared config lives in `src/config.py`.

Important values:

```python
BASE_DIR = pathlib.Path(__file__).parent.parent
CORPUS_DIR = str(BASE_DIR / "data" / "corpus")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
KB_REGISTRY_PATH = str(BASE_DIR / "data" / "kb_registry.json")

COLLECTION = "codon_docs"
DISTANCE = "cosine"

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "ministral-3:3b"

CHUNK_STRATEGY = "paragraph"
FIXED_CHUNK_SIZE = 200
FIXED_CHUNK_STEP = 150

TOP_K = 5
RELEVANCE_THRESHOLD = 0.65
```

The default prompt is also in `src/config.py`:

```python
SYSTEM_PROMPT = """You are a helpful assistant for Codon Consulting.
Answer questions using only the provided context chunks.
If the context does not contain the answer, say clearly: "I don't have information about that."
Always cite the source document names at the end of your answer."""
```

Operational note:

- `src/config.py` centralizes many defaults.
- `workbench/main.py`, `src/rag.py`, `src/eval.py`, and `src/eval_grid.py` still
  use hardcoded Ollama base URLs in their own code paths.
- A production/container version should introduce a shared `OLLAMA_BASE_URL`
  environment variable.

## 7. Data Inventory

Current registry snapshot:

```text
+----------+----------------+----------------------+---------+---------+---------------------------+
| KB       | Collection     | Corpus dir           | Sources | Chunks  | Built at                  |
+----------+----------------+----------------------+---------+---------+---------------------------+
| codon    | codon_docs     | data/corpus          | 8       | 329     | 2026-05-22T18:52:34 UTC   |
| buddhism | buddhism_docs  | data/corpus/buddhism | 6       | 529     | 2026-05-19T21:49:34 UTC   |
+----------+----------------+----------------------+---------+---------+---------------------------+
```

The `codon` corpus contains eight files:

- Company overview
- Services
- Industries
- Industrial Edge and OT/IT integration
- Fourteen case studies
- Blog/resources/Data Science Friday
- Contact and careers
- Sitemap and link inventory

The `buddhism` corpus contains six files:

- Class 1 script
- Chapter 18 text/notes
- Vimalakirti chapter 1
- Chapter 1 theism/Tibetan verse notes
- Vimalakirti seminar memory
- Upaya passage extractions

Evaluation data:

- `data/test_questions.txt` has 5 plain questions.
- `data/test_questions.json` has 15 structured questions with `id`,
  `question`, and `ground_truth`.
- `data/eval_log.jsonl` records scored RAGAS runs.
- `data/eval_responses_*.jsonl` files are checkpoints from phase 1 eval
  collection.

ChromaDB data:

- `chroma_db/chroma.sqlite3`
- HNSW index directories with binary files such as `data_level0.bin`,
  `header.bin`, and `length.bin`

## 8. Ingestion Pipeline

There are two ingestion modes.

### 8.1 CLI incremental ingest

Entry point:

```bash
python src/ingest.py --kb codon
```

The CLI path:

```text
read registry
  |
  v
resolve corpus dir and collection
  |
  v
for each .md/.txt file:
  - read UTF-8 text
  - hash file text
  - derive stable doc_id from kb_id + filename
  - skip if existing metadata has same file_hash
  - delete old chunks if file_hash changed
  - chunk new/changed file
  |
  v
embed changed chunks with SentenceTransformer
  |
  v
collection.add(ids, embeddings, documents, metadatas)
```

Metadata written per chunk:

```json
{
  "kb_id": "codon",
  "source_file": "01_company_overview.md",
  "source_path": "data/corpus/01_company_overview.md",
  "doc_id": "...sha256...",
  "file_hash": "...sha256...",
  "chunk_index": 0,
  "total_chunks": 10,
  "ingested_at": "2026-05-23T..."
}
```

Deletion commands:

```bash
python src/ingest.py --kb codon --delete-file 01_company_overview.md
python src/ingest.py --kb codon --delete-kb
```

### 8.2 Workbench full rebuild

Entry point:

```http
POST /api/kbs/{kb_id}/build
```

The workbench path:

```text
browser sends BuildRequest
  |
  v
FastAPI starts daemon ingest thread
  |
  v
thread discovers corpus files
  |
  v
drop old Chroma collection
  |
  v
create new collection with requested distance metric
  |
  v
chunk all files
  |
  v
embed all chunks
  |
  v
write chunks in batches of 500
  |
  v
stamp build manifest into kb_registry.json
  |
  v
stream done event to browser
```

SSE event examples:

```json
{"type": "progress", "msg": "Found 8 files - chunking..."}
{"type": "progress", "msg": "Embedding 329 chunks..."}
{"type": "done", "msg": "329 chunks written to 'codon_docs'", "manifest": {}}
{"type": "error", "msg": "...", "traceback": "..."}
```

The workbench rebuild is intentionally destructive for the collection. It drops
and recreates the collection so a changed distance metric or embedding model
cannot leave stale vectors behind.

## 9. Chunking

Current strategies:

```text
paragraph:
  split text on blank lines

fixed:
  take chunk_size characters
  advance by chunk_step characters
```

Code paths:

- `src/ingest.py`: `chunk_paragraph()`, `chunk_fixed()`, `get_chunks()`
- `workbench/main.py`: `_chunk_paragraph()`, `_chunk_fixed()`

ELI5 fixed-chunk example:

```text
text = "ABCDEFGHIJ"
chunk_size = 4
chunk_step = 2

chunks:
  ABCD
  CDEF
  EFGH
  GHIJ
  IJ

true overlap = chunk_size - chunk_step = 2
```

Important detail:

- The workbench UI labels one slider as "Overlap (chars)" but sends that value
  as `chunk_step`.
- In the backend, `chunk_step` means "how far the window moves forward", not
  "how much overlap to preserve".
- If `chunk_size = 200` and `chunk_step = 50`, true overlap is 150 characters.

Known limitation:

- Paragraph chunking is currently a blank-line split.
- It does not understand Markdown headers, lists, tables, or fenced code blocks.
- This is the highest-impact chunking improvement candidate.

## 10. Retrieval And Answering

Main raw RAG code path:

```text
question
  |
  v
SentenceTransformer.encode([question])
  |
  v
Chroma collection.query(n_results=top_k)
  |
  v
filter chunks where distance <= threshold
  |
  v
build context block
  |
  v
POST to Ollama /api/chat
  |
  v
answer with source citations
```

ELI5 retrieval example:

```text
Question: "Who founded Codon?"

Chroma returns:
  1. company_overview.md, distance 0.31 -> keep
  2. careers.md,          distance 0.72 -> discard if threshold is 0.65
  3. services.md,         distance 0.59 -> keep

The LLM sees only kept chunks.
```

### 10.1 `src/query.py`

Purpose:

- Embed a question.
- Retrieve top-K chunks.
- Print ranked results with distances and metadata.
- Does not call an LLM.

Use:

```bash
python src/query.py --kb codon "What industries does Codon work in?"
```

### 10.2 `src/rag.py`

Purpose:

- Full RAG loop with raw Python, `chromadb`, `SentenceTransformer`, and
  `urllib.request` against Ollama.
- Provides importable helpers used by eval code:
  - `retrieve_from()`
  - `generate()`

Ollama behavior:

- Calls `POST http://127.0.0.1:11434/api/chat`
- Uses `cfg.OLLAMA_MODEL`
- Retries network-level failures up to 3 times with exponential backoff
- Exits immediately for HTTP-level errors

### 10.3 `src/rag_lc.py`

Purpose:

- LangChain LCEL version of the RAG path.
- Uses:
  - `HuggingFaceEmbeddings`
  - `langchain_chroma.Chroma`
  - `ChatOllama`
  - `RunnableLambda` for relevance filtering

This is a parallel implementation, not the main active workbench path.

### 10.4 `POST /api/ask`

Workbench ask endpoint:

```json
{
  "kb_id": "codon",
  "question": "Who founded Codon?",
  "top_k": 5,
  "threshold": 0.65,
  "model": "ministral-3:3b",
  "system_prompt": "optional override"
}
```

SSE sequence:

```text
data: {"type": "chunks", "chunks": [...kept...], "all_chunks": [...all...]}
data: {"type": "token", "content": "Codon"}
data: {"type": "token", "content": " was"}
data: {"type": "done", "elapsed_ms": 1843}
```

No relevant chunks case:

```text
data: {"type": "chunks", "chunks": [], "all_chunks": [...]}
data: {"type": "done", "elapsed_ms": 0, "msg": "No chunks passed the relevance threshold"}
```

## 11. Workbench Backend API

File:

```text
workbench/main.py
```

Routes:

```text
GET    /                         serve workbench/static/index.html
GET    /api/kbs                  list KBs with live chunk counts
POST   /api/kbs                  create KB registry entry and corpus dir
DELETE /api/kbs/{kb_id}          delete collection and registry entry
GET    /api/kbs/{kb_id}/files    list corpus files
POST   /api/kbs/{kb_id}/files    upload .md/.txt file
DELETE /api/kbs/{kb_id}/files/{filename}
POST   /api/kbs/{kb_id}/build    SSE full rebuild
GET    /api/ollama/models        query Ollama /api/tags
POST   /api/ask                  SSE chunks plus streamed answer
POST   /api/eval/run             SSE two-phase RAGAS eval
GET    /api/eval/log             return eval_log.jsonl newest first
```

Create KB request:

```json
{
  "kb_id": "client_x",
  "description": "Client X support docs",
  "corpus_subdir": "client_x"
}
```

Build request:

```json
{
  "embed_model": "all-MiniLM-L6-v2",
  "chunk_strategy": "paragraph",
  "chunk_size": 200,
  "chunk_step": 150,
  "distance": "cosine",
  "top_k": 5,
  "threshold": 0.65
}
```

Eval request:

```json
{
  "kb_id": "codon",
  "top_k": 5,
  "threshold": 0.65,
  "note": "baseline"
}
```

Implementation details:

- `SentenceTransformer` instances are cached in `_model_cache`.
- Every new ChromaDB client goes through `_chroma_client()`, which clears
  Chroma's shared system cache first to avoid SQLite cross-thread errors.
- Build and eval work run in daemon threads and communicate back through
  `queue.Queue`.
- SSE streams include keepalive comments during long blocking phases.

## 12. Workbench Frontend

File:

```text
workbench/static/index.html
```

The UI is a single HTML file with:

- HTML
- CSS custom-property theme system
- Vanilla JavaScript
- Native `fetch()` calls
- Native readable stream parsing for SSE-style responses

There is no bundler, framework, package manager, or frontend build step.

### 12.1 Top bar

- App logo/title
- Tab navigation:
  - Build KB
  - Ask
  - Evaluate
- Ollama status pill
- Light/dark theme toggle persisted in `localStorage`

### 12.2 Build tab

Capabilities:

- Select an existing KB.
- Create a new KB.
- Upload `.md` and `.txt` files through drag/drop or file picker.
- Delete corpus files.
- Choose chunking strategy.
- Choose embedding model.
- Choose Chroma distance metric.
- Set default top-K and threshold.
- Build the KB and watch live progress.
- View last build manifest.

Build page data flow:

```text
load /api/kbs
  |
  v
select KB
  |
  +--> GET /api/kbs/{kb_id}/files
  |
  +--> render manifest from registry build block
  |
  v
POST /api/kbs/{kb_id}/build
  |
  v
stream progress into build log
```

### 12.3 Ask tab

Capabilities:

- Select a KB from card list.
- Select Ollama model from `/api/ollama/models`.
- Adjust top-K and threshold per question.
- Send question with Enter.
- Stream assistant response token by token.
- Render retrieved chunks before generation starts.
- Show all retrieved chunks, with threshold-filtered ones dimmed.

Ask page data flow:

```text
user question
  |
  v
POST /api/ask
  |
  +--> chunks event renders retrieval panel immediately
  |
  +--> token events append to assistant bubble
  |
  +--> done event updates elapsed time
```

### 12.4 Evaluate tab

Capabilities:

- Select a KB.
- Choose top-K and threshold.
- Add a note for the run.
- Run a two-phase RAGAS evaluation over `data/test_questions.json`.
- Watch live Q/A rows during collection.
- See final score cards for:
  - faithfulness
  - answer relevancy
  - context recall
- View eval history from `data/eval_log.jsonl`.

Current limitations:

- The test set is hardcoded to `data/test_questions.json`.
- It displays 15 questions as static UI text.
- It cannot upload a question set yet.
- It does not expose a prompt editor.
- It does not call `src/eval_grid.py`; it calls the single-run eval endpoint.

## 13. Evaluation Subsystem

Evaluation has three layers.

### 13.1 `src/eval.py`

Two-phase evaluator:

```text
Phase 1: collect
  for each question:
    retrieve chunks
    generate answer
    write one JSONL checkpoint row

Phase 2: score
  read checkpoint
  run RAGAS metrics
  append summary to eval_log.jsonl
```

Metrics:

- `Faithfulness`
- `AnswerRelevancy`
- `ContextRecall`

Judge:

- LLM: `gemma4:31b-cloud`
- Embeddings: `all-MiniLM-L6-v2`

Useful commands:

```bash
python src/eval.py --kb codon --note "baseline"
python src/eval.py --kb codon --collect-only
python src/eval.py --score-file data/eval_responses_2026-05-23T14-42-06Z.jsonl
```

### 13.2 `src/eval_grid.py`

Parameter sweep evaluator:

```text
for each chunk strategy:
  build ephemeral Chroma collection
  for each top_k:
    for each threshold:
      collect checkpoint
      score with RAGAS
      record result
  delete ephemeral collection
```

Default grid:

```text
chunk_strategies: paragraph, fixed
fixed chunk_size: 200
top_k_values: 3, 5, 8
threshold_values: 0.50, 0.65, 0.80
```

Use:

```bash
python src/eval_grid.py --kb codon
```

Important detail:

- `eval_grid.py` exists and is importable.
- The current workbench does not expose `/api/eval/grid`.

### 13.3 Current eval log signal

Recent logged Codon runs show:

```text
top_k=5 threshold=0.65  -> faithfulness 0.8000, answer_relevancy 0.6643, context_recall 0.4333
top_k=8 threshold=0.65  -> faithfulness 0.8621, answer_relevancy 0.6039, context_recall 0.4333
top_k=8 threshold=0.80  -> faithfulness 0.8356, answer_relevancy 0.6100, context_recall 0.4333
top_k=3 threshold=0.45  -> faithfulness 0.7106, answer_relevancy 0.5298, context_recall 0.3667
```

Interpretation:

- Raising top-K helped faithfulness in one run but did not improve context recall.
- Lowering threshold to 0.45 hurt context recall.
- The persistent low context recall supports prioritizing better chunking and
  better question/context alignment before advanced agent work.

## 14. Legacy Streamlit UI

Files:

```text
app.py
pages/1_Knowledge_Bases.py
pages/2_Ask.py
```

Current role:

- Legacy UI and scratchpad.
- Still useful for basic KB management and chat.
- Not the active target for new feature development.

`pages/1_Knowledge_Bases.py`:

- Shows KB overview.
- Creates KB entries.
- Uploads files and invokes `src/ingest.py`.
- Deletes files and KBs.
- Can re-ingest a KB by deleting and recreating registry state.

`pages/2_Ask.py`:

- Uses LangChain Chroma + ChatOllama.
- Clears chat when switching KB.
- Applies relevance threshold.
- Shows retrieved chunks in an expander.
- Reloads `src.config` on rerun to avoid stale Streamlit config values.

Known Streamlit/Chroma lesson:

```text
Do not cache a Chroma PersistentClient across Streamlit reruns.
Clear Chroma's SharedSystemClient cache before creating a new client.
```

## 15. CLI Reference

Setup:

```bash
bash setup.sh
source .venv/bin/activate
```

```bat
setup.bat
.venv\Scripts\activate
```

Run workbench:

```bash
uvicorn workbench.main:app --port 8000 --reload
```

Run legacy Streamlit:

```bash
streamlit run app.py
```

Build:

```bash
python src/ingest.py --kb codon
python src/ingest.py --kb codon --delete-file 01_company_overview.md
python src/ingest.py --kb codon --delete-kb
```

Retrieve:

```bash
python src/query.py --kb codon "What industries does Codon work in?"
```

Ask:

```bash
python src/rag.py --kb codon "Who founded Codon?"
python src/rag_lc.py --kb codon "Who founded Codon?"
```

Evaluate:

```bash
python src/eval.py --kb codon --note "baseline"
python src/eval_grid.py --kb codon
```

Optional tunnel:

```bash
ngrok http 8000
ngrok http 8501
```

## 16. Operational Constraints

Local-only by default:

- No cloud LLM or embedding APIs are required for normal ask/build operation.
- RAGAS evaluation uses an Ollama model named `gemma4:31b-cloud`; despite the
  name, the code calls the local Ollama daemon.

Model caching:

- Workbench caches `SentenceTransformer` models in process.
- First use may take seconds.
- Later calls are warm.

ChromaDB threading:

- ChromaDB uses SQLite under the hood.
- The workbench clears Chroma's shared system cache before creating a client.
- This mitigates same-process cross-thread SQLite errors.
- It does not make two separate Python processes fully safe against concurrent
  writes to the same ChromaDB path.

File writes:

- Workbench upload writes `UploadFile.filename` into the KB corpus directory.
- Production code should sanitize filenames before writing.

No auth:

- The workbench allows CORS from any origin.
- There is no user auth, RBAC, CSRF protection, or per-KB access control.
- This is fine for a local tool, not for public deployment.

No automated tests:

- There is no `pytest` suite.
- Important behavior is currently verified manually.

## 17. Known Limitations And Risks

Markdown structure loss:

- Blank-line chunking can separate headers from body text.
- Lists, tables, and code fences can be split badly.
- This likely contributes to weak context recall.

Chunk-step naming:

- UI wording says "overlap", backend uses "step".
- The math should be clarified before serious chunk experiments.

Prompt experimentation gap:

- `AskRequest.system_prompt` exists.
- UI does not expose it.
- Eval endpoint does not accept it.
- Eval logs only record `cfg.PROMPT_HASH`, not request-level prompt overrides.

Eval dataset rigidity:

- Workbench eval uses a hardcoded test file.
- No upload, selection, validation, or dataset registry exists.

Workbench/CLI ingest divergence:

- Workbench always full-rebuilds and stamps a manifest.
- CLI ingest is incremental and uses `src/config.py` defaults.
- Mixing them can make registry manifests and actual collection state harder to
  reason about.

Ollama base URL hardcoding:

- Several code paths use `http://127.0.0.1:11434` directly.
- Docker or remote Ollama needs env-based config.

Broad exception handling:

- Some paths catch broad exceptions and convert them into empty results or SSE
  errors.
- This is user-friendly but can hide infrastructure faults during development.

## 18. Documentation Notes

Existing docs are useful but partially stale:

- `README.md` is broadly accurate but describes the workbench as having two
  tabs; the current SPA has Build, Ask, and Evaluate.
- `docs/assessment_and_roadmap.md` says the Codon build manifest was missing;
  the current registry now has a Codon build block.
- `chapter2.md` describes future work and is incorporated into
  `EXTENSIONS_ROADMAP.md`.

This spec should be treated as the current baseline as of 2026-05-23.

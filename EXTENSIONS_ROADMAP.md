# codon-rag Extensions, Improvements, And Generalizations

Date: 2026-05-23

This roadmap incorporates the ideas from `chapter2.md`, corrects them against
the current codebase, and generalizes them into a staged plan.

The companion factual spec is `PROJECT_SPEC.md`.

## Roadmap Principles

1. Improve retrieval quality before adding agentic complexity.
2. Keep the workbench as the active UI surface.
3. Make experiments measurable through eval logs, not anecdotes.
4. Preserve local-first operation while making deployment configurable.
5. Split "Codon-specific assistant" from "general RAG workbench" over time.

High-level dependency map:

```text
better chunks
  |
  v
trustworthy evals
  |
  v
prompt/model/dataset experiments
  |
  v
docker + stable services
  |
  v
agentic ingest/eval workflows
  |
  v
general RAG platform
```

ELI5 version:

```text
Before teaching the system to drive itself, make sure:
1. It cuts documents into sensible pieces.
2. It can grade answers fairly.
3. It records what changed in each experiment.
4. It starts the same way on every machine.
```

## Soon

Goal: make the current workbench more trustworthy, easier to experiment with,
and less fragile. These are the changes that should happen before larger
architecture work.

### 1. Markdown-aware chunking

Current state:

- `paragraph` chunking splits on blank lines.
- This can detach a heading from the section it names.
- It can split lists, tables, and fenced code blocks in awkward places.
- RAGAS context recall is currently weak enough that this should be the first
  quality improvement.

Target:

```text
Markdown file
  |
  v
parse structure
  |
  v
section chunks:
  - heading path
  - section body
  - intact lists
  - intact code fences
  - intact tables when possible
  |
  v
metadata includes section title and heading path
```

Implementation options:

- Use `markdown-it-py` or `mistune` for AST-level control.
- Or use LangChain's Markdown splitters if that fits the existing dependency
  stack and produces predictable chunks.
- Keep a small internal interface so more strategies can be added later.

Suggested interface:

```python
class Chunker:
    name: str

    def split(self, text: str, source_file: str) -> list[dict]:
        return [
            {
                "text": "...",
                "metadata": {
                    "section_title": "...",
                    "heading_path": "H1 > H2 > H3",
                    "chunk_kind": "markdown_section",
                },
            }
        ]
```

Acceptance criteria:

- Headers and their body stay together.
- Fenced code blocks are not split.
- Markdown tables are either kept whole or split only at row boundaries.
- Chunk metadata includes at least `section_title` and `heading_path`.
- `python src/eval.py --kb codon --note "after markdown-aware chunking"` is
  run and compared with the current baseline.

Generalization:

- Treat this as a chunking plugin system, not a one-off patch.
- Future corpora may include HTML, PDFs, Word docs, or transcripts. The system
  should be able to select a chunker by content type.

### 2. Fix chunk-step semantics in the UI

Current state:

- The Build tab labels the slider as overlap.
- The backend expects `chunk_step`, meaning how far the window advances.
- True overlap is `chunk_size - chunk_step`.

Fix:

- Either rename the UI field to `Step (chars)`.
- Or keep `Overlap (chars)` in the UI and convert before sending:

```javascript
chunk_step = chunk_size - overlap
```

Recommended:

- Expose "Overlap" to the user because it is easier to reason about.
- Store both values in the manifest:
  - `chunk_size`
  - `chunk_step`
  - `chunk_overlap`

ELI5:

```text
If each card is 200 letters long and you want 50 letters repeated:
  step = 200 - 50 = 150
```

### 3. Prompt editor in Ask and Eval

Current state:

- `POST /api/ask` already accepts `system_prompt`.
- The Ask UI does not expose it.
- `POST /api/eval/run` does not accept it.
- `src/eval.py` always uses `cfg.SYSTEM_PROMPT`.

Target:

- Add a collapsible prompt editor to Ask.
- Add a collapsible prompt editor to Evaluate.
- Prepopulate both with `cfg.SYSTEM_PROMPT`.
- Send the edited prompt in request bodies.
- Log prompt hash and prompt label in eval records.

API changes:

```python
class EvalRequest(BaseModel):
    kb_id: str
    top_k: int = 5
    threshold: float = 0.65
    note: str = ""
    system_prompt: str | None = None
```

Eval record addition:

```json
{
  "prompt_source": "request",
  "prompt_hash": "abcd1234",
  "prompt_text_preview": "You are a helpful assistant..."
}
```

Acceptance criteria:

- Ask tab can run the same question with two different prompts without editing
  files.
- Eval tab can score a run with a custom prompt.
- Eval logs make prompt differences visible.

### 4. Uploadable and selectable eval question sets

Current state:

- Workbench eval is hardcoded to `data/test_questions.json`.
- The UI displays "15" statically.
- `chapter2.md` asks for uploadable `questions.json`; that is not implemented.

Target:

- Add a question-set picker.
- Add an upload input for `.json`.
- Validate client-side and server-side.
- Store uploaded sets under a controlled directory such as
  `data/eval_sets/`.

Required schema:

```json
[
  {
    "question": "When was Codon founded?",
    "ground_truth": "Codon Consulting was founded in 2019."
  }
]
```

Validation:

- Top-level value must be a non-empty array.
- Each item must have `question` and `ground_truth`.
- Strings must be non-empty.
- Optional fields such as `id`, `topic`, and `difficulty` should be preserved.

Generalization:

- Support named eval sets:

```text
data/eval_sets/
  codon_baseline_15.json
  codon_edge_cases_25.json
  buddhism_seminar_20.json
```

### 5. Wire eval grid into the workbench

Current state:

- `src/eval_grid.py` exists.
- The workbench has `/api/eval/run`, but no `/api/eval/grid`.
- `chapter2.md` mentions `/api/eval/grid`; that is a future route, not current
  reality.

Target:

- Add `POST /api/eval/grid`.
- Stream `build_start`, `build_done`, `combo_start`, `question`,
  `combo_done`, `build_cleanup`, `done`, and `error`.
- Render a results table as combos complete.
- Add running average bars for:
  - faithfulness
  - answer relevancy
  - context recall

Suggested request:

```json
{
  "kb_id": "codon",
  "chunk_configs": [
    {"strategy": "paragraph", "chunk_size": null},
    {"strategy": "fixed", "chunk_size": 200}
  ],
  "top_k_values": [3, 5, 8],
  "threshold_values": [0.50, 0.65, 0.80],
  "questions_path": "data/test_questions.json",
  "note": "grid after markdown chunking",
  "system_prompt": null
}
```

UI sketch:

```text
+-------------------------------------------------------------+
| Eval Grid                                                   |
+----------------------+--------------------------------------+
| Strategies            | paragraph [x] fixed [x] markdown [ ] |
| top_k values          | 3, 5, 8                              |
| thresholds            | 0.50, 0.65, 0.80                     |
| question set          | codon_baseline_15                    |
| prompt                | default                              |
+----------------------+--------------------------------------+
| combo results stream                                         |
| strategy   k   threshold   faithful   relevancy   recall     |
| paragraph  5   0.65        0.80       0.66       0.43       |
+-------------------------------------------------------------+
```

### 6. Safety hardening for files and paths

Current state:

- Uploads accept `.md` and `.txt`.
- The backend writes `UploadFile.filename` under the corpus directory.
- A hostile filename should not be trusted even in a local tool.

Fix:

- Use `Path(filename).name`.
- Reject names containing path separators, drive prefixes, NUL bytes, or empty
  basenames.
- Normalize extensions to lowercase.
- Enforce a max upload size.
- Reject zero-byte files.
- Read text with explicit UTF-8 and a clear error for invalid encoding, or use
  `errors="replace"` with a warning.

### 7. Logging and visible diagnostics

Current state:

- CLI scripts use `print()`.
- Workbench mostly emits SSE errors and relies on the uvicorn console.

Target:

- Add structured Python logging.
- Write logs to `logs/codon-rag.log`.
- Include request ids for build, ask, and eval runs.
- Surface log file path in error events.

Minimum useful log fields:

```text
timestamp
level
request_id
kb_id
route_or_command
message
exception
```

### 8. Align README and docs with current code

Current state:

- `README.md` still describes two workbench tabs.
- `docs/assessment_and_roadmap.md` contains some stale state.
- `PROJECT_SPEC.md` is now the factual baseline.

Fix:

- Update README to reference Build, Ask, and Evaluate.
- Link to `PROJECT_SPEC.md`.
- Link to `EXTENSIONS_ROADMAP.md`.
- Mark older assessment docs as historical.

### 9. Add first automated tests

Start small and focus on contracts that break easily:

- Chunking:
  - paragraph split
  - fixed split
  - markdown section split once implemented
- Registry:
  - read missing registry
  - create KB
  - build manifest shape
- API:
  - `GET /api/kbs`
  - file upload validation
  - `POST /api/ask` no-collection error
- Eval:
  - question-set validation
  - log record schema

Suggested test layout:

```text
tests/
  test_chunking.py
  test_registry.py
  test_workbench_api.py
  test_eval_schema.py
```

## In 3 Months

Goal: make the system portable, measurable, and capable of semi-automated
corpus onboarding.

### 1. Dockerize the workbench

`chapter2.md` proposes Dockerization with Ollama staying on the host. That is
the right direction.

Target architecture:

```text
Host machine
  |
  +-- Ollama at localhost:11434
  |
  +-- Docker container: codon-rag
      |
      +-- FastAPI workbench
      +-- src/ modules
      +-- mounted chroma_db/
      +-- mounted data/
      +-- mounted HuggingFace cache
```

Required config change:

```python
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
```

Apply this to:

- `workbench/main.py`
- `src/rag.py`
- `src/rag_lc.py`
- `src/eval.py`
- `src/eval_grid.py`

Dockerfile sketch:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt workbench/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r workbench/requirements.txt

COPY . .

CMD ["uvicorn", "workbench.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Compose sketch:

```yaml
services:
  codon-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./data:/app/data
      - ~/.cache/huggingface:/root/.cache/huggingface
```

Acceptance criteria:

- `docker compose up --build` starts the workbench.
- Build, Ask, and Evaluate work against host Ollama.
- No source code contains a required hardcoded Ollama base URL.

### 2. Move ChromaDB behind a service boundary

Current issue:

- `PersistentClient` is convenient but not ideal when multiple Python processes
  can access the same SQLite-backed Chroma path.

Target:

```text
workbench process
  |
  v
Chroma HttpClient
  |
  v
chroma server process
  |
  v
chroma_db/
```

Benefits:

- Cleaner concurrency model.
- Easier Docker composition.
- A future worker can build KBs without sharing SQLite internals with the API
  process.

### 3. Add background jobs and run records

Current state:

- Build and eval use daemon threads.
- If the browser disconnects, the operation may continue but there is no
  durable job record.

Target:

- Introduce job ids.
- Persist job state to `data/jobs/*.json` or a lightweight SQLite DB.
- Let the UI reconnect to running jobs.
- Separate "start job" from "stream job events".

Job state sketch:

```json
{
  "job_id": "build_20260523_abc123",
  "kind": "build",
  "kb_id": "codon",
  "status": "running",
  "started_at": "...",
  "finished_at": null,
  "events_path": "data/jobs/build_20260523_abc123.jsonl"
}
```

### 4. LangGraph ingestion and auto-eval pipeline

`chapter2.md` proposes a LangGraph agent that receives a KB id and a corpus
folder, builds the KB, generates eval questions, runs a lite grid, and returns
a recommended config.

This should happen after chunking and eval grid are stable.

Graph:

```text
input: kb_id + corpus_path
  |
  v
[ingest]
  |
  v
[generate grounded questions]
  |
  v
[run lite eval grid]
  |
  v
[synthesize recommendation]
  |
  v
output: best config + explanation + eval log records
```

Lite eval set:

```python
AGENT_EVAL_COMBOS = {
    "chunk_strategies": ["markdown", "paragraph", "fixed"],
    "top_k_values": [5, 8],
    "threshold_values": [0.50, 0.65],
}
```

Correction to the chapter:

- Do not use generated questions as the only evaluation set.
- Generated questions are useful for smoke tests and coverage discovery.
- A trusted human-curated set should remain the primary benchmark.

Question generation guardrails:

- Sample chunks across documents and sections.
- Require answer spans or source chunk ids.
- Reject questions that cannot be answered from sampled chunks.
- Mix factual, synthesis, boundary, and "not in corpus" questions.

### 5. Metadata filtering

Add filtering by metadata:

- source file
- section title
- heading path
- tags
- date
- document kind
- KB id

Target API shape:

```json
{
  "kb_id": "codon",
  "question": "What does Codon say about edge deployment?",
  "filters": {
    "source_file": ["04_industrial_edge.md"],
    "tags": ["industrial edge"]
  }
}
```

UI:

- Add optional filter drawer in Ask.
- Let retrieved chunk cards display metadata.

### 6. Model and prompt experiment registry

Current state:

- Embedding model and prompt hash are logged.
- Prompt text variants are not managed.
- Ollama model choice is per request but not part of a formal experiment
  registry.

Target:

```text
data/experiments/
  prompts.json
  model_profiles.json
  retrieval_presets.json
```

Prompt registry example:

```json
{
  "codon-sales-v1": {
    "description": "Concise sales assistant with source citations",
    "text": "You are a helpful assistant...",
    "created_at": "2026-05-23T..."
  }
}
```

### 7. Retrieval improvements before knowledge graphs

Add simple hybrid retrieval first:

```text
BM25 keyword stream
  |
  +-- exact terms and proper nouns
  |
  v
RRF fusion ----> reranked top_k chunks
  ^
  |
Vector semantic stream
```

Implementation:

- Add `rank_bm25`.
- Build BM25 index at ingest time or lazily per collection.
- Fuse with vector ranks using Reciprocal Rank Fusion.
- Evaluate with the existing grid.

Rationale:

- This catches vocabulary mismatch without jumping straight to a graph store.

### 8. CI and release hygiene

Add:

- `pytest`
- formatting check
- import smoke tests
- small API smoke test
- docs link check
- optional eval smoke test with a tiny fixture corpus

Do not run full RAGAS in ordinary CI. It is too slow and depends on local model
availability.

## In 6 Months To A Year

Goal: turn the local Codon RAG workbench into a general, reliable RAG and
document-intelligence platform.

### 1. Generalize beyond Codon-specific assumptions

Current identity:

- The default prompt and corpus are Codon-focused.
- The architecture is already mostly generic.

Target:

```text
codon-rag today:
  local multi-KB RAG workbench

generalized platform:
  local-first document intelligence workbench
  with pluggable corpora, chunkers, retrievers, prompts, models, and evals
```

Rename at the concept level:

- "KB" -> "workspace" or "collection"
- "Codon assistant" -> "RAG assistant profile"
- "test_questions" -> "eval set"
- "build manifest" -> "index manifest"

Keep repo/package names only if desired; the conceptual model should not be
Codon-specific.

### 2. Cross-KB and federated search

Add queries across multiple KBs:

```text
question
  |
  v
route to KBs or search all selected KBs
  |
  v
retrieve per KB
  |
  v
normalize and fuse ranks
  |
  v
answer with citations grouped by KB
```

UI:

- Multi-select KBs in Ask.
- Show result provenance:

```text
[codon/04_industrial_edge.md]
[buddhism/vimalakirti_seminar_memory.md]
```

Advanced:

- Add a router that predicts which KBs to search.
- Keep manual override available.

### 3. Full hybrid retrieval and reranking

Move from pure vector retrieval to a layered stack:

```text
              +------------------+
query ------> | query analyzer   |
              +---------+--------+
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
  BM25 search      vector search    metadata search
       |                |                |
       +----------------+----------------+
                        |
                        v
                RRF rank fusion
                        |
                        v
                 cross-encoder rerank
                        |
                        v
                  final context set
```

Possible components:

- BM25: `rank_bm25`
- Vector: existing ChromaDB
- Metadata: Chroma where filters or separate SQLite metadata table
- Reranker: local cross-encoder, BGE reranker, or Ollama-based lightweight
  reranking

Acceptance criteria:

- Improves recall on the curated eval set.
- Keeps context length bounded.
- Logs per-stream contributions so retrieval failures are debuggable.

### 4. Knowledge graph retrieval

Only add this when the corpus is large or heavily cross-referential.

Graph build:

```text
chunks
  |
  v
entity and relation extraction
  |
  v
nodes: people, orgs, products, concepts, files, sections
edges: mentions, part_of, related_to, cites, same_as
  |
  v
graph store
```

Query flow:

```text
query entities
  |
  v
graph neighborhood expansion
  |
  v
candidate chunks
  |
  v
fuse with BM25/vector candidates
```

Start with NetworkX or SQLite tables before adopting a graph database.

### 5. Production-grade deployment

Needed pieces:

- Docker Compose with separate services:
  - workbench API
  - Chroma server
  - optional worker
  - optional reverse proxy
- Auth for non-local deployment
- HTTPS
- CORS restricted by environment
- Upload size limits
- Backups for `data/` and `chroma_db/`
- Health checks:
  - `/health`
  - `/health/ollama`
  - `/health/chroma`
  - `/health/models`

Deployment diagram:

```text
browser
  |
  v
reverse proxy
  |
  v
FastAPI workbench
  |
  +-- worker queue
  |
  +-- Chroma server
  |
  +-- Ollama
  |
  +-- data volume
```

### 6. Document ingestion beyond Markdown and text

Add loaders for:

- PDF
- DOCX
- HTML
- CSV/TSV
- slide decks
- web crawls
- copied URLs

Pipeline:

```text
raw file
  |
  v
loader
  |
  v
normalized document model
  |
  v
chunker by content type
  |
  v
metadata extraction
  |
  v
embed and index
```

Normalized document model:

```json
{
  "doc_id": "...",
  "source_uri": "...",
  "title": "...",
  "mime_type": "application/pdf",
  "pages": [],
  "sections": [],
  "metadata": {}
}
```

### 7. Evaluation as a product feature

Turn evals into a first-class workflow:

- Eval set library.
- Run comparison view.
- Regression alerts.
- Per-question failure inspection.
- Retrieved-context viewer for each eval row.
- Export to CSV/JSON.
- Prompt and model diffing.
- "Promote this config" button that writes a build/retrieval preset.

Dashboard sketch:

```text
+-------------------------------------------------------------+
| Eval Runs                                                   |
+--------------------+---------+---------+--------+-----------+
| run                | prompt  | top_k   | recall | faithful  |
| baseline           | v1      | 5       | 0.43   | 0.80      |
| markdown chunks    | v1      | 5       | 0.61   | 0.84      |
| markdown + BM25    | v1      | 8       | 0.72   | 0.82      |
+--------------------+---------+---------+--------+-----------+
```

### 8. Agentic operations

After the LangGraph ingest/eval pipeline is reliable, add agents for:

- Corpus audit:
  - find stale docs
  - find duplicate docs
  - find weak metadata
- Eval set expansion:
  - propose new test questions
  - flag ambiguous ground truth
- Retrieval diagnosis:
  - explain why a question failed
  - suggest chunking or filter changes
- Release assistant:
  - run smoke evals before promoting a config

Guardrail:

- Agents should recommend and prepare changes.
- Human approval should remain required for deleting corpora, replacing eval
  sets, or promoting production configs.

### 9. Multi-user governance

If this becomes team-facing:

- User accounts
- Per-KB permissions
- Audit logs
- Prompt change approvals
- Eval result signing or freezing
- Dataset versioning
- Redaction policies for sensitive corpora

### 10. Productized variants

The current system can become several products by configuration:

DocSense:

- Domain document assistant for HSE, policies, SOPs, contracts, or technical
  manuals.
- Requires metadata filtering, document loaders, role-aware access, and strong
  citation UX.

Sales assistant:

- Codon-facing sales and case-study assistant.
- Requires polished prompt profiles and strong source citation.

Research seminar assistant:

- Works well for the Buddhism corpus.
- Requires long-form source navigation, section-aware citations, and careful
  "unknown" handling.

Internal knowledge workbench:

- General local-first RAG tool for experiments.
- Requires robust ingestion, evals, presets, and exportable configs.

### 11. Long-range architecture target

```text
                 +----------------------+
                 | Workbench UI         |
                 | Build Ask Eval Admin |
                 +----------+-----------+
                            |
                            v
                 +----------+-----------+
                 | FastAPI API          |
                 | auth, jobs, configs  |
                 +----+----------+------+
                      |          |
          +-----------+          +----------------+
          v                                     v
+---------+---------+                 +---------+---------+
| Worker service    |                 | Retrieval service |
| ingest, eval, KG  |                 | BM25/vector/RRF   |
+---------+---------+                 +---------+---------+
          |                                     |
          v                                     v
+---------+---------+                 +---------+---------+
| Document store    |                 | Chroma server     |
| data + metadata   |                 | vector indexes    |
+---------+---------+                 +---------+---------+
          |
          v
+---------+---------+
| Model providers   |
| Ollama/local/cloud|
+-------------------+
```

The north star is not "more agents everywhere." It is a reliable local-first
RAG platform where every answer can be traced back to:

- corpus version
- chunker version
- embedding model
- retrieval config
- prompt version
- generator model
- eval score history

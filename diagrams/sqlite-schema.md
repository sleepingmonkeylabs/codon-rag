# codon-rag — SQLite Backend Schema

Replaces `kb_registry.json`, `test_questions.json`, `eval_responses_*.jsonl`,
`grid_responses_*.jsonl`, and `eval_log.jsonl` with a single `codon_rag.db`
file in the project root. ChromaDB remains unchanged at this stage.

---

## Tables

### `kbs`
Single source of truth for every knowledge base. Replaces `kb_registry.json`.

```sql
CREATE TABLE kbs (
    kb_id        TEXT PRIMARY KEY,
    description  TEXT,
    config       TEXT NOT NULL,   -- JSON: embed_model, chunk_strategy, chunk_size,
                                  --       chunk_step, distance, top_k, threshold
    build_stats  TEXT,            -- JSON: built_at, total_chunks, sources
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Notes:**
- `config` stores the build parameters used for the last ingest. On rebuild, update in place.
- `build_stats` is NULL until the first `ingest` run completes successfully.
- `kb_id` is the natural key (e.g. `"codon"`, `"buddhism"`); no surrogate needed.

---

### `documents`
Every uploaded file, decoupled from `data/corpus/`. The corpus directory becomes
a **derived cache** that ingest.py materializes from this table — it is no longer
the source of truth.

```sql
CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    kb_id       TEXT NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    content     BLOB NOT NULL,
    size        INTEGER NOT NULL,
    hash        TEXT NOT NULL,    -- SHA-256; ingest uses this for change detection
    uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(kb_id, filename)
);

CREATE INDEX documents_kb_id_idx ON documents(kb_id);
CREATE INDEX documents_hash_idx  ON documents(hash);
```

**Notes:**
- `content` as BLOB keeps everything in one place. For very large corpora (100s of MB),
  swap to an external storage path and store a relative path in `content` instead.
- The `UNIQUE(kb_id, filename)` constraint replaces the implicit file-per-directory uniqueness.
- `hash` replaces `ingest.py`'s current file-hash cache; incremental ingest queries this
  instead of walking the filesystem.

---

### `test_questions`
Eval question sets, replacing `test_questions.json`. Supports per-KB question sets
and optional ground-truth answers for RAGAS `ContextRecall`.

```sql
CREATE TABLE test_questions (
    q_id         TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    kb_id        TEXT REFERENCES kbs(kb_id) ON DELETE SET NULL,
    question     TEXT NOT NULL,
    ground_truth TEXT,
    tags         TEXT,            -- JSON array of strings, e.g. ["retrieval","factual"]
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX test_questions_kb_id_idx ON test_questions(kb_id);
```

---

### `eval_runs`
One row per `eval.py` invocation. Replaces the `eval_responses_*.jsonl` filename
as run identifier; `status` tracks partial completion so Phase 2 can be re-run
safely against a `done` Phase 1.

```sql
CREATE TABLE eval_runs (
    run_id      TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    kb_id       TEXT NOT NULL REFERENCES kbs(kb_id),
    note        TEXT,
    params      TEXT NOT NULL,    -- JSON: top_k, threshold, chunk_strategy, model
    status      TEXT NOT NULL DEFAULT 'collecting',
                                  -- collecting | collected | scoring | done | failed
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);

CREATE INDEX eval_runs_kb_id_idx ON eval_runs(kb_id);
```

---

### `eval_samples`
One row per question per eval run. Replaces the JSONL checkpoint lines written
during Phase 1 and the scored rows from Phase 2. Appended incrementally — a crash
during Phase 1 leaves partial rows that Phase 2 picks up cleanly.

```sql
CREATE TABLE eval_samples (
    sample_id  TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    run_id     TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    q_id       TEXT REFERENCES test_questions(q_id),
    question   TEXT NOT NULL,
    answer     TEXT,
    contexts   TEXT,              -- JSON array of retrieved chunk strings
    scores     TEXT,              -- JSON: {faithfulness, answer_relevancy, context_recall}
    error      TEXT,              -- non-NULL if this sample failed
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX eval_samples_run_id_idx ON eval_samples(run_id);
```

---

### `grid_runs`
One row per `eval_grid.py` invocation (the outer loop). Replaces the implicit
grouping of `grid_responses_*.jsonl` files under a single timestamp.

```sql
CREATE TABLE grid_runs (
    grid_run_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    kb_id       TEXT NOT NULL REFERENCES kbs(kb_id),
    params_grid TEXT NOT NULL,    -- JSON: {chunk_strategies, top_k_values,
                                  --        threshold_values, chunk_sizes}
    status      TEXT NOT NULL DEFAULT 'running',
                                  -- running | done | failed
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
```

---

### `grid_cells`
One row per parameter combination. Replaces the per-cell JSONL files and the
append-only `eval_log.jsonl`. The `scores` column holds the same aggregate
metrics that were previously written to `eval_log.jsonl`.

```sql
CREATE TABLE grid_cells (
    cell_id        TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    grid_run_id    TEXT NOT NULL REFERENCES grid_runs(grid_run_id) ON DELETE CASCADE,
    chunk_strategy TEXT NOT NULL,
    chunk_size     INTEGER,
    top_k          INTEGER NOT NULL,
    threshold      REAL NOT NULL,
    scores         TEXT,          -- JSON: {faithfulness, answer_relevancy,
                                  --        context_recall, avg_score}
    samples        TEXT,          -- JSON array: per-question detail rows
    duration_s     REAL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX grid_cells_grid_run_id_idx ON grid_cells(grid_run_id);
```

---

## Relationship Summary

```
kbs ──< documents
kbs ──< test_questions
kbs ──< eval_runs ──< eval_samples >── test_questions
kbs ──< grid_runs ──< grid_cells
```

---

## Migration from JSON/JSONL

| Old file                          | New table(s)                      |
|-----------------------------------|-----------------------------------|
| `data/kb_registry.json`           | `kbs`                             |
| `data/corpus/<kb>/*.md`           | `documents` (content as BLOB)     |
| `data/test_questions.json`        | `test_questions`                  |
| `data/eval_responses_*.jsonl`     | `eval_runs` + `eval_samples`      |
| `data/grid_responses_*.jsonl`     | `grid_runs` + `grid_cells`        |
| `data/eval_log.jsonl`             | `grid_cells.scores` (queryable)   |

Migration script: read each JSON/JSONL, insert rows, verify counts, then archive
(don't delete) the originals.

---

## Enabling WAL Mode

Add this to `src/config.py` (or a new `src/db.py` connection helper):

```python
import sqlite3, pathlib

DB_PATH = pathlib.Path(__file__).parent.parent / "codon_rag.db"

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
```

WAL mode allows concurrent reads during a write — important since uvicorn's
thread pool can have ingest writing while the UI is reading KB metadata.

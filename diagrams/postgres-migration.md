# codon-rag — Postgres + pgvector Migration

Building on top of the SQLite schema. This step collapses ChromaDB into the
same database, giving you one connection string, one backup target, and one
transaction boundary for the entire application state.

---

## The Key Question: Do Manifests and Embeddings Live in the Same DB?

**Yes — and that's the point.**

In the SQLite step, ChromaDB still holds embeddings in a separate `chroma_db/`
directory. In the Postgres step, you add a `chunks` table with a `vector(384)`
column via the `pgvector` extension. ChromaDB is removed entirely.

Everything lives in one Postgres database:

| Concern              | SQLite step              | Postgres step             |
|----------------------|--------------------------|---------------------------|
| KB metadata          | `kbs` table (SQLite)     | `kbs` table (Postgres)    |
| Uploaded files       | `documents` table        | `documents` table         |
| **Vector embeddings**| ChromaDB (separate)      | **`chunks` table (same DB)**|
| Eval results         | `eval_samples` etc.      | Same, JSONB columns        |

---

## Speed

For your corpus sizes (hundreds to low thousands of chunks), pgvector query
latency is **effectively zero** — sub-millisecond for exact search, 1–3ms with
an IVFFlat approximate index. The LLM generation step (1–30 seconds on local
Ollama) will always dominate.

At scale:
- **IVFFlat** (`lists=100`): ~2ms at 100K vectors, recall ~97%.
- **HNSW** (`m=16, ef_construction=64`): ~1ms at 1M vectors, recall ~99%. Better
  default once you care about retrieval quality at scale.

The local Docker container adds ~1–2ms round-trip overhead vs SQLite file I/O.
For a RAG workbench this is completely irrelevant.

---

## Concurrency

This is where Postgres wins decisively over both SQLite and ChromaDB:

**SQLite + WAL:** Allows one writer + multiple concurrent readers. Fine for a
single-user local workbench. Would become a bottleneck if multiple ingest jobs
run simultaneously (e.g., rebuilding two KBs at once).

**ChromaDB:** Uses its own SQLite internally, inheriting the same single-writer
limitation. The threading gotchas documented in `CLAUDE.md` go away entirely
when ChromaDB is removed.

**Postgres:** Full MVCC — reads never block writes, writes never block reads.
Multiple simultaneous ingest jobs, eval runs, and UI queries are all fine.
`pgvector` index builds (`CREATE INDEX ... USING hnsw`) can run concurrently
with reads (Postgres builds the index without locking the table). For a local
workbench this headroom is overkill, but it means the architecture won't need
revisiting when you add background jobs or multiple concurrent users.

---

## Latency

| Operation                     | SQLite step    | Postgres step  | Delta        |
|-------------------------------|----------------|----------------|--------------|
| KB metadata read              | ~0.1ms         | ~1ms           | +0.9ms       |
| Vector similarity search      | ChromaDB ~5ms  | pgvector ~2ms  | −3ms         |
| Eval sample insert (row)      | ~0.2ms         | ~1ms           | +0.9ms       |
| Full RAG round-trip           | ~15s (Ollama)  | ~15s (Ollama)  | negligible   |

The LLM is the latency. Everything else is noise.

---

## New Table: `chunks`

This is the only net-new table. It replaces all ChromaDB collections.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    chunk_id    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID        NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kb_id       TEXT        NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
    content     TEXT        NOT NULL,
    embedding   vector(384),           -- all-MiniLM-L6-v2 output dimension
    chunk_index INTEGER     NOT NULL,  -- position within document
    metadata    JSONB,                 -- chunk_strategy, chunk_size, file_hash, etc.
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Approximate nearest-neighbor index (build after bulk insert, not before)
CREATE INDEX chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_kb_id_idx ON chunks(kb_id);
CREATE INDEX chunks_doc_id_idx ON chunks(doc_id);
```

Similarity query replacing `chromadb.Collection.query()`:

```python
# top-K chunks for a query vector, filtered to one KB
SELECT chunk_id, content, metadata,
       1 - (embedding <=> %s::vector) AS score
FROM   chunks
WHERE  kb_id = %s
ORDER  BY embedding <=> %s::vector
LIMIT  %s;
```

The `<=>` operator is cosine distance. `1 - distance = similarity`, matching
the existing threshold semantics in `src/config.py`.

---

## Schema Changes from SQLite → Postgres

All column types tighten up, JSON becomes JSONB (indexed, queryable):

| SQLite type          | Postgres type    | Notes                              |
|----------------------|------------------|------------------------------------|
| `TEXT PRIMARY KEY`   | `UUID PRIMARY KEY DEFAULT gen_random_uuid()` | except `kb_id` which stays TEXT |
| `TEXT (JSON)`        | `JSONB`          | indexed, operators available       |
| `TEXT (datetime)`    | `TIMESTAMPTZ`    | timezone-aware                     |
| `BLOB`               | `BYTEA`          | document content                   |
| `REAL`               | `DOUBLE PRECISION` |                                  |
| —                    | `vector(384)`    | new, requires pgvector extension   |

The `kbs.kb_id` stays `TEXT` (natural key like `"codon"`) — no UUID needed there.

---

## Docker Compose

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: codon_rag
      POSTGRES_USER: codon
      POSTGRES_PASSWORD: codon_local
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./diagrams/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql

  workbench:
    build: .
    command: uvicorn workbench.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://codon:codon_local@db:5432/codon_rag
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    depends_on:
      - db
    volumes:
      - .:/app

volumes:
  pgdata:
```

Ollama stays on the host (`host.docker.internal`) — no need to containerize the
GPU-bound LLM.

---

## Migration Path (SQLite → Postgres)

The SQLite step is the right first move. Migration to Postgres is then a
two-script operation:

**Step 1 — Structural migration** (`scripts/migrate_to_postgres.py`):
Read every row from SQLite, insert into Postgres. Rows are identical in shape;
only type coercions are needed (TEXT→UUID, TEXT JSON→JSONB, TEXT→TIMESTAMPTZ).

**Step 2 — Embedding migration** (`scripts/migrate_embeddings.py`):
For each KB, re-run `build_kb()` targeting Postgres `chunks` instead of
ChromaDB. Or, if you want to avoid re-embedding: dump ChromaDB vectors
(`collection.get(include=["embeddings","documents","metadatas"])`), then
bulk-insert into `chunks` with `COPY`.

After both scripts pass row-count assertions, flip `DATABASE_URL` in
`src/config.py` and archive `chroma_db/`.

---

## What Goes Away

- `chroma_db/` directory and all ChromaDB infrastructure
- `chromadb.api.client.SharedSystemClient.clear_system_cache()` workaround
- Cross-thread SQLite errors (the whole section in CLAUDE.md)
- Scattered `.jsonl` files in `data/`
- `data/corpus/` as source of truth
- `data/kb_registry.json`

One connection string. One `docker compose up`.

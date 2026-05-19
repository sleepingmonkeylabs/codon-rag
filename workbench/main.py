"""
workbench/main.py
=================
FastAPI backend for the codon-rag Workbench UI.

Runs ALONGSIDE the existing Streamlit app — it does not replace it.

  Streamlit  →  http://localhost:8501   (streamlit run app.py)
  Workbench  →  http://localhost:8000   (uvicorn workbench.main:app --reload)

Both apps share the same src/ modules, chroma_db/, and data/ directories.
No logic is duplicated — this file only wires HTTP endpoints around existing code.

Design notes
------------
* All KB state lives in data/kb_registry.json — the single source of truth.
  When a KB is built we stamp a "build" manifest into its registry entry
  (embed_model, chunk_strategy, params, built_at, etc.).  That manifest is
  what the UI renders as the frozen "last build" record.

* Ingest runs in a background thread (blocking CPU + I/O) and streams progress
  back to the browser via Server-Sent Events (SSE).  The thread puts dicts onto
  a stdlib queue.Queue; the SSE generator polls that queue and yields events.

* The /api/ask SSE endpoint embeds the question, retrieves chunks from ChromaDB,
  sends a "chunks" event immediately (so the UI can render them), then opens a
  streaming Ollama connection and forwards each token as a "token" event.

* SentenceTransformer models are cached in a module-level dict so they are only
  loaded once per process rather than on every request.

* ChromaDB's SharedSystemClient cache is cleared before every client creation to
  avoid SQLite cross-thread errors (see CLAUDE.md → Streamlit & ChromaDB Gotchas).
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import hashlib
import json
import os
import queue as sync_queue
import sys
import threading
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── third-party ───────────────────────────────────────────────────────────────
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── local: add repo root so we can "import src.config" ───────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import src.config as cfg  # noqa: E402  (after sys.path manipulation)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="codon-rag Workbench", version="0.1.0")

# Allow any origin in dev — tighten this for production deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory that holds index.html (and any future static assets).
STATIC_DIR = Path(__file__).parent / "static"

# Local Ollama base URL — keep in sync with src/rag.py.
OLLAMA_BASE = "http://127.0.0.1:11434"

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model cache
# ─────────────────────────────────────────────────────────────────────────────

# SentenceTransformer models are expensive to load (~1-3 s).  Cache them by
# name so subsequent requests in the same process pay zero load time.
# Models are thread-safe for inference, so sharing across threads is fine.
_model_cache: dict[str, SentenceTransformer] = {}


def _get_embed_model(name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading it on first use."""
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_registry() -> dict:
    """Read data/kb_registry.json and return it as a dict.  Returns {} if missing."""
    path = Path(cfg.KB_REGISTRY_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(registry: dict) -> None:
    """Write the registry dict back to disk (pretty-printed, UTF-8)."""
    Path(cfg.KB_REGISTRY_PATH).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _chroma_client() -> chromadb.PersistentClient:
    """
    Return a fresh ChromaDB PersistentClient, clearing the shared system cache
    first.  This prevents the SQLite cross-thread error described in CLAUDE.md
    ("SQLite objects created in a thread can only be used in that same thread").
    """
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    return chromadb.PersistentClient(path=cfg.CHROMA_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Root — serve the SPA
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the single-page workbench application."""
    return FileResponse(STATIC_DIR / "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# KB management endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/kbs", summary="List all knowledge bases")
def list_kbs():
    """
    Return all KBs from the registry, each enriched with a live chunk_count
    pulled directly from ChromaDB (so the UI always shows the real number).

    Response shape:
        [
          {
            "kb_id": "codon",
            "collection": "codon_docs",
            "corpus_dir": "data/corpus",
            "description": "...",
            "build": { ...manifest... } | null,
            "chunk_count": 312
          },
          ...
        ]
    """
    registry = _read_registry()
    client = _chroma_client()
    existing_collections = {c.name for c in client.list_collections()}

    result = []
    for kb_id, kb in registry.items():
        cname = kb.get("collection", kb_id)
        chunk_count = 0
        if cname in existing_collections:
            try:
                chunk_count = client.get_collection(cname).count()
            except Exception:
                pass  # collection might be empty or broken — treat as 0

        result.append({
            "kb_id":       kb_id,
            "collection":  cname,
            "corpus_dir":  kb.get("corpus_dir", ""),
            "description": kb.get("description", ""),
            "build":       kb.get("build"),   # None until first successful build
            "chunk_count": chunk_count,
        })

    return result


class CreateKBRequest(BaseModel):
    kb_id:       str
    description: str = ""
    # Optional subdirectory under data/corpus/.  Defaults to kb_id.
    corpus_subdir: str = ""


@app.post("/api/kbs", status_code=201, summary="Register a new KB (no ingest)")
def create_kb(body: CreateKBRequest):
    """
    Add a new entry to kb_registry.json and create the corpus directory.
    Does NOT run ingest — call /api/kbs/{kb_id}/build for that.

    The collection name is derived as "{kb_id}_docs" and stored in the registry.
    """
    kb_id = body.kb_id.strip()
    if not kb_id:
        raise HTTPException(400, "kb_id must not be empty")

    registry = _read_registry()
    if kb_id in registry:
        raise HTTPException(400, f"KB '{kb_id}' already exists")

    # Resolve corpus directory
    subdir = body.corpus_subdir.strip() or kb_id
    corpus_rel = f"data/corpus/{subdir}"
    corpus_abs = Path(cfg.BASE_DIR) / corpus_rel
    corpus_abs.mkdir(parents=True, exist_ok=True)

    entry = {
        "kb_id":       kb_id,
        "collection":  f"{kb_id}_docs",
        "corpus_dir":  corpus_rel,
        "description": body.description,
        # "build" key is absent until first successful ingest
    }
    registry[kb_id] = entry
    _write_registry(registry)

    return entry


@app.delete("/api/kbs/{kb_id}", summary="Drop a KB (collection + registry entry)")
def delete_kb(kb_id: str):
    """
    Delete the ChromaDB collection and remove the KB from kb_registry.json.
    Corpus files are left untouched (they live in data/corpus/).
    """
    registry = _read_registry()
    if kb_id not in registry:
        raise HTTPException(404, f"KB '{kb_id}' not found")

    cname = registry[kb_id].get("collection", kb_id)
    client = _chroma_client()
    existing = {c.name for c in client.list_collections()}
    if cname in existing:
        client.delete_collection(cname)

    del registry[kb_id]
    _write_registry(registry)

    return {"deleted": kb_id}


# ─────────────────────────────────────────────────────────────────────────────
# Corpus file management
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/kbs/{kb_id}/files", summary="List corpus files for a KB")
def list_files(kb_id: str):
    """Return [{name, size}] for every .md / .txt file in the KB's corpus dir."""
    registry = _read_registry()
    if kb_id not in registry:
        raise HTTPException(404, f"KB '{kb_id}' not found")

    corpus_dir = Path(cfg.BASE_DIR) / registry[kb_id]["corpus_dir"]
    if not corpus_dir.exists():
        return []

    return [
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(corpus_dir.iterdir())
        if f.is_file() and f.suffix in (".md", ".txt")
    ]


@app.post("/api/kbs/{kb_id}/files", summary="Upload a file to the corpus")
async def upload_file(kb_id: str, file: UploadFile = File(...)):
    """
    Accept a multipart upload and save it to the KB's corpus directory.
    Only .md and .txt extensions are accepted.
    """
    registry = _read_registry()
    if kb_id not in registry:
        raise HTTPException(404, f"KB '{kb_id}' not found")
    if not (file.filename or "").endswith((".md", ".txt")):
        raise HTTPException(400, "Only .md and .txt files are accepted")

    corpus_dir = Path(cfg.BASE_DIR) / registry[kb_id]["corpus_dir"]
    corpus_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    dest = corpus_dir / file.filename
    dest.write_bytes(content)

    return {"uploaded": file.filename, "size": len(content)}


@app.delete("/api/kbs/{kb_id}/files/{filename}", summary="Remove a file from the corpus")
def delete_file(kb_id: str, filename: str):
    """Delete a single corpus file.  The KB index is NOT automatically rebuilt."""
    registry = _read_registry()
    if kb_id not in registry:
        raise HTTPException(404, f"KB '{kb_id}' not found")

    target = Path(cfg.BASE_DIR) / registry[kb_id]["corpus_dir"] / filename
    if not target.exists():
        raise HTTPException(404, f"File '{filename}' not found")

    target.unlink()
    return {"deleted": filename}


# ─────────────────────────────────────────────────────────────────────────────
# Ingest (build) — SSE streaming
# ─────────────────────────────────────────────────────────────────────────────

class BuildRequest(BaseModel):
    """Parameters for a KB build.  All are stored in the manifest on success."""
    embed_model:     str   = "all-MiniLM-L6-v2"
    chunk_strategy:  str   = "paragraph"   # "paragraph" | "fixed"
    chunk_size:      int   = 200           # chars — used only when strategy=="fixed"
    chunk_step:      int   = 150           # overlap step — used only when strategy=="fixed"
    distance:        str   = "cosine"      # ChromaDB hnsw:space
    top_k:           int   = 5
    threshold:       float = 0.65


# ── Chunkers (standalone — do not read from cfg so params come from the request)

def _chunk_paragraph(text: str) -> list[str]:
    """Split on blank lines; skip empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _chunk_fixed(text: str, size: int, step: int) -> list[str]:
    """Sliding window: `size`-char windows advancing by `step` chars (overlap = size - step)."""
    return [text[i: i + size] for i in range(0, len(text), step)]


def _blocking_ingest(
    kb_id:     str,
    kb_config: dict,
    params:    BuildRequest,
    q:         sync_queue.Queue,
) -> None:
    """
    Full ingest pipeline — intended to run in a daemon thread.

    Puts SSE event dicts onto `q`.  The calling generator drains the queue and
    yields them as SSE events.  A None sentinel on the queue means the thread
    has finished (success or error).

    On success the build manifest is stamped into kb_registry.json so it
    persists across restarts.

    Progress events:  {"type": "progress", "msg": "..."}
    Done event:       {"type": "done",     "msg": "...", "manifest": {...}}
    Error event:      {"type": "error",    "msg": "...", "traceback": "..."}
    """
    def progress(msg: str):
        q.put({"type": "progress", "msg": msg})

    try:
        corpus_dir     = Path(cfg.BASE_DIR) / kb_config["corpus_dir"]
        collection_name = kb_config["collection"]

        # ── 1. Discover source files ──────────────────────────────────────
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        md_files = sorted(
            f for f in os.listdir(corpus_dir)
            if f.endswith((".md", ".txt"))
        )
        if not md_files:
            raise ValueError(f"No .md or .txt files found in {corpus_dir}")

        progress(f"Found {len(md_files)} files — chunking...")

        # ── 2. Connect to ChromaDB ────────────────────────────────────────
        # Clear the system cache here (in the ingest thread) to get a fresh,
        # thread-local SQLite connection — avoids the cross-thread crash.
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)

        # Drop and recreate the collection so the distance metric is applied
        # fresh.  This is intentionally destructive — the UI warns the user.
        existing = {c.name for c in client.list_collections()}
        if collection_name in existing:
            client.delete_collection(collection_name)
            progress(f"Dropped old collection '{collection_name}'")

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": params.distance},
        )
        progress(f"Created collection '{collection_name}' (metric={params.distance})")

        # ── 3. Chunk all files ────────────────────────────────────────────
        all_chunks: list[str]  = []
        all_ids:    list[str]  = []
        all_meta:   list[dict] = []
        ingested_at = datetime.now(timezone.utc).isoformat()

        for fname in md_files:
            fpath = corpus_dir / fname
            text  = fpath.read_text(encoding="utf-8")

            # Stable IDs derived from kb_id + filename (not content) so that
            # chunk IDs survive content edits without orphaning old entries.
            doc_id    = hashlib.sha256(f"{kb_id}{fname}".encode()).hexdigest()
            file_hash = hashlib.sha256(text.encode()).hexdigest()

            # Choose chunking strategy from request params (not cfg globals).
            if params.chunk_strategy == "fixed":
                chunks = _chunk_fixed(text, params.chunk_size, params.chunk_step)
            else:
                chunks = _chunk_paragraph(text)  # default / "paragraph"

            progress(f"  {fname} → {len(chunks)} chunks ({params.chunk_strategy})")

            try:
                source_path = str(fpath.relative_to(cfg.BASE_DIR))
            except ValueError:
                source_path = str(fpath)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_id}::{i}")
                all_meta.append({
                    "kb_id":        kb_id,
                    "source_file":  fname,
                    "source_path":  source_path,
                    "doc_id":       doc_id,
                    "file_hash":    file_hash,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "ingested_at":  ingested_at,
                })

        total = len(all_chunks)
        progress(f"Total: {total} chunks — loading embedding model '{params.embed_model}'...")

        # ── 4. Embed ──────────────────────────────────────────────────────
        # Use the module-level cache so we don't reload the model between
        # requests (warm requests save ~1-3 s).
        model      = _get_embed_model(params.embed_model)
        progress(f"Embedding {total} chunks...")
        embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()

        # ── 5. Write to ChromaDB in batches ──────────────────────────────
        # ChromaDB recommends batches ≤ 5 000 to avoid memory spikes.
        BATCH = 500
        for start in range(0, total, BATCH):
            end = min(start + BATCH, total)
            collection.add(
                ids=       all_ids[start:end],
                embeddings=embeddings[start:end],
                documents= all_chunks[start:end],
                metadatas= all_meta[start:end],
            )
            progress(f"  Written {end}/{total} chunks to ChromaDB...")

        # ── 6. Stamp manifest into the registry ──────────────────────────
        # This is the "frozen build params" record.  From here on the UI
        # shows these values as what the current index was built with.
        manifest = {
            "embed_model":    params.embed_model,
            "chunk_strategy": params.chunk_strategy,
            # chunk_size / chunk_step are only meaningful for fixed strategy;
            # store null for paragraph so the UI can dim those fields.
            "chunk_size":     params.chunk_size if params.chunk_strategy == "fixed" else None,
            "chunk_step":     params.chunk_step if params.chunk_strategy == "fixed" else None,
            "distance":       params.distance,
            "top_k":          params.top_k,
            "threshold":      params.threshold,
            "built_at":       ingested_at,
            "total_chunks":   total,
            "sources":        len(md_files),
        }

        registry = _read_registry()
        registry[kb_id]["build"] = manifest
        _write_registry(registry)

        q.put({
            "type":     "done",
            "msg":      f"✓ {total} chunks written to '{collection_name}'",
            "manifest": manifest,
        })

    except Exception as exc:
        q.put({
            "type":      "error",
            "msg":       str(exc),
            "traceback": traceback.format_exc(),
        })


@app.post("/api/kbs/{kb_id}/build", summary="Build (ingest) a KB — SSE progress stream")
def build_kb(kb_id: str, body: BuildRequest):
    """
    Runs the full ingest pipeline in a background thread and streams progress
    back as Server-Sent Events.

    SSE event types
    ---------------
    {"type": "progress", "msg": "..."}          — status update (many)
    {"type": "done",     "msg": "...", "manifest": {...}}  — success
    {"type": "error",    "msg": "...", "traceback": "..."} — failure

    The browser reads these with the Fetch Streams API (see index.html).
    """
    registry = _read_registry()
    if kb_id not in registry:
        raise HTTPException(404, f"KB '{kb_id}' not found")

    kb_config = registry[kb_id]
    q = sync_queue.Queue()

    # Daemon thread: dies automatically if the server process exits.
    thread = threading.Thread(
        target=_blocking_ingest,
        args=(kb_id, kb_config, body, q),
        daemon=True,
        name=f"ingest-{kb_id}",
    )
    thread.start()

    def event_stream():
        """Poll the queue and yield SSE events until done/error or timeout."""
        while True:
            try:
                item = q.get(timeout=300)   # 5-minute max for very large corpora
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
            except sync_queue.Empty:
                # Send a keepalive comment so the connection doesn't time out.
                yield ": keepalive\n\n"
        thread.join()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx proxy buffering if deployed behind one
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ollama/models", summary="List locally available Ollama models")
def list_ollama_models():
    """
    Query the local Ollama daemon's GET /api/tags endpoint.
    Returns {"models": [...], "online": bool}.
    If Ollama is not running, returns the fallback model from cfg and online=False.
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data   = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models, "online": True}
    except Exception as exc:
        # Ollama might not be running — return a safe fallback so the UI
        # can still render (the user will see the "offline" status pill).
        return {"models": [cfg.OLLAMA_MODEL], "online": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Ask — SSE: retrieve chunks then stream LLM answer
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    kb_id:         str
    question:      str
    top_k:         int   = 5
    threshold:     float = 0.65
    model:         str   = "ministral-3:3b"
    system_prompt: Optional[str] = None   # override the default system prompt


@app.post("/api/ask", summary="Ask a question — SSE: chunks then streamed answer")
def ask(body: AskRequest):
    """
    Two-phase SSE response:

    Phase 1 — Retrieval (fast, ~100–500 ms):
        Embeds the question, queries ChromaDB, applies the threshold filter,
        then emits ONE "chunks" event so the UI can render the retrieval panel
        immediately — before the LLM has produced a single token.

    Phase 2 — Generation (slow, 1–30 s):
        Streams the Ollama response token-by-token as "token" events.
        Ends with a "done" event carrying elapsed_ms.

    SSE event types
    ---------------
    {"type": "chunks",  "chunks": [...kept...], "all_chunks": [...all...]}
    {"type": "token",   "content": "..."}
    {"type": "done",    "elapsed_ms": 1234}
    {"type": "error",   "msg": "...", "traceback": "..."}
    """
    registry = _read_registry()
    if body.kb_id not in registry:
        raise HTTPException(404, f"KB '{body.kb_id}' not found")

    kb_config      = registry[body.kb_id]
    build_manifest = kb_config.get("build", {})

    # Use the embed model from the KB's build manifest when available.
    # This ensures we embed queries with the same model that was used at ingest.
    embed_model     = build_manifest.get("embed_model", cfg.EMBED_MODEL)
    collection_name = kb_config.get("collection", body.kb_id)

    def event_stream():
        start = datetime.now(timezone.utc)

        try:
            # ── Phase 1a: embed the question ──────────────────────────────
            model  = _get_embed_model(embed_model)
            q_emb  = model.encode([body.question]).tolist()

            # ── Phase 1b: query ChromaDB ──────────────────────────────────
            client = _chroma_client()
            try:
                collection = client.get_collection(collection_name)
            except Exception:
                # Build the error payload separately — nested dicts inside
                # f-strings confuse the parser in Python < 3.12.
                err = json.dumps({"type": "error", "msg": "Collection not found — build the KB first"})
                yield f"data: {err}\n\n"
                return

            results = collection.query(
                query_embeddings=q_emb,
                n_results=body.top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Build a flat list of all retrieved chunks with threshold flag.
            all_chunks = []
            if results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    all_chunks.append({
                        "text":             doc,
                        "source":           meta.get("source_file", "unknown"),
                        "dist":             round(dist, 4),
                        "passed_threshold": dist <= body.threshold,
                    })

            kept = [c for c in all_chunks if c["passed_threshold"]]

            # ── Emit chunks event (UI renders retrieval panel immediately) ─
            yield (
                "data: "
                + json.dumps({"type": "chunks", "chunks": kept, "all_chunks": all_chunks})
                + "\n\n"
            )

            if not kept:
                yield (
                    "data: "
                    + json.dumps({
                        "type":       "done",
                        "elapsed_ms": 0,
                        "msg":        "No chunks passed the relevance threshold",
                    })
                    + "\n\n"
                )
                return

            # ── Phase 2: build prompt and stream from Ollama ──────────────
            description = kb_config.get("description", body.kb_id)
            system = body.system_prompt or (
                f"You are a helpful assistant that answers questions about {description}. "
                "Answer using ONLY the context provided below. Do not use prior knowledge. "
                "If the context does not contain enough information, say so clearly. "
                "Be concise. Cite which source document(s) your answer draws from."
            )

            # Assemble the context block from kept chunks, numbered for citation.
            context_lines = [
                f"[{i}] source={c['source']} (dist={c['dist']})\n{c['text']}"
                for i, c in enumerate(kept, 1)
            ]
            context = "\n\n".join(context_lines)

            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {body.question}"},
            ]

            payload = json.dumps({
                "model":    body.model,
                "messages": messages,
                "stream":   True,   # Ollama returns NDJSON, one object per line
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            # Read Ollama's NDJSON stream line-by-line and forward each token.
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = obj.get("message", {}).get("content", "")
                    if content:
                        yield (
                            "data: "
                            + json.dumps({"type": "token", "content": content})
                            + "\n\n"
                        )

                    if obj.get("done"):
                        elapsed = int(
                            (datetime.now(timezone.utc) - start).total_seconds() * 1000
                        )
                        yield (
                            "data: "
                            + json.dumps({"type": "done", "elapsed_ms": elapsed})
                            + "\n\n"
                        )
                        break

        except Exception as exc:
            yield (
                "data: "
                + json.dumps({
                    "type":      "error",
                    "msg":       str(exc),
                    "traceback": traceback.format_exc(),
                })
                + "\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Use reload=True in dev so file changes restart the server automatically.
    # In production: uvicorn workbench.main:app --host 0.0.0.0 --port 8000 --workers 1
    uvicorn.run("workbench.main:app", host="0.0.0.0", port=8000, reload=True)

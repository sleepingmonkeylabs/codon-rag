"""
Full RAG loop: retrieve from ChromaDB → build prompt → call Ollama → print answer.

Usage:
    python src/rag.py "What industries does Codon work in?"
    python src/rag.py          # interactive
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = cfg.OLLAMA_MODEL

# ── Embedding model cache ─────────────────────────────────────────────────────
_model_cache: dict = {}

def _get_embed_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]




# ── Registry ──────────────────────────────────────────────────────────────────

def load_kb_config(kb_id: str) -> dict:
    if not os.path.exists(cfg.KB_REGISTRY_PATH):
        print(f"Error: KB registry not found at {cfg.KB_REGISTRY_PATH}")
        sys.exit(1)
    with open(cfg.KB_REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    if kb_id not in registry:
        print(f"Error: KB '{kb_id}' not found in registry.")
        sys.exit(1)
    return registry[kb_id]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, model: SentenceTransformer, collection_name: str) -> list[dict]:
    q_emb = model.encode([question]).tolist()
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)

    try:
        collection = client.get_collection(collection_name)
    except ValueError:
        print(f"Error: Collection '{collection_name}' does not exist.")
        sys.exit(1)

    results = collection.query(
        query_embeddings=q_emb,
        n_results=cfg.TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    if not results["documents"][0]:
        return chunks

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta.get("source_file", meta.get("source", "unknown"))
        chunks.append({"text": doc, "source": source, "dist": dist, "meta": meta})

    # ── Filter noise ──────────────────────────────────────────────────────────
    chunks = [c for c in chunks if c["dist"] <= cfg.RELEVANCE_THRESHOLD]
    return chunks


def retrieve_from(
    collection_name: str,
    question: str,
    top_k: int = None,
    threshold: float = None,
) -> list[str]:
    """
    Importable retrieval function for eval runners.

    Returns a list of chunk text strings (not dicts) so RAGAS can consume
    them directly as the `contexts` field.

    Args:
        collection_name: ChromaDB collection to query (may be ephemeral).
        question:        The query string.
        top_k:           Number of chunks to retrieve (defaults to cfg.TOP_K).
        threshold:       Cosine distance cutoff (defaults to cfg.RELEVANCE_THRESHOLD).
    """
    if top_k is None:
        top_k = cfg.TOP_K
    if threshold is None:
        threshold = cfg.RELEVANCE_THRESHOLD

    model = _get_embed_model(cfg.EMBED_MODEL)
    q_emb = model.encode([question]).tolist()

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise ValueError(f"Collection '{collection_name}' not found: {e}")

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "distances"],
    )

    chunks = []
    if results["documents"][0]:
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist <= threshold:
                chunks.append(doc)
    return chunks


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] source={c['source']} (dist={c['dist']:.4f})\n{c['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[dict], system_prompt: str) -> list[dict]:
    context = build_context_block(chunks)
    user_content = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "user",   "content": user_content},
    ]


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict], max_retries: int = 3) -> str:
    """
    Call Ollama with retry + exponential backoff.

    Retries on connection errors (e.g. laptop woke from sleep and the socket
    is dead, or Ollama hasn't fully resumed yet).  Does NOT retry on HTTP 4xx
    errors (bad model name, etc.) — those are immediately fatal.
    """
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
    }).encode("utf-8")

    backoff = 2  # seconds — doubles each retry: 2, 4, 8
    last_exc = None

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["message"]["content"]

        except urllib.error.HTTPError as e:
            # HTTP-level error (4xx/5xx) — not a network issue, don't retry.
            print(f"\nError: Ollama API returned HTTP {e.code}: {e.reason}")
            print(f"This usually means the model '{OLLAMA_MODEL}' is missing, corrupted, or unsupported.")
            try:
                err_details = json.loads(e.read())
                if "error" in err_details:
                    print(f"Details: {err_details['error']}")
            except Exception:
                pass
            sys.exit(1)

        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            # Network-level error — Ollama unreachable or socket died (sleep/wake).
            last_exc = e
            wait = backoff * (2 ** (attempt - 1))
            print(f"\n⚠ Connection error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"  Retrying in {wait}s (Ollama may still be waking up)...")
                time.sleep(wait)
            else:
                print(f"\n✗ Ollama unreachable after {max_retries} attempts.")
                print("  Make sure Ollama is running: ollama serve")
                sys.exit(1)


# ── High-level importable functions for eval runners ─────────────────────────

def generate(system_prompt: str, chunks: list[str], question: str) -> str:
    """
    Build a prompt from chunk texts and call Ollama.  Returns the answer string.

    `chunks` is a list of raw text strings (as returned by retrieve_from()).
    This is the importable entry point used by eval.py and eval_grid.py —
    the same code path as the live system, no duplication.
    """
    chunk_dicts = [{"text": c, "source": "context", "dist": 0.0} for c in chunks]
    messages = build_messages(question, chunk_dicts, system_prompt)
    return call_ollama(messages)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query Codon RAG")
    parser.add_argument("--kb", type=str, default="codon", help="Knowledge base ID")
    parser.add_argument("question", nargs="*", help="The question to ask")
    args = parser.parse_args()
    
    question = " ".join(args.question)
    if not question:
        question = input("Question: ").strip()
    if not question:
        sys.exit(1)
        
    kb_id = args.kb
    kb_config = load_kb_config(kb_id)
    collection_name = kb_config.get("collection", cfg.COLLECTION)

    print(f"\nLoading embedding model...")
    model = SentenceTransformer(cfg.EMBED_MODEL)

    print(f"Retrieving top-{cfg.TOP_K} chunks from KB '{kb_id}'...")
    chunks = retrieve(question, model, collection_name)
    
    if not chunks:
        print("No relevant chunks found in the knowledge base for that question.")
        sys.exit(0)

    description = kb_config.get("description", kb_id)
    system_prompt = f"""You are a helpful assistant that answers questions about {description}.
Answer using ONLY the context provided below. Do not use prior knowledge.
If the context does not contain enough information to answer, say so clearly.
Be concise. Cite which source document(s) your answer draws from."""

    print(f"Calling {OLLAMA_MODEL}...\n")
    messages = build_messages(question, chunks, system_prompt)
    answer = call_ollama(messages)

    # ── Output ────────────────────────────────────────────────────────────────
    print("═" * 72)
    print(f"Q: {question}")
    print("═" * 72)
    print(answer)
    print("\n── Retrieved chunks ─────────────────────────────────────────────────")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{i}] dist={c['dist']:.4f}  {c['source']}")
        print(f"       {preview}")
        
        # Display new metadata fields for visibility (Step 1 requirement)
        meta = c.get("meta", {})
        debug_meta = {k: v for k, v in meta.items() if k not in ["source_file", "source"]}
        if debug_meta:
             print(f"       meta: {debug_meta}")

if __name__ == "__main__":
    main()
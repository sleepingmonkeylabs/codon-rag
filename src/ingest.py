"""
Load .md files from CORPUS_DIR, chunk, embed, write to ChromaDB.
Run this whenever you change CHUNK_STRATEGY or add new files.
"""

import os
import sys
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Allow running from repo root: python src/ingest.py
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


# ── Chunkers ──────────────────────────────────────────────────────────────────

def chunk_paragraph(text: str) -> list[str]:
    """Split on blank lines. Each paragraph = one chunk."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_fixed(text: str, size: int) -> list[str]:
    """Hard split at `size` characters. No overlap (yet)."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def get_chunks(text: str) -> list[str]:
    if cfg.CHUNK_STRATEGY == "paragraph":
        return chunk_paragraph(text)
    elif cfg.CHUNK_STRATEGY == "fixed":
        return chunk_fixed(text, cfg.FIXED_CHUNK_SIZE)
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {cfg.CHUNK_STRATEGY!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load files
    md_files = sorted(
        f for f in os.listdir(cfg.CORPUS_DIR) if f.endswith(".md")
    )
    if not md_files:
        print(f"No .md files found in {cfg.CORPUS_DIR}")
        sys.exit(1)
    print(f"Found {len(md_files)} files: {md_files}")

    # Build chunk list with metadata
    all_chunks, all_ids, all_meta = [], [], []
    for fname in md_files:
        path = os.path.join(cfg.CORPUS_DIR, fname)
        text = open(path, encoding="utf-8").read()
        chunks = get_chunks(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{fname}::{i}")
            all_meta.append({"source": fname, "chunk_index": i})

    print(f"Total chunks ({cfg.CHUNK_STRATEGY}): {len(all_chunks)}")

    # Embed
    print(f"Embedding with {cfg.EMBED_MODEL} ...")
    model = SentenceTransformer(cfg.EMBED_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # ChromaDB — persistent, cosine distance
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)

    # Drop and recreate so re-runs are idempotent
    existing = [c.name for c in client.list_collections()]
    if cfg.COLLECTION in existing:
        client.delete_collection(cfg.COLLECTION)
        print(f"Dropped existing collection '{cfg.COLLECTION}'")

    collection = client.create_collection(
        name=cfg.COLLECTION,
        metadata={"hnsw:space": cfg.DISTANCE},
    )

    # Write in one batch (fine at this scale)
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_meta,
    )

    print(f"Done. Collection '{cfg.COLLECTION}' has {collection.count()} chunks.")
    print(f"Persisted to: {cfg.CHROMA_DIR}")


if __name__ == "__main__":
    main()
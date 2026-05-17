"""
Load .md files from CORPUS_DIR, chunk, embed, write to ChromaDB.
Run this whenever you change CHUNK_STRATEGY or add new files.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# Allow running from repo root: python src/ingest.py
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Chunkers ──────────────────────────────────────────────────────────────────

def chunk_paragraph(text: str) -> list[str]:
    """Split on blank lines. Each paragraph = one chunk."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def chunk_fixed(text: str, size: int, step: int = None) -> list[str]:
    """Hard split at `size` characters. Uses `step` for overlap."""
    if step is None:
        step = size
    return [text[i : i + size] for i in range(0, len(text), step)]

def get_chunks(text: str) -> list[str]:
    if cfg.CHUNK_STRATEGY == "paragraph":
        return chunk_paragraph(text)
    elif cfg.CHUNK_STRATEGY == "fixed":
        return chunk_fixed(text, cfg.FIXED_CHUNK_SIZE, getattr(cfg, "FIXED_CHUNK_STEP", cfg.FIXED_CHUNK_SIZE))
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {cfg.CHUNK_STRATEGY!r}")

# ── Helper functions ──────────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=str, default="codon", help="Knowledge base ID")
    parser.add_argument("--delete-file", type=str, help="Filename to delete from the KB")
    parser.add_argument("--delete-kb", action="store_true", help="Delete the entire KB")
    args = parser.parse_args()

    kb_id = args.kb
    kb_config = load_kb_config(kb_id)
    collection_name = kb_config.get("collection", cfg.COLLECTION)
    corpus_dir = kb_config.get("corpus_dir", cfg.CORPUS_DIR)
    
    # Resolve relative corpus_dir to absolute if needed
    if not os.path.isabs(corpus_dir):
        corpus_dir = os.path.join(cfg.BASE_DIR, corpus_dir)

    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)

    # ── KB-level deletion ──
    if args.delete_kb:
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
            print(f"Deleted collection '{collection_name}' for KB '{kb_id}'.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
        
        # Remove from registry
        with open(cfg.KB_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        if kb_id in registry:
            del registry[kb_id]
            with open(cfg.KB_REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            print(f"Removed '{kb_id}' from registry.")
        sys.exit(0)

    # Make sure collection exists
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": cfg.DISTANCE},
        )
    else:
        collection = client.get_collection(name=collection_name)

    # ── File-level deletion ──
    if args.delete_file:
        doc_id = hashlib.sha256(f"{kb_id}{args.delete_file}".encode()).hexdigest()
        collection.delete(where={"doc_id": doc_id})
        print(f"Deleted file '{args.delete_file}' (doc_id: {doc_id}) from KB '{kb_id}'.")
        sys.exit(0)

    # ── Ingest (Incremental) ──
    if not os.path.exists(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    md_files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md") or f.endswith(".txt"))
    if not md_files:
        print(f"No .md or .txt files found in {corpus_dir}")
        sys.exit(1)
    print(f"Found {len(md_files)} files in '{corpus_dir}'.")

    # We will only embed chunks that need updating
    to_embed_chunks = []
    to_embed_ids = []
    to_embed_meta = []
    docs_to_delete = []

    for fname in md_files:
        path = os.path.join(corpus_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
            
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        doc_id = hashlib.sha256(f"{kb_id}{fname}".encode()).hexdigest()
        
        # Check if already exists and unchanged
        existing_docs = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        if existing_docs and existing_docs["metadatas"]:
            # Check file_hash
            old_hash = existing_docs["metadatas"][0].get("file_hash")
            if old_hash == file_hash:
                print(f"Skipping {fname} (unchanged)")
                continue
            else:
                print(f"Updating {fname} (changed)")
                docs_to_delete.append(doc_id)
        else:
            print(f"Adding {fname} (new)")

        chunks = get_chunks(text)
        total_chunks = len(chunks)
        ingested_at = datetime.utcnow().isoformat() + "Z"
        
        # relative path from BASE_DIR
        try:
            source_path = os.path.relpath(path, cfg.BASE_DIR)
        except ValueError:
            source_path = path

        for i, chunk in enumerate(chunks):
            to_embed_chunks.append(chunk)
            # Use doc_id and index for chunk id
            to_embed_ids.append(f"{doc_id}::{i}")
            to_embed_meta.append({
                "kb_id": kb_id,
                "source_file": fname,
                "source_path": source_path,
                "doc_id": doc_id,
                "file_hash": file_hash,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "ingested_at": ingested_at,
            })

    if docs_to_delete:
        for doc_id in docs_to_delete:
            collection.delete(where={"doc_id": doc_id})
        print(f"Deleted old chunks for {len(docs_to_delete)} updated files.")

    if not to_embed_chunks:
        print("No new or updated files to ingest.")
        sys.exit(0)

    print(f"Total chunks to ingest ({cfg.CHUNK_STRATEGY}): {len(to_embed_chunks)}")
    print(f"Embedding with {cfg.EMBED_MODEL} ...")
    model = SentenceTransformer(cfg.EMBED_MODEL)
    embeddings = model.encode(to_embed_chunks, show_progress_bar=True).tolist()

    collection.add(
        ids=to_embed_ids,
        embeddings=embeddings,
        documents=to_embed_chunks,
        metadatas=to_embed_meta,
    )

    print(f"Done. Collection '{collection_name}' has {collection.count()} chunks.")
    print(f"Persisted to: {cfg.CHROMA_DIR}")

if __name__ == "__main__":
    main()
"""
Embed a question, retrieve top-K chunks from ChromaDB, print ranked results.
Usage:
    python src/query.py "What industries does Codon work in?"
    python src/query.py   # prompts interactively
"""

import os
import sys
import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

PREVIEW_LEN = 160   # chars to show per chunk in output

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

def query(question: str, kb_id: str, collection_name: str) -> None:
    model = SentenceTransformer(cfg.EMBED_MODEL)
    q_embedding = model.encode([question]).tolist()

    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    try:
        collection = client.get_collection(collection_name)
    except ValueError:
        print(f"Error: Collection '{collection_name}' does not exist.")
        sys.exit(1)

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=cfg.TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nQuery: {question}")
    print(f"KB: {kb_id} | Strategy: {cfg.CHUNK_STRATEGY} | Top-{cfg.TOP_K} | Distance: {cfg.DISTANCE}\n")
    print("─" * 72)

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        preview = doc[:PREVIEW_LEN].replace("\n", " ")
        if len(doc) > PREVIEW_LEN:
            preview += "..."
        source = meta.get("source_file", meta.get("source", "unknown"))
        chunk_idx = meta.get("chunk_index", "?")
        print(f"[{rank}]  dist={dist:.4f}  source={source}  chunk={chunk_idx}")
        print(f"     {preview}")
        # Print additional metadata for visibility
        debug_meta = {k: v for k, v in meta.items() if k not in ["source_file", "source", "chunk_index"]}
        if debug_meta:
            print(f"     meta={debug_meta}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Query Codon ChromaDB")
    parser.add_argument("--kb", type=str, default="codon", help="Knowledge base ID")
    parser.add_argument("question", nargs="*", help="The question to ask")
    args = parser.parse_args()

    question = " ".join(args.question)
    if not question:
        question = input("Question: ").strip()

    if not question:
        print("No question provided.")
        sys.exit(1)

    kb_id = args.kb
    kb_config = load_kb_config(kb_id)
    collection_name = kb_config.get("collection", cfg.COLLECTION)

    query(question, kb_id, collection_name)


if __name__ == "__main__":
    main()
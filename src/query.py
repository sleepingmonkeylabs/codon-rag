"""
Embed a question, retrieve top-K chunks from ChromaDB, print ranked results.
Usage:
    python src/query.py "What industries does Codon work in?"
    python src/query.py   # prompts interactively
"""

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

PREVIEW_LEN = 160   # chars to show per chunk in output


def query(question: str) -> None:
    model = SentenceTransformer(cfg.EMBED_MODEL)
    q_embedding = model.encode([question]).tolist()

    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    collection = client.get_collection(cfg.COLLECTION)

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=cfg.TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nQuery: {question}")
    print(f"Strategy: {cfg.CHUNK_STRATEGY}  |  Top-{cfg.TOP_K}  |  Distance: {cfg.DISTANCE}\n")
    print("─" * 72)

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        preview = doc[:PREVIEW_LEN].replace("\n", " ")
        if len(doc) > PREVIEW_LEN:
            preview += "..."
        print(f"[{rank}]  dist={dist:.4f}  source={meta['source']}  chunk={meta['chunk_index']}")
        print(f"     {preview}")
        print()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    if not question:
        print("No question provided.")
        sys.exit(1)

    query(question)


if __name__ == "__main__":
    main()
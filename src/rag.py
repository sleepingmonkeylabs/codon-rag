"""
Full RAG loop: retrieve from ChromaDB → build prompt → call Ollama → print answer.

Usage:
    python src/rag.py "What industries does Codon work in?"
    python src/rag.py          # interactive
"""

import os
import sys
import json
import urllib.request

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
#OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "ministral-3:3b"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Codon Consulting.
Answer using ONLY the context provided below. Do not use prior knowledge.
If the context does not contain enough information to answer, say so clearly.
Be concise. Cite which source document(s) your answer draws from."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, model: SentenceTransformer) -> list[dict]:
    q_emb = model.encode([question]).tolist()
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    collection = client.get_collection(cfg.COLLECTION)
    results = collection.query(
        query_embeddings=q_emb,
        n_results=cfg.TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "source": meta["source"], "dist": dist})

    # ── Filter noise ──────────────────────────────────────────────────────────
    chunks = [c for c in chunks if c["dist"] <= cfg.RELEVANCE_THRESHOLD]
    return chunks    


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] source={c['source']} (dist={c['dist']:.4f})\n{c['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context = build_context_block(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict]) -> str:
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()
    if not question:
        sys.exit(1)

    print(f"\nLoading embedding model...")
    model = SentenceTransformer(cfg.EMBED_MODEL)

    print(f"Retrieving top-{cfg.TOP_K} chunks...")
    chunks = retrieve(question, model)
    
    if not chunks:
     print("No relevant chunks found in the knowledge base for that question.")
     sys.exit(0)

    print(f"Calling {OLLAMA_MODEL}...\n")
    messages = build_messages(question, chunks)
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


if __name__ == "__main__":
    main()
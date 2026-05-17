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
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = cfg.OLLAMA_MODEL




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


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] source={c['source']} (dist={c['dist']:.4f})\n{c['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[dict], system_prompt: str) -> list[dict]:
    context = build_context_block(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
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
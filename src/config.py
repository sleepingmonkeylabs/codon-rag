"""
All tunable knobs in one place. Change here, not in the scripts.
"""

import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.parent
CORPUS_DIR = str(BASE_DIR / "data" / "corpus")
CHROMA_DIR = str(BASE_DIR / "chroma_db")
KB_REGISTRY_PATH = str(BASE_DIR / "data" / "kb_registry.json")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION = "codon_docs"
DISTANCE   = "cosine"          # "cosine" | "l2" | "ip"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, local, no API key needed

# ── Ollama model ──────────────────────────────────────────────────────────────
OLLAMA_MODEL = "ministral-3:3b"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_STRATEGY   = "paragraph"   # "paragraph" | "fixed"
FIXED_CHUNK_SIZE = 200           # chars — only used when CHUNK_STRATEGY == "fixed"
FIXED_CHUNK_STEP = 150           # chars — overlap step for fixed chunking

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
RELEVANCE_THRESHOLD = 0.65

# ── Prompt versioning ─────────────────────────────────────────────────────────
import hashlib

SYSTEM_PROMPT = """You are a helpful assistant for Codon Consulting.
Answer questions using only the provided context chunks.
If the context does not contain the answer, say clearly: "I don't have information about that."
Always cite the source document names at the end of your answer."""

PROMPT_VERSION = "v1.0"
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]
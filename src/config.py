"""
All tunable knobs in one place. Change here, not in the scripts.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
CORPUS_DIR = r"C:\Users\DmitriApassov\Documents\codon\codon-rag\data\corpus"
CHROMA_DIR = r"C:\Users\DmitriApassov\Documents\codon\codon-rag\chroma_db"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION = "codon_docs"
DISTANCE   = "cosine"          # "cosine" | "l2" | "ip"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, local, no API key needed

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_STRATEGY   = "paragraph"   # "paragraph" | "fixed"
FIXED_CHUNK_SIZE = 200           # chars — only used when CHUNK_STRATEGY == "fixed"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
RELEVANCE_THRESHOLD = 0.45
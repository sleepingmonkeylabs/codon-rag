import streamlit as st
import json
import os
import subprocess
import chromadb
import sys
from datetime import datetime

sys.path.insert(0, "src")
import config as cfg

st.set_page_config(page_title="Knowledge Bases", page_icon="📚")
st.title("Knowledge Bases")

# Helper to load registry
def load_registry():
    if not os.path.exists(cfg.KB_REGISTRY_PATH):
        return {}
    with open(cfg.KB_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Helper to save registry
def save_registry(registry):
    with open(cfg.KB_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

registry = load_registry()

# ── KB Overview Table ────────────────────────────────────────────────────────
st.subheader("Overview")
if not registry:
    st.info("No knowledge bases found. Create one below.")
else:
    client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
    kb_data = []
    for kb_id, config in registry.items():
        collection_name = config.get("collection", cfg.COLLECTION)
        
        chunk_count = 0
        last_ingested = "Never"
        try:
            col = client.get_collection(collection_name)
            chunk_count = col.count()
            
            # Try to get last_ingested from first document metadata
            result = col.get(limit=1, include=["metadatas"])
            if result and result["metadatas"]:
                last_ingested = result["metadatas"][0].get("ingested_at", "Unknown")
        except Exception:
            pass
            
        kb_data.append({
            "ID": kb_id,
            "Description": config.get("description", ""),
            "Chunks": chunk_count,
            "Last Ingested": last_ingested
        })
    
    st.table(kb_data)

st.divider()

# ── Active KB Selector ───────────────────────────────────────────────────────
if registry:
    active_kb = st.selectbox("Select Active KB for Management", list(registry.keys()))
    active_config = registry[active_kb]
    corpus_dir = active_config.get("corpus_dir", cfg.CORPUS_DIR)
    if not os.path.isabs(corpus_dir):
        corpus_dir = os.path.join(cfg.BASE_DIR, corpus_dir)
else:
    active_kb = None
    corpus_dir = None

# ── Create KB ────────────────────────────────────────────────────────────────
with st.expander("Create New Knowledge Base"):
    with st.form("create_kb_form"):
        new_kb_id = st.text_input("KB ID (e.g. client_x)")
        new_kb_desc = st.text_input("Description")
        new_kb_corpus = st.text_input("Corpus Directory (relative to repo root)", value="data/corpus/client_x")
        
        if st.form_submit_button("Create KB"):
            if new_kb_id in registry:
                st.error("KB ID already exists!")
            elif new_kb_id:
                registry[new_kb_id] = {
                    "kb_id": new_kb_id,
                    "collection": f"{new_kb_id}_docs",
                    "corpus_dir": new_kb_corpus,
                    "description": new_kb_desc
                }
                save_registry(registry)
                
                # Create directory if it doesn't exist
                abs_dir = os.path.join(cfg.BASE_DIR, new_kb_corpus)
                os.makedirs(abs_dir, exist_ok=True)
                
                st.success(f"Created KB '{new_kb_id}'. Running initial ingest...")
                # trigger ingest
                subprocess.run([sys.executable, "src/ingest.py", "--kb", new_kb_id])
                st.rerun()

if active_kb:
    # ── Add Documents ────────────────────────────────────────────────────────────
    with st.expander("Add Documents"):
        uploaded_files = st.file_uploader("Upload Markdown (.md) or Text (.txt) files", type=["md", "txt"], accept_multiple_files=True)
        if st.button("Upload and Ingest") and uploaded_files:
            os.makedirs(corpus_dir, exist_ok=True)
            for uf in uploaded_files:
                file_path = os.path.join(corpus_dir, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
            st.info("Ingesting new files...")
            res = subprocess.run([sys.executable, "src/ingest.py", "--kb", active_kb], capture_output=True, text=True)
            st.success("Ingestion complete.")
            with st.expander("Ingest Log"):
                st.text(res.stdout)
            st.rerun()

    # ── Delete Document ──────────────────────────────────────────────────────────
    with st.expander("Delete Document"):
        if os.path.exists(corpus_dir):
            files = sorted(f for f in os.listdir(corpus_dir) if f.endswith((".md", ".txt")))
        else:
            files = []
            
        if not files:
            st.info("No documents in corpus.")
        else:
            file_to_del = st.selectbox("Select file to delete", files)
            if st.button("Delete File"):
                res = subprocess.run([sys.executable, "src/ingest.py", "--kb", active_kb, "--delete-file", file_to_del], capture_output=True, text=True)
                # also remove from disk
                file_path = os.path.join(corpus_dir, file_to_del)
                if os.path.exists(file_path):
                    os.remove(file_path)
                st.success(f"Deleted {file_to_del}.")
                st.rerun()

    # ── Delete KB ────────────────────────────────────────────────────────────────
    with st.expander("Delete KB", expanded=False):
        st.warning(f"Are you sure you want to delete the '{active_kb}' knowledge base? This will drop the collection and remove it from the registry.")
        if st.button("Confirm Delete KB"):
            subprocess.run([sys.executable, "src/ingest.py", "--delete-kb", "--kb", active_kb])
            st.success(f"Deleted KB '{active_kb}'.")
            st.rerun()

    # ── Re-ingest KB ─────────────────────────────────────────────────────────────
    with st.expander("Re-ingest KB"):
        st.warning("This will drop the existing collection and perform a full re-ingest of the corpus directory.")
        if st.button("Confirm Re-ingest"):
            # First delete the KB collection
            subprocess.run([sys.executable, "src/ingest.py", "--delete-kb", "--kb", active_kb])
            
            # Add it back to registry (since delete-kb removes it from registry)
            registry[active_kb] = active_config
            save_registry(registry)
            
            st.info("Running full re-ingest...")
            res = subprocess.run([sys.executable, "src/ingest.py", "--kb", active_kb], capture_output=True, text=True)
            st.success("Re-ingestion complete.")
            with st.expander("Ingest Log"):
                st.text(res.stdout)
            st.rerun()

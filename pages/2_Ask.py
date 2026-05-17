import streamlit as st
import os
import json
import sys
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

sys.path.insert(0, "src")
import config as cfg
import importlib
importlib.reload(cfg)

st.set_page_config(page_title="Ask Anything against local knowledge base", page_icon="🤖")
# ── Load registry ────────────────────────────────────────────────────────────

def load_registry():
    if not os.path.exists(cfg.KB_REGISTRY_PATH):
        return {}
    with open(cfg.KB_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

registry = load_registry()

if not registry:
    st.error("No knowledge bases found. Please go to the 'Knowledge Bases' page to create one.")
    st.stop()

# ── KB selector ──────────────────────────────────────────────────────────────

kb_id = st.sidebar.selectbox("Select Knowledge Base", list(registry.keys()))
active_config = registry[kb_id]
collection_name = active_config.get("collection", cfg.COLLECTION)
description = active_config.get("description", "the selected topic")

st.title("Ask Anything against local knowledge base")
st.caption(f"Ask anything about {description} — powered by local RAG")

# Clear chat button
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# Automatically clear chat when switching KB
if "current_kb" not in st.session_state:
    st.session_state.current_kb = kb_id

if st.session_state.current_kb != kb_id:
    st.session_state.messages = []
    st.session_state.current_kb = kb_id
    st.rerun()

# ── Build chain (cached per KB) ──────────────────────────────────────────────

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)

def load_chain(kb_id: str, collection_name: str, description: str = ""):
    import chromadb.api.client
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    
    embeddings = load_embedding_model()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(cfg.CHROMA_DIR),
    )
    
    def retrieve_and_filter(question: str):
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(question, k=cfg.TOP_K)
        except Exception:
            return []
            
        unique_docs = []
        seen_distances = set()
        for doc, score in docs_with_scores:
            rounded_score = round(score, 4)
            if rounded_score not in seen_distances and score <= cfg.RELEVANCE_THRESHOLD:
                seen_distances.add(rounded_score)
                unique_docs.append(doc)
        return unique_docs

    retriever = RunnableLambda(retrieve_and_filter)
    llm = ChatOllama(model=cfg.OLLAMA_MODEL, base_url="http://127.0.0.1:11434")
    
    intro = f"You are a helpful assistant that answers questions about {description}." if description else "You are a helpful assistant."
    prompt = ChatPromptTemplate.from_template(f"""
{intro}
Answer using ONLY the context below. If the context is insufficient, say so.
Cite which source document(s) your answer draws from.

Context:
{{context}}

Question: {{question}}
""")

    def format_docs(docs):
        return "\n\n".join(
            f"[source: {d.metadata.get('source_file', d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, vectorstore

description = active_config.get("description", "")
chain, vectorstore = load_chain(kb_id, collection_name, description)

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────

if question := st.chat_input(f"Ask about {kb_id}..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # First, do the retrieval manually to check if we have any valid chunks
            try:
                raw_docs = vectorstore.similarity_search_with_score(question, k=cfg.TOP_K)
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Chroma Search Error: {e}")
                raw_docs = []
            
            # Deduplicate by identical cosine distances (rounded to 4 decimal places)
            docs = []
            seen_distances = set()
            for doc, score in raw_docs:
                rounded_score = round(score, 4)
                if rounded_score not in seen_distances:
                    seen_distances.add(rounded_score)
                    docs.append((doc, score))

            valid_docs = [d for d in docs if d[1] <= cfg.RELEVANCE_THRESHOLD]
            invalid_count = len(docs) - len(valid_docs)

            # Short-circuit if no relevant context was found
            if not valid_docs:
                answer = "I couldn't find any relevant information in the knowledge base to answer your question."
            else:
                answer = chain.invoke(question)

        st.markdown(answer)

        with st.expander("Retrieved chunks"):
            for doc, score in valid_docs:
                source = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
                st.markdown(f"**{source}** — dist `{score:.4f}`")
                st.caption(doc.page_content[:300])
                
            if invalid_count > 0:
                st.caption(f"*(...and {invalid_count} other chunks were above the `{cfg.RELEVANCE_THRESHOLD}` distance threshold)*")

    st.session_state.messages.append({"role": "assistant", "content": answer})

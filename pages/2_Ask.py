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

st.set_page_config(page_title="Ask Codon", page_icon="🤖")
st.title("Ask Codon")
st.caption("Ask anything about Codon Consulting — powered by local RAG")

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

# Clear chat button
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# ── Build chain (cached per KB) ──────────────────────────────────────────────

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)

@st.cache_resource
def load_chain(kb_id: str, collection_name: str):
    embeddings = load_embedding_model()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.CHROMA_DIR,
    )
    
    def retrieve_and_filter(question: str):
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=cfg.TOP_K)
        return [doc for doc, score in docs_with_scores if score <= cfg.RELEVANCE_THRESHOLD]

    retriever = RunnableLambda(retrieve_and_filter)
    llm = ChatOllama(model=cfg.OLLAMA_MODEL, base_url="http://127.0.0.1:11434")
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about Codon Consulting.
Answer using ONLY the context below. If the context is insufficient, say so.
Cite which source document(s) your answer draws from.

Context:
{context}

Question: {question}
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

chain, vectorstore = load_chain(kb_id, collection_name)

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
            answer = chain.invoke(question)

            # Show retrieved chunks in expander
            docs = vectorstore.similarity_search_with_score(question, k=cfg.TOP_K)

        st.markdown(answer)

        with st.expander("Retrieved chunks"):
            for doc, score in docs:
                source = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
                st.markdown(f"**{source}** — dist `{score:.4f}`")
                st.caption(doc.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import sys
sys.path.insert(0, "src")
import config as cfg

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Codon Sales Assistant", page_icon="🤖")
st.title("Codon Sales Assistant")
st.caption("Ask anything about Codon Consulting — powered by local RAG")

# ── Build chain (cached so it loads once) ────────────────────────────────────

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=cfg.COLLECTION,
        embedding_function=embeddings,
        persist_directory=cfg.CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.TOP_K})
    llm = ChatOllama(model="ministral-3:3b", base_url="http://127.0.0.1:11434")
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
            f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="ministral-3:3b", base_url="http://127.0.0.1:11434")
        | StrOutputParser()
    )
    return chain, vectorstore

chain, vectorstore = load_chain()

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask about Codon..."):
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
                st.markdown(f"**{doc.metadata.get('source')}** — dist `{score:.4f}`")
                st.caption(doc.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})
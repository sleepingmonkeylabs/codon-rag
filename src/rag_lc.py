from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ── Components ────────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)

vectorstore = Chroma(
    collection_name=cfg.COLLECTION,
    embedding_function=embeddings,
    persist_directory=cfg.CHROMA_DIR,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": cfg.TOP_K},
)

llm = ChatOllama(model="ministral-3:3b", base_url="http://127.0.0.1:11434")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about Codon Consulting.
Answer using ONLY the context below. If the context is insufficient, say so.
Be concise. Cite which source document(s) your answer draws from.

Context:
{context}

Question: {question}
""")

parser = StrOutputParser()

# ── Chain ─────────────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    print(chain.invoke(question))
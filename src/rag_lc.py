import argparse
import sys, os
import json
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

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

def main():
    parser = argparse.ArgumentParser(description="Query Codon RAG")
    parser.add_argument("--kb", type=str, default="codon", help="Knowledge base ID")
    parser.add_argument("question", nargs="*", help="The question to ask")
    args = parser.parse_args()
    
    question = " ".join(args.question) or input("Question: ")
    if not question:
        sys.exit(1)
        
    kb_id = args.kb
    kb_config = load_kb_config(kb_id)
    collection_name = kb_config.get("collection", cfg.COLLECTION)

    # ── Components ────────────────────────────────────────────────────────────────
    
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
    
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
    You are a helpful assistant.
    Answer using ONLY the context below. If the context is insufficient, say so.
    Be concise. Cite which source document(s) your answer draws from.
    
    Context:
    {context}
    
    Question: {question}
    """)
    
    parser_obj = StrOutputParser()
    
    # ── Chain ─────────────────────────────────────────────────────────────────────
    
    def format_docs(docs):
        return "\n\n".join(
            f"[source: {d.metadata.get('source_file', d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
            for d in docs
        )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser_obj
    )
    
    print(chain.invoke(question))

if __name__ == "__main__":
    main()
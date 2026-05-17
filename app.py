import streamlit as st

st.set_page_config(
    page_title="Codon Sales Assistant",
    page_icon="👋",
)

st.write("# Welcome to Codon Sales Assistant! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is the Codon Sales Assistant, powered by local RAG.
    
    **👈 Select a page from the sidebar** to navigate:
    
    - **Knowledge Bases**: Manage what the assistant knows. Create new KBs, ingest documents, and delete old files.
    - **Ask**: Chat with the assistant using your active knowledge bases.
"""
)
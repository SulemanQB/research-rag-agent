from __future__ import annotations

import streamlit as st

from src.pipeline import answer_question

st.set_page_config(page_title="Research RAG Agent", layout="wide")

st.title("Research RAG Agent")
st.markdown("Search the indexed local research corpus and inspect the selected route and source citations.")

question = st.text_input("Ask a question")

if st.button("Generate answer") and question:
    result = answer_question(question, top_k=3)
    st.subheader("Selected route")
    st.write(f"Route: {result['route']}")
    st.write(f"Reason: {result['reason']}")

    if result.get("sources"):
        st.subheader("Sources")
        for item in result["sources"]:
            st.write(f"- {item}")

    st.subheader("Answer")
    st.write(result["answer"])

import streamlit as st
import requests

import os
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Document Q&A Agent", page_icon="📄")
st.title("📄 Document Q&A Agent")
st.caption("Upload a PDF, then ask questions about it — grounded in your document, not general knowledge.")

# --- Upload section ---
st.subheader("1. Upload a document")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Ingest document"):
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Ingested '{data['filename']}' — {data['chunks_stored']} chunks stored.")
        else:
            st.error(f"Upload failed: {response.text}")

st.divider()

# --- Q&A section ---
st.subheader("2. Ask a question")
query = st.text_input("Your question")

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/ask", json={"query": query})
    if response.status_code == 200:
        data = response.json()
        st.markdown("**Answer:**")
        st.write(data["answer"])
        st.caption(f"Based on {data['sources_used']} retrieved passages.")
    else:
        st.error(f"Request failed: {response.text}")
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_split(pdf_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} pages, split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks, persist_directory: str = CHROMA_DIR):
    """Embed chunks and store them in a persistent Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{persist_directory}'.")
    return vectordb


if __name__ == "__main__":
    test_pdf = "data/sample.pdf"
    if not os.path.exists(test_pdf):
        print(f"Put a PDF at {test_pdf} to test this script.")
    else:
        doc_chunks = load_and_split(test_pdf)
        build_vectorstore(doc_chunks)
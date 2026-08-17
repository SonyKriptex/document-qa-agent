import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingest import load_and_split, build_vectorstore
from app.rag import load_vectorstore, retrieve, generate_answer

app = FastAPI(title="Document Q&A Agent")

UPLOAD_DIR = "data"
CHROMA_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class Question(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_split(save_path)
    build_vectorstore(chunks, persist_directory=CHROMA_DIR)

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "message": "Document ingested successfully.",
    }


@app.post("/ask")
def ask_question(payload: Question):
    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(status_code=400, detail="No document has been ingested yet.")

    vectordb = load_vectorstore(CHROMA_DIR)
    chunks = retrieve(payload.query, vectordb)
    answer = generate_answer(payload.query, chunks)

    return {
        "query": payload.query,
        "answer": answer,
        "sources_used": len(chunks),
    }
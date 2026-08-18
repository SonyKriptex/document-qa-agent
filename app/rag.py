import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def load_vectorstore(persist_directory: str = CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def retrieve(query: str, vectordb, k: int = 4):
    """Return the top-k most relevant chunks for a query."""
    results = vectordb.similarity_search(query, k=k)
    return results


def generate_answer(query: str, chunks):
    """Send retrieved chunks + query to the LLM and get a grounded answer."""
    context = "\n\n---\n\n".join(c.page_content for c in chunks)

    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
If the answer isn't in the context, say you don't know — do not make things up.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    vectordb = load_vectorstore()
    print("Document Q&A ready. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ")
        if question.strip().lower() in ("exit", "quit"):
            break
        top_chunks = retrieve(question, vectordb)
        answer = generate_answer(question, top_chunks)
        print("\n--- Answer ---")
        print(answer)
        print()
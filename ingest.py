"""
INGEST SCRIPT
--------------
Run this ONCE (or whenever your document changes) to:
1. Read your document
2. Split it into small chunks
3. Convert each chunk into an embedding (Module 3 concept!)
4. Store everything in a local ChromaDB vector database (Module 10 concept!)

Usage:
    python ingest.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENT_PATH = "sample_docs/project_readme.md"
CHUNK_SIZE = 500        # characters per chunk (roughly ~100 words)
CHUNK_OVERLAP = 50      # overlap between chunks so we don't cut sentences awkwardly

def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    print("Step 1: Loading document...")
    text = load_document(DOCUMENT_PATH)

    print("Step 2: Splitting into chunks...")
    chunks = chunk_text(text)
    print(f"   -> Created {len(chunks)} chunks")

    print("Step 3: Loading embedding model (downloads once, then cached locally)...")
    # This is a small, free, local model — no API key needed for embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Step 4: Generating embeddings for each chunk...")
    embeddings = embedder.encode(chunks).tolist()

    print("Step 5: Storing in ChromaDB (local vector database)...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="project_docs")

    # Clear old data if re-running, so we don't get duplicates
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    print(f"\nDone! {len(chunks)} chunks embedded and stored in ./chroma_db")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()

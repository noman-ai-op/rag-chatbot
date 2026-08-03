"""
RAG CHATBOT — Streamlit App
-----------------------------
Chat with your own document! This app:
1. Takes your question
2. Embeds it and searches ChromaDB for the most relevant chunks (Module 10)
3. Sends those chunks + your question to Groq's LLM (Module 8/9)
4. Displays a grounded answer, based on YOUR document

Usage:
    streamlit run app.py
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------- CONFIG ----------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]   # reads the key from Streamlit's secure "Secrets" storage, never written in this file
MODEL_NAME = "llama-3.3-70b-versatile"
TOP_K = 3   # how many relevant chunks to retrieve per question

# Distance threshold: ChromaDB returns a "distance" for each result — LOWER means MORE similar.
# Calibrated using real test data: relevant questions scored ~1.34, irrelevant ones ~2.0.
# 1.6 sits safely in between.
DISTANCE_THRESHOLD = 1.6

# ---------- SETUP (cached so it only loads once, not on every question) ----------
DOCUMENT_PATH = "sample_docs/project_readme.md"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker (same logic as ingest.py)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Self-building: if this is a fresh environment (e.g. a new deployment
    # where chroma_db doesn't exist yet), build it automatically instead
    # of requiring ingest.py to have been run manually beforehand.
    try:
        collection = chroma_client.get_collection(name="project_docs")
    except Exception:
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()

        collection = chroma_client.get_or_create_collection(name="project_docs")
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )

    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, collection, groq_client

embedder, collection, groq_client = load_resources()

# ---------- RAG PIPELINE ----------
def retrieve_relevant_chunks(question, top_k=TOP_K):
    """Embed the question, find the most similar chunks, and filter out
    anything that isn't actually close enough to be relevant (Module 3 + 10)."""
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = results["documents"][0]
    distances = results["distances"][0]

    # Pair each chunk with its distance, then keep only ones under the threshold
    relevant = [
        (chunk, dist) for chunk, dist in zip(chunks, distances)
        if dist <= DISTANCE_THRESHOLD
    ]
    return relevant  # list of (chunk_text, distance) tuples, possibly empty

def generate_answer(question, context_chunks):
    """Build the augmented prompt and call the LLM (Module 8 + 9)."""
    context = "\n\n".join(chunk for chunk, dist in context_chunks)

    prompt = f"""Answer the user's question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information in the document."
Be concise and clear.

<context>
{context}
</context>

<question>
{question}
</question>"""

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.3,   # low temperature = focused, factual answers (Module 7)
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Chat With My Project")
st.caption("Ask questions about the Customer Churn Prediction project — answers are grounded in the actual project docs, not guesswork.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_question = st.chat_input("Ask something about the project...")

if user_question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            relevant_chunks = retrieve_relevant_chunks(user_question)

            if not relevant_chunks:
                # Nothing close enough was found — skip calling the LLM entirely.
                # This is stricter and more reliable than hoping the LLM notices.
                answer = "I don't have that information in the document — nothing relevant was found."
                st.markdown(answer)
                st.caption("No chunk passed the similarity threshold, so the LLM wasn't even called for this one.")
            else:
                answer = generate_answer(user_question, relevant_chunks)
                st.markdown(answer)

                with st.expander("See retrieved context (what the model actually read)"):
                    for i, (chunk, dist) in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {i+1}** (distance: {dist:.3f}): {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

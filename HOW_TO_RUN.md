# How to Run This Project (On Your Own Laptop)

I built and structured this project here, but couldn't fully test the embedding
step in this environment because it doesn't have internet access to huggingface.co
(where the free local embedding model lives). Your laptop has normal internet access,
so this will work fine there. Follow these steps exactly.

## Step 1: Open the project folder
Extract/move the `rag_chatbot` folder to your usual projects location, e.g.:
```
C:\Users\SST\Documents\New folder\gitdemo\rag_chatbot
```

## Step 2: Open Anaconda Prompt (or terminal) and navigate there
```
cd "C:\Users\SST\Documents\New folder\gitdemo\rag_chatbot"
```

## Step 3: Install the required packages
```
pip install -r requirements.txt
```
This may take a few minutes the first time (sentence-transformers pulls in some
larger dependencies like PyTorch).

## Step 4: Run the ingestion script (only needs to run once)
```
python ingest.py
```
This will:
- Read `sample_docs/project_readme.md`
- Split it into chunks
- Download the free embedding model (~90MB, one-time download)
- Store everything in a local `chroma_db` folder

You should see output ending with:
```
Done! 6 chunks embedded and stored in ./chroma_db
```

## Step 5: Run the chatbot app
```
streamlit run app.py
```
This will open a browser tab with your chatbot. Try asking things like:
- "What model did you use and what accuracy did it get?"
- "What bug did you fix during deployment?"
- "What's this project about?"

You'll see the AI answer using ONLY the actual document content — click
"See retrieved context" under any answer to see exactly which chunks it used.

## Step 6 (Optional): Swap in your OWN document
Replace `sample_docs/project_readme.md` with any `.md` or `.txt` file of your own
(your actual project docs, a client's FAQ, anything), then re-run:
```
python ingest.py
streamlit run app.py
```

## Important Security Note
Your Groq API key is currently hardcoded in `app.py` for convenience. Before
uploading this project to GitHub:
1. Delete the current key at console.groq.com/keys (since it was shared in our chat)
2. Create a fresh key
3. Move it into a `.env` file instead of hardcoding it (ask me and I'll show you how)
4. Add `.env` to your `.gitignore` so it never gets pushed to GitHub

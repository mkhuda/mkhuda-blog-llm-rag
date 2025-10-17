from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

# Path ke folder index-mu
INDEX_DIR = "mkhuda_faiss_index"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di file .env")

# --- 1) Inspect pure FAISS index (binary) ---
idx_path = os.path.join(INDEX_DIR, "index.faiss")
idx = faiss.read_index(idx_path)
print(f"[FAISS] ntotal vectors: {idx.ntotal}, dimension: {idx.d}")

# reconstruct beberapa vector pertama
for i in range(min(20, idx.ntotal)):
    v = idx.reconstruct(i)           # butuh IndexFlat atau reconstruct-able index
    print(f" vector #{i} first 5 dims:", v[:5])

# --- 2) Load LangChain FAISS store utk metadata & text ---
# Agar FAISS.load_local bisa jalan, perlu dummy embeddings
dummy_emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
vs = FAISS.load_local(INDEX_DIR, dummy_emb, allow_dangerous_deserialization=True)

# docstore di vs.docstore._dict: key=internal_id, value=Document
docs_dict = vs.docstore._dict
print(f"\n[LangChain FAISS] docs loaded: {len(docs_dict)}")

# print 3 sample entries
for idx_id, doc in list(docs_dict.items())[:20]:
    print(f"\n--- ID {idx_id} ---")
    print("page_content:", doc.page_content[:200].replace("\n"," "), "…")
    print("metadata   :", doc.metadata)

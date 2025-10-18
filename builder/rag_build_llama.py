"""
RAG Index Builder untuk mkhuda.com (versi LlamaIndex)
------------------------------------------------------
Ambil data dari database WordPress, bersihkan teks,
buat embedding menggunakan OpenAI, dan simpan ke FAISS vectorstore.
"""

from dotenv import load_dotenv
load_dotenv()

import os, re, json
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# --- 1️⃣ Load ENV ---
api_key = os.getenv("OPENAI_API_KEY")
mysql_database = os.getenv("MYSQL_DATABASE")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_port = os.getenv("MYSQL_PORT", "3306")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di file .env")

index_path = Path(__file__).resolve().parent.parent / "mkhuda_faiss_index"

# --- 2️⃣ Setup embedding dan FAISS vectorstore ---
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

# Jika index lama ada → load; jika tidak → buat baru
index_file = index_path / "index.faiss"
if index_file.exists():
    print("📂 Memuat FAISS index lama...")
    faiss_index = faiss.read_index(str(index_file))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print("🆕 Tidak ada index lama, membuat index baru...")
    # OpenAI embedding dimension: 1536 for "text-embedding-3-small"
    dim = 1536  
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)

# --- 3️⃣ Ambil data WordPress ---
print("🔌 Menghubungkan ke database WordPress...")
conn = mysql.connector.connect(
    host=mysql_host,
    port=mysql_port,
    user=mysql_user,
    password=mysql_password,
    database=mysql_database
)

query = """
SELECT ID, post_title, post_content, post_date
FROM wp_posts
WHERE post_status = 'publish' AND post_type = 'post'
ORDER BY post_date DESC;
"""
df = pd.read_sql(query, conn)
conn.close()
print(f"✅ {len(df)} post berhasil dimuat dari database.\n")

# --- 4️⃣ Bersihkan konten ---
def clean_html(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

df["clean_content"] = df["post_content"].apply(clean_html)
df["url"] = "https://mkhuda.com/?p=" + df["ID"].astype(str)

# --- 5️⃣ Convert ke dokumen baru ---
new_docs = [
    Document(
        text=row.clean_content,
        metadata={"title": row.post_title, "url": row.url, "date": str(row.post_date)}
    )
    for _, row in df.iterrows()
    if row.clean_content.strip()
]

with open("docs.json", "w", encoding="utf-8") as f:
    json.dump([d.to_dict() for d in new_docs], f, ensure_ascii=False, indent=2)

print(f"🧩 Menemukan {len(new_docs)} artikel untuk ditambahkan ke index.")

# --- 6️⃣ Tambahkan ke index dengan progress bar ---
for doc in tqdm(new_docs, desc="Embedding dokumen..."):
    index.insert(doc)

# --- 7️⃣ Simpan FAISS index ---
index_path.mkdir(parents=True, exist_ok=True)
faiss.write_index(vector_store.faiss_index, str(index_file))
print(f"💾 Index disimpan di: {index_path}")
print("🎯 Selesai — semua artikel telah di-embed.")

"""
RAG Index Builder untuk mkhuda.com
----------------------------------
Ambil data dari database WordPress, bersihkan teks,
buat embedding menggunakan OpenAI, dan simpan ke FAISS vectorstore.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup
from tqdm import tqdm
import faiss

# --- Compatibility patch untuk FAISS >= 1.11 ---
if not hasattr(faiss, "IndexFlatL2"):
    try:
        # Untuk FAISS 1.11–1.12
        import faiss.class_wrappers as class_wrappers
        faiss.IndexFlatL2 = class_wrappers.IndexFlatL2
        faiss.IndexIDMap = class_wrappers.IndexIDMap
    except (ImportError, AttributeError):
        # Fallback untuk build tertentu
        import faiss.swigfaiss as swigfaiss
        faiss.IndexFlatL2 = swigfaiss.IndexFlatL2
        faiss.IndexIDMap = swigfaiss.IndexIDMap

# LangChain modern imports (2025)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1️⃣ Ambil variabel environment ---
api_key = os.getenv("OPENAI_API_KEY")
mysql_database = os.getenv("MYSQL_DATABASE")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_port = os.getenv("MYSQL_PORT")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di file .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
index_path = "mkhuda_faiss_index"

# --- 2️⃣ Load vectorstore lama (jika ada) ---
if os.path.exists(index_path):
    print("📂 Memuat vectorstore lama...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    indexed_urls = {m.metadata.get("url") for m in vectorstore.docstore._dict.values() if m.metadata.get("url")}
    print(f"✅ Index lama dimuat ({len(indexed_urls)} dokumen sudah di-index).")
else:
    print("🆕 Tidak ada index lama, membuat index baru...")
    # Buat index FAISS manual dengan dimensi sesuai model embedding
    dim = 1536  # untuk model text-embedding-3-small
    index = faiss.IndexFlatL2(dim)
    docstore = InMemoryDocstore({})  # ✅ docstore yang bisa di-append
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={}
    )
    indexed_urls = set()

# --- 2️⃣ Koneksi ke database WordPress ---
print("🔌 Menghubungkan ke database WordPress...")

conn = mysql.connector.connect(
    host     = mysql_host,
    port     = mysql_port,
    user     = mysql_user,
    password = mysql_password,
    database = mysql_database
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

# # --- 3️⃣ Bersihkan konten HTML & shortcode ---
def clean_html(text):
    text = re.sub(r'\[.*?\]', '', text)  # hapus shortcode [gallery], dll
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.strip()

df["clean_content"] = df["post_content"].apply(clean_html)
df["url"] = "https://mkhuda.com/?p=" + df["ID"].astype(str)

# # --- 4️⃣ Split menjadi potongan teks ---
# --- 4️⃣ Filter artikel baru ---
new_docs = []
for _, row in df.iterrows():
    if row.url not in indexed_urls:
        new_docs.append({
            "page_content": row.clean_content,
            "metadata": {
                "title": row.post_title,
                "url": row.url,
                "date": str(row.post_date)
            }
        })

# debug: simpan semua docs ke file docs.json
with open("docs.json", "w", encoding="utf-8") as f:
    json.dump(new_docs, f, ensure_ascii=False, indent=2)

print("📝 File docs.json berhasil dibuat—cek di direktori kerja kamu.")

if not new_docs:
    print("🎉 Tidak ada artikel baru untuk di-index.")
    exit()

print(f"🧩 Menemukan {len(new_docs)} artikel baru untuk ditambahkan ke index.")

# --- 5️⃣ Embedding dengan progress halus ---
total = len(new_docs)
batch_size = 10

print("🧠 Membuat embedding dan menambahkan ke FAISS...")

for i in range(0, total, batch_size):
    batch = new_docs[i:i+batch_size]
    texts = [d["page_content"] for d in batch]
    metas = [d["metadata"] for d in batch]

    # buat objek Document baru
    docs_batch = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
    
    # tambahkan langsung ke vectorstore
    vectorstore.add_documents(docs_batch, embedding=embeddings)

    pct = ((i + len(batch)) / total) * 100
    print(f"\rProgress: {i + len(batch)}/{total} ({pct:.1f}%)", end="")

print("\n✅ Embedding selesai, menyimpan kembali FAISS index...")

# --- 6️⃣ Simpan ulang index ---
os.makedirs(index_path, exist_ok=True)
vectorstore.save_local(index_path)

print(f"💾 Index berhasil diperbarui dan disimpan di: {index_path}")
print("🎯 Selesai — semua artikel baru telah di-embed.")


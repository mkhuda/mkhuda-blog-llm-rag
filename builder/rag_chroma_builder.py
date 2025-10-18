"""
RAG Index Builder mkhuda.com — Versi ChromaDB Final (Stable)
------------------------------------------------------------
• Aman lintas OS & Python
• Explicit collection_name ("mkhuda_articles")
• Menyimpan metadata build (jumlah dokumen, tanggal)
• Auto rebuild jika index kosong / tidak kompatibel
"""

from dotenv import load_dotenv
load_dotenv()

import os, json, re
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb import PersistentClient

# === Konfigurasi dasar ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

mysql_database = os.getenv("MYSQL_DATABASE")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_port = os.getenv("MYSQL_PORT", "3306")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "mkhuda_chroma"
CHROMA_DOCS_PATH = BASE_DIR / "docs_chroma.json"

collection_name = "mkhuda_articles"

# === Coba deteksi koleksi yang sudah ada ===
indexed_urls = set()
client = PersistentClient(path=str(CHROMA_DIR))
collections = [c.name for c in client.list_collections()]

if collection_name in collections:
    print(f"📂 Memuat koleksi '{collection_name}' yang sudah ada...")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    all_docs = vectorstore.get()
    for meta in all_docs["metadatas"]:
        if meta and "url" in meta:
            indexed_urls.add(meta["url"])
    print(f"✅ Koleksi dimuat ({len(indexed_urls)} dokumen sudah di-index).")
else:
    print("🆕 Koleksi baru akan dibuat...")
    vectorstore = None

# === Siapkan dokumen ===
def clean_html(text):
    text = re.sub(r"\[.*?\]", "", text or "")
    return BeautifulSoup(text, "html.parser").get_text().strip()


def fetch_posts_from_db():
    print("📡 Menghubungkan ke database WordPress...")
    conn = None
    try:
        conn = mysql.connector.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )
        df = pd.read_sql(
            """
            SELECT ID, post_title, post_content, post_date
            FROM wp_posts
            WHERE post_status='publish' AND post_type='post'
            ORDER BY post_date DESC;
            """,
            conn,
        )
        print(f"✅ {len(df)} post berhasil dimuat dari database.")
        df["clean_content"] = df["post_content"].apply(clean_html)
        df["url"] = "https://mkhuda.com/?p=" + df["ID"].astype(str)
        candidate_docs = [
            {
                "page_content": row.clean_content,
                "metadata": {
                    "title": row.post_title,
                    "url": row.url,
                    "date": str(row.post_date),
                },
            }
            for _, row in df.iterrows()
        ]
        return candidate_docs
    finally:
        if conn:
            conn.close()


try:
    candidate_docs = fetch_posts_from_db()
except mysql.connector.Error as err:
    if CHROMA_DOCS_PATH.exists():
        print(f"⚠️  Gagal mengambil data dari database ({err}).")
        print(f"📄 Menggunakan cadangan {CHROMA_DOCS_PATH}...")
        with open(CHROMA_DOCS_PATH, "r", encoding="utf-8") as f:
            candidate_docs = json.load(f)
    else:
        raise

new_docs = [
    doc for doc in candidate_docs if doc["metadata"].get("url") not in indexed_urls
]

with open(CHROMA_DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(new_docs, f, ensure_ascii=False, indent=2)

if not new_docs:
    print("🎉 Tidak ada artikel baru untuk di-index.")
    exit()

docs_objs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in new_docs]

# === Tambah atau buat index ===
if vectorstore is None:
    print("🧱 Membuat koleksi baru...")
    vectorstore = Chroma.from_documents(
        documents=docs_objs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name,
    )
else:
    print("🧠 Menambahkan embedding ke koleksi...")
    vectorstore.add_documents(docs_objs)

# vectorstore.persist()
print(f"💾 Index tersimpan di: {CHROMA_DIR}/{collection_name}")

# === Simpan metadata build ===
meta_info = {
    "collection_name": collection_name,
    "total_indexed": len(indexed_urls) + len(new_docs),
    "new_added": len(new_docs),
    "build_time": datetime.now().isoformat(),
}
meta_path = BASE_DIR / "mkhuda_chroma_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta_info, f, ensure_ascii=False, indent=2)

print(f"📘 Metadata build disimpan di {meta_path}")
print("🎯 Selesai — index Chroma siap digunakan lintas-device.")

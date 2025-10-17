"""
RAG Index Builder mkhuda.com — Versi ChromaDB Cross-Platform
-------------------------------------------------------------
• Tidak lagi menggunakan FAISS/pickle (lebih aman lintas OS)
• Otomatis rebuild index dari docs.json jika belum ada
• Bisa digunakan untuk advanced filtering (mis. by date)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# === 1️⃣ Konfigurasi dasar ===
api_key = os.getenv("OPENAI_API_KEY")
mysql_database = os.getenv("MYSQL_DATABASE")
mysql_host = os.getenv("MYSQL_HOST")
mysql_user = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_port = os.getenv("MYSQL_PORT", "3306")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di file .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
chroma_dir = Path("mkhuda_chroma")  # direktori penyimpanan ChromaDB
docs_path = Path("docs_chroma.json")  # versi json portable

# === 2️⃣ Coba load index Chroma lama ===
vectorstore = None
indexed_urls = set()

if chroma_dir.exists():
    try:
        print("📂 Memuat Chroma index lama...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(chroma_dir)
        )
        # ambil semua metadata untuk deteksi incremental
        all_docs = vectorstore.get()
        for meta in all_docs["metadatas"]:
            if meta and "url" in meta:
                indexed_urls.add(meta["url"])
        print(f"✅ Index lama dimuat ({len(indexed_urls)} dokumen sudah di-index).")
    except Exception as e:
        print(f"⚠️ Gagal memuat index lama ({type(e).__name__}: {e})")
        print("➡️ Akan membangun index baru dari docs.json (jika ada)...")

# === 3️⃣ Buat atau load docs.json ===
if not docs_path.exists():
    print("📡 Menghubungkan ke database WordPress...")

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
    print(f"✅ {len(df)} post berhasil dimuat dari database.")

    # Bersihkan HTML
    def clean_html(text):
        text = re.sub(r'\[.*?\]', '', text)
        text = BeautifulSoup(text or "", "html.parser").get_text()
        return text.strip()

    df["clean_content"] = df["post_content"].apply(clean_html)
    df["url"] = "https://mkhuda.com/?p=" + df["ID"].astype(str)

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

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(new_docs, f, ensure_ascii=False, indent=2)
    print(f"📝 File {docs_path} berhasil dibuat ({len(new_docs)} dokumen).")
else:
    print(f"📄 Menggunakan {docs_path} yang sudah ada...")
    with open(docs_path, "r", encoding="utf-8") as f:
        new_docs = json.load(f)

if not new_docs:
    print("🎉 Tidak ada artikel baru untuk di-index.")
    exit()

print(f"🧩 Siap menambahkan {len(new_docs)} artikel baru ke Chroma index.")

# === 4️⃣ Rebuild atau tambah ke Chroma ===
if vectorstore is None:
    print("🧱 Membuat index Chroma baru dari docs.json...")
    docs_objs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in new_docs]
    vectorstore = Chroma.from_documents(
        documents=docs_objs,
        embedding=embeddings,
        persist_directory=str(chroma_dir)
    )
else:
    print("🧠 Menambahkan embedding ke Chroma index...")
    docs_objs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in new_docs]
    vectorstore.add_documents(docs_objs)

# === 5️⃣ Simpan index ===
vectorstore.persist()
print(f"💾 Index tersimpan di: {chroma_dir}")

# === 6️⃣ Simpan backup JSON ===
backup_path = Path("mkhuda_chroma_backup.json")
backup = [
    {"page_content": d.page_content, "metadata": d.metadata}
    for d in docs_objs
]
with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(backup, f, ensure_ascii=False, indent=2)

print(f"📦 Backup JSON dibuat ({backup_path})")
print("🎯 Selesai — index Chroma siap digunakan lintas-device.")

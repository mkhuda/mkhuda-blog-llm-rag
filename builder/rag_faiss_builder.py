# rag_index_builder_json.py
"""
RAG Index Builder mkhuda.com — Cross-Platform + Self-Healing
- Selalu tulis docs.json (FULL korpus) dari DB.
- Jika FAISS lama gagal diload, rebuild dari:
  1) mkhuda_faiss_backup.json (jika ada)
  2) docs.json
  3) DB (full export)
"""

from dotenv import load_dotenv
load_dotenv()

import os, re, json
from pathlib import Path
import pandas as pd
import mysql.connector
from bs4 import BeautifulSoup

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "mkhuda_faiss_index"
DOCS_JSON = BASE_DIR / "docs.json"                   # full korpus
BACKUP_JSON = BASE_DIR / "mkhuda_faiss_backup.json"  # dump dari FAISS terakhir

# --- ENV ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

mysql_database = os.getenv("MYSQL_DATABASE")
mysql_host     = os.getenv("MYSQL_HOST")
mysql_user     = os.getenv("MYSQL_USER")
mysql_password = os.getenv("MYSQL_PASSWORD")
mysql_port     = int(os.getenv("MYSQL_PORT", "3306"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

def clean_html(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)  # hapus shortcode
    return BeautifulSoup(text or "", "html.parser").get_text().strip()

def fetch_full_corpus_from_db() -> list[dict]:
    if not all([mysql_database, mysql_host, mysql_user]) or mysql_password is None:
        raise RuntimeError("❌ Variabel koneksi MySQL belum lengkap.")
    conn = mysql.connector.connect(
        host=mysql_host, port=mysql_port, user=mysql_user,
        password=mysql_password, database=mysql_database
    )
    query = """
      SELECT ID, post_title, post_content, post_date
      FROM wp_posts
      WHERE post_status='publish' AND post_type='post'
      ORDER BY post_date DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["clean_content"] = df["post_content"].apply(clean_html)
    df["url"] = "https://mkhuda.com/?p=" + df["ID"].astype(str)
    docs = []
    for _, r in df.iterrows():
        docs.append({
            "page_content": r.clean_content,
            "metadata": {"title": r.post_title, "url": r.url, "date": str(r.post_date)}
        })
    return docs

def load_docs_from_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_docs_json(docs: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

def docs_to_objects(docs: list[dict]) -> list[Document]:
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]

# 1) Coba load FAISS lama untuk incremental
vectorstore = None
indexed_urls: set[str] = set()
if INDEX_DIR.exists():
    try:
        print("📂 Memuat FAISS lama…")
        vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        indexed_urls = {
            d.metadata.get("url")
            for d in vectorstore.docstore._dict.values()
            if d.metadata.get("url")
        }
        print(f"✅ FAISS lama dimuat ({len(indexed_urls)} dokumen).")
    except Exception as e:
        print(f"⚠️ Gagal memuat FAISS lama ({type(e).__name__}: {e})")

# 2) Pastikan kita punya FULL docs.json (bukan “new_docs”)
full_docs: list[dict] = []
if DOCS_JSON.exists():
    try:
        full_docs = load_docs_from_json(DOCS_JSON)
        # Heuristik: kalau docs.json kosong/terlalu kecil, anggap tidak valid
        if len(full_docs) == 0:
            print("⚠️ docs.json kosong — akan rebuild dari sumber lain.")
            full_docs = []
    except Exception as e:
        print(f"⚠️ docs.json tidak bisa dibaca ({e})")
        full_docs = []

# 3) Fallback sumber dokumen: BACKUP_JSON → DB
if not full_docs and BACKUP_JSON.exists():
    try:
        print("📄 Menggunakan mkhuda_faiss_backup.json …")
        full_docs = load_docs_from_json(BACKUP_JSON)
    except Exception as e:
        print(f"⚠️ Backup JSON rusak ({e})")

if not full_docs:
    print("🗄️ Mengambil FULL korpus dari database…")
    full_docs = fetch_full_corpus_from_db()
    print(f"✅ Dapat {len(full_docs)} dokumen dari DB.")
    save_docs_json(full_docs, DOCS_JSON)
    print(f"💾 FULL docs.json tersimpan → {DOCS_JSON}")

if not full_docs:
    raise RuntimeError("❌ Tidak ada dokumen untuk di-index (DB/JSON kosong).")

# 4) Tentukan dokumen yang perlu ditambahkan (incremental), tapi
#    kalau FAISS belum ada (atau gagal load), kita build dari NOL.
if vectorstore is None:
    print("🧱 Membangun FAISS BARU dari FULL docs.json …")
    vectorstore = FAISS.from_documents(docs_to_objects(full_docs), embeddings)
else:
    # incremental add: hanya dokumen yang url-nya belum pernah di-index
    to_add = [d for d in full_docs if d["metadata"].get("url") not in indexed_urls]
    if not to_add:
        print("🎉 Tidak ada artikel baru untuk ditambahkan.")
    else:
        print(f"🧩 Menambahkan {len(to_add)} artikel baru…")
        # batch kecil agar stabil
        BATCH = 16
        for i in range(0, len(to_add), BATCH):
            batch = to_add[i : i + BATCH]
            vectorstore.add_documents(docs_to_objects(batch), embedding=embeddings)
            done = i + len(batch)
            print(f"\rProgress: {done}/{len(to_add)}", end="")
        print("")

# 5) Simpan FAISS + backup JSON dari FAISS (ground truth portable)
INDEX_DIR.mkdir(exist_ok=True)
vectorstore.save_local(str(INDEX_DIR))
print(f"✅ FAISS tersimpan di {INDEX_DIR}")

backup = [
    {"page_content": d.page_content, "metadata": d.metadata}
    for d in vectorstore.docstore._dict.values()
]
save_docs_json(backup, BACKUP_JSON)
print(f"📦 Backup JSON tersimpan di {BACKUP_JSON}")
print("🎯 Selesai.")

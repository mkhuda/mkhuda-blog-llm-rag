"""
RAG FastAPI mkhuda.com
----------------------
Backend ringan untuk asisten mkhuda.com berbasis FAISS lokal dan OpenAI API.
Diuji untuk berjalan di VPS 1 GB (Python 3.12 + FastAPI + Uvicorn).

Endpoint utama:
POST /ask  →  { "message": "artikel tentang HTMX" }
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from rag_pre_reasoning import pre_reasoning

from datetime import datetime
import os

# ---------- SETUP ENV ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

# ---------- INIT FASTAPI ----------
app = FastAPI(title="mkhuda.com RAG API", version="1.0")

# Aktifkan CORS agar bisa diakses dari web / localhost / domain mkhuda.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa diganti dengan ["https://mkhuda.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODEL & RETRIEVER ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

INDEX_PATH = "mkhuda_faiss_index"
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ---------- PROMPT ----------
today = datetime.now().strftime("%Y-%m-%d")
# ---------- PROMPT ----------
prompt_body = """
Kamu adalah asisten cerdas untuk situs web mkhuda.com — blog teknologi berisi artikel seputar AI, web development, dan tutorial modern.

Gunakan konteks {context} yang berisi kumpulan artikel dari mkhuda.com. 
Setiap artikel memiliki metadata berikut:
- title → judul artikel
- url → tautan artikel
- date → tanggal publikasi (format YYYY-MM-DD HH:MM:SS)

Tugasmu adalah membantu pengguna menemukan artikel yang sesuai.

---

🧩 Jenis permintaan yang perlu kamu tangani:

1️⃣ **Pencarian artikel berdasarkan topik atau kata kunci**
   - Jika user menanyakan sesuatu seperti "artikel tentang htmx", "apa itu prompt engineering", atau "framework ringan", 
     carikan artikel yang relevan.
   - Jawaban ideal berisi penjelasan singkat, lalu daftar artikel relevan dengan format HTML:
     <a href="{{url}}" target="_blank">{{title}}</a>

2️⃣ **Pencarian artikel berdasarkan waktu**
   - Jika user menyebut waktu, seperti “artikel bulan Juli 2024”, “artikel tahun ini”, “artikel terbaru”, atau “artikel terlama”:
     - Gunakan metadata `date` untuk menentukan artikel yang dimaksud.
     - Urutkan hasil:
       • “terbaru” → tanggal paling baru di atas  
       • “terlama” → tanggal paling lama di atas
     - Jika user menyebut bulan/tahun → tampilkan artikel dengan tanggal yang cocok.

3️⃣ **Ringkasan artikel**
   - Jika user menyebut judul artikel (mis. “ringkas/rangkum/kesimpulan artikel tentang HTMX atau React”), 
     anggap mereka mencari artikel itu atau topik yang serupa.
   - Jika artikel dengan judul itu ada di konteks, tampilkan ringkasan berupa point penting dan kesimpulan
   - Lalu, tampilkan juga artikel lain dengan tema yang mirip dan tautan langsungnya.
---
💬 Gaya jawaban:
- Bahasa Indonesia santai, informatif, dan sopan.
- Jangan tautkan situs lain selain mkhuda.com.
- Gunakan HTML aman (tanpa <script>).
- Jika tidak ada hasil relevan, jawab sopan: “Sepertinya belum ada artikel tentang itu di mkhuda.com.”
"""

system_prompt = f"Tanggal hari ini: {today}\n\n{prompt_body}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Pertanyaan: {input}\n\nKonteks:\n{context}")
])

combine_docs_chain = create_stuff_documents_chain(
    llm=llm, prompt=prompt, document_variable_name="context"
)

# ---------- HELPER ----------
def format_docs_with_meta(docs, max_chars=1000):
    """Ringkas dokumen agar hemat token"""
    parts = []
    for d in docs:
        m = d.metadata
        text = d.page_content.strip()[:max_chars]
        date = m.get("date", "(tanpa tanggal)")[:10]
        parts.append(
            f"Judul: {m.get('title','(tanpa judul)')}\n"
            f"URL: {m.get('url','(tanpa url)')}\n"
            f"Tanggal: {date}\n"
            f"Teks:\n{text}\n"
        )
    return "\n---\n".join(parts)

# ---------- ROUTES ----------
@app.get("/")
async def root():
    return {"message": "🤖 mkhuda.com RAG API aktif", "status": "ok"}

@app.post("/ask")
async def ask(request: Request):
    """Terima pertanyaan dari user dan kembalikan jawaban HTML-ready"""
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"reply": "Tolong masukkan pertanyaan."}

    # 1️⃣ Intent detection (pre-reasoning)
    intent = pre_reasoning(message)
    if intent["intent"] == "out_of_scope":
        return {"reply": intent.get("message", "Pertanyaan di luar cakupan mkhuda.com.")}

    # 2️⃣ Retrieve artikel relevan
    docs = retriever.invoke(message)
    context_text = format_docs_with_meta(docs)
    context_doc = [Document(page_content=context_text)]

    # 3️⃣ Generate jawaban dari LLM
    answer = combine_docs_chain.invoke({"context": context_doc, "input": message})
    return {"reply": answer}

# ---------- RUN LOCAL ----------
# Jalankan:  uvicorn rag_fast_api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

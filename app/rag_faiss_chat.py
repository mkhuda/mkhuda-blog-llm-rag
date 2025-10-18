import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.pydantic_langchain_fix import patch_langchain_models
patch_langchain_models(verbose=False)

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document

# 1) Keys / models
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

# 2) Load FAISS
INDEX_PATH = BASE_DIR / "mkhuda_faiss_index"
vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def debug_faiss_retriever(query):
    results = vectorstore.similarity_search_with_score(query, k=3)
    print(f"\n🔍 [DEBUG FAISS] Hasil retrieval untuk query: '{query}'")
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        title = meta.get("title", "(tanpa judul)")
        url = meta.get("url", "-")
        score_display = f"{score:.4f}"  # FAISS pakai L2 distance
        # Biasanya FAISS pakai L2 distance (semakin kecil semakin mirip)
        print(f"{rank:02d}. {title}")
        print(f"    Jarak : {score:.4f}")  # bukan similarity, tapi distance
        print(f"    URL   : {url}\n")
    print("-" * 40)

# 3) Prompt: WAJIB punya {context} + {question}

system_prompt = """
    Kamu adalah asisten situs web mkhuda.com.

    Setiap bagian konteks memiliki format seperti:
    Judul: ...
    URL: ...
    Teks: ...

    Gunakan URL dan judul yang tertera di konteks untuk membuat tautan HTML dengan format:
    <a href="{{url}}" target="_blank">{{title}}</a>

    Jika ada beberapa artikel relevan, tampilkan semuanya dalam daftar tautan.
    Gunakan bahasa Indonesia yang santai tapi sopan.
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Konteks:\n{context}\n\nPertanyaan: {input}")
])

# 2️⃣ Fungsi formatter context agar metadata ikut dikirim
def format_docs_with_meta(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(
            f"Judul: {meta.get('title', '(tanpa judul)')}\n"
            f"URL: {meta.get('url', '(tidak ada url)')}\n"
            f"Teks:\n{d.page_content}\n"
        )
    return "\n---\n".join(formatted)

# 4) Bangun chain modern (tanpa StuffDocumentsChain class)
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context"  # default "context", eksplisitkan saja
)

rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
# rag_chain = rag_chain.with_config({"document_formatter": format_docs_with_meta})

# 5) Loop interaktif
print("\n💬 RAG Chat mkhuda.com siap. Ketik pertanyaan, atau 'exit' untuk keluar.\n")
while True:
    q = input("🧠 Pertanyaan: ").strip()
    if q.lower() in {"exit", "quit", "keluar"}:
        print("👋 Keluar.")
        break

    # 1️⃣ ambil hasil dari retriever (versi baru)
    docs = retriever.invoke(q)

    ## debugging only: tampilkan hasil retrieval
    # debug_faiss_retriever(q)  

    # 2️⃣ format ulang jadi string gabungan (dengan metadata)
    context_text = format_docs_with_meta(docs)

    # 3️⃣ bungkus kembali jadi Document tunggal
    context_doc = [Document(page_content=context_text)]
    result = combine_docs_chain.invoke({
        "context": context_doc,
        "input": q
    })
    print("\n🤖 Jawaban:\n", result, "\n")

from dotenv import load_dotenv
load_dotenv()

import os
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
INDEX_PATH = "mkhuda_faiss_index"
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

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

    # 2️⃣ format ulang jadi string gabungan (dengan metadata)
    context_text = format_docs_with_meta(docs)

    # 3️⃣ bungkus kembali jadi Document tunggal
    context_doc = [Document(page_content=context_text)]
    result = combine_docs_chain.invoke({
        "context": context_doc,
        "input": q
    })
    print("\n🤖 Jawaban:\n", result, "\n")


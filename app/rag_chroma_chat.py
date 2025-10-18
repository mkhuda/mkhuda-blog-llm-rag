from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document

# 1️⃣ Keys / models
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)

# 2️⃣ Load Chroma
chroma_dir = "mkhuda_chroma"
collection_name = "mkhuda_articles"

vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=chroma_dir,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

def debug_chroma_retriever(query):
    print(f"\n🔍 [DEBUG Chroma] Hasil retrieval untuk query: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=3)
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        title = meta.get("title", "(tanpa judul)")
        url = meta.get("url", "-")
        print(f"{rank:02d}. {title}")
        print(f"    Skor : {score:.4f}")
        print(f"    URL  : {url}\n")
    print("-" * 40)

# 3️⃣ Prompt (WAJIB punya {context} + {question})
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

# 4️⃣ Formatter untuk context agar metadata ikut dikirim
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

# 5️⃣ Bangun chain modern
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context"
)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 6️⃣ Loop interaktif
print("\n💬 RAG Chat mkhuda.com (Chroma) siap. Ketik pertanyaan, atau 'exit' untuk keluar.\n")
while True:
    q = input("🧠 Pertanyaan: ").strip()
    if q.lower() in {"exit", "quit", "keluar"}:
        print("👋 Keluar.")
        break

    # Ambil hasil retrieval
    # docs = retriever.invoke(q)

    debug_chroma_retriever(q)  # debug: tampilkan hasil retrieval

    # break

    # # Format ulang jadi gabungan teks
    # context_text = format_docs_with_meta(docs)

    # # Bungkus ulang jadi Document tunggal
    # context_doc = [Document(page_content=context_text)]

    # # Jalankan LLM
    # result = combine_docs_chain.invoke({
    #     "context": context_doc,
    #     "input": q
    # })

    # print("\n🤖 Jawaban:\n", result, "\n")

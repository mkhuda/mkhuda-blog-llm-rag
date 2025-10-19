"""
RAG Chat mkhuda.com — versi praktis
-----------------------------------
Tanpa filter where, tanpa date range.
Fokus pada semantic similarity dan relevansi pertanyaan.
"""

from utils.pydantic_langchain_fix import patch_langchain_models
patch_langchain_models(verbose=False)

from dotenv import load_dotenv
load_dotenv()

import os, gradio as gr
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from rag_pre_reasoning2 import pre_reasoning
from utils.rag_prompts import mkhuda_system_prompt


# ---------- KONFIGURASI ----------
chroma_dir = "mkhuda_chroma"
collection_name = "mkhuda_articles"
today = datetime.now().strftime("%Y-%m-%d")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

# ---------- LOAD CHROMA ----------
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=chroma_dir,
    embedding_function=embeddings,
)

# ---------- PROMPT ----------
system_prompt = mkhuda_system_prompt(today)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Pertanyaan: {input}\n\nKonteks:\n{context}")
])
combine_docs_chain = create_stuff_documents_chain(
    llm=llm, prompt=prompt, document_variable_name="context"
)


# ---------- HELPER ----------
def format_docs_with_meta(docs, max_chars=400):
    """Format isi dokumen hasil Chroma."""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        text = d.page_content.strip()[:max_chars]
        parts.append(
            f"📝 **{meta.get('title','(tanpa judul)')}**\n"
            f"📅 {meta.get('date','(tanpa tanggal)')[:10]}\n"
            f"🔗 [{meta.get('url','(tanpa url)')}]({meta.get('url','')})\n\n"
            f"{text}\n"
        )
    return "\n---\n".join(parts)


# ---------- MAIN FUNCTION ----------
def rag_answer(message, history):
    """Jawaban utama untuk chat interface."""
    # 1️⃣ Deteksi apakah pertanyaan relevan
    intent_result = pre_reasoning(message)
    if intent_result.get("intent") == "out_of_scope":
        return intent_result.get("message") or "Maaf, saya hanya menjawab seputar teknologi & AI di mkhuda.com."

    # 2️⃣ Cari konteks dengan similarity search
    try:
        docs = vectorstore.similarity_search(message, k=3)
    except Exception as e:
        return f"❌ Gagal mencari artikel: {e}"

    if not docs:
        return "Belum ada artikel yang relevan untuk pertanyaan itu."

    # 3️⃣ Kirim ke LLM untuk membangun jawaban
    context_text = format_docs_with_meta(docs)
    context_doc = [Document(page_content=context_text)]
    try:
        answer = combine_docs_chain.invoke({"context": context_doc, "input": message})
    except Exception as e:
        return f"⚠️ Gagal memproses jawaban: {e}"

    return answer


# ---------- GRADIO ----------
demo = gr.ChatInterface(
    fn=rag_answer,
    title="🤖 Asisten mkhuda.com",
    description="Tanyakan apa saja tentang artikel teknologi & AI di mkhuda.com.",
    theme=gr.themes.Soft(),
    examples=[
        ["Apa itu HTMX?"],
        ["Framework ringan apa yang dibahas di mkhuda.com?"],
        ["Tips pakai Alpine.js untuk interaksi web?"],
        ["AI tools terbaru yang direview di mkhuda.com?"],
    ],
    type="messages",
    analytics_enabled=True,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

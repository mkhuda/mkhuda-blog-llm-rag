"""
RAG Chat mkhuda.com — Gradio 5.49.1
-----------------------------------
- Fully compatible dengan LangChain 0.3+
- HTML output <a href> langsung bisa diklik
- Gaya chat modern pakai gr.ChatInterface
"""

from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from rag_pre_reasoning import pre_reasoning
from utils.rag_prompts import mkhuda_system_prompt

from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")

# ---------- SETUP ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

INDEX_PATH = "mkhuda_faiss_index"
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

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
def format_docs_with_meta(docs, max_chars=1000):
    parts = []
    for d in docs:
        meta = d.metadata
        text = d.page_content.strip()[:max_chars]
        date_str = meta.get("date", "(tanpa tanggal)")[:10]  # potong jamnya
        parts.append(
            f"Judul: {meta.get('title','(tanpa judul)')}\n"
            f"URL: {meta.get('url','(tanpa url)')}\n"
            f"Tanggal: {date_str}\n"
            f"Teks:\n{text}\n"
        )
    return "\n---\n".join(parts)

def rag_answer(message, history):
    # 0️⃣ Cek maksud dulu
    intent_result = pre_reasoning(message)
    if intent_result.get("intent") == "out_of_scope":
        return intent_result.get(
            "message",
            "Maaf, saya hanya bisa membantu menjawab pertanyaan seputar teknologi dan artikel di mkhuda.com."
        )
    
    docs = retriever.invoke(message)
    context_text = format_docs_with_meta(docs)
    context_doc = [Document(page_content=context_text)]
    answer = combine_docs_chain.invoke({"context": context_doc, "input": message})
    return answer  # langsung string HTML-friendly


# ---------- GRADIO CHATINTERFACE ----------
demo = gr.ChatInterface(
    fn=rag_answer,
    title="🤖 Asisten mkhuda.com",
    description="Tanyakan apa saja tentang artikel di mkhuda.com — asisten akan mencarikan artikel yang relevan dan memberikan tautan langsung.",
    theme=gr.themes.Soft(),
    examples=[
        ["Apa itu HTMX?"],
        ["Framework ringan apa yang dibahas di mkhuda.com?"],
        ["Ada artikel tentang PHP modern?"],
    ],
    type="messages",     # gunakan input gaya percakapan
    multimodal=False,    # hanya teks
    analytics_enabled=True,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

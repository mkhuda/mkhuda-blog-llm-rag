"""
RAG FastAPI mkhuda.com
----------------------
- Self-healing: builds FAISS index on first run if missing.
- Auto-updating: schedules a background job to rebuild the index every 2 days.
"""
import os
import sys
import re
import threading
import datetime
from pathlib import Path
import subprocess
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger("uvicorn")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_community.callbacks.manager import get_openai_callback

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.rag_pre_reasoning import pre_reasoning
from utils.rag_prompts import mkhuda_system_prompt

# ---------- SETUP & PATHS ----------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

INDEX_PATH = BASE_DIR / "mkhuda_faiss_index"
BUILDER_PATH = BASE_DIR / "builder" / "rag_faiss_builder.py"
scheduler = BackgroundScheduler()

# ---------- FAISS INDEX & SCHEDULER LOGIC ----------
def build_faiss_index():
    """Builds or rebuilds the FAISS index by running the builder script."""
    print(f"[{datetime.datetime.now()}] 🔄 Starting FAISS index build...")
    try:
        # Use sys.executable to ensure we're using the python from the correct venv
        subprocess.run([sys.executable, str(BUILDER_PATH)], check=True, capture_output=True, text=True)
        print("✅ FAISS index built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build FAISS index. Error: {e.stderr}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during FAISS build: {e}")

def ensure_faiss_index():
    """Ensures FAISS index is available, building it automatically if not."""
    if not INDEX_PATH.exists() or not any(INDEX_PATH.iterdir()):
        print("⚙️ FAISS index not found — building automatically (this may take a moment)...")
        build_faiss_index() # This is a blocking call for the very first startup
    else:
        print("🧠 FAISS index found — ready to use.")

def scheduled_rebuild_job():
    """Wrapper function for the scheduler to run the build process."""
    print("🗓️ Kicked off by scheduler: Rebuilding FAISS index in the background.")
    # Run the build in a separate thread to avoid blocking the main app
    threading.Thread(target=build_faiss_index, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI app."""
    print("🚀 FastAPI starting up...")
    # 1. On startup, ensure the index exists (blocking).
    ensure_faiss_index()

    # 2. Add the recurring job to the scheduler. It will run every 2 days.
    scheduler.add_job(scheduled_rebuild_job, "interval", days=2, id="faiss_rebuild_job")
    
    # 3. Start the background scheduler.
    scheduler.start()
    print(f"✅ Scheduler started. Next FAISS rebuild is scheduled in 2 days.")
    
    try:
        yield
    finally:
        # 4. On shutdown, cleanly stop the scheduler.
        print("🛑 Shutting down scheduler...")
        scheduler.shutdown()

# ---------- INIT FASTAPI ----------
app = FastAPI(
    title="mkhuda.com RAG API",
    version="1.1",
    lifespan=lifespan # Use the new lifespan manager
)

ENV = os.getenv("MODE", "production").lower()

ORIGINS = {
    "development": "*",
    "production": "https://mkhuda.com",
}

# fallback to production if somebody sets MODE to something unexpected
ORIGIN = ORIGINS.get(ENV, ORIGINS["production"])

logger.info(f"🌐 Running in '{ENV}' mode. CORS allowed origin: {ORIGIN}")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", lambda: print("✅ FastAPI app is up and running."))

# ---------- MODEL & RETRIEVER ----------
# This part remains the same
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

vectorstore = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ---------- PROMPT ----------
# This part remains the same
today = datetime.datetime.now().strftime("%Y-%m-%d")
system_prompt = mkhuda_system_prompt(today)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Pertanyaan: {input}\n\nKonteks:\n{context}")
])

combine_docs_chain = create_stuff_documents_chain(
    llm=llm, prompt=prompt, document_variable_name="context"
)

# ---------- HELPER ----------
# This part remains the same
def format_docs_with_meta(docs, max_chars=1000):
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

# get IP address and user agent from request
def get_request_info(request: Request):
    ip = request.client.host
    user_agent = request.headers.get("User-Agent", "unknown")
    return ip, user_agent

# ---------- ROUTES ----------
@app.get("/")
async def root():
    next_run = scheduler.get_job('faiss_rebuild_job').next_run_time.strftime('%Y-%m-%d %H:%M:%S')
    return {
        "message": "🤖 mkhuda.com RAG API aktif",
        "status": "ok",
        "faiss_rebuild_scheduler": "active",
        "next_scheduled_rebuild": next_run
    }

@app.post("/ask")
async def ask(request: Request):
    # This part remains the same
    ip, user_agent = get_request_info(request)
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return {"reply": "Tolong masukkan pertanyaan."}
    
    intent, usage_metadata = pre_reasoning(message)
    logger.info(f"🧠 [PRE-REASONING] Tokens used: {usage_metadata.total_tokens}")

    if intent["intent"] == "out_of_scope":
        return {"reply": intent.get("message", "Pertanyaan di luar cakupan mkhuda.com.")}
    
    docs = retriever.invoke(message)
    context_text = format_docs_with_meta(docs)
    context_doc = [Document(page_content=context_text)]
    
    with get_openai_callback() as cb:
        answer = combine_docs_chain.invoke({"context": context_doc, "input": message})

    logger.info(f"🧾 [RAG ANSWER] Tokens used: {cb.total_tokens}")
    logger.info(f"🧾 [ALL] Tokens used: {cb.total_tokens + usage_metadata.total_tokens} | from {ip} - {user_agent}")

    response_text = re.sub(r'\\n', '\n', answer).strip()
    return {"reply": response_text}

@app.post("/rebuild")
def manual_rebuild():
    """Endpoint to manually trigger a rebuild in the background."""
    scheduled_rebuild_job()
    return {"status": "ok", "message": "Manual FAISS index rebuild triggered in the background."}


# ---------- RUN LOCAL ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

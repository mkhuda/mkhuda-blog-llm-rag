import subprocess
import datetime
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

BASE_DIR = Path(__file__).resolve().parent.parent

def _builder_path():
    return str(BASE_DIR / "builder" / "rag_faiss_builder.py")

def rebuild_faiss():
    print(f"[{datetime.datetime.now()}] 🔄 Building FAISS index (initial)...")
    print(f"📂 Builder path: {_builder_path()}")
    try:
        subprocess.run([sys.executable, _builder_path()], check=True)
        print("✅ FAISS index built successfully.")
    except Exception as e:
        print(f"❌ Failed to build FAISS: {e}")

def rebuild_faiss_async():
    threading.Thread(target=rebuild_faiss, daemon=True).start()

scheduler = BackgroundScheduler()
scheduler.add_job(rebuild_faiss, "interval", days=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting FastAPI — building FAISS index before serving API...")
    rebuild_faiss()  # 🔒 BLOCKING — pastikan index ready sebelum serve
    print("🧠 FAISS ready — starting scheduler...")
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FAISS auto rebuild active (blocking startup).",
        "next_rebuild": "every 2 days",
    }

@app.post("/rebuild")
def manual_rebuild():
    rebuild_faiss_async()
    return {"status": "manual rebuild triggered"}

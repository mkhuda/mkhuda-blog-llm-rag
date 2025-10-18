"""
pre_reasoning.py — Layer Intent Routing untuk RAG mkhuda.com
------------------------------------------------------------
Tahapan:
1️⃣ Menerima pertanyaan user (string)
2️⃣ Menggunakan model ringan (gpt-4o-mini) untuk menilai maksud pertanyaan
3️⃣ Mengembalikan JSON:
   {
     "intent": "rag_search" | "out_of_scope",
     "message": "..."  # opsional saran ke user
   }
"""

from openai import OpenAI
import os, json
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

client = OpenAI(api_key=api_key)

def pre_reasoning(user_query: str) -> dict:
    """
    Analisis maksud pertanyaan user.
    - Jika masih relevan dengan artikel mkhuda.com → intent = "rag_search"
    - Jika di luar topik → intent = "out_of_scope" + message
    """

    system_prompt = """
    Kamu adalah asisten untuk situs mkhuda.com.
    Tugasmu: pahami maksud pertanyaan user dan tentukan apakah perlu pencarian artikel (RAG) atau tidak. bisa jadi user menggunakan bahasa gaul atau formal.

    Panduan:
    - Jawab "rag_search" jika pertanyaan berkaitan dengan teknologi, veo, AI, pemrograman, web development,
      atau hal-hal yang kemungkinan dibahas di mkhuda.com.
    - Jawab "out_of_scope" jika pertanyaannya tidak relevan (mis. cuaca, puisi, gosip, motivasi, dll).

    Selalu jawab dalam format JSON seperti:
    {
      "intent": "rag_search" | "out_of_scope",
      "message": "jika out_of_scope, beri saran singkat"
    }
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    try:
        result = json.loads(completion.choices[0].message.content)
    except Exception:
        result = {"intent": "rag_search"}  # fallback aman

    return result


# --- Quick test (jalankan file langsung) ---
if __name__ == "__main__":
    while True:
        q = input("🧠 Pertanyaan: ")
        if q.lower() in {"exit", "quit"}:
            break
        res = pre_reasoning(q)
        print(json.dumps(res, indent=2, ensure_ascii=False))

"""
pre_reasoning.py — versi praktis
--------------------------------
Tugas: deteksi apakah pertanyaan user masih relevan dengan konten blog mkhuda.com
Tanpa niat rumit (date, keywords, dsb)
"""

from openai import OpenAI
import os, json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def pre_reasoning(user_query: str) -> dict:
    system_prompt = """
    Kamu adalah asisten untuk situs mkhuda.com.
    Tugasmu hanya menentukan apakah pertanyaan user relevan dengan topik situs ini atau tidak.

    Topik mkhuda.com meliputi:
    - teknologi modern, AI, prompt, web development, Laravel, Next.js, Alpine.js, HTMX,
    - produktivitas developer, framework ringan, dan tools teknologi.

    Kalau pertanyaan user di luar itu (cuaca, puisi, gosip, motivasi, makanan, dll),
    jawab dengan intent:"out_of_scope" dan berikan pesan sopan.
    Kalau masih relevan dengan topik di atas, balas dengan intent:"rag_search".

    Selalu output JSON seperti:
    {
    "intent": "rag_search" | "out_of_scope",
    "message": "pesan singkat ke user bila out_of_scope"
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
        return json.loads(completion.choices[0].message.content)
    except Exception:
        # fallback aman
        return {"intent": "rag_search", "message": ""}
        

if __name__ == "__main__":
    while True:
        q = input("🧠 Pertanyaan: ")
        if q.lower() in {"exit", "quit"}:
            break
        res = pre_reasoning(q)
        print(json.dumps(res, indent=2, ensure_ascii=False))

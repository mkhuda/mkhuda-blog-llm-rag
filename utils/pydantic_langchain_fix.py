"""
patch_langchain_fix.py
----------------------
Auto-patcher untuk bug Pydantic LangChain seperti:
❌ `ChatOpenAI is not fully defined`
❌ `name 'BaseCache' is not defined`
❌ `name 'Callbacks' is not defined`

Kompatibel dengan:
- LangChain 0.3.x
- LangChain-OpenAI 0.2.x
- Python 3.10–3.12
- Pydantic 2.x

Gunakan:
    from patch_langchain_fix import patch_langchain_models
    patch_langchain_models()
"""

def patch_langchain_models(verbose: bool = True):
    try:
        # --- Import dependency LangChain ---
        from langchain_openai import ChatOpenAI
        from langchain_core.language_models import BaseChatModel
        from langchain_core.caches import BaseCache

        # --- Tambahkan placeholder Callback bila belum ada ---
        import sys
        from types import SimpleNamespace

        if "Callbacks" not in globals():
            class Callbacks:
                """Dummy placeholder supaya Pydantic tidak error."""
                pass

        # Pastikan modul dikenali
        BaseCache.__module__ = "langchain_core.caches"
        globals()["Callbacks"] = Callbacks

        # Daftarkan dummy ke sys.modules biar bisa ditemukan oleh eval()
        sys.modules.setdefault("Callbacks", SimpleNamespace(Callbacks=Callbacks))

        # --- Rebuild model hanya jika perlu ---
        for model in (BaseChatModel, ChatOpenAI):
            if not getattr(model, "__pydantic_complete__", False):
                model.model_rebuild(force=True)

        if verbose:
            print("✅ LangChain Pydantic patch fully applied (BaseCache + Callbacks OK).")

    except Exception as e:
        if verbose:
            print(f"⚠️ LangChain patch failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    patch_langchain_models()

# mkhuda.com RAG Toolkit

This repository powers Retrieval-Augmented Generation (RAG) features for mkhuda.com. It contains builders that sync WordPress posts into vector stores (FAISS and Chroma) plus several chat front ends (CLI, Gradio, FastAPI) that query those vectors with OpenAI models.

## Project Layout

- `builder/` – scripts that pull content from WordPress and build FAISS or Chroma indexes.
- `app/` – chat experiences (CLI, Gradio UI, FastAPI) that read from the prepared indexes.
- `utils/` – helpers, including `pydantic_langchain_fix.py` that patches LangChain models for Pydantic v2.
- `mkhuda_faiss_index/`, `mkhuda_chroma/` – persisted vector stores and accompanying JSON backups/metadata.
- `docs.json`, `docs_chroma.json` – cached exports of the latest WordPress corpus used for incremental builds.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager (`pip install uv` or follow Astral’s instructions)
- Had access to your WordPress MySQL database
- OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`

## Environment Setup

1. Clone the repository and install dependencies with `uv`:

   ```bash
   git clone https://github.com/mkhuda/mkhuda-blog-llm-rag.git
   cd mkhuda-blog-llm-rag
   uv sync
   ```

2. Create a `.env` file in the project root (copy `.env.example` if you have one) and fill in your credentials:

   ```ini
   OPENAI_API_KEY=sk-...
   MYSQL_HOST=your-db-host
   MYSQL_PORT=port
   MYSQL_DATABASE=wordpress_db
   MYSQL_USER=db_user
   MYSQL_PASSWORD=db_password
   ```

   The scripts load these variables via `python-dotenv`.

3. (Optional) if you need to install extra tools into the virtual environment later, use `uv pip install <package>`.

## Building Indexes

All builder scripts assume you are in the project root. They will persist files to the top-level `mkhuda_*` directories.

- **Chroma index** (primary for LangChain apps):

  ```bash
  uv run python builder/rag_chroma_builder.py
  ```

  - Pulls the latest WordPress posts, cleans HTML, writes `docs_chroma.json`, updates the `mkhuda_chroma` directory, and logs metadata to `mkhuda_chroma_meta.json`.
  - Handles incremental updates by skipping URLs already present in the existing collection. Falls back to cached JSON if the DB is unreachable.

- **FAISS index (LangChain)**:

  ```bash
  uv run python builder/rag_faiss_builder.py
  ```

  - Loads the previous FAISS store if available, otherwise rebuilds from scratch.
  - Uses `docs.json` (latest full corpus) and `mkhuda_faiss_backup.json` as fallbacks to keep the vector store in sync.

- **FAISS index (LlamaIndex)**:

  ```bash
  uv run python builder/rag_build_llama.py
  ```

  - Maintains a FAISS index compatible with LlamaIndex using the same WordPress source.

- **Legacy FAISS builder** (`builder/rag_build.py`) is kept for backwards compatibility; prefer the scripts above.

## Running Chat Clients

- **CLI with FAISS (LangChain)**:

  ```bash
  uv run python app/rag_faiss_chat.py
  ```

  - Interactive terminal prompt that retrieves with FAISS and answers with `gpt-4o-mini`.

- **CLI with Chroma (LangChain)**:

  ```bash
  uv run python app/rag_chroma_chat.py
  ```

- **Gradio web app**:

  ```bash
  uv run python app/rag_gradio.py          # FAISS-backed
  uv run python app/rag_gradio_chroma.py   # Chroma-backed
  ```

  Visit the local URL printed in the terminal.

- **FastAPI service**:

  ```bash
  uv run python app/rag_fastapi.py
  ```

  Exposes an HTTP endpoint for programmatic access.

All chat apps expect the corresponding index directories to exist before launch. Run the builder scripts first if you see missing index errors.

## Maintenance & Tips

- Re-run the builder scripts whenever new posts are published on mkhuda.com.
- `docs.json` and `mkhuda_faiss_backup.json` are safe to commit to backups but contain full article text; handle according to your data policies.
- If LangChain breaks due to Pydantic updates, ensure `utils/pydantic_langchain_fix.py` is imported before other LangChain modules (already handled in the app scripts).
- The repository uses `pyproject.toml` + `uv.lock` to pin dependencies. Use `uv lock` to refresh the lockfile when upgrading packages.

## License

This project is licensed under the MIT License. Pull requests and suggestions are welcome. Reach out before introducing large structural changes to align on direction.***

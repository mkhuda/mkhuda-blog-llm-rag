# === Builder Stage ===
# This stage installs dependencies, pre-builds the FAISS index, and creates a virtual env.
# --- FIX: Use 'bullseye' (Debian 11 Stable) to fix compatibility with old Docker versions ---
FROM python:3.12-slim-bullseye AS builder
WORKDIR /app

# --- FIX: Add apt.conf.d fix for "Post-Invoke" error ---
# This disables the problematic script that fails on some Docker versions.
RUN echo 'APT::Update::Post-Invoke "true";' > /etc/apt/apt.conf.d/99-no-post-invoke

# Install system dependencies needed for building Python packages
# --- FIX: Separate apt-get commands to isolate the issue ---
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential libgomp1
RUN rm -rf /var/lib/apt/lists/*

# Install uv, a fast python package manager
RUN pip install --upgrade pip uv --no-cache-dir

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Create a virtual environment and install ALL dependencies from the lockfile
# This ensures the builder script has access to necessary libraries
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
RUN uv sync --frozen --no-dev --no-progress

# Copy all application and builder source code
COPY app ./app
COPY builder ./builder
COPY utils ./utils

# ---> PRE-BUILD THE FAISS INDEX <---
# This is a key optimization. The index is built once and baked into the image.
# We mount the .env file as a secret to provide the necessary API key.
# --- FIX: Point to the correct builder script 'rag_faiss_builder.py' ---
RUN --mount=type=secret,id=dotenv,target=.env \
    echo "--- Building FAISS index during Docker image creation ---" && \
    python builder/rag_faiss_builder.py && \
    echo "--- FAISS index successfully built ---"

# === Runtime Stage ===
# This is the final, lean image that will be deployed.
# --- FIX: Use 'bullseye' (Debian 11 Stable) on the final stage as well ---
FROM python:3.12-slim-bullseye
WORKDIR /app

# --- FIX: Add apt.conf.d fix for "Post-Invoke" error ---
# This disables the problematic script that fails on some Docker versions.
RUN echo 'APT::Update::Post-Invoke "true";' > /etc/apt/apt.conf.d/99-no-post-invoke

# Install only the necessary runtime system dependencies
# --- FIX: Separate apt-get commands to isolate the issue ---
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgomp1
RUN rm -rf /var/lib/apt/lists/*

# Copy the virtual environment with all dependencies from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
# --- FIX: Corrected typo '--from-builder' to '--from=builder' ---
COPY --from=builder /app/app ./app
COPY --from=builder /app/utils ./utils

# ---> COPY THE PRE-BUILT INDEX <---
# Bring the generated index from the builder stage into our final image.
# --- FIX: Corrected typo 'faISS_index' to 'faiss_index' ---
COPY --from=builder /app/mkhuda_faiss_index ./mkhuda_faiss_index

# Set environment variables so the app uses the virtual environment
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV=/app/.venv

# --- FIX: Force Python to run unbuffered so logs appear immediately ---
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ---> THE CORRECTED AND FINAL COMMAND <---
# --- FIX: Added --access-logfile and --error-logfile flags to stream logs ---
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.rag_fastapi:app", \
    "--workers", "2", \
    "--bind", "0.0.0.0:8000", \
    "--timeout", "60", \
    "--keep-alive", "5", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]


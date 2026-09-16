# Dockerfile — packages the backend into a single image.
#
# Build:  docker build -t askpage-backend .
# Run:    docker run -p 8000:8000 --env-file .env askpage-backend

FROM python:3.11-slim

WORKDIR /app

# Install the CPU-only build of PyTorch first. The default build pulls in
# ~2 GB of CUDA libraries we don't need on a CPU server.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model and the reranker during the build so the
# container starts instantly instead of downloading ~210 MB on first request.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

EXPOSE 8000

# --host 0.0.0.0 makes the server reachable from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── gpu_tts_service/Dockerfile ─────────────────────────────────────────
# Use CUDA‑enabled Python base (change to a different tag if your host
# GPU needs CUDA 11.* instead of 12.*)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps ─ ffmpeg for audio I/O, git for pulling the model repo
RUN apt-get update && \
    apt-get install -y git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the FastAPI server code
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
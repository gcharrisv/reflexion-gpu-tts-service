# ─── gpu_tts_service/Dockerfile ──────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------- system deps + Python ------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools \
        git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps ---------------------------------------------
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# ---------- App code ------------------------------------------------
COPY . .

EXPOSE 8000
CMD ["python3", "server.py"]
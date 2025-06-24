# --- gpu_tts_service/Dockerfile -------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------- system deps + Python 3.11 --------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common git ffmpeg curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps ----------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- App code -------------------------------------------------
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]

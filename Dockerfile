# ---------- gpu_tts_service/Dockerfile --------------------------------
# CUDA 11.8 runtime (works with cu118 wheels)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# --- system deps  +  Python 3.10 ---------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common git ffmpeg curl wget unzip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-distutils && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch CUDA wheels ----------------------------------------------
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118

# --- Download OpenVoice checkpoints (≈2 GB) ----------------------------
RUN mkdir -p /root/.cache/openvoice/checkpoints && \
    wget -qO /tmp/en.zip \
        https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/base_speakers/EN.zip && \
    unzip -q /tmp/en.zip -d /root/.cache/openvoice/checkpoints && rm /tmp/en.zip && \
    wget -qO /tmp/conv.zip \
        https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter.zip && \
    unzip -q /tmp/conv.zip -d /root/.cache/openvoice/checkpoints && rm /tmp/conv.zip

# --- Python deps -------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ----------------------------------------------------------
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
# ---------- gpu_tts_service/Dockerfile --------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# --- system deps + Python 3.10 ----------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common git git-lfs ffmpeg curl wget unzip && \
    git lfs install && \
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

# --- Clone OpenVoice checkpoints (≈2 GB) -------------------------------
RUN mkdir -p /tmp && \
    git clone --depth 1 https://huggingface.co/myshell-ai/OpenVoice /tmp/ov && \
    mkdir -p /root/.cache/openvoice && \
    cp -r /tmp/ov/checkpoints /root/.cache/openvoice && \
    rm -rf /tmp/ov

# --- Python deps -------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code ----------------------------------------------------------
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]

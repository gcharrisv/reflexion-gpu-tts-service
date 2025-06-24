FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ---- system deps + Python 3.11 ----
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

# ---- install PyTorch CUDA wheel first ----
RUN pip install --no-cache-dir \
      torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu128

# ---- the rest of the deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- copy app code ----
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]

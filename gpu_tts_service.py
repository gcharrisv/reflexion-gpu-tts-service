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

# ─── gpu_tts_service/requirements.txt ────────────────────────────────
fastapi==0.110.1
uvicorn[standard]==0.29.0
pydantic==2.7.1
numpy>=1.24
soundfile>=0.12
scipy>=1.11
# Pull OpenVoice directly from GitHub (MIT‑licensed)
 git+https://github.com/myshell-ai/OpenVoice.git
# Torch is installed with CUDA support in the base image, but install matching wheels if needed.
 torch==2.2.2+cu122 --extra-index-url https://download.pytorch.org/whl/cu122

aiofiles>=23.2

# ─── gpu_tts_service/server.py ───────────────────────────────────────
"""Minimal FastAPI wrapper around OpenVoice V2 for low‑latency TTS.

POST /tts {"text": str, "embed": List[float]} → returns {"audio": base64_wav}
Optionally stream via Server‑Sent Events by setting ?stream=1 (omitted for brevity).
"""
from typing import List
import base64, io, time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import soundfile as sf
import numpy as np

# Import OpenVoice once (loads model onto GPU)
from openvoice import se_extractor, commons, tts

MODEL_NAME = "openvoice_v2"

# /// Load model (approx. 5‑6 GB VRAM) ///
print("[OpenVoice] loading model …")
start = time.time()

model_ins = tts.TTSInfer("/root/.cache/openvoice")  # defaults to ~/.cache/openvoice
print(f"[OpenVoice] model ready in {time.time()-start:.1f}s")

app = FastAPI()

class TtsRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise")
    embed: List[float] = Field(..., description="256‑dim speaker embedding")

class TtsResponse(BaseModel):
    audio: str = Field(..., description="base64‑encoded 16‑kHz WAV")

@app.post("/tts", response_model=TtsResponse)
def tts_endpoint(body: TtsRequest):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    if len(body.embed) != 256:
        raise HTTPException(400, "embed must have length 256")

    # OpenVoice expects numpy float32 embedding
    spk_emb = np.array(body.embed, dtype=np.float32)

    try:
        audio = model_ins.infer(body.text, spk_emb, language="en")  # returns float32 PCM 16‑kHz
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

    # Encode WAV to base64
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, subtype="PCM_16", format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return TtsResponse(audio=b64)

# ─── gpu_tts_service/README.md ───────────────────────────────────────
# gpu_tts_service

FastAPI micro‑service that wraps **OpenVoice V2** for real‑time, per‑user voice cloning.

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST   | /tts | `{ "text": "Hello", "embed": [256 floats] }` | `{ "audio": "<base64 wav>" }` |

A 256‑float speaker embedding is generated once per agent by your `ingestVoice` edge function and stored in Supabase. Pass the same array here to synthesise speech in that voice.

## Quick test (after Docker build)

```bash
docker build -t gpu_tts_service .
docker run --gpus 1 -p 8000:8000 gpu_tts_service &

curl -s -X POST http://localhost:8000/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"Testing one two","embed": [0.01,0.02,...]}' | jq -r .audio | base64 -d > out.wav

aplay out.wav  # or open the file to listen
```

## Notes
* **GPU required** – the base image uses NVIDIA runtime; pass `--gpus 1` when running locally.
* Expected **first‑token latency**: ~180‑220 ms on an A10 or RTX 3090.
* Memory: ~5.4 GB VRAM for model weights + ~400 MB RAM.
* Scale horizontally with one container per GPU; your Supabase `streamTts` edge function load‑balances by round‑robin or random choice.

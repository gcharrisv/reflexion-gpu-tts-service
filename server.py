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
from openvoice import se_extractor, commons
from openvoice.tts import TTSInfer

MODEL_NAME = "openvoice_v2"

# /// Load model (approx. 5‑6 GB VRAM) ///
print("[OpenVoice] loading model …")
start = time.time()

model_ins = TTSInfer("/root/.cache/openvoice")  # defaults to ~/.cache/openvoice
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
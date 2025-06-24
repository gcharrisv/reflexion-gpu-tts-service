# ─── gpu_tts_service/server.py ──────────────────────────────────────
"""
FastAPI wrapper around OpenVoice V2.

POST /tts
Body  : { "text": str, "embed": List[float] }  # 256-float speaker embedding
Return: { "audio": base64_wav }                # 16-kHz mono, PCM-16
"""
from __future__ import annotations
from typing import List
import base64, io, os, tempfile, time, pathlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf
import torch

# -------------------------------------------------------------------
# 1.  Try the one-step wrapper first …
# -------------------------------------------------------------------
try:
    from openvoice.api import TTS as _TTS
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _CKPT = pathlib.Path(os.getenv("OPENVOICE_HOME", "~/.cache/openvoice")).expanduser()
    print("[OpenVoice] using wrapper openvoice.api.TTS")
    _model = _TTS(checkpoint_root=str(_CKPT), device=_DEVICE)

    def _infer(text: str, spk_emb: np.ndarray) -> np.ndarray:
        return _model.infer(text, spk_emb, language="en")  # float32 PCM 16 kHz

# -------------------------------------------------------------------
# 2.  …otherwise build the two-stage pipeline once.
# -------------------------------------------------------------------
except ImportError:
    print("[OpenVoice] wrapper absent → falling back to BaseSpeakerTTS + ToneColorConverter")
    from openvoice.api import BaseSpeakerTTS, ToneColorConverter
    from openvoice import se_extractor

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _ROOT = pathlib.Path(os.getenv("OPENVOICE_HOME", "~/.cache/openvoice")).expanduser()
    _BASE = _ROOT / "base_speakers" / "EN"
    _CONV = _ROOT / "converter"

    _base_tts = BaseSpeakerTTS(_BASE / "config.json", device=_DEVICE)
    _base_tts.load_ckpt(_BASE / "checkpoint.pth")

    _conv = ToneColorConverter(_CONV / "config.json", device=_DEVICE)
    _conv.load_ckpt(_CONV / "checkpoint.pth")

    _src_se = torch.load(_BASE / "en_default_se.pth").to(_DEVICE)

    def _infer(text: str, spk_emb: np.ndarray) -> np.ndarray:
        """Neutral TTS ➜ tone-color transfer ➜ float32 PCM 16 kHz in RAM"""
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_neutral, \
             tempfile.NamedTemporaryFile(suffix=".wav") as tmp_final:

            # stage 1: base speaker
            _base_tts.tts(text, tmp_neutral.name,
                          speaker="default", language="English", speed=1.0)

            # stage 2: colour transfer
            _conv.convert(
                audio_src_path=tmp_neutral.name,
                src_se=_src_se,
                tgt_se=torch.from_numpy(spk_emb).to(_DEVICE),
                output_path=tmp_final.name,
                message="@Reflexion")                      # optional watermark

            audio, _ = sf.read(tmp_final.name, dtype="float32")
            return audio

# -------------------------------------------------------------------
# 3.  API setup
# -------------------------------------------------------------------
print("[OpenVoice] model ready")

app = FastAPI()

class TtsRequest(BaseModel):
    text : str          = Field(..., description="Text to synthesise")
    embed: List[float]  = Field(..., description="256-float speaker embedding")

class TtsResponse(BaseModel):
    audio: str = Field(..., description="base64-encoded 16-kHz WAV")

@app.post("/tts", response_model=TtsResponse)
def tts_endpoint(body: TtsRequest):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    if len(body.embed) != 256:
        raise HTTPException(400, "embed must have length 256")

    spk_emb = np.asarray(body.embed, dtype=np.float32)

    try:
        audio = _infer(body.text, spk_emb)           # float32 PCM 16 kHz
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

    buf = io.BytesIO()
    sf.write(buf, audio, 16000, subtype="PCM_16", format="WAV")
    return TtsResponse(audio=base64.b64encode(buf.getvalue()).decode())

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
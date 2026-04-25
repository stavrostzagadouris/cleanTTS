import asyncio
import base64
import io
import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("cleanTTS")

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")  # fallback only
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_bella")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Keep responses short and natural for speech.",
)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

KOKORO_SAMPLE_RATE = 24000


def _parse_backends() -> list[dict]:
    raw = os.getenv("LLM_BACKENDS", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            out = []
            for item in data:
                if isinstance(item, dict) and item.get("name") and item.get("url"):
                    out.append({"name": str(item["name"]), "url": str(item["url"]).rstrip("/")})
            if out:
                return out
        except json.JSONDecodeError as e:
            log.warning("LLM_BACKENDS JSON could not be parsed: %s", e)
    return [{"name": "default", "url": LLM_BASE_URL.rstrip("/")}]


LLM_BACKENDS = _parse_backends()


def _find_backend(name: str | None) -> dict:
    if name:
        for b in LLM_BACKENDS:
            if b["name"] == name:
                return b
    return LLM_BACKENDS[0]

# Curated set of well-known Kokoro voices. Users can pass any voice id Kokoro
# supports; this list is just for the UI dropdown and /v1/voices discovery.
KOKORO_VOICES_BY_LANG = {
    "a": [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "b": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Whisper: %s on %s", WHISPER_MODEL, WHISPER_DEVICE)
    compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    state["whisper"] = WhisperModel(
        WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute_type
    )
    log.info("Whisper loaded.")

    log.info("Loading Kokoro: lang=%s", KOKORO_LANG)
    state["kokoro"] = KPipeline(lang_code=KOKORO_LANG)
    log.info("Kokoro loaded.")

    yield

    state.clear()


app = FastAPI(title="cleanTTS", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Run Kokoro and return WAV bytes (mono PCM_16 @ 24kHz)."""
    pipeline: KPipeline = state["kokoro"]
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        elif hasattr(audio, "numpy"):
            audio = audio.numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return b""
    wav = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=KOKORO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# OpenAI-compatible TTS API (for external tools, e.g. the GT race app)
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize.")
    model: str = Field("kokoro", description="Ignored; we always use Kokoro-82M.")
    voice: str = Field(DEFAULT_VOICE, description="Voice id, e.g. af_bella.")
    response_format: str = Field("wav", description="Only 'wav' is supported.")
    speed: float = Field(1.0, ge=0.25, le=4.0)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.response_format.lower() != "wav":
        raise HTTPException(400, f"response_format '{req.response_format}' not supported. Use 'wav'.")
    if not req.input.strip():
        raise HTTPException(400, "input must not be empty.")
    try:
        wav_bytes = await asyncio.to_thread(synthesize, req.input, req.voice, req.speed)
    except Exception as e:
        log.exception("TTS synthesis failed")
        raise HTTPException(500, f"TTS failed: {e}")
    if not wav_bytes:
        raise HTTPException(500, "TTS produced no audio (Kokoro returned no chunks).")
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "kokoro", "object": "model", "owned_by": "hexgrad"}],
    }


@app.get("/v1/voices")
async def list_voices():
    voices = KOKORO_VOICES_BY_LANG.get(KOKORO_LANG, [])
    return {"voices": voices, "default": DEFAULT_VOICE, "lang": KOKORO_LANG}


# ---------------------------------------------------------------------------
# Voice assistant: UI + WebSocket pipeline
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/llm-backends")
async def llm_backends():
    return {
        "backends": [{"name": b["name"]} for b in LLM_BACKENDS],
        "default": LLM_BACKENDS[0]["name"],
    }


@app.get("/api/llm-models")
async def llm_models(backend: str | None = None):
    b = _find_backend(backend)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{b['url']}/models")
            r.raise_for_status()
            data = r.json()
            return {"models": [m["id"] for m in data.get("data", [])], "backend": b["name"]}
    except Exception as e:
        log.warning("LLM model list failed (%s): %s", b["name"], e)
        return {"models": [], "backend": b["name"], "error": str(e)}


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(\s+|$)")


def drain_sentences(buf: str) -> tuple[list[str], str]:
    """Pull complete sentences from buf; return (sentences, leftover)."""
    sentences: list[str] = []
    while True:
        m = SENTENCE_BOUNDARY.search(buf)
        if not m:
            break
        end = m.end()
        s = buf[:end].strip()
        if s:
            sentences.append(s)
        buf = buf[end:]
    return sentences, buf


async def transcribe_bytes(audio_bytes: bytes) -> str:
    """Write to a temp file (so ffmpeg can decode webm/ogg/etc) and run Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        whisper: WhisperModel = state["whisper"]
        segments, _ = await asyncio.to_thread(
            whisper.transcribe, path, beam_size=5
        )
        return "".join(s.text for s in segments).strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    log.info("WS connected")
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def send(payload: dict):
        await socket.send_text(json.dumps(payload))

    try:
        while True:
            raw = await socket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                await send({"type": "reset_ack"})
                continue

            if "audio" not in msg:
                continue

            voice = msg.get("voice") or DEFAULT_VOICE
            llm_model = msg.get("llm_model")
            backend = _find_backend(msg.get("llm_backend"))
            if not llm_model:
                await send({"type": "error", "message": "Pick an LLM model first."})
                continue

            # 1. STT
            try:
                audio_bytes = base64.b64decode(msg["audio"])
                transcription = await transcribe_bytes(audio_bytes)
            except Exception as e:
                log.exception("Whisper failed")
                await send({"type": "error", "message": f"Whisper failed: {e}"})
                continue

            if not transcription:
                await send({"type": "transcription", "text": "(no speech detected)"})
                continue

            await send({"type": "transcription", "text": transcription})
            history.append({"role": "user", "content": transcription})

            # 2. LLM stream + sentence-chunked TTS
            tts_q: asyncio.Queue = asyncio.Queue()

            async def tts_worker():
                while True:
                    sentence = await tts_q.get()
                    if sentence is None:
                        return
                    try:
                        wav = await asyncio.to_thread(synthesize, sentence, voice)
                        if wav:
                            await send({
                                "type": "tts_audio",
                                "audio": base64.b64encode(wav).decode("ascii"),
                            })
                    except Exception as e:
                        log.exception("TTS chunk failed")
                        await send({"type": "error", "message": f"TTS chunk failed: {e}"})

            tts_task = asyncio.create_task(tts_worker())

            full_response = ""
            buf = ""
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    payload = {
                        "model": llm_model,
                        "messages": history,
                        "stream": True,
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    }
                    async with client.stream(
                        "POST", f"{backend['url']}/chat/completions", json=payload
                    ) as r:
                        r.raise_for_status()
                        async for line in r.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                obj = json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}

                            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            if reasoning:
                                await send({"type": "llm_chunk", "text": reasoning, "is_reasoning": True})

                            text_piece = delta.get("content") or ""
                            if not text_piece:
                                continue

                            full_response += text_piece
                            buf += text_piece
                            await send({"type": "llm_chunk", "text": text_piece, "is_reasoning": False})

                            sentences, buf = drain_sentences(buf)
                            for s in sentences:
                                await tts_q.put(s)

                if buf.strip():
                    await tts_q.put(buf.strip())
            except Exception as e:
                log.exception("LLM call failed")
                await send({"type": "error", "message": f"LLM call failed: {e}"})

            await tts_q.put(None)
            await tts_task

            if full_response:
                history.append({"role": "assistant", "content": full_response})
            await send({"type": "llm_done"})

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception:
        log.exception("WS handler error")
        try:
            await socket.send_text(json.dumps({"type": "error", "message": "Server error; see logs."}))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")

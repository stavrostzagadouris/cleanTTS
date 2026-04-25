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


def _parse_langs() -> list[str]:
    """KOKORO_LANGS=a,b loads multiple pipelines (one per language). Falls back
    to KOKORO_LANG (singular) for backward compat."""
    plural = os.getenv("KOKORO_LANGS", "").strip()
    if plural:
        return [c.strip() for c in plural.split(",") if c.strip()]
    return [os.getenv("KOKORO_LANG", "a").strip()]


KOKORO_LANGS = _parse_langs()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_bella")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Replies are spoken aloud, so keep them "
    "short and natural for speech (no markdown, no bullet lists, no headers). Input is "
    "transcribed from the user's voice and may contain occasional mis-heard words or odd "
    "artifacts — if something looks like a transcription glitch, infer what they likely "
    "meant when context makes it obvious, otherwise ask a brief clarifying question "
    "rather than answering the literal nonsense.",
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
                    backend = {
                        "name": str(item["name"]),
                        "url": str(item["url"]).rstrip("/"),
                    }
                    # Optional: chat_template_kwargs forwarded into each
                    # /v1/chat/completions request for this backend. Use this to
                    # disable thinking on Qwen3 ({"enable_thinking": false}), etc.
                    ctk = item.get("chat_template_kwargs")
                    if isinstance(ctk, dict):
                        backend["chat_template_kwargs"] = ctk
                    out.append(backend)
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

# Full Kokoro v1 voice catalog, keyed by lang_code. Source: hexgrad/Kokoro-82M.
# Only the entries listed in KOKORO_LANGS get loaded into VRAM at startup.
KOKORO_VOICES_BY_LANG = {
    "a": [  # American English
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "b": [  # British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "e": ["ef_dora", "em_alex", "em_santa"],                   # Spanish
    "f": ["ff_siwis"],                                          # French
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],         # Hindi
    "i": ["if_sara", "im_nicola"],                              # Italian
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi",
          "jf_tebukuro", "jm_kumo"],                            # Japanese
    "p": ["pf_dora", "pm_alex", "pm_santa"],                    # Brazilian Portuguese
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
          "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"], # Mandarin
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

    log.info("Loading Kokoro pipelines: langs=%s", KOKORO_LANGS)
    state["kokoro"] = {}
    for lang in KOKORO_LANGS:
        state["kokoro"][lang] = KPipeline(lang_code=lang)
    log.info("Kokoro loaded: %d pipeline(s) — %s.", len(state["kokoro"]), ",".join(KOKORO_LANGS))

    yield

    state.clear()


app = FastAPI(title="cleanTTS", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Run Kokoro and return WAV bytes (mono PCM_16 @ 24kHz). Picks the right
    pipeline based on the voice's lang_code prefix (e.g. 'bf_emma' → 'b')."""
    pipelines: dict[str, KPipeline] = state["kokoro"]
    lang = voice[0] if voice else KOKORO_LANGS[0]
    pipeline = pipelines.get(lang)
    if pipeline is None:
        raise ValueError(
            f"Voice {voice!r} requires KOKORO_LANGS to include {lang!r}; "
            f"loaded: {sorted(pipelines.keys())}"
        )
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
    except ValueError as e:
        raise HTTPException(400, str(e))
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
    voices: list[str] = []
    for lang in KOKORO_LANGS:
        voices.extend(KOKORO_VOICES_BY_LANG.get(lang, []))
    return {"voices": voices, "default": DEFAULT_VOICE, "langs": KOKORO_LANGS}


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


async def run_pipeline(msg: dict, history: list[dict], send) -> None:
    """One full STT → LLM → TTS turn. Cancellable: on CancelledError we save
    whatever assistant text we already streamed to history, then re-raise."""
    voice = msg.get("voice") or DEFAULT_VOICE
    llm_model = msg.get("llm_model")
    backend = _find_backend(msg.get("llm_backend"))

    if not llm_model:
        await send({"type": "error", "message": "Pick an LLM model first."})
        return

    # 1. STT
    try:
        audio_bytes = base64.b64decode(msg["audio"])
        transcription = await transcribe_bytes(audio_bytes)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.exception("Whisper failed")
        await send({"type": "error", "message": f"Whisper failed: {e}"})
        return

    if not transcription:
        await send({"type": "transcription", "text": "(no speech detected)"})
        await send({"type": "llm_done"})
        return

    await send({"type": "transcription", "text": transcription})
    history.append({"role": "user", "content": transcription})

    # 2. LLM stream + sentence-chunked TTS
    tts_q: asyncio.Queue = asyncio.Queue()
    full_response = ""

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
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.exception("TTS chunk failed")
                await send({"type": "error", "message": f"TTS chunk failed: {e}"})

    tts_task = asyncio.create_task(tts_worker())
    cancelled = False

    try:
        buf = ""
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {
                "model": llm_model,
                "messages": history,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            if backend.get("chat_template_kwargs"):
                payload["chat_template_kwargs"] = backend["chat_template_kwargs"]
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
    except asyncio.CancelledError:
        cancelled = True
        log.info("Pipeline cancelled (barge-in or new turn)")
        raise
    except Exception as e:
        log.exception("LLM call failed")
        try:
            await send({"type": "error", "message": f"LLM call failed: {e}"})
        except Exception:
            pass
    finally:
        # Always shut down the TTS worker so we don't leak tasks
        await tts_q.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(tts_task), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            tts_task.cancel()

        # Save whatever assistant text we generated (full or partial) so the
        # LLM remembers what it had said before being interrupted.
        if full_response.strip():
            history.append({"role": "assistant", "content": full_response.strip()})

        if cancelled:
            try:
                await send({"type": "cancelled"})
            except Exception:
                pass
        else:
            try:
                await send({"type": "llm_done"})
            except Exception:
                pass


async def _cancel_and_wait(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    log.info("WS connected")
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    pipeline_task: asyncio.Task | None = None

    async def send(payload: dict):
        await socket.send_text(json.dumps(payload))

    try:
        while True:
            raw = await socket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "reset":
                await _cancel_and_wait(pipeline_task)
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                await send({"type": "reset_ack"})
                continue

            if mtype == "cancel":
                await _cancel_and_wait(pipeline_task)
                continue

            if "audio" not in msg:
                continue

            # New turn: cancel any in-flight pipeline (barge-in or just sequential turns).
            await _cancel_and_wait(pipeline_task)
            pipeline_task = asyncio.create_task(run_pipeline(msg, history, send))

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception:
        log.exception("WS handler error")
        try:
            await socket.send_text(json.dumps({"type": "error", "message": "Server error; see logs."}))
        except Exception:
            pass
    finally:
        await _cancel_and_wait(pipeline_task)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")

import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
from contextlib import asynccontextmanager

import httpx
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
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
# beam_size=1 (greedy) is ~2x faster and near-identical quality on distil
# models; 5 is the faster-whisper default and safest for accuracy.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
# Silero VAD pre-pass. Without it, faster-whisper reports every segment as
# starting at 0.00 when the clip has leading silence (measured up to 20s off),
# and it hallucinates a phrase — "you", "Thank you." — out of speech-free
# audio. With it, onsets land within 0.1s and speech-free audio yields no
# segments. Real speech is unaffected down to at least -50dB, and it costs
# ~7ms. Set WHISPER_VAD_FILTER=false for the pre-0.2 behaviour.
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").strip().lower() not in (
    "0", "false", "no", "off",
)


def _auto_device() -> str:
    """Pick the best available accelerator. Prefer CUDA, then MPS, then CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_AUTO_DEVICE = _auto_device()

# Whisper: faster-whisper / CTranslate2 has no MPS backend. If auto-detect
# returns mps (Apple Silicon) we silently downgrade Whisper to CPU; Kokoro
# can still use MPS independently. Explicit "mps" is also remapped, with a
# warning so the user knows.
_w_raw = os.getenv("WHISPER_DEVICE", "auto").lower()
if _w_raw == "auto":
    WHISPER_DEVICE = "cpu" if _AUTO_DEVICE == "mps" else _AUTO_DEVICE
elif _w_raw == "mps":
    log.warning("faster-whisper has no MPS backend; using CPU for Whisper instead.")
    WHISPER_DEVICE = "cpu"
else:
    WHISPER_DEVICE = _w_raw

# Kokoro: PyTorch-based, supports cuda/mps/cpu directly.
_k_raw = os.getenv("KOKORO_DEVICE", "auto").lower()
KOKORO_DEVICE = _AUTO_DEVICE if _k_raw == "auto" else _k_raw

# Diarization (pyannote): plain PyTorch, so unlike Whisper it runs on MPS. On
# Apple Silicon this is the one stage the GPU genuinely accelerates, so "auto"
# is left at the raw auto-detect (no CPU downgrade).
_d_raw = os.getenv("DIARIZE_DEVICE", "auto").lower()
DIARIZE_DEVICE = _AUTO_DEVICE if _d_raw == "auto" else _d_raw
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
# The pyannote model is gated: the token's HF account must have accepted its
# terms. Falls back to whatever `huggingface-cli login` cached.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


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

    log.info("Loading Kokoro pipelines: langs=%s on %s", KOKORO_LANGS, KOKORO_DEVICE)
    state["kokoro"] = {}
    for lang in KOKORO_LANGS:
        state["kokoro"][lang] = KPipeline(lang_code=lang, device=KOKORO_DEVICE)
    log.info("Kokoro loaded: %d pipeline(s) — %s.", len(state["kokoro"]), ",".join(KOKORO_LANGS))

    # One shared client for all outbound LLM traffic: keeps connections alive
    # between turns instead of a TCP handshake per request. Timeouts are set
    # per-request at the call sites.
    state["http"] = httpx.AsyncClient()

    yield

    await state["http"].aclose()
    state.clear()


app = FastAPI(title="cleanTTS", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    """Parse a voice spec into (voice_id, weight) pairs.

    Supports blending:
      "af_bella"                     -> [("af_bella", 1.0)]              single
      "af_bella,af_jessica"          -> equal blend (both weight 1.0)
      "af_bella*0.7,af_jessica*0.3"  -> weighted blend

    A voice with no '*weight' suffix defaults to 1.0, so an unweighted spec
    behaves exactly as before. Weights are relative — they're normalized to
    sum to 1 at blend time, so '2,1' means the same as '0.67,0.33'.
    """
    pairs: list[tuple[str, float]] = []
    for piece in (spec or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        name, sep, w = piece.partition("*")
        name = name.strip()
        if not name:
            continue
        if sep:
            try:
                weight = float(w)
            except ValueError:
                raise ValueError(
                    f"Invalid weight in {piece!r}; expected e.g. 'af_bella*0.7'."
                )
            if weight < 0:
                raise ValueError(f"Voice weight must be >= 0 (got {weight!r} in {piece!r}).")
        else:
            weight = 1.0
        pairs.append((name, weight))
    if not pairs:
        raise ValueError("No voice specified.")
    return pairs


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Run Kokoro and return WAV bytes (mono PCM_16 @ 24kHz).

    `voice` is a spec that supports blending (see parse_voice_spec):
      - "af_bella"                      single voice
      - "af_bella,af_jessica"           equal 50/50 blend
      - "af_bella*0.7,af_jessica*0.3"   weighted blend (normalized)

    The pipeline only phonemizes the *text*, so it's selected from the FIRST
    voice's lang_code prefix (e.g. 'bf_emma' → 'b'), and that language must be
    loaded. Voice embeddings themselves are language-agnostic tensors of
    identical shape, so a blend can freely mix voices across languages
    (e.g. 'am_adam,bm_lewis') — Kokoro just averages the style vectors. We
    don't pre-validate voice ids; an unknown one surfaces as Kokoro's own
    error rather than a guess here.
    """
    pipelines: dict[str, KPipeline] = state["kokoro"]
    pairs = parse_voice_spec(voice)

    lang = pairs[0][0][0]
    pipeline = pipelines.get(lang)
    if pipeline is None:
        raise ValueError(
            f"Voice {pairs[0][0]!r} requires KOKORO_LANGS to include {lang!r}; "
            f"loaded: {sorted(pipelines.keys())}"
        )

    # A single unweighted voice is passed through as its id (Kokoro handles it
    # normally). Anything else is a blend: load each voice's full embedding
    # pack and combine into a normalized weighted-mean tensor, which the
    # pipeline accepts directly in place of a voice id.
    if len(pairs) == 1 and pairs[0][1] == 1.0:
        voice_arg: object = pairs[0][0]
    else:
        total = sum(w for _, w in pairs)
        if total <= 0:
            raise ValueError("Voice weights sum to zero; nothing to blend.")
        packs = [pipeline.load_single_voice(name) for name, _ in pairs]
        voice_arg = torch.stack(
            [pack * (w / total) for pack, (_, w) in zip(packs, pairs)]
        ).sum(dim=0)

    chunks = []
    for _, _, audio in pipeline(text, voice=voice_arg, speed=speed):
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

# Maps OpenAI response_format → (Content-Type, ffmpeg args).
# ffmpeg args is None for native WAV (no transcode). For everything else we
# pipe Kokoro's WAV through ffmpeg. -c:a is set explicitly where the muxer
# would otherwise pick a wrong default (e.g. ogg → vorbis instead of opus).
SUPPORTED_FORMATS: dict[str, tuple[str, list[str] | None]] = {
    "wav":  ("audio/wav",  None),
    "opus": ("audio/ogg",  ["-c:a", "libopus", "-f", "ogg"]),
    "mp3":  ("audio/mpeg", ["-c:a", "libmp3lame", "-f", "mp3"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "aac":  ("audio/aac",  ["-c:a", "aac", "-f", "adts"]),
    "pcm":  ("audio/pcm",  ["-f", "s16le"]),  # raw 16-bit LE @ 24kHz mono
}


def transcode(wav_bytes: bytes, ffmpeg_args: list[str]) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         *ffmpeg_args, "pipe:1"],
        input=wav_bytes, capture_output=True, check=True,
    )
    return proc.stdout


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize.")
    model: str = Field("kokoro", description="Ignored; we always use Kokoro-82M.")
    voice: str = Field(
        DEFAULT_VOICE,
        description=(
            "Voice id (e.g. af_bella), or a blend: 'af_bella,af_jessica' for an "
            "equal mix, or 'af_bella*0.7,af_jessica*0.3' for a weighted one. "
            "Voices may be mixed across languages; the first voice's language "
            "must be loaded (it phonemizes the text)."
        ),
    )
    response_format: str = Field("wav", description="wav | opus | mp3 | flac | aac | pcm")
    speed: float = Field(1.0, ge=0.25, le=4.0)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    fmt = req.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"response_format '{req.response_format}' not supported. "
            f"Use one of: {', '.join(SUPPORTED_FORMATS)}.",
        )
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

    media_type, ffmpeg_args = SUPPORTED_FORMATS[fmt]
    if ffmpeg_args is None:
        return Response(content=wav_bytes, media_type=media_type)

    try:
        encoded = await asyncio.to_thread(transcode, wav_bytes, ffmpeg_args)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg transcode to %s failed: %s", fmt, e.stderr.decode(errors="ignore"))
        raise HTTPException(500, f"transcode to {fmt} failed")
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not installed; required for non-WAV formats.")

    return Response(content=encoded, media_type=media_type)


def segments_text(segments: list) -> str:
    """Join Whisper segments into one transcript, as the /ws pipeline expects."""
    return "".join(s.text for s in segments).strip()


def verbose_payload(segments: list, info) -> dict:
    """Render (segments, info) in OpenAI's verbose_json shape.

    The scores are the point of this format, not decoration: clients filter
    Whisper hallucinations on no_speech_prob (background room speech scores
    ~0.67 where real speech scores ~0.12) and gauge confidence on avg_logprob.
    """
    return {
        "task": "transcribe",
        "language": info.language,
        "duration": round(info.duration, 3),
        "text": segments_text(segments),
        "segments": [
            # id is renumbered from 0 to match OpenAI; faster-whisper's own
            # segment ids start at 1.
            {
                "id": i,
                "seek": s.seek,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "tokens": list(s.tokens),
                "temperature": s.temperature,
                "avg_logprob": s.avg_logprob,
                "compression_ratio": s.compression_ratio,
                "no_speech_prob": s.no_speech_prob,
            }
            for i, s in enumerate(segments)
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    vad_filter: bool | None = Form(None),
):
    """OpenAI-compatible Whisper STT: multipart/form-data in, JSON {text} out.

    response_format: "json" (default) | "text" | "verbose_json". The default
    stays exactly {"text": ...}; verbose_json additionally exposes per-segment
    timings and scores. Anything unrecognised still falls back to the default,
    as before.

    Note `model` is accepted for OpenAI compatibility but ignored — the server
    loads WHISPER_MODEL once at startup and serves every request from it.

    vad_filter overrides WHISPER_VAD_FILTER for this request. Leave it unset
    unless you specifically want the other behaviour: with VAD off, segment
    timings collapse to 0.00 on clips with leading silence.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "file must not be empty.")

    fmt = (response_format or "json").lower()
    try:
        segments, info = await transcribe_segments(
            audio_bytes, language=language, initial_prompt=prompt,
            vad_filter=vad_filter,
        )
    except Exception as e:
        log.exception("STT transcription failed")
        raise HTTPException(500, f"transcription failed: {e}")

    if fmt == "text":
        return Response(content=segments_text(segments), media_type="text/plain")
    if fmt == "verbose_json":
        return verbose_payload(segments, info)
    return {"text": segments_text(segments)}


# ---------------------------------------------------------------------------
# Speaker diarization (pyannote) — "who spoke when"
# ---------------------------------------------------------------------------

# pyannote's models expect 16 kHz; diarization is speaker-level, so a mono
# downmix is what we want regardless of how many channels came in.
DIARIZE_SAMPLE_RATE = 16000

# Serializes the one-time lazy load so two simultaneous first requests don't
# both pull the model. Deliberately NOT shared with the Whisper path — the /ws
# voice pipeline never waits on this.
_diarize_load_lock = asyncio.Lock()


def load_diarization_pipeline():
    """Build the pyannote pipeline. Blocking — always call via to_thread."""
    from pyannote.audio import Pipeline as PyannotePipeline

    if not HF_TOKEN:
        raise HTTPException(
            503,
            f"diarization needs an HF token: {DIARIZE_MODEL} is a gated model. "
            "Set HF_TOKEN in .env (the account must have accepted the model's "
            "terms on huggingface.co) and restart.",
        )

    pipeline = PyannotePipeline.from_pretrained(DIARIZE_MODEL, token=HF_TOKEN)
    # from_pretrained returns None instead of raising when the token can't
    # reach the repo, so an unchecked return would fail later as a confusing
    # 'NoneType is not callable'.
    if pipeline is None:
        raise HTTPException(
            503,
            f"could not load {DIARIZE_MODEL}: pyannote returned no pipeline. "
            "Usually the HF token is invalid, or its account hasn't accepted "
            "the model's terms on huggingface.co.",
        )

    pipeline.to(torch.device(DIARIZE_DEVICE))
    return pipeline


async def get_diarization_pipeline():
    """Return the shared pyannote pipeline, loading it on first use.

    Loaded lazily rather than in lifespan: diarization is occasional, and the
    voice bot shouldn't pay for it in startup time or resident memory. A failed
    load isn't cached, so fixing the token and retrying just works.
    """
    pipeline = state.get("diarize")
    if pipeline is not None:
        return pipeline
    async with _diarize_load_lock:
        pipeline = state.get("diarize")
        if pipeline is not None:
            return pipeline
        log.info("Loading diarization: %s on %s", DIARIZE_MODEL, DIARIZE_DEVICE)
        pipeline = await asyncio.to_thread(load_diarization_pipeline)
        state["diarize"] = pipeline
        log.info("Diarization loaded.")
        return pipeline


def decode_mono_f32(audio_bytes: bytes) -> np.ndarray:
    """Decode any ffmpeg-readable audio to mono float32 @ DIARIZE_SAMPLE_RATE.

    We decode here and hand pyannote an in-memory waveform. Given a *path* it
    decodes via torchcodec, which needs FFmpeg's shared libraries installed
    next to the Python package; the ffmpeg CLI we already require for
    /v1/audio/speech is enough for this route.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(DIARIZE_SAMPLE_RATE), "pipe:1"],
        input=audio_bytes, capture_output=True, check=True,
    )
    # frombuffer is read-only and torch needs to own its storage; copy().
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def diarize_bytes(
    pipeline,
    audio_bytes: bytes,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Run diarization on raw audio bytes. Blocking — call via to_thread."""
    wav = decode_mono_f32(audio_bytes)
    if wav.size == 0:
        raise ValueError("no decodable audio in file.")

    # pyannote wants (channel, time).
    waveform = torch.from_numpy(wav).unsqueeze(0)

    # Only forward the hints that were actually given; pyannote picks the
    # speaker count itself otherwise, and rejects bad combinations better than
    # a guess here would.
    hints = {
        k: v for k, v in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        ) if v is not None
    }

    output = pipeline(
        {"waveform": waveform, "sample_rate": DIARIZE_SAMPLE_RATE}, **hints
    )
    # pyannote 4 wraps the result: DiarizeOutput.speaker_diarization. Older
    # pipelines return the Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)

    return [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


@app.post("/v1/audio/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    """Speaker diarization: multipart audio in, {"turns": [...]} out.

    Optional num_speakers pins the count exactly; min_speakers/max_speakers
    bound it. Omit all three to let pyannote decide.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "file must not be empty.")

    pipeline = await get_diarization_pipeline()

    try:
        turns = await asyncio.to_thread(
            diarize_bytes, pipeline, audio_bytes,
            num_speakers, min_speakers, max_speakers,
        )
    except subprocess.CalledProcessError as e:
        log.error("diarize decode failed: %s", e.stderr.decode(errors="ignore"))
        raise HTTPException(400, "could not decode audio file.")
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not installed; required for diarization.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("diarization failed")
        raise HTTPException(500, f"diarization failed: {e}")

    return {"turns": turns}


@app.get("/health")
async def health():
    """Cheap liveness probe for external tools — models are loaded at startup,
    so a 200 here means both Whisper and Kokoro are ready."""
    return {
        "status": "ok",
        "whisper": {
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            # Additive field; "model" and "device" are unchanged.
            "vad_filter": WHISPER_VAD_FILTER,
        },
        "kokoro": {"langs": KOKORO_LANGS, "device": KOKORO_DEVICE},
        # Additive: diarization loads lazily, so "loaded" says whether the
        # first request has paid for it yet. Existing keys are untouched.
        "diarization": {
            "model": DIARIZE_MODEL,
            "device": DIARIZE_DEVICE,
            "loaded": state.get("diarize") is not None,
        },
    }


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
        client: httpx.AsyncClient = state["http"]
        r = await client.get(f"{b['url']}/models", timeout=5.0)
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


async def transcribe_segments(
    audio_bytes: bytes,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool | None = None,
) -> tuple[list, object]:
    """Write to a temp file (so ffmpeg can decode webm/ogg/etc), run Whisper and
    return (segments, info) with every field faster-whisper produced.

    faster-whisper's `transcribe()` only *builds* a lazy generator; the actual
    decode happens while iterating it. So the iteration is drained inside the
    same to_thread call — otherwise the decode runs on the event loop and stalls
    the /ws pipeline sharing it.

    vad_filter=None uses the WHISPER_VAD_FILTER default; pass a bool to override
    it per call.
    """
    use_vad = WHISPER_VAD_FILTER if vad_filter is None else vad_filter
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    def _run() -> tuple[list, object]:
        whisper: WhisperModel = state["whisper"]
        segments, info = whisper.transcribe(
            path, beam_size=WHISPER_BEAM_SIZE,
            language=language, initial_prompt=initial_prompt,
            vad_filter=use_vad,
        )
        return list(segments), info

    try:
        return await asyncio.to_thread(_run)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def transcribe_bytes(
    audio_bytes: bytes,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool | None = None,
) -> str:
    """Plain-text transcription. Contract is `-> str`: this is what the /ws voice
    pipeline and the default /v1/audio/transcriptions response use."""
    segments, _ = await transcribe_segments(
        audio_bytes, language, initial_prompt, vad_filter
    )
    return segments_text(segments)


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
        client: httpx.AsyncClient = state["http"]
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
            "POST", f"{backend['url']}/chat/completions", json=payload,
            timeout=180.0,
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

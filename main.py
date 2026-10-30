import asyncio
import base64
import importlib.util
import io
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("cleanTTS")

load_dotenv()

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
# Names a client may send in the OpenAI `model` field that mean "whatever this
# server is configured for" rather than naming a real checkpoint. OpenAI SDKs
# hardcode "whisper-1", so it must not be taken literally and looked up on the
# Hub. Anything else is passed to faster-whisper as-is — either one of its
# built-in names (tiny, small.en, distil-small.en, large-v3-turbo, turbo, ...)
# or a full HF repo id / local path.
WHISPER_MODEL_ALIASES = {"", "whisper-1", "whisper", "default"}

# Parakeet (NVIDIA FastConformer-TDT) via MLX. Unlike faster-whisper this runs
# on the GPU through Metal, which is why it is ~4.6x faster than distil-small.en
# and ~36x faster than large-v3-turbo on this hardware. Selected per request by
# naming a model containing "parakeet"; "parakeet" alone means PARAKEET_MODEL.
PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
PARAKEET_ALIASES = {"parakeet", "parakeet-tdt", "parakeet-mlx"}

# The model used when a request names none. Defaults to WHISPER_MODEL so
# existing installs are unchanged; set it to "parakeet" to move the default
# onto the MLX engine.
STT_MODEL = os.getenv("STT_MODEL", "").strip() or WHISPER_MODEL


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


def _kokoro_backend() -> str:
    """KOKORO_BACKEND=torch|mlx|auto. "mlx" is mlx-audio's port of the same
    model onto Apple's Metal stack: on an M-series Mac it measured ~3x faster
    than torch-on-CPU per sentence, with identical output, and without MPS's
    ~1s per-new-input-shape penalty. "auto" picks it on Apple Silicon when
    mlx-audio is installed, and torch everywhere else."""
    raw = os.getenv("KOKORO_BACKEND", "auto").strip().lower()
    if raw != "auto":
        return raw
    if (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_audio") is not None):
        return "mlx"
    return "torch"


KOKORO_BACKEND = _kokoro_backend()
KOKORO_MLX_MODEL = os.getenv("KOKORO_MLX_MODEL", "mlx-community/Kokoro-82M-bf16")
# MLX state is thread-affine (see the Parakeet note in lifespan), so every MLX
# Kokoro call — load, warm-up, synthesis — runs on this one thread. Serializing
# TTS costs nothing in practice: a sentence takes ~0.1s.
KOKORO_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro-mlx")

# Qwen3-TTS: a second, optional TTS engine (mlx-audio, Apple Silicon only),
# picked per request with model="qwen3-tts" or via /v1/audio/speech/qwen.
# Kokoro stays the default. Loads on first use: the model needs several GB
# while generating, which nothing should pay for unless it's asked for.
QWEN_TTS_MODEL = os.getenv("QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
QWEN_TTS_VOICE = os.getenv("QWEN_TTS_VOICE", "ryan")
# "auto" sends no language id at all; naming the language is what the model
# was benchmarked with here.
QWEN_TTS_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "english")
# Sampling temperature for every Qwen model. Unset = the checkpoint's own
# generation_config.json (0.9). Lower is steadier but flatter; a request's
# `temperature` overrides it.
QWEN_TTS_TEMPERATURE = float(os.getenv("QWEN_TTS_TEMPERATURE")) if os.getenv("QWEN_TTS_TEMPERATURE") else None
QWEN_TTS_PRELOAD = os.getenv("QWEN_TTS_PRELOAD", "false").strip().lower() in ("1", "true", "yes", "on")
QWEN_TTS_ALIASES = {"qwen3-tts", "qwen3_tts", "qwen-tts", "qwen"}
# VoiceDesign: invents a voice from a text description in `instructions`. It
# designs a NEW voice on every call, so it's for auditioning, not for a voice
# you want to stay the same. To keep one, save a clip it made into
# QWEN_TTS_VOICES_DIR and use it by name (cloned by the Base model).
QWEN_TTS_DESIGN_MODEL = os.getenv(
    "QWEN_TTS_DESIGN_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
)
QWEN_TTS_DESIGN_ALIASES = {"qwen3-tts-design", "qwen3_tts_design", "qwen-tts-design", "qwen-design"}
# Base: clones a saved voice from a reference clip. Output is sampled, so
# delivery varies call to call (see QWEN_TTS_TEMPERATURE).
QWEN_TTS_BASE_MODEL = os.getenv("QWEN_TTS_BASE_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
# Saved voices: <name>.wav (10-20s of speech) next to <name>.txt (its exact
# transcript; optional). Read per request, so a new voice works without a restart.
QWEN_TTS_VOICES_DIR = os.getenv("QWEN_TTS_VOICES_DIR", "voices")
# Speaker-only (x-vector) cloning uses just the clip's voice fingerprint. It
# keeps one identity on short lines, where full-clip (ICL) cloning drifts in
# age/gender/accent and can run on with the clip's last words. A request's
# `xvec_only` overrides; a voice with no .txt is always speaker-only.
QWEN_TTS_XVEC_ONLY = os.getenv("QWEN_TTS_XVEC_ONLY", "true").strip().lower() in ("1", "true", "yes", "on")
# Language for saved voices. "auto" keeps the clip's accent; "english"
# flattens it toward General American.
QWEN_TTS_CLONE_LANGUAGE = os.getenv("QWEN_TTS_CLONE_LANGUAGE", "auto")
# Its own MLX thread, for the same thread-affinity reason as Kokoro's. Separate
# from Kokoro's so a long Qwen generation never queues a Kokoro request.
QWEN_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-tts")

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
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

KOKORO_SAMPLE_RATE = 24000


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
    # Keyed by resolved model name. Only the default engine's model is loaded
    # here, so startup cost tracks what default requests use; anything else a
    # client asks for loads lazily on first use.
    state["whisper_models"] = {}
    state["parakeet_models"] = {}
    _default = resolve_whisper_model(None)
    if is_parakeet(_default):
        # NOT preloaded here: building the model on the event-loop thread and
        # then running inference in a to_thread worker fails inside mx.eval()
        # ("There is no Stream(cpu, 1) in current thread"). get_parakeet_model
        # loads it inside a worker on first use, which works.
        log.info("Default STT is Parakeet (%s); loads on first use.", _default)
    else:
        log.info("Loading Whisper: %s on %s", _default, WHISPER_DEVICE)
        state["whisper_models"][_default] = load_whisper_model(_default)
        log.info("Whisper loaded.")

    if KOKORO_BACKEND == "mlx":
        log.info("Loading Kokoro (MLX): %s, langs=%s", KOKORO_MLX_MODEL, KOKORO_LANGS)
        state["kokoro"] = await asyncio.get_running_loop().run_in_executor(
            KOKORO_MLX_EXECUTOR, load_kokoro_mlx_pipelines
        )
    else:
        log.info("Loading Kokoro pipelines: langs=%s on %s", KOKORO_LANGS, KOKORO_DEVICE)
        state["kokoro"] = {}
        for lang in KOKORO_LANGS:
            state["kokoro"][lang] = KPipeline(lang_code=lang, device=KOKORO_DEVICE)
    log.info("Kokoro loaded: %d pipeline(s) — %s.", len(state["kokoro"]), ",".join(KOKORO_LANGS))

    # One throwaway synthesis per language so the first real request doesn't
    # pay for voice-pack download/load and first-call setup. A failure here is
    # only logged; the request path will surface the real error.
    for lang in KOKORO_LANGS:
        voice = DEFAULT_VOICE if DEFAULT_VOICE.startswith(lang) else (
            KOKORO_VOICES_BY_LANG.get(lang) or [DEFAULT_VOICE]
        )[0]
        try:
            await synthesize_async("Warming up.", voice)
        except Exception as e:
            log.warning("Kokoro warm-up for lang %r (%s) failed: %s", lang, voice, e)
    log.info("Kokoro warmed up.")

    if QWEN_TTS_PRELOAD:
        for repo in (QWEN_TTS_MODEL, QWEN_TTS_BASE_MODEL, QWEN_TTS_DESIGN_MODEL):
            await asyncio.get_running_loop().run_in_executor(QWEN_TTS_EXECUTOR, qwen_tts_model, repo)

    yield

    state.clear()


app = FastAPI(title="cleanTTS", version="0.1.0", lifespan=lifespan)


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


def render_speech(
    text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0
) -> tuple[np.ndarray | None, list[dict]]:
    """Run Kokoro; return (mono float32 audio @ 24kHz or None, chunks).

    Kokoro synthesizes in chunks; each entry of `chunks` records where one
    landed in the joined audio ({"offset", "duration", "graphemes",
    "tokens"}). `tokens` carries Kokoro's own per-word start_ts/end_ts,
    relative to that chunk, for English (lang a/b); other languages have none.

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
    pipelines: dict = state["kokoro"]
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
    # pack and combine into a normalized weighted-mean tensor. The torch
    # pipeline accepts that directly in place of a voice id; the MLX one only
    # takes names, so the blend is cached in its voice table under the spec
    # string (the cache is checked before the name is parsed) and passed by name.
    if len(pairs) == 1 and pairs[0][1] == 1.0:
        voice_arg: object = pairs[0][0]
    else:
        total = sum(w for _, w in pairs)
        if total <= 0:
            raise ValueError("Voice weights sum to zero; nothing to blend.")
        packs = [pipeline.load_single_voice(name) for name, _ in pairs]
        blend = sum(pack * (w / total) for pack, (_, w) in zip(packs, pairs))
        if KOKORO_BACKEND == "mlx":
            pipeline.voices[voice] = blend
            voice_arg = voice
        else:
            voice_arg = blend

    audio_chunks = []
    chunks: list[dict] = []
    offset = 0
    for result in pipeline(text, voice=voice_arg, speed=speed):
        audio = result.audio
        if audio is None:
            continue
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        elif hasattr(audio, "numpy"):
            audio = audio.numpy()
        # MLX yields (1, n) mx.arrays; np.asarray reads them via the buffer
        # protocol, and reshape flattens both backends to 1-D.
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_chunks.append(audio)
        chunks.append({
            "offset": offset / KOKORO_SAMPLE_RATE,
            "duration": len(audio) / KOKORO_SAMPLE_RATE,
            "graphemes": result.graphemes,
            "tokens": result.tokens,
        })
        offset += len(audio)
    if not audio_chunks:
        return None, []
    return np.concatenate(audio_chunks), chunks


def to_wav_bytes(wav: np.ndarray, sample_rate: int = KOKORO_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# SSML-style cue markers for /v1/audio/speech/timed: <mark name="slide2"/>.
# Stripped before synthesis on every route so they're never read aloud.
MARK_RE = re.compile(r"""<mark\s+name\s*=\s*["']([^"']*)["']\s*/?>""", re.IGNORECASE)


def split_marks(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Remove <mark name="..."/> tags; return (clean text, [(name, char
    offset into the clean text)])."""
    parts: list[str] = []
    marks: list[tuple[str, int]] = []
    pos = 0
    length = 0
    for m in MARK_RE.finditer(text):
        parts.append(text[pos:m.start()])
        length += m.start() - pos
        marks.append((m.group(1), length))
        pos = m.end()
    parts.append(text[pos:])
    return "".join(parts), marks


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Run Kokoro and return WAV bytes (mono PCM_16 @ 24kHz); b"" if Kokoro
    produced nothing. See render_speech for the voice spec."""
    wav, _ = render_speech(split_marks(text)[0], voice, speed)
    return b"" if wav is None else to_wav_bytes(wav)


# A token that is only sentence-final punctuation closes a sentence.
SENTENCE_END_RE = re.compile(r"^[.!?…]+$")
WORDLIKE_RE = re.compile(r"\w")


def timing_from_chunks(clean: str, chunks: list[dict], marks: list[tuple[str, int]]) -> dict:
    """Build {duration, words, sentences, cues} from render_speech's chunks.

    Words come straight from Kokoro's predicted durations — the times they are
    actually spoken in this audio, not an estimate. Chunk-relative stamps are
    shifted by each chunk's offset. Languages without word stamps fall back
    to one "sentence" per Kokoro chunk, and their cues to chunk boundaries.
    """
    words: list[dict] = []
    sentences: list[dict] = []
    # (char offset in `clean`, start time) per timed token, for placing cues.
    anchors: list[tuple[int, float]] = []
    cursor = 0
    # (text, start, end, trailing space, char span in `clean` or None)
    pending: list[tuple[str, float, float, str, tuple[int, int] | None]] = []

    def close_sentence():
        if pending:
            # Prefer the caller's own text: misaki normalizes some tokens
            # (e.g. "$3.50" comes back as "3.50"), so rebuilding from tokens
            # is only the fallback when a span couldn't be located.
            first, last = pending[0][4], pending[-1][4]
            if first and last:
                text = clean[first[0]:last[1]].strip()
            else:
                text = "".join(t + ws for t, _, _, ws, _ in pending).strip()
            sentences.append({
                "text": text,
                "start": round(pending[0][1], 3),
                "end": round(pending[-1][2], 3),
            })
            pending.clear()

    for ch in chunks:
        tokens = ch["tokens"]
        if not tokens:
            close_sentence()
            at = clean.find(ch["graphemes"].strip()[:20], cursor) if ch["graphemes"] else -1
            if at >= 0:
                anchors.append((at, ch["offset"]))
                cursor = at
            sentences.append({
                "text": (ch["graphemes"] or "").strip(),
                "start": round(ch["offset"], 3),
                "end": round(ch["offset"] + ch["duration"], 3),
            })
            continue
        for tok in tokens:
            if tok.start_ts is None or tok.end_ts is None:
                continue
            start = ch["offset"] + tok.start_ts
            end = ch["offset"] + tok.end_ts
            # Tokens don't carry their source position, so find each one in
            # order; misaki keeps the original spelling in tok.text.
            at = clean.find(tok.text, cursor) if tok.text else -1
            span = None
            if at >= 0:
                anchors.append((at, start))
                cursor = at + len(tok.text)
                span = (at, cursor)
            if WORDLIKE_RE.search(tok.text or ""):
                words.append({"word": tok.text, "start": round(start, 3), "end": round(end, 3)})
            pending.append((tok.text, start, end, " " if tok.whitespace else "", span))
            if SENTENCE_END_RE.match(tok.text or ""):
                close_sentence()
    close_sentence()

    duration = (chunks[-1]["offset"] + chunks[-1]["duration"]) if chunks else 0.0
    cues = []
    for name, char_at in marks:
        # A cue fires when the first word at or after it starts; a trailing
        # cue fires at the end of the audio.
        at = next((t for c, t in anchors if c >= char_at), duration)
        cues.append({"name": name, "at": round(at, 3)})

    return {
        "duration": round(duration, 3),
        "words": words,
        "sentences": sentences,
        "cues": cues,
    }


def synthesize_timed(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> tuple[bytes, dict]:
    """synthesize(), plus the timing dict from timing_from_chunks."""
    clean, marks = split_marks(text)
    wav, chunks = render_speech(clean, voice, speed)
    if wav is None:
        return b"", {}
    return to_wav_bytes(wav), timing_from_chunks(clean, chunks, marks)


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render_subtitles(sentences: list[dict], kind: str) -> str:
    """One cue per sentence, as WebVTT or SRT."""
    if kind == "vtt":
        blocks = ["WEBVTT"] + [
            f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}"
            for s in sentences
        ]
    else:
        blocks = [
            f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}"
            for i, s in enumerate(sentences, 1)
        ]
    return "\n\n".join(blocks) + "\n"


def load_kokoro_mlx_pipelines() -> dict:
    """Load the MLX model and one pipeline per language. Must run on
    KOKORO_MLX_EXECUTOR, the thread that will also run inference.

    The pipelines are called directly rather than through model.generate(),
    because generate() clears the pipeline's voice cache on every call — that
    would re-read voice packs each sentence and drop cached blends.
    """
    from mlx_audio.tts.utils import load

    model = load(KOKORO_MLX_MODEL)
    # _get_pipeline is private, but it's the one place that wires a pipeline to
    # the model's own repo/snapshot so voices load from the same weights.
    return {lang: model._get_pipeline(lang) for lang in KOKORO_LANGS}


async def on_tts_thread(fn, *args):
    """Run a Kokoro call off the event loop, on the thread its backend needs."""
    if KOKORO_BACKEND == "mlx":
        return await asyncio.get_running_loop().run_in_executor(KOKORO_MLX_EXECUTOR, fn, *args)
    return await asyncio.to_thread(fn, *args)


async def synthesize_async(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    return await on_tts_thread(synthesize, text, voice, speed)


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
    model: str = Field("kokoro", description="kokoro (default), qwen3-tts or qwen3-tts-design.")
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
    # Additive, Qwen3-TTS only; Kokoro ignores these.
    instructions: str | None = Field(
        None,
        description="Qwen3-TTS: speaking style/emotion for preset voices, e.g. "
        "'Calm and warm.' (1.7B models only; saved voices ignore it). For "
        "qwen3-tts-design, the description of the voice to invent.",
    )
    stream: bool = Field(
        False,
        description="Qwen3-TTS: stream audio as it is generated (wav or pcm only).",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        description="Qwen3-TTS: sampling temperature (default QWEN_TTS_TEMPERATURE, "
        "else 0.9). Lower = more consistent delivery; 0 = greedy.",
    )
    xvec_only: bool | None = Field(
        None,
        description="Qwen3-TTS saved voices: true = clone from the clip's voice "
        "fingerprint only (steady identity, no run-on); false = full-clip "
        "(ICL) clone. Default QWEN_TTS_XVEC_ONLY.",
    )


def check_speech_request(req: SpeechRequest) -> str:
    """Validate the fields both speech routes share; return the format."""
    fmt = req.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"response_format '{req.response_format}' not supported. "
            f"Use one of: {', '.join(SUPPORTED_FORMATS)}.",
        )
    if not split_marks(req.input)[0].strip():
        raise HTTPException(400, "input must not be empty.")
    return fmt


async def run_tts(fn, req: SpeechRequest):
    """Run a synthesize function for a request, mapping failures to HTTP."""
    try:
        return await on_tts_thread(fn, req.input, req.voice, req.speed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("TTS synthesis failed")
        raise HTTPException(500, f"TTS failed: {e}")


async def encode_audio(wav_bytes: bytes, fmt: str) -> bytes:
    if not wav_bytes:
        raise HTTPException(500, "TTS produced no audio (Kokoro returned no chunks).")
    ffmpeg_args = SUPPORTED_FORMATS[fmt][1]
    if ffmpeg_args is None:
        return wav_bytes
    try:
        return await asyncio.to_thread(transcode, wav_bytes, ffmpeg_args)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg transcode to %s failed: %s", fmt, e.stderr.decode(errors="ignore"))
        raise HTTPException(500, f"transcode to {fmt} failed")
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg not installed; required for non-WAV formats.")


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    fmt = check_speech_request(req)
    if req.model.strip().lower() in QWEN_TTS_ALIASES | QWEN_TTS_DESIGN_ALIASES:
        return await speak_qwen(req, fmt)
    wav_bytes = await run_tts(synthesize, req)
    encoded = await encode_audio(wav_bytes, fmt)
    return Response(content=encoded, media_type=SUPPORTED_FORMATS[fmt][0])


# ---------------------------------------------------------------------------
# Qwen3-TTS
# ---------------------------------------------------------------------------

def qwen_tts_model(repo: str = QWEN_TTS_MODEL):
    """Return a Qwen3-TTS model, loading it on first call. Must run on
    QWEN_TTS_EXECUTOR — its single thread is also what serializes loads."""
    models: dict = state.setdefault("qwen_tts", {})
    model = models.get(repo)
    if model is None:
        from mlx_audio.tts.utils import load

        log.info("Loading Qwen3-TTS: %s (MLX/Metal)", repo)
        model = load(repo)
        models[repo] = model
        log.info("Qwen3-TTS %s loaded.", repo)
    return model


def qwen_saved_voices() -> dict[str, tuple[str, str | None]]:
    """Saved voices in QWEN_TTS_VOICES_DIR: name -> (wav path, transcript, or
    None for a speaker-only voice with no .txt). Only names found on disk are
    ever used as paths."""
    voices = {}
    try:
        names = os.listdir(QWEN_TTS_VOICES_DIR)
    except FileNotFoundError:
        return voices
    for fname in names:
        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".wav":
            continue
        txt = os.path.join(QWEN_TTS_VOICES_DIR, stem + ".txt")
        transcript = None
        if os.path.exists(txt):
            with open(txt) as f:
                transcript = f.read().strip() or None
        voices[stem.lower()] = (os.path.join(QWEN_TTS_VOICES_DIR, fname), transcript)
    return voices


def resolve_qwen(req: SpeechRequest) -> tuple[str, dict]:
    """Pick the Qwen model and generate() kwargs for a request:
    model=qwen3-tts-design -> VoiceDesign, described by `instructions`;
    a saved voice name    -> Base, cloning that voice's clip;
    anything else         -> CustomVoice preset speaker (+ `instructions` style)."""
    if req.model.strip().lower() in QWEN_TTS_DESIGN_ALIASES:
        # mlx-audio raises its own ValueError (-> 400) if instructions is empty.
        return QWEN_TTS_DESIGN_MODEL, {"instruct": req.instructions}
    # A client that didn't pick a voice gets Qwen's default, not Kokoro's.
    voice = req.voice if "voice" in req.model_fields_set else QWEN_TTS_VOICE
    saved = qwen_saved_voices().get(voice.strip().lower())
    if saved:
        wav, transcript = saved
        xvec_only = QWEN_TTS_XVEC_ONLY if req.xvec_only is None else req.xvec_only
        if xvec_only or transcript is None:
            if req.xvec_only is False:
                raise HTTPException(400, f"Saved voice {voice!r} has no transcript; it is speaker-only.")
            # Speaker-only: just the clip's voice fingerprint, so every line is
            # the same person and nothing from the clip can run on.
            return QWEN_TTS_BASE_MODEL, {"ref_audio": wav, "lang_code": QWEN_TTS_CLONE_LANGUAGE}
        # Full clip (ICL): continues from the clip, copying its delivery.
        return QWEN_TTS_BASE_MODEL, {"ref_audio": wav, "ref_text": transcript, "lang_code": QWEN_TTS_CLONE_LANGUAGE}
    kwargs = {"voice": voice}
    if req.instructions:
        kwargs["instruct"] = req.instructions
    return QWEN_TTS_MODEL, kwargs


def qwen_generate(repo: str, text: str, kwargs: dict, stream: bool):
    """Yield float32 audio chunks from Qwen3-TTS. Unknown preset voices raise
    ValueError from mlx-audio itself, listing the valid ones."""
    model = qwen_tts_model(repo)
    kwargs = dict({"lang_code": QWEN_TTS_LANGUAGE}, **kwargs, text=split_marks(text)[0])
    if kwargs.get("temperature") is None:
        kwargs["temperature"] = QWEN_TTS_TEMPERATURE
    if stream:
        # ~6 codec frames at 12.5Hz: first audio in ~0.25s on an M-series Mac.
        kwargs.update(stream=True, streaming_interval=0.48)
    for result in model.generate(**kwargs):
        if result.audio is not None:
            yield np.asarray(result.audio, dtype=np.float32).reshape(-1)


def synthesize_qwen(repo: str, text: str, kwargs: dict) -> bytes:
    """Run Qwen3-TTS to completion; return WAV bytes, or b"" if it produced nothing."""
    chunks = list(qwen_generate(repo, text, kwargs, False))
    if not chunks:
        return b""
    return to_wav_bytes(np.concatenate(chunks), state["qwen_tts"][repo].sample_rate)


def streaming_wav_header(sample_rate: int) -> bytes:
    """A PCM_16 mono WAV header for a stream of unknown length: the size fields
    are set to the maximum, which players read as "until the data ends"."""
    unknown = 0xFFFFFFFF
    return (
        b"RIFF" + unknown.to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little") + (sample_rate * 2).to_bytes(4, "little")
        + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        + b"data" + unknown.to_bytes(4, "little")
    )


async def stream_qwen(req: SpeechRequest, repo: str, kwargs: dict, fmt: str) -> StreamingResponse:
    """Stream Qwen3-TTS audio as it's generated. The first chunk is pulled
    before responding, so a bad voice still becomes a 400 rather than a
    truncated 200."""
    loop = asyncio.get_running_loop()
    gen = qwen_generate(repo, req.input, kwargs, True)
    done = object()

    def next_chunk():
        return next(gen, done)

    try:
        first = await loop.run_in_executor(QWEN_TTS_EXECUTOR, next_chunk)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Qwen3-TTS synthesis failed")
        raise HTTPException(500, f"TTS failed: {e}")
    if first is done:
        raise HTTPException(500, "TTS produced no audio (Qwen3-TTS returned no chunks).")
    sample_rate = state["qwen_tts"][repo].sample_rate

    def pcm16(chunk: np.ndarray) -> bytes:
        return (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    async def body():
        # One chunk per executor hop, so a client that disconnects stops
        # generation at the next chunk instead of holding the thread.
        try:
            if fmt == "wav":
                yield streaming_wav_header(sample_rate)
            chunk = first
            while chunk is not done:
                yield pcm16(chunk)
                chunk = await loop.run_in_executor(QWEN_TTS_EXECUTOR, next_chunk)
        finally:
            await loop.run_in_executor(QWEN_TTS_EXECUTOR, gen.close)

    return StreamingResponse(body(), media_type=SUPPORTED_FORMATS[fmt][0])


async def speak_qwen(req: SpeechRequest, fmt: str) -> Response:
    repo, kwargs = resolve_qwen(req)
    kwargs["temperature"] = req.temperature
    if req.stream:
        if fmt not in ("wav", "pcm"):
            raise HTTPException(400, "stream=true supports response_format wav or pcm only.")
        return await stream_qwen(req, repo, kwargs, fmt)
    loop = asyncio.get_running_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            QWEN_TTS_EXECUTOR, synthesize_qwen, repo, req.input, kwargs
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Qwen3-TTS synthesis failed")
        raise HTTPException(500, f"TTS failed: {e}")
    encoded = await encode_audio(wav_bytes, fmt)
    return Response(content=encoded, media_type=SUPPORTED_FORMATS[fmt][0])


@app.post("/v1/audio/speech/qwen")
async def speech_qwen(req: SpeechRequest):
    """Same as /v1/audio/speech with model="qwen3-tts" (or qwen3-tts-design,
    if that's what `model` says)."""
    if req.model.strip().lower() not in QWEN_TTS_DESIGN_ALIASES:
        req.model = "qwen3-tts"
    return await speak_qwen(req, check_speech_request(req))


class TimedSpeechRequest(SpeechRequest):
    subtitles: str | None = Field(
        None, description="vtt | srt — also return subtitles, one cue per sentence."
    )


@app.post("/v1/audio/speech/timed")
async def speech_timed(req: TimedSpeechRequest):
    """Like /v1/audio/speech, but JSON: base64 audio plus when each word and
    sentence is spoken, for syncing slides, captions or highlighting.

    Put <mark name="..."/> anywhere in `input` to get a cue back with the time
    the following word starts (or the end of the audio, for a trailing mark).
    Marks are never spoken.

    Word timings exist for English voices (lang a/b) only; other languages
    return no words, and one sentence per Kokoro chunk. wav/opus/flac/pcm keep
    the timings sample-exact. mp3/aac encoders add a few tens of ms of leading
    padding that some decoders don't strip, which shifts playback slightly.
    """
    fmt = check_speech_request(req)
    subs = (req.subtitles or "").lower() or None
    if subs not in (None, "vtt", "srt"):
        raise HTTPException(400, "subtitles must be 'vtt' or 'srt'.")
    wav_bytes, timing = await run_tts(synthesize_timed, req)
    encoded = await encode_audio(wav_bytes, fmt)
    payload = {
        "audio": base64.b64encode(encoded).decode("ascii"),
        "format": fmt,
        "media_type": SUPPORTED_FORMATS[fmt][0],
        "sample_rate": KOKORO_SAMPLE_RATE,
        **timing,
    }
    if subs:
        payload["subtitles"] = render_subtitles(timing["sentences"], subs)
    return payload


def segments_text(segments: list) -> str:
    """Join Whisper segments into one plain-text transcript."""
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


async def transcribe_parakeet(
    audio_bytes: bytes, model_name: str
) -> tuple[list, float]:
    """Run Parakeet on raw audio bytes; return (sentences, duration).

    Parakeet reads a file, so the bytes are staged the same way the Whisper path
    stages them. Inference happens in a thread for the same reason: MLX releases
    the GIL, but the surrounding work shouldn't sit on the event loop.
    """
    model = await get_parakeet_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        result = await asyncio.to_thread(model.transcribe, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    sentences = list(getattr(result, "sentences", []) or [])
    duration = max((sn.end for sn in sentences), default=0.0)
    return sentences, duration


def parakeet_verbose_payload(sentences: list, duration: float, model_name: str) -> dict:
    """Render Parakeet output in the same verbose_json shape as Whisper.

    Two fields cannot be produced honestly by this engine and are documented as
    such in the README's Parakeet section:

    - no_speech_prob is always 0.0. Parakeet's transducer emits nothing where
      there is no speech, so there is no per-segment non-speech probability to
      report. Clients must NOT treat 0.0 as "confidently speech" — use
      `confidence` instead (measured: hallucinations ~0.78, real speech >0.96,
      so ~0.9 separates them).
    - avg_logprob is derived as log(confidence) so it lands in the same units
      and direction as Whisper's, for clients that already threshold on it.

    `confidence` is the native signal and the one to prefer.
    """
    segments = []
    for i, sn in enumerate(sentences):
        conf = float(getattr(sn, "confidence", 0.0) or 0.0)
        segments.append({
            "id": i,
            "seek": 0,
            "start": round(sn.start, 3),
            "end": round(sn.end, 3),
            "text": sn.text,
            "tokens": [int(t.id) for t in getattr(sn, "tokens", []) if hasattr(t, "id")],
            "temperature": 0.0,
            "avg_logprob": math.log(conf) if conf > 0 else float("-inf"),
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0,
            # Native to this engine; absent from Whisper responses.
            "confidence": round(conf, 6),
            "words": [
                {"start": round(t.start, 3), "end": round(t.end, 3),
                 "word": t.text, "confidence": round(float(t.confidence), 6)}
                for t in getattr(sn, "tokens", [])
            ],
        })
    return {
        "task": "transcribe",
        "language": "auto",
        "duration": round(duration, 3),
        "text": "".join(sn.text for sn in sentences).strip(),
        "segments": segments,
        "engine": "parakeet",
        "model": model_name,
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

    `model` selects the checkpoint per request. "whisper-1" (what OpenAI SDKs
    send), "default" or an empty value all mean WHISPER_MODEL; anything else is
    a faster-whisper built-in name ("large-v3-turbo", "turbo", "tiny",
    "distil-small.en", ...) or a full HF repo id. Non-default models are loaded
    on first use and then stay resident.

    vad_filter overrides WHISPER_VAD_FILTER for this request. Leave it unset
    unless you specifically want the other behaviour: with VAD off, segment
    timings collapse to 0.00 on clips with leading silence.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "file must not be empty.")

    fmt = (response_format or "json").lower()

    resolved = resolve_whisper_model(model)

    # Parakeet (MLX/Metal) is a separate engine selected by model name.
    if is_parakeet(resolved):
        try:
            sentences, duration = await transcribe_parakeet(audio_bytes, resolved)
        except Exception as e:
            log.exception("Parakeet transcription failed")
            raise HTTPException(500, f"transcription failed: {e}")
        text = "".join(sn.text for sn in sentences).strip()
        if fmt == "text":
            return Response(content=text, media_type="text/plain")
        if fmt == "verbose_json":
            return parakeet_verbose_payload(sentences, duration, resolved)
        return {"text": text}

    # Resolve/load the model first so an unusable `model` value reports as a
    # 400 against the request rather than a 500 against the server.
    try:
        await get_whisper_model(model)
    except Exception as e:
        log.warning("Whisper model %r could not be loaded: %s", model, e)
        raise HTTPException(
            400,
            f"model {resolve_whisper_model(model)!r} could not be loaded: {e}",
        )

    try:
        segments, info = await transcribe_segments(
            audio_bytes, language=language, initial_prompt=prompt,
            vad_filter=vad_filter, model=model,
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
# both pull the model. Deliberately NOT shared with the Whisper path — plain
# transcription requests never wait on this.
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
            # "model" is the default used when a request names none; it is not
            # the only model that can be served. Unchanged for existing clients.
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            # Additive fields.
            "vad_filter": WHISPER_VAD_FILTER,
            "loaded": sorted(state.get("whisper_models", {})),
        },
        # Additive: the model used when a request names none .
        "stt_default": resolve_whisper_model(None),
        # Additive: a second STT engine, selected per request via `model`.
        # Runs on the GPU via MLX/Metal, unlike Whisper.
        "parakeet": {
            "model": PARAKEET_MODEL,
            "device": "mps",
            "loaded": sorted(state.get("parakeet_models", {})),
        },
        "kokoro": {
            "langs": KOKORO_LANGS,
            # MLX always runs on the GPU via Metal, whatever KOKORO_DEVICE says.
            "device": "mps" if KOKORO_BACKEND == "mlx" else KOKORO_DEVICE,
            # Additive.
            "backend": KOKORO_BACKEND,
            **({"model": KOKORO_MLX_MODEL} if KOKORO_BACKEND == "mlx" else {}),
        },
        # Additive: optional second TTS engine, loaded on first use.
        "qwen_tts": {
            "model": QWEN_TTS_MODEL,
            "voice": QWEN_TTS_VOICE,
            "temperature": QWEN_TTS_TEMPERATURE,
            "xvec_only": QWEN_TTS_XVEC_ONLY,
            "clone_language": QWEN_TTS_CLONE_LANGUAGE,
            "device": "mps",
            "design_model": QWEN_TTS_DESIGN_MODEL,
            "base_model": QWEN_TTS_BASE_MODEL,
            "saved_voices": sorted(qwen_saved_voices()),
            "loaded": sorted(state.get("qwen_tts", {})),
        },
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
        "data": [
            {"id": "kokoro", "object": "model", "owned_by": "hexgrad"},
            {"id": "qwen3-tts", "object": "model", "owned_by": "qwen"},
            {"id": "qwen3-tts-design", "object": "model", "owned_by": "qwen"},
        ],
    }


@app.get("/v1/voices")
async def list_voices(model: str | None = None):
    if (model or "").strip().lower() in QWEN_TTS_ALIASES:
        qwen = await asyncio.get_running_loop().run_in_executor(QWEN_TTS_EXECUTOR, qwen_tts_model)
        return {
            "voices": qwen.get_supported_speakers(),
            # Additive: cloned voices from QWEN_TTS_VOICES_DIR, same `voice` field.
            "saved_voices": sorted(qwen_saved_voices()),
            "default": QWEN_TTS_VOICE,
        }
    voices: list[str] = []
    for lang in KOKORO_LANGS:
        voices.extend(KOKORO_VOICES_BY_LANG.get(lang, []))
    return {"voices": voices, "default": DEFAULT_VOICE, "langs": KOKORO_LANGS}


def is_parakeet(name: str) -> bool:
    """Whether a resolved model name should be served by the Parakeet engine."""
    return "parakeet" in (name or "").lower()


_parakeet_load_lock = asyncio.Lock()


async def get_parakeet_model(name: str):
    """Return the Parakeet model, loading it on first use. Same cache-first
    pattern as Whisper: a hit never takes the lock, so a slow load can't stall
    requests on an already-loaded model."""
    models: dict = state.setdefault("parakeet_models", {})
    cached = models.get(name)
    if cached is not None:
        return cached
    async with _parakeet_load_lock:
        cached = models.get(name)
        if cached is not None:
            return cached

        def _load():
            from parakeet_mlx import from_pretrained
            return from_pretrained(name)

        log.info("Loading Parakeet on demand: %s (MLX/Metal)", name)
        model = await asyncio.to_thread(_load)
        models[name] = model
        log.info("Parakeet %s loaded.", name)
        return model


def load_whisper_model(name: str) -> WhisperModel:
    """Instantiate one WhisperModel. Blocking — off the event loop for non-default
    models, which may also have to download."""
    compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    return WhisperModel(name, device=WHISPER_DEVICE, compute_type=compute_type)


def resolve_whisper_model(name: str | None) -> str:
    """Map a client-supplied `model` field to a canonical checkpoint name.

    Distinct spellings of the same checkpoint are collapsed so they share one
    cache entry — faster-whisper accepts both "turbo" and "large-v3-turbo" for
    the same 1.5GB repo, and caching it twice would just waste memory.
    """
    cleaned = (name or "").strip()
    if cleaned.lower() in PARAKEET_ALIASES:
        return PARAKEET_MODEL
    if cleaned.lower() in WHISPER_MODEL_ALIASES:
        cleaned = STT_MODEL
        if cleaned.lower() in PARAKEET_ALIASES:
            return PARAKEET_MODEL
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(cleaned, cleaned)
    except Exception:
        # Private API; if it moves, fall back to the name as given.
        return cleaned


# Serializes loads of *new* models. Cache hits never touch it, so a slow load
# triggered by one client can't stall requests on the default model.
_whisper_load_lock = asyncio.Lock()


async def get_whisper_model(name: str | None) -> WhisperModel:
    """Return the requested model, loading and caching it on first use.

    Models stay resident once loaded: distil-small.en is ~0.3GB and
    large-v3-turbo ~1.5GB, so a couple of them is cheap, but nothing evicts,
    and the set is driven by what clients ask for.
    """
    resolved = resolve_whisper_model(name)
    models: dict[str, WhisperModel] = state["whisper_models"]
    cached = models.get(resolved)
    if cached is not None:
        return cached
    async with _whisper_load_lock:
        cached = models.get(resolved)
        if cached is not None:
            return cached
        log.info("Loading Whisper on demand: %s on %s", resolved, WHISPER_DEVICE)
        model = await asyncio.to_thread(load_whisper_model, resolved)
        models[resolved] = model
        log.info("Whisper %s loaded (resident: %s).", resolved, ", ".join(sorted(models)))
        return model


async def transcribe_segments(
    audio_bytes: bytes,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool | None = None,
    model: str | None = None,
) -> tuple[list, object]:
    """Write to a temp file (so ffmpeg can decode webm/ogg/etc), run Whisper and
    return (segments, info) with every field faster-whisper produced.

    faster-whisper's `transcribe()` only *builds* a lazy generator; the actual
    decode happens while iterating it. So the iteration is drained inside the
    same to_thread call — otherwise the decode runs on the event loop and stalls
    every other request sharing it.

    vad_filter=None uses the WHISPER_VAD_FILTER default; pass a bool to override
    it per call. model=None uses WHISPER_MODEL.
    """
    use_vad = WHISPER_VAD_FILTER if vad_filter is None else vad_filter
    whisper = await get_whisper_model(model)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    def _run() -> tuple[list, object]:
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
    model: str | None = None,
) -> str:
    """Plain-text transcription. Contract is `-> str`: this is what the default
    /v1/audio/transcriptions response uses.

    Routes to whichever engine the resolved model belongs to, so setting
    STT_MODEL=parakeet moves default requests onto MLX with no other change.
    """
    resolved = resolve_whisper_model(model)
    if is_parakeet(resolved):
        sentences, _ = await transcribe_parakeet(audio_bytes, resolved)
        return "".join(sn.text for sn in sentences).strip()
    segments, _ = await transcribe_segments(
        audio_bytes, language, initial_prompt, vad_filter, model
    )
    return segments_text(segments)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")

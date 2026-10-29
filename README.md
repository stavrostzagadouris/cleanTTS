# cleanTTS

A small Python service that exposes OpenAI-compatible TTS and STT APIs for
any other tool to use.

Now also has a companion app to hit this endpoint from linux when holding a hotkey (super + space) to talk straight to your text line (sdl2stt). And the windows companion is sdl2stt-win.

NOTE this is all a Claude vibecode special. Claude, take it from here:

- **STT**: `faster-whisper` (local; CUDA or CPU), or Parakeet via
  `parakeet-mlx` on Apple Silicon
- **TTS**: Kokoro-82M, via `mlx-audio` on Apple Silicon or the `kokoro` pip
  package (PyTorch) everywhere else; optionally Qwen3-TTS via `mlx-audio`
  (Apple Silicon only)
- **Diarization**: `pyannote.audio` (optional, loads on first use)
- **Web**: FastAPI

One process, one port (default `5000`). No Docker, no vLLM-omni, no Voxtral.

---

## What you get

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible TTS — call this from any external tool |
| `POST /v1/audio/speech/qwen` | Same TTS on Qwen3-TTS instead of Kokoro (or send `model: "qwen3-tts"` to `/v1/audio/speech`) — see [Qwen3-TTS](#qwen3-tts-second-tts-engine) |
| `POST /v1/audio/speech/timed` | Same TTS, as JSON with word/sentence timings, cue markers and optional VTT/SRT subtitles |
| `POST /v1/audio/transcriptions` | OpenAI-compatible STT — two engines (Whisper on CPU, Parakeet on GPU), picked per request |
| `POST /v1/audio/diarize` | Speaker diarization — "who spoke when" (pyannote; CUDA or MPS) |
| `GET /v1/voices` | Available Kokoro voices (`?model=qwen3-tts` for Qwen's preset and saved voices) |
| `GET /v1/models` | OpenAI-compat model list (`kokoro`, `qwen3-tts`, `qwen3-tts-design`) |
| `GET /health` | Liveness + which models/devices are loaded |

### Changes on 2026-10-29

- **Removed: the browser voice assistant.** The `GET /` page, the `WS /ws`
  pipeline, `GET /api/llm-backends` / `GET /api/llm-models`, and the
  `LLM_BACKENDS`, `LLM_BASE_URL` and `SYSTEM_PROMPT` settings are gone, along
  with the `templates/` page and the `httpx`/`jinja2` dependencies. cleanTTS no
  longer talks to any LLM; clients (e.g. an agent) decide what to say and call
  the REST routes. Every REST route above is unchanged.
- **Added: Qwen3-TTS** as an opt-in second TTS engine, with preset voices,
  style `instructions`, streaming, voice design, and saved voices cloned from a
  reference clip. Kokoro stays the default. See
  [Qwen3-TTS](#qwen3-tts-second-tts-engine).

---

## Setup

The Python install differs slightly per platform — pick the one that matches
your machine. After install, the same `main.py` runs on all of them; device
selection is handled automatically (`WHISPER_DEVICE=auto`, `KOKORO_BACKEND=auto`,
`KOKORO_DEVICE=auto`).

### Linux + NVIDIA (CUDA)

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
./venv/bin/pip install -r requirements.txt
sudo apt install ffmpeg
```

### macOS (Apple Silicon — M1/M2/M3/M4)

No CUDA index URL — the default macOS PyTorch wheel ships with MPS support
built in.

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch
./venv/bin/pip install -r requirements.txt
brew install ffmpeg
```

`parakeet-mlx` and `mlx-audio` in `requirements.txt` are Apple-Silicon-only
(they pull `mlx`/`mlx-metal`). Both are optional — drop those lines on
Linux/CUDA. Without `parakeet-mlx`, Whisper still serves STT; without
`mlx-audio`, Kokoro falls back to the PyTorch backend.

On Apple Silicon, Kokoro runs through `mlx-audio` on Apple's GPU via Metal
(`KOKORO_BACKEND=auto` picks it when installed). Measured on this Mac, per
sentence: ~0.1s on MLX vs ~0.3s for PyTorch on CPU and ~1.1s for PyTorch on
MPS. MPS is slowest because it pays ~1s for every input length it hasn't seen
before, and a voice reply is all new sentences. If you use the PyTorch
backend on a Mac, set `KOKORO_DEVICE=cpu`, not `mps`. `faster-whisper` /
CTranslate2 has **no** MPS backend, so Whisper transparently falls back to
CPU at int8 — fast enough on M-series for short utterances. If you'd
rather force a different combo, set `WHISPER_DEVICE` and `KOKORO_DEVICE`
explicitly in `.env` (see `.env.example`).

### Linux/macOS, CPU-only

```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch
./venv/bin/pip install -r requirements.txt
# ffmpeg via your package manager
```

Set `WHISPER_DEVICE=cpu` and `KOKORO_DEVICE=cpu` in `.env`. Expect slower
TTS / STT than the accelerated paths, but it works for testing.

> `ffmpeg` is required for decoding uploaded and browser (WebM) audio for
> STT and diarization, and for non-WAV TTS response formats
> (opus/mp3/flac/aac/pcm).

---

## Configure

Copy `.env.example` to `.env` and edit it (`.env` is gitignored):

```bash
cp .env.example .env
```

```env
WHISPER_MODEL=distil-large-v3                  # or 'small', 'medium', etc.
WHISPER_DEVICE=auto                            # auto | cuda | cpu (mps not supported by faster-whisper)
KOKORO_LANGS=a,b                               # comma-separated; one KPipeline per language
KOKORO_BACKEND=auto                            # auto | torch | mlx (auto = mlx on Apple Silicon if installed)
KOKORO_DEVICE=auto                             # auto | cuda | mps | cpu (torch backend only)
DEFAULT_VOICE=af_bella
QWEN_TTS_PRELOAD=false                         # true = load the Qwen3-TTS models at startup
HOST=0.0.0.0
PORT=5000
```

That's the common subset. `.env.example` documents every option, including
`STT_MODEL`, `WHISPER_BEAM_SIZE`, `WHISPER_VAD_FILTER`, `KOKORO_MLX_MODEL`,
`DIARIZE_DEVICE`, `HF_TOKEN` and the `QWEN_TTS_*` settings.

---

## Run

```bash
./venv/bin/python main.py
```

First start: the Kokoro weights (~300MB) and the default STT model (e.g.
~1.5GB for `distil-large-v3`; Parakeet instead loads on first request)
download from HuggingFace into the local cache. Subsequent starts are fast.
Startup ends with one throwaway synthesis per language, so the first real
request doesn't pay Kokoro's setup cost. Qwen3-TTS weights (~3 GB per model)
download on first use, or at startup with `QWEN_TTS_PRELOAD=true`.

---

## Updating

### Python packages (Kokoro, Whisper, etc.)

`requirements.txt` pins exact versions (torch has only a floor, because its
wheel comes from a platform-specific index). Unpinned upgrades have broken
this app before: pyannote 3→4, and `kokoro`/`misaki` drifting apart. So
upgrades are deliberate. To try newer releases:

```bash
./venv/bin/pip list --outdated              # see what's newer
./venv/bin/pip install -U kokoro misaki     # upgrade specific packages
```

Then smoke-test (below). If it works, write the new versions back into
`requirements.txt`. Keep `kokoro` and `misaki` on matching versions; `misaki`
is Kokoro's text→phoneme step, and `mlx-audio` uses it too.

After any Kokoro/Whisper update, restart `main.py` and run a quick smoke
test before relying on it:

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"update check","voice":"af_bella"}' --output /tmp/test.wav
```

Almost all Kokoro updates within the `0.x` line are drop-in. Synthesis goes
through one function, `render_speech()` in `main.py`, which both backends
share; the MLX model is loaded in `load_kokoro_mlx_pipelines()`. If a future
release changes the pipeline API or the 24 kHz sample rate, those two
functions (plus `KOKORO_SAMPLE_RATE`) are the places to adjust.

### New voices / new languages

New Kokoro voices and languages usually ship as weights on HuggingFace
(`hexgrad/Kokoro-82M`) and download automatically on first use — no package
update required. To expose them:

- **New voice in a language you already load** → add its id to
  `KOKORO_VOICES_BY_LANG` in `main.py` (used for the `/v1/voices` listing;
  synthesis routes by the voice's first letter regardless).
- **New language** → add its code to `KOKORO_VOICES_BY_LANG` *and* to
  `KOKORO_LANGS` in `.env`. A new pipeline loads for it at startup (~300 MB
  each on the torch backend; on MLX the languages share one model).

---

## Running as a service (macOS, launchd)

`deploy/com.cleantts.plist` is a LaunchAgent template. Substitute `REPO_DIR`,
`VENV_DIR` and `HOME_DIR`, then:

```bash
cp deploy/com.cleantts.plist ~/Library/LaunchAgents/
mkdir -p ~/Library/Logs/cleanTTS
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cleantts.plist
```

Managing it:

```bash
launchctl print gui/$(id -u)/com.cleantts        # state, pid, restart count
launchctl kickstart -k gui/$(id -u)/com.cleantts # restart (after a code change)
launchctl bootout gui/$(id -u)/com.cleantts      # stop and disable
tail -f ~/Library/Logs/cleanTTS/stderr.log       # logs
```

Three things that are easy to get wrong:

- **Agent, not daemon.** Kokoro, Parakeet and pyannote need the GPU, and Metal is only
  reachable from inside a logged-in user session. A LaunchDaemon runs in the
  system context before login and cannot use the GPU.
- **`PATH` must be set explicitly.** launchd hands a job
  `/usr/bin:/bin:/usr/sbin:/sbin` with no Homebrew, so `ffmpeg` would be
  invisible — breaking `/v1/audio/diarize` and every non-WAV
  `/v1/audio/speech` format, at request time rather than at startup.
- **`WorkingDirectory` must be the repo.** `main.py` resolves `.env`
  relatively.

The agent starts at **login**, not at boot. With FileVault enabled that is the
best available: the volume is encrypted until someone enters the password at
the boot screen, so nothing — agent or daemon — can run before that. After the
unlock (which logs you in) the service comes up on its own and `KeepAlive`
keeps it up.

### Binding, and what it means for `localhost`

`HOST=0.0.0.0` binds every interface, which is only safe behind a NAT or
firewall — cleanTTS has no authentication. On a machine whose `en0` holds a globally routable address (check
with `ipconfig getifaddr en0`) with the macOS firewall off, prefer binding a
single VPN address, e.g. this host's Tailscale IP:

```bash
HOST=100.x.y.z    # `tailscale ip -4`
```

Two consequences worth knowing:

- **uvicorn binds exactly one address**, so `localhost:5000` and `127.0.0.1:5000`
  stop resolving. Local clients must use the Tailscale IP or hostname too.
- **The Tailscale interface appears at login**, roughly when the agent starts,
  so the first bind can lose the race and fail with
  `[Errno 49] can't assign requested address`. `KeepAlive` retries and it
  self-heals within seconds — but uvicorn runs the app's lifespan *before*
  binding, so each failed attempt loads Whisper and Kokoro first, making a
  retry cycle ~15s. If you'd rather not have that startup dependency, keep
  `HOST=0.0.0.0` and block port 5000 on the public interface with a `pf` rule
  instead.

launchd does not rotate `StandardOutPath` / `StandardErrorPath`. The logs are
low-volume, but they grow without bound; truncate them occasionally or add a
`newsyslog.d` rule.

---

## Speech-to-text and diarization

### `POST /v1/audio/transcriptions`

OpenAI-compatible, multipart:

| field | notes |
|---|---|
| `file` | **required.** Anything ffmpeg can decode. |
| `model` | Picks the checkpoint **and engine** — see below. Omit for `STT_MODEL`. |
| `language` | e.g. `en`. Ignored by `.en` models and by Parakeet. |
| `prompt` | Vocabulary bias. **Whisper only, and see the warning below.** |
| `response_format` | `json` (default), `text`, `verbose_json`. |
| `vad_filter` | Per-request override of `WHISPER_VAD_FILTER`. Whisper only. |

`json` returns just `{"text": "..."}`. `verbose_json` additionally returns
per-segment timings and confidence scores:

```bash
curl -X POST http://localhost:5000/v1/audio/transcriptions \
  -F file=@clip.wav -F response_format=verbose_json
```

```json
{"task": "transcribe", "language": "en", "duration": 12.3, "text": "...",
 "segments": [{"id": 0, "start": 1.2, "end": 3.4, "text": "...",
               "avg_logprob": -0.61, "no_speech_prob": 0.116,
               "compression_ratio": 1.4, "temperature": 0.0}]}
```

### Choosing an STT engine

Two engines are available on `POST /v1/audio/transcriptions`, selected by the
`model` field. Whisper (faster-whisper/CTranslate2) has no Metal backend and
runs on CPU; Parakeet (NVIDIA FastConformer-TDT via MLX) runs on the GPU.

Measured on an M1 Max — same clips, same harness:

| `model=` | 5s clip | 64-min recording | names/jargon WER |
|---|---|---|---|
| *omitted* → `distil-small.en` | 0.66s | ~2.1 min | 29.6% |
| `parakeet` | **0.14s** | **~51 s** | 25.9% |
| `turbo` | 5.15s | ~13.6 min | 14.8% |
| `turbo` + glossary `prompt` | 5.15s | ~13.6 min | **3.7%** |

Pick by the job:

- **Lowest latency** — `parakeet`. It is both faster *and* slightly more
  accurate than the default, and gives word-level timestamps. Setting
  `STT_MODEL=parakeet` makes it the default for requests that name no model.
- **Best accuracy on names and jargon** — `turbo` with a glossary `prompt`.
  Nothing else comes close, but it is ~36x slower than Parakeet.
- Never `medium` (1.2x realtime); `distil-large-v3` is dominated by `turbo`.

Parakeet differences that matter to clients:

- **No `no_speech_prob`** — it is reported as `0.0` and means nothing. Parakeet
  emits a native `confidence` per segment instead. Measured: hallucinated
  output ~0.78, real speech >0.96 even at -50dB, so ~0.9 separates them.
  `avg_logprob` is derived as `log(confidence)` for clients that threshold on it.
- **`prompt` is ignored** — there is no `initial_prompt` equivalent. Vocabulary
  biasing is Whisper-only.
- **`language` is ignored** — 25 European languages with auto-detection.
- **Word-level timestamps** are included in `verbose_json` under `words`.
- Responses carry `"engine": "parakeet"`; Whisper responses have no `engine`
  key, so existing clients are unaffected.
- **It is not hallucination-proof.** It returns nothing on digital silence, but
  on some pink-noise levels it produced `"El turismo."` at confidence 0.78.
  Filter on `confidence`, don't assume immunity.

### The VAD pre-pass

`WHISPER_VAD_FILTER` defaults to **on**, and affects every Whisper request. Without it, faster-whisper has two failure modes on real
recordings:

| input | `vad_filter=false` | `vad_filter=true` |
|---|---|---|
| speech after 10s of silence | segment reads `[0.00-12.00]` | `[9.94-…]` |
| speech after 20s of silence | segment reads `[0.00-13.00]` | `[19.95-…]` |
| 12s of digital silence | `"you"` | `""` |
| 12s of room noise, no speech | `"Thank you."` | `""` |
| real speech, even at -50dB | correct | correct (unchanged) |

So segment timings are unusable on any clip with leading silence, and
speech-free audio produces a confabulated phrase that a voice assistant will
then answer. Cost is ~7ms per request. `POST /v1/audio/transcriptions` accepts
a per-request `vad_filter` form field if you need the old behaviour for one
call; `GET /health` reports the server default under `whisper.vad_filter`.

Note the VAD changes `no_speech_prob` slightly (it removes the silence the
score was partly computed over), so calibrate any threshold against the
setting you actually run.

`no_speech_prob` is the useful one for filtering Whisper hallucinations —
real speech scores low (~0.1), while faint background speech picked up by the
OS-level capture scores high (~0.67), so a threshold around `0.6` separates
them. `avg_logprob` gives overall confidence.

### `POST /v1/audio/diarize`

Multipart `file`, plus optional `num_speakers` (pins the count exactly) or
`min_speakers` / `max_speakers` (bound it). Omit all three to let pyannote
decide.

```bash
curl -X POST http://localhost:5000/v1/audio/diarize -F file=@meeting.wav
```

```json
{"turns": [{"start": 0.5, "end": 4.2, "speaker": "SPEAKER_00"}]}
```

Notes:

- The default model `pyannote/speaker-diarization-community-1` is **gated**.
  `HF_TOKEN` must belong to an account that has accepted its terms on
  huggingface.co, or the route answers `503`.
- Unlike faster-whisper (CTranslate2, which has no Metal backend — hence
  `/health` reporting Whisper on `cpu`), pyannote is plain PyTorch and runs on
  **MPS**. On Apple Silicon this is the one stage the GPU really accelerates:
  roughly 28x realtime, i.e. ~9s for a 4-minute recording.
- The pipeline loads **lazily on the first request**, so it costs nothing in
  startup time or memory until used. `/health` reports whether it is loaded.
- Kokoro also uses the GPU (MLX/Metal, or MPS on the torch backend), so a
  long diarization and TTS compete for it. On MPS synthesis slowed roughly 5x
  while one was running; this hasn't been re-measured on the MLX backend.
  Whisper STT is unaffected (it is on CPU) and the server stays responsive
  throughout.

---

## Calling the TTS API from your own tools

The endpoint is OpenAI-compatible: `POST /v1/audio/speech` with JSON
`{input, voice, speed}` returns WAV bytes. From any peer on your network
or Tailscale, swap `localhost` for the host's IP (port `5000`).

### Python — minimal (just `httpx` and the stdlib)

```python
import httpx

r = httpx.post("http://localhost:5000/v1/audio/speech", json={
    "input": "Hello from cleanTTS.",
    "voice": "af_bella",
})
r.raise_for_status()
with open("out.wav", "wb") as f:
    f.write(r.content)
```

That's it — `out.wav` is a standard 24 kHz mono PCM WAV file. Open it in
any audio player, drop it into Discord, attach it to an email, etc.

### Python — speak it immediately (`sounddevice` + `soundfile`)

```python
import io, httpx, sounddevice as sd, soundfile as sf

r = httpx.post("http://localhost:5000/v1/audio/speech", json={
    "input": "Third place, two seconds behind the leader.",
    "voice": "af_bella",
    "speed": 1.1,
})
r.raise_for_status()
audio, sr = sf.read(io.BytesIO(r.content))
sd.play(audio, sr); sd.wait()
```

### Python — drop-in OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="unused")
client.audio.speech.create(
    model="kokoro", voice="af_bella", input="P1, you've got the lead!"
).stream_to_file("out.wav")
```

### curl

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world","voice":"af_bella"}' \
  --output hello.wav
```

### Word and sentence timings — `POST /v1/audio/speech/timed`

Same request body as `/v1/audio/speech`, but the response is JSON: the audio
(base64) plus when every word and sentence is spoken. Use it to advance
slides, show captions or highlight words in sync with the voice.

The timings aren't estimates. Kokoro predicts a duration for every sound in
order to generate the audio, and these are those durations. Checked against
Parakeet's own word timestamps on the same audio, they agreed to within 0.08s
on average.

Put `<mark name="..."/>` anywhere in the text to get a **cue** back: the time
the next word starts. Marks are never spoken, and the plain
`/v1/audio/speech` route strips them too.

```bash
curl -s http://localhost:5000/v1/audio/speech/timed \
  -H 'Content-Type: application/json' \
  -d '{"input": "<mark name=\"title\"/>Welcome to the quarterly update. <mark name=\"slide2\"/>Revenue grew twelve percent.",
       "voice": "bm_lewis", "subtitles": "vtt"}'
```

```jsonc
{
  "audio": "UklGRi...",          // base64, in response_format (default wav)
  "format": "wav", "media_type": "audio/wav", "sample_rate": 24000,
  "duration": 4.9,
  "cues":      [{"name": "title", "at": 0.4}, {"name": "slide2", "at": 2.413}],
  "sentences": [{"text": "Welcome to the quarterly update.", "start": 0.4, "end": 2.413}, ...],
  "words":     [{"word": "Welcome", "start": 0.4, "end": 0.85}, ...],
  "subtitles": "WEBVTT\n\n00:00:00.400 --> 00:00:02.413\nWelcome to ..."  // only if asked
}
```

| Field | Notes |
|---|---|
| `subtitles` | Optional: `vtt` or `srt`, one subtitle per sentence. |
| `response_format` | `wav`, `opus`, `flac` and `pcm` keep timings exact. `mp3`/`aac` encoders add a few tens of ms of leading padding that some players don't strip, so playback can run slightly late. |

- A cue placed at the very end fires at `duration`.
- A sentence's `end` includes the pause after it, so it equals the next
  sentence's `start`. A word's `end` is when the word itself stops.
- **Word timings exist for English voices only** (languages `a`/`b`). Other
  languages return no `words`, and `sentences` falls back to Kokoro's own
  chunks (split at line breaks and in long passages), so put one sentence
  per line if you need sentence timings. Only English has been tested.

### Discoverability

- `GET /v1/voices` — voices currently loaded on the server (depends on
  `KOKORO_LANGS`)
- `GET /v1/voices?model=qwen3-tts` — Qwen3-TTS speakers (loads the model)
- `GET /v1/models` — OpenAI-compat model list (`kokoro`, `qwen3-tts`)

---

## Qwen3-TTS (second TTS engine)

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) runs alongside Kokoro through
`mlx-audio` (Apple Silicon only). Kokoro stays the default; pick Qwen per request:

```bash
# Either name the model on the usual route...
curl -s http://localhost:5000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"qwen3-tts","input":"Hello there.","voice":"ryan"}' -o out.wav
# ...or use the dedicated route, where `model` is optional.
curl -s http://localhost:5000/v1/audio/speech/qwen -H 'content-type: application/json' \
  -d '{"input":"Hello there.","stream":true}' -o out.wav
# A saved (cloned) voice, and a style instruction on a preset voice:
curl -s http://localhost:5000/v1/audio/speech/qwen -H 'content-type: application/json' \
  -d '{"input":"Hello there.","voice":"myvoice"}' -o out.wav
curl -s http://localhost:5000/v1/audio/speech/qwen -H 'content-type: application/json' \
  -d '{"input":"We won!","voice":"aiden","instructions":"Extremely excited."}' -o out.wav
```

- **Voices** (`GET /v1/voices?model=qwen3-tts`): `ryan` and `aiden` are the
  English speakers; also `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`,
  `ono_anna`, `sohee`. A request that sets no `voice` gets `QWEN_TTS_VOICE`
  (`ryan`), not Kokoro's default. An unknown voice is a 400 listing the valid ones.
- **`instructions`** (the OpenAI field name): speaking style, e.g.
  `"Calm and warm."`. Honoured by the 1.7B models; the 0.6B ones ignore it.
- **`stream: true`**: audio is sent as it's generated — `wav` (header with
  unknown length) or `pcm` only. This only helps a client that plays audio
  while it downloads (a live voice channel, a player on a stream URL); one
  that needs the whole file first (a chat attachment) gains nothing, and should
  leave `stream` off to get a WAV with correct size fields. Hanging up stops
  generation.
- **Saved voices** (`"voice": "myvoice"`): any `<name>.wav` + `<name>.txt`
  (5–15s of speech and its exact transcript) in `voices/` (`QWEN_TTS_VOICES_DIR`)
  is a voice, cloned by the Base model on every request. New files work without
  a restart; they're listed under `saved_voices` in `/v1/voices?model=qwen3-tts`.
  `instructions` is ignored for them. ~1.9s for a one-liner (vs ~1.3s preset);
  the first request after a restart adds ~2.5s while the clip is encoded.
- **Designing a voice** (`"model": "qwen3-tts-design"`): `instructions` is the
  description of the voice to invent ("a gravelly old Scottish sea captain").
  It invents a *different* voice on every call, so use it to audition, then
  save the clip you like into `voices/` (with its transcript) to keep it.
- **Not supported on Qwen:** `speed` (ignored), voice blends, and
  `/v1/audio/speech/timed` (always Kokoro — Qwen gives no word timings).
- **Three models** back these, all 1.7B 8bit: CustomVoice (preset voices),
  VoiceDesign (`qwen3-tts-design`) and Base (saved voices).
- Each of the three models (preset, design, saved-voice) loads on its first
  request (~5s from the local cache; `QWEN_TTS_PRELOAD=true` loads all three
  at startup instead — ~9 GB resident). It runs on its own
  MLX thread, so a long Qwen job doesn't queue Kokoro requests.

### Adding a voice from a real recording

A real recording clones far better than a described voice. Measured on one
speaker: six designed takes from a careful description all landed at
160–239 Hz median pitch against the real speaker's 96 Hz, and scored *lower*
speaker similarity to the real voice (0.91) than an unrelated Kokoro voice (0.925). The
clone of a 15s real clip scored 0.97–0.98, and every test line transcribed
back word-for-word.

To add one:

1. **Pick the clip:** one speaker, no music, laughter or applause, natural
   talking. Run it through `POST /v1/audio/diarize` if unsure; a second voice
   would be blended into the clone.
2. **Trim to 5–15s**, cutting in a pause, and convert to 24kHz mono WAV:
   `ffmpeg -i in.wav -t 15 -ac 1 -ar 24000 -c:a pcm_s16le voices/<name>.wav`
3. **Write its exact transcript** to `voices/<name>.txt`. Transcribe it with
   `/v1/audio/transcriptions` and fix any misheard words by ear. The transcript
   has to match the audio.
4. Use it as `"voice": "<name>"`. No restart needed.

About 1 take in 16 drifts noticeably higher in pitch. Lowering the sampling
temperature (0.6 vs the default 0.9, 16 takes each) didn't change that.

**Don't commit voices of real people.** The whole `voices/` folder is
gitignored, so saved voices stay on the machine that made them.

### Kokoro vs Qwen3-TTS, measured

M-series Mac, MLX, median of 3 after warm-up. "First audio" is when the first
chunk is ready when streaming; without streaming it equals the total.
RTF = compute time ÷ audio length (below 1 is faster than real time).

| model | 1-line reply (~4s audio) | 3 sentences (~12s) | paragraph (~38s) | first audio, streamed | RTF | peak mem |
|---|---|---|---|---|---|---|
| **Kokoro-82M** (bf16) | **0.14s** | **0.40s** | **1.26s** | — (not streamed) | **0.035** | 1.4–3.3 GB |
| Qwen 0.6B CustomVoice bf16 | 1.69s | 5.25s | 18.1s | 0.23s | 0.40 | 5.5–9.1 GB |
| Qwen 0.6B CustomVoice 8bit | 1.42s | 3.79s | 14.1s | 0.19s | 0.32 | 5.1–8.7 GB |
| Qwen 1.7B CustomVoice bf16 | 1.78s | 5.69s | 20.2s | 0.28s | 0.48 | 7.5–10.8 GB |
| **Qwen 1.7B CustomVoice 8bit** (default) | 1.28s | 3.70s | 14.1s | 0.22s | 0.36 | 6.2–9.6 GB |
| Qwen 1.7B CustomVoice 4bit | 1.37s | 3.98s | 18.8s* | 0.20s | 0.33 | 6.6–10.8 GB |
| Qwen 1.7B VoiceDesign 8bit | 1.64s | 3.76s | 15.2s | 0.26s | 0.37 | 6.2–9.7 GB |

\* 4bit read the paragraph much more slowly (55s of audio vs ~38s).

- **Accuracy:** every model read the scripts correctly — 0–7% WER when
  transcribed back with Parakeet, the same as Kokoro (the residue is number
  formatting, e.g. "five fifteen" → "5:15").
- **Kokoro is ~10x faster** in total, in every size. Qwen is still ~3x faster
  than real time, so **streamed** it starts in ~0.2s and never stalls playback;
  **not streamed**, a client waits 1.3s for one line and ~14s for a paragraph.
- **8bit is the sweet spot**: faster than bf16 at no measurable accuracy cost,
  and the 1.7B 8bit is as fast as the 0.6B 8bit while also honouring `instructions`.
- **Alongside Kokoro:** during a 15s Qwen job, Kokoro requests went from
  0.24s to 0.30s and Parakeet STT was unaffected.

---

## Voices

Set `KOKORO_LANGS` to a comma-separated list of language codes; cleanTTS
loads one `KPipeline` per language and routes synthesis by the voice's
prefix letter. Default is `a,b` (US + UK English).

| code | language | voices | count |
|---|---|---|---|
| `a` | American English | `af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa` | 20 |
| `b` | British English | `bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis` | 8 |
| `e` | Spanish | `ef_dora em_alex em_santa` | 3 |
| `f` | French | `ff_siwis` | 1 |
| `h` | Hindi | `hf_alpha hf_beta hm_omega hm_psi` | 4 |
| `i` | Italian | `if_sara im_nicola` | 2 |
| `j` | Japanese | `jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo` | 5 |
| `p` | Brazilian Portuguese | `pf_dora pm_alex pm_santa` | 3 |
| `z` | Mandarin | `zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang` | 8 |

On the torch backend each loaded pipeline costs ~300 MB of (V)RAM; on MLX
the languages share one model. Asking for a voice whose
language isn't loaded returns `400` with a message telling you which
code to add to `KOKORO_LANGS`.

Note: Kokoro v1 has no Australian English. British is the closest fit.

### Blending voices

A "voice" is just an embedding tensor, so two or more can be averaged into a
new one. The `voice` field accepts a blend spec anywhere a voice id is used —
the `/v1/audio/speech` API, the UI's voice picker (click **+ Add voice to
blend**), and the assistant:

| Spec | Result |
|---|---|
| `af_bella` | single voice (unchanged) |
| `af_bella,af_jessica` | equal 50/50 blend |
| `af_bella*0.7,af_jessica*0.3` | weighted blend |
| `af_bella*2,af_jessica` | weights are relative; a missing weight defaults to 1 (→ ~67% / 33%) |
| `am_adam,bm_lewis` | voices may even be mixed across languages |

Weights don't need to sum to 1 — they're normalized for you. Voice embeddings
are language-agnostic tensors, so you can blend across languages too
(`am_adam,bm_lewis`). The only requirement is that the **first** voice's
language is loaded (per `KOKORO_LANGS`), since that pipeline phonemizes the
text.

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"half Bella, half Jessica","voice":"af_bella*0.5,af_jessica*0.5"}' \
  --output blend.wav
```

---

## Architecture notes

**Kokoro is pure TTS.** It has no understanding of the text — give it a
string, get a waveform. Your code (or your agent's LLM) decides what to say
and POSTs the string to `/v1/audio/speech`.

**Five engines, two devices.** On Apple Silicon the split is not obvious and
drives most of the performance decisions here:

| stage | runtime | device | why |
|---|---|---|---|
| Whisper STT | faster-whisper / CTranslate2 | **CPU** | no Metal backend exists |
| Parakeet STT | parakeet-mlx / MLX | **GPU** | MLX targets Metal |
| Kokoro TTS | mlx-audio / MLX | **GPU** | MLX targets Metal (PyTorch on CPU is the fallback; MPS is slower than either) |
| Diarization | pyannote / PyTorch | **GPU** | MPS |
| Qwen3-TTS (optional) | mlx-audio / MLX | **GPU** | MLX targets Metal; its own thread, so it never queues Kokoro |

So Whisper is the only stage that cannot use the GPU, which is why bigger
Whisper models cost so much here (`large-v3-turbo` is ~36x slower than
Parakeet) and why Parakeet exists as an option. It also means a long
diarization competes with Kokoro for the GPU (measured ~5x slower synthesis
on the old MPS backend while one runs) but does not slow Whisper, which is on
the other device. A long Qwen3-TTS job costs Kokoro ~25% (0.24s → 0.30s per
request) and left Parakeet unaffected.

---

## Troubleshooting

**Transcripts get much worse, and slower, when I pass `prompt`.** You are
sending it to a distil model. `initial_prompt` biasing is a large win on
`large-v3-turbo` (25.9% → 3.7% WER on names and jargon) and actively harmful on
`distil-small.en`, which was not trained for prompt conditioning and degenerates
into repetition loops at roughly 4x the latency. Send `prompt` only with
`model=turbo`. Parakeet ignores it entirely.

**`no_speech_prob` is always 0.0.** You are on the Parakeet engine (the response
carries `"engine": "parakeet"`). Parakeet has no equivalent score; the field
exists only so Whisper-shaped clients do not crash. Filter on the per-segment
`confidence` instead — around 0.9 separates real speech (>0.96) from
hallucinated output (~0.78).

**The transcript invented "Thank you." out of silence.** That is Whisper
hallucinating on non-speech, and `WHISPER_VAD_FILTER` (on by default) fixes it.
If you turned it off, turn it back on. Parakeet is much more resistant but not
immune — it produced `"El turismo."` on some pink noise.


- **CUDA OOM on the 3080**: switch `WHISPER_MODEL=small` or `tiny`.
- **Kokoro voice not found**: voice ids are case-sensitive, and the voice's
  language must be in `KOKORO_LANGS` (e.g. `bf_emma` requires `b`).
- **Kokoro on MLX fails with an espeak `phontab` error**: espeak-ng can't
  handle very long install paths. Put the venv somewhere with a shorter path.

# cleanTTS

A small Python webapp that runs a local voice assistant **and** exposes an
OpenAI-compatible TTS API for any other tool to use.

Now also has a companion app to his this endpoint from linux when holding a hotkey (super + space) to talk straight to your text line.

NOTE this is all a Claude vibecode special. Claude, take it from here:

- **STT**: `faster-whisper` (local, CUDA)
- **TTS**: Kokoro-82M via the `kokoro` pip package (local, CUDA, ~300MB VRAM)
- **LLM**: any OpenAI-compatible chat endpoint (vLLM, LM Studio, Ollama with
  OpenAI shim, OpenRouter, etc.)
- **Web**: FastAPI + a single HTML page

One process, one port (default `5000`). No Docker, no vLLM-omni, no Voxtral.

---

## What you get

| Endpoint | Purpose |
|---|---|
| `GET /` | Voice assistant UI (mic → STT → LLM → TTS) |
| `WS /ws` | WebSocket for the voice assistant pipeline |
| `POST /v1/audio/speech` | OpenAI-compatible TTS — call this from any external tool |
| `POST /v1/audio/transcriptions` | OpenAI-compatible STT — two engines (Whisper on CPU, Parakeet on GPU), picked per request |
| `POST /v1/audio/diarize` | Speaker diarization — "who spoke when" (pyannote, runs on MPS) |
| `GET /v1/voices` | Available Kokoro voices |
| `GET /v1/models` | OpenAI-compat model list (always returns `kokoro`) |
| `GET /health` | Liveness + which models/devices are loaded |
| `GET /api/llm-models` | Discovery for the configured LLM backend |

---

## Setup

The Python install differs slightly per platform — pick the one that matches
your machine. After install, the same `main.py` runs on all of them; device
selection is handled automatically (`WHISPER_DEVICE=auto`, `KOKORO_DEVICE=auto`).

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

`parakeet-mlx` in `requirements.txt` is Apple-Silicon-only (it pulls
`mlx`/`mlx-metal`). It is optional — Whisper is the default and the server runs
without it; drop that line on Linux/CUDA.

On Apple Silicon, Kokoro runs on MPS (Apple's GPU). `faster-whisper` /
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

> `ffmpeg` is required by Whisper (decoding the browser's WebM audio) and
> by `/v1/audio/speech` for non-WAV response formats (opus/mp3/flac/aac).

---

## Configure

Copy `.env.example` to `.env` and edit it (`.env` is gitignored):

```bash
cp .env.example .env
```

```env
# JSON list of LLM backends. Each entry needs a name (shown in the UI dropdown)
# and a URL pointing at any OpenAI-compatible /v1 endpoint (vLLM, LM Studio, Ollama, etc.).
LLM_BACKENDS=[{"name":"5090","url":"http://100.x.y.z:1234/v1"},{"name":"3080","url":"http://100.x.y.z:1234/v1"}]

# Single-backend fallback (used only if LLM_BACKENDS is unset or invalid)
# LLM_BASE_URL=http://100.x.y.z:1234/v1

WHISPER_MODEL=distil-large-v3                  # or 'small', 'medium', etc.
WHISPER_DEVICE=auto                            # auto | cuda | cpu (mps not supported by faster-whisper)
KOKORO_LANGS=a,b                               # comma-separated; one KPipeline per language
KOKORO_DEVICE=auto                             # auto | cuda | mps | cpu
DEFAULT_VOICE=af_bella
SYSTEM_PROMPT=You are a helpful, concise voice assistant. Keep responses short and natural for speech.
HOST=0.0.0.0
PORT=5000
```

The UI shows a "LLM server" dropdown; switching it refetches that backend's
model list. The first backend in the list is the default.

### Disabling thinking on Qwen3 / reasoning models

Reasoning models (Qwen3 etc.) are too slow for natural voice — by the time
they finish thinking, the conversation has moved on. You can disable
thinking **for this app only** by adding `chat_template_kwargs` to a
backend entry:

```json
{"name":"5090","url":"http://...","chat_template_kwargs":{"enable_thinking":false}}
```

This kwarg is forwarded per-request, so other clients hitting the same
vLLM/LM Studio endpoint still get thinking enabled. Works on any model
whose chat template supports `enable_thinking` (Qwen3 family). Other
models silently ignore unknown kwargs.

---

## Run

```bash
./venv/bin/python main.py
```

First start: Kokoro (~300MB) and Whisper (~1.5GB for `distil-large-v3`)
download from HuggingFace into the local cache. Subsequent starts are fast.

### Accessing the UI

The browser **must** load the app from `localhost` (or HTTPS) — anything
else and it'll block mic access. How to get there depends on where the
browser is running:

- **Same machine (Linux desktop, or macOS)**: just open
  `http://localhost:5000`.
- **Windows browser → WSL2**: still just `http://localhost:5000`. WSL2
  forwards `localhost` from Windows to your WSL distro automatically — no
  SSH needed. Don't use the WSL IP (e.g. `192.168.64.x`) or the Tailscale
  IP, those will trigger the mic block. If forwarding ever flakes, run
  `wsl --shutdown` in PowerShell and start WSL again.
- **Phone / tablet / another machine on your LAN or Tailscale**: SSH-tunnel
  so the device sees the app on its own `localhost`:
  ```bash
  ssh -L 5000:localhost:5000 stavros@<host-ip>
  ```
  Then open `http://localhost:5000` on the phone. Termius (iOS) and Termux
  (Android) can both do this. Requires an SSH server running on the host —
  either Windows OpenSSH server, or `sshd` inside WSL.

External tools calling `POST /v1/audio/speech` are unaffected by all of
this — they don't need a mic, so the LAN/Tailscale IP works fine for them.

---

## Updating

### Python packages (Kokoro, Whisper, etc.)

`requirements.txt` is intentionally unpinned, so you track the latest
releases. To pull newer versions of everything:

```bash
./venv/bin/pip install -U -r requirements.txt
```

The TTS model lives in the `kokoro` package (with `misaki`, its
grapheme→phoneme companion — keep the two in sync, a version mismatch is a
common breakage). To update just those:

```bash
./venv/bin/pip install -U kokoro misaki
```

After any Kokoro/Whisper update, restart `main.py` and run a quick smoke
test before relying on it:

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"update check","voice":"af_bella"}' --output /tmp/test.wav
```

Almost all Kokoro updates within the `0.x` line are drop-in. The app only
touches Kokoro through one function — `synthesize()` in `main.py` — so if a
future release changes the `KPipeline` API or the 24 kHz sample rate, that
one function (plus `KOKORO_SAMPLE_RATE`) is the only place you'd need to
adjust.

### New voices / new languages

New Kokoro voices and languages usually ship as weights on HuggingFace
(`hexgrad/Kokoro-82M`) and download automatically on first use — no package
update required. To expose them:

- **New voice in a language you already load** → add its id to
  `KOKORO_VOICES_BY_LANG` in `main.py` (used for the `/v1/voices` listing;
  synthesis routes by the voice's first letter regardless).
- **New language** → add its code to `KOKORO_VOICES_BY_LANG` *and* to
  `KOKORO_LANGS` in `.env`. A new pipeline (~300 MB) loads for it at startup.

---

## Conversation mode

The mic button toggles **Start conversation / Stop conversation**. Once
started, [Silero VAD](https://github.com/snakers4/silero-vad) (loaded
in-browser via [`@ricky0123/vad-web`](https://github.com/ricky0123/vad))
listens continuously: when you stop speaking it sends the utterance for
processing, then resumes listening after the assistant replies.

**Barge-in is supported.** While the assistant is speaking, you can just
start talking — it cuts off mid-word, the server cancels the in-flight
LLM/TTS pipeline, and the assistant remembers what it had said up to that
point so the next turn picks up coherently.

The first time you click *Start conversation*, the browser downloads the
~1.5 MB Silero ONNX model (cached after that).

### Echo / feedback loop

Barge-in relies on the browser's built-in **acoustic echo cancellation**
(AEC) so your speakers don't trigger the mic. Modern Chrome/Edge/Firefox
do this by default for `getUserMedia`, and it works well in most setups.
If you find the assistant interrupts itself:

- **Easiest fix**: use headphones — physically removes the loop.
- Lower your speaker volume so AEC has an easier time.
- Some Linux audio stacks (PipeWire) need extra config; `echo-cancel` is
  a separate module there.

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

- **Agent, not daemon.** Kokoro and pyannote need MPS, and Metal is only
  reachable from inside a logged-in user session. A LaunchDaemon runs in the
  system context before login and cannot use the GPU.
- **`PATH` must be set explicitly.** launchd hands a job
  `/usr/bin:/bin:/usr/sbin:/sbin` with no Homebrew, so `ffmpeg` would be
  invisible — breaking `/v1/audio/diarize` and every non-WAV
  `/v1/audio/speech` format, at request time rather than at startup.
- **`WorkingDirectory` must be the repo.** `main.py` resolves `.env`,
  `templates/` and `static/` relatively.

The agent starts at **login**, not at boot. With FileVault enabled that is the
best available: the volume is encrypted until someone enters the password at
the boot screen, so nothing — agent or daemon — can run before that. After the
unlock (which logs you in) the service comes up on its own and `KeepAlive`
keeps it up.

### Binding, and what it means for `localhost`

`HOST=0.0.0.0` binds every interface, which is only safe behind a NAT or
firewall — cleanTTS has no authentication and proxies to the configured LLM
backends. On a machine whose `en0` holds a globally routable address (check
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
  `STT_MODEL=parakeet` moves `/ws` onto it too: measured 0.44s earlier
  first-audio on a voice turn.
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

`WHISPER_VAD_FILTER` defaults to **on**, and affects both this endpoint and the
`/ws` voice pipeline. Without it, faster-whisper has two failure modes on real
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

The `model` field is accepted for compatibility but ignored: the server loads
`WHISPER_MODEL` once at startup and serves every request from it.

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
- Kokoro also uses MPS, so a long diarization and TTS compete for the GPU:
  synthesis slows roughly 5x while one is running. STT is unaffected (it is on
  CPU) and the server stays responsive throughout.

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

### Discoverability

- `GET /v1/voices` — voices currently loaded on the server (depends on
  `KOKORO_LANGS`)
- `GET /v1/models` — OpenAI-compat model list (always returns `kokoro`)

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

Each loaded pipeline costs ~300 MB VRAM. Asking for a voice whose
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
string, get a waveform. All "thinking" happens in the LLM. This means:

- For the voice assistant, the LLM (vLLM on your remote machine) generates
  the reply, and Kokoro speaks it.
- For external tools (race telemetry, alerts, etc.), your code already knows
  what to say — just POST the string to `/v1/audio/speech`. **No LLM
  required** for that path.

**Three models, two devices.** On Apple Silicon the split is not obvious and
drives most of the performance decisions here:

| stage | runtime | device | why |
|---|---|---|---|
| Whisper STT | faster-whisper / CTranslate2 | **CPU** | no Metal backend exists |
| Parakeet STT | parakeet-mlx / MLX | **GPU** | MLX targets Metal |
| Kokoro TTS | PyTorch | **GPU** | MPS |
| Diarization | pyannote / PyTorch | **GPU** | MPS |

So Whisper is the only stage that cannot use the GPU, which is why bigger
Whisper models cost so much here (`large-v3-turbo` is ~36x slower than
Parakeet) and why Parakeet exists as an option. It also means a long
diarization competes with Kokoro for the GPU — measured ~5x slower synthesis
while one runs — but does not slow Whisper, which is on the other device.

**Sentence-level streaming.** During the voice assistant flow, LLM tokens
are accumulated until a sentence boundary, at which point that sentence is
queued for TTS while the LLM keeps generating. First audio plays before the
LLM has finished thinking.

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


- **No mic in browser**: you're on `http://<lan-ip>:5000`. Use `localhost`
  or HTTPS.
- **CUDA OOM on the 3080**: switch `WHISPER_MODEL=small` or `tiny`.
- **`OSError: PortAudio` / no sound**: check the browser's tab volume; the
  audio is sent as base64 WAV over the WebSocket.
- **Kokoro voice not found**: voice ids are case-sensitive and must match
  the configured `KOKORO_LANG` (e.g. `bf_emma` requires `KOKORO_LANG=b`).

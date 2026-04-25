# cleanTTS

A small Python webapp that runs a local voice assistant **and** exposes an
OpenAI-compatible TTS API for any other tool to use.

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
| `GET /v1/voices` | Available Kokoro voices |
| `GET /v1/models` | OpenAI-compat model list (always returns `kokoro`) |
| `GET /api/llm-models` | Discovery for the configured LLM backend |

---

## Setup

```bash
# create a venv inside the repo
python3 -m venv venv

# install PyTorch with CUDA wheels first (required so kokoro/whisper see CUDA)
./venv/bin/pip install --upgrade pip
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126

# everything else
./venv/bin/pip install -r requirements.txt
```

System packages: `ffmpeg` is required so Whisper can decode the browser's
WebM audio. Install with `sudo apt install ffmpeg` on Debian/Ubuntu.

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
WHISPER_DEVICE=cuda                            # or 'cpu'
KOKORO_LANG=a                                  # a=US-English, b=UK, j=JP, z=Mandarin, ...
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

## Calling the TTS API from your own tools

Any OpenAI TTS client works. Plain `httpx`:

```python
import httpx, sounddevice as sd, soundfile as sf, io

r = httpx.post("http://localhost:5000/v1/audio/speech", json={
    "input": "Third place, two seconds behind the leader.",
    "voice": "af_bella",
    "speed": 1.1,
})
r.raise_for_status()
audio, sr = sf.read(io.BytesIO(r.content))
sd.play(audio, sr); sd.wait()
```

Or the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
audio = client.audio.speech.create(model="kokoro", voice="af_bella",
                                   input="P1, you've got the lead!")
audio.stream_to_file("out.wav")
```

Or `curl`:

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world","voice":"af_bella"}' \
  --output hello.wav
```

---

## Voices

US English (`KOKORO_LANG=a`):
`af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa`

UK English (`KOKORO_LANG=b`):
`bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis`

Other Kokoro languages (`j`, `z`, `e`, `f`, `h`, `i`, `p`) work too — set
`KOKORO_LANG` and pass any voice id Kokoro supports for that language.

---

## Architecture notes

**Kokoro is pure TTS.** It has no understanding of the text — give it a
string, get a waveform. All "thinking" happens in the LLM. This means:

- For the voice assistant, the LLM (vLLM on your remote machine) generates
  the reply, and Kokoro speaks it.
- For external tools (race telemetry, alerts, etc.), your code already knows
  what to say — just POST the string to `/v1/audio/speech`. **No LLM
  required** for that path.

**Sentence-level streaming.** During the voice assistant flow, LLM tokens
are accumulated until a sentence boundary, at which point that sentence is
queued for TTS while the LLM keeps generating. First audio plays before the
LLM has finished thinking.

---

## Troubleshooting

- **No mic in browser**: you're on `http://<lan-ip>:5000`. Use `localhost`
  or HTTPS.
- **CUDA OOM on the 3080**: switch `WHISPER_MODEL=small` or `tiny`.
- **`OSError: PortAudio` / no sound**: check the browser's tab volume; the
  audio is sent as base64 WAV over the WebSocket.
- **Kokoro voice not found**: voice ids are case-sensitive and must match
  the configured `KOKORO_LANG` (e.g. `bf_emma` requires `KOKORO_LANG=b`).

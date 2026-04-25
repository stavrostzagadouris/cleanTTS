---
name: cleanTTS
description: Convert text to speech via a self-hosted Kokoro-82M server. POST text, get WAV bytes back. Use when audio output is needed — Discord file attachments, voice-channel playback, alerts, race callouts, anything that should sound like a human voice.
---

# cleanTTS — Text-to-Speech Skill

Self-hosted, OpenAI-compatible TTS server running on the home network. No
internet round-trip, no metering, ~75–300 ms warm latency on a 3080.

> **Replace `100.x.y.z` everywhere below with the Tailscale IP of the
> machine running cleanTTS.** That's the only configuration this skill
> needs.

---

## When to use this skill

- The user asks for a voice message, an audio file, or "say it out loud".
- You want to send a Discord audio attachment instead of plain text.
- Bot is in a voice channel and needs to speak.
- Any time you have text that the user would rather hear than read.

## When NOT to use this skill

- You need transcription (this is TTS only — no STT).
- You need voice cloning (Kokoro doesn't support custom voices).
- You need live token-by-token audio streaming (this returns the full WAV
  at once; sub-sentence streaming is not implemented yet).

---

## Endpoint

```
POST http://100.x.y.z:5000/v1/audio/speech
Content-Type: application/json
```

No auth header needed (open on the Tailscale network).

### Request body

| field             | type   | default     | notes                                                      |
| ----------------- | ------ | ----------- | ---------------------------------------------------------- |
| `input`           | string | (required)  | Text to synthesize. Use punctuation for natural pauses.    |
| `voice`           | string | `af_bella`  | See voice list below.                                      |
| `speed`           | float  | `1.0`       | Range 0.25–4.0. ~1.1–1.2 sounds energetic for short calls. |
| `response_format` | string | `wav`       | Only `wav` is supported.                                   |
| `model`           | string | `kokoro`    | Ignored; always Kokoro-82M.                                |

### Response

- **200 OK** — body is `audio/wav`, mono PCM_16 @ 24 kHz. Length depends
  on input; ~3 KB per second of audio plus a 44-byte header.
- **400** — bad input (empty text, unsupported `response_format`).
- **500** — synthesis failure (rare; usually invalid voice id for the
  configured language).

---

## Voices

Voice ids are `{lang}{gender}_{name}`. Prefix letter = language:
`a`=US English, `b`=British English. (No Australian — Kokoro v1 doesn't
have it; British is the closest stylistic match.)

US English (always available — default config):
```
af_alloy   af_aoede   af_bella   af_heart   af_jessica
af_kore    af_nicole  af_nova    af_river   af_sarah   af_sky
am_adam    am_echo    am_eric    am_fenrir  am_liam
am_michael am_onyx    am_puck    am_santa
```

British English (available when server's `KOKORO_LANGS` includes `b`):
```
bf_alice  bf_emma   bf_isabella bf_lily
bm_daniel bm_fable  bm_george   bm_lewis
```

Kokoro also supports Spanish (`e`), French (`f`), Hindi (`h`), Italian
(`i`), Japanese (`j`), Brazilian Portuguese (`p`), and Mandarin (`z`) —
the host has to opt in by adding those codes to `KOKORO_LANGS`. Hit
`GET http://100.x.y.z:5000/v1/voices` to see what's actually loaded right
now.

---

## Examples

### Python (httpx, sync)
```python
import httpx

r = httpx.post("http://100.x.y.z:5000/v1/audio/speech", json={
    "input": "Hello from cleanTTS.",
    "voice": "af_bella",
})
r.raise_for_status()
with open("out.wav", "wb") as f:
    f.write(r.content)
```

### Python (aiohttp, async — typical Discord bot)
```python
import aiohttp

async def synthesize(text: str, voice: str = "af_bella") -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://100.x.y.z:5000/v1/audio/speech",
            json={"input": text, "voice": voice},
        ) as r:
            r.raise_for_status()
            return await r.read()  # WAV bytes
```

### curl
```bash
curl -X POST http://100.x.y.z:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello there","voice":"af_bella"}' \
  --output out.wav
```

### OpenAI SDK (drop-in)
```python
from openai import OpenAI
client = OpenAI(base_url="http://100.x.y.z:5000/v1", api_key="unused")
client.audio.speech.create(
    model="kokoro", voice="af_bella", input="Hi"
).stream_to_file("out.wav")
```

---

## Discord integration

### As a file attachment (works in any channel)
```python
import io, aiohttp, discord

async def send_tts(channel: discord.abc.Messageable, text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://100.x.y.z:5000/v1/audio/speech",
            json={"input": text, "voice": "af_bella"},
        ) as r:
            r.raise_for_status()
            wav = await r.read()
    await channel.send(file=discord.File(io.BytesIO(wav), filename="speech.wav"))
```

### Playing in a voice channel
```python
import tempfile, aiohttp, discord

async def speak_in_voice(voice_client: discord.VoiceClient, text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://100.x.y.z:5000/v1/audio/speech",
            json={"input": text, "voice": "af_bella"},
        ) as r:
            r.raise_for_status()
            wav = await r.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        path = f.name
    voice_client.play(discord.FFmpegPCMAudio(path))
```
Requires `ffmpeg` on the bot host.

---

## Operational notes

- **Latency**: 75–300 ms warm for typical sentence-length input. Cold call
  (first hit after server start) can be ~700 ms.
- **Throughput**: 25–65× realtime on the host's RTX 3080. Concurrent
  requests serialize through a single Kokoro instance — for bursts,
  batch into one call where possible.
- **Failure to connect**: the host might be down, asleep, or off the
  Tailscale net. If `aiohttp` raises `ClientConnectorError` or the
  request times out, tell the user "the TTS host isn't reachable" rather
  than retrying forever.
- **Long passages**: split into shorter calls — multiple short
  syntheses finish faster than one long one and let you start playback
  on the first chunk.

## Tips for natural-sounding output

- Punctuation matters. Periods/commas give clean pauses; missing
  punctuation produces a rushed read.
- Acronyms: "API" reads as "A-P-I" letter-by-letter. If you want "appy",
  spell it phonetically.
- Numbers usually read fine ("P3", "two seconds behind"). For specific
  pronunciations, spell things out.
- For race-style short callouts, `speed: 1.1` to `1.15` adds urgency
  without making it sound chipmunky.

# Next steps — cross-platform support (CUDA / Apple Silicon)

Picking this back up later. Goal: same `main.py` runs on a CUDA box (the
current setup) and an M1/M2/M3 Mac. No Docker. No second copy of the repo.

## Why a single codebase, not an `M1mac/` folder

The original idea was to fork the repo into `M1mac/` with rewrites for
Apple Silicon. Rejected because every API change, voice list update, or
barge-in fix would have to be done twice and the two copies will drift.

Instead: keep one `main.py`, auto-detect the platform at startup
(CUDA → MPS → CPU), and put the install commands that differ per OS into
the README (or a short `README-macos.md`).

## What actually changes per concern

### 1. PyTorch install

- **CUDA box (current):** `pip install torch --index-url https://download.pytorch.org/whl/cu126`
- **Mac (Apple Silicon):** `pip install torch` — no special index. MPS
  support is built into the default macOS wheel.

This is a README-only change; nothing in `main.py` cares.

### 2. Whisper (STT)

`faster-whisper` uses CTranslate2 under the hood, which has **no MPS
backend**. Two options on Mac:

- **a) Stay on `faster-whisper`, run on CPU with `int8`.** M-series CPUs
  are fast enough that `distil-large-v3` at int8 still hits real-time on
  short utterances. Zero code change beyond device detection.
- **b) Swap to `mlx-whisper` on Mac** for true GPU (Apple Neural
  Engine / Metal) acceleration. Different API, so this means a small
  abstraction layer in `main.py`.

Recommended start: **option (a)**. Add `mlx-whisper` later if perf is
unacceptable.

### 3. Kokoro (TTS)

The `kokoro` pip package uses PyTorch. On MPS it works but a couple of
ops fall back to CPU (well-known issue), so latency is mediocre. Two
options:

- **a) Use `kokoro` on MPS as-is.** Slower than CUDA but acceptable for
  short replies; no code change.
- **b) Use `kokoro-onnx` with CoreML execution provider** on Mac. Faster
  on Apple Silicon and avoids the PyTorch op-fallback problem. Different
  API → small abstraction.

Recommended start: **option (a)**. Switch to (b) if first-token latency
is bad.

### 4. ffmpeg

Currently a system dep on Linux (`apt install ffmpeg`). On Mac:
`brew install ffmpeg`. README change only.

### 5. Whisper compute_type

Already partially handled: `compute_type = "float16" if WHISPER_DEVICE
== "cuda" else "int8"`. This logic is fine for CPU on Mac. Just need
`WHISPER_DEVICE=cpu` (or auto) in the Mac `.env`.

## Concrete plan when we pick this up

1. **Add device auto-detection** in `main.py`:
   ```python
   def _auto_device() -> str:
       try:
           import torch
           if torch.cuda.is_available():
               return "cuda"
           if torch.backends.mps.is_available():
               return "mps"
       except Exception:
           pass
       return "cpu"
   ```
   Use this when `WHISPER_DEVICE` (and a new optional `KOKORO_DEVICE`)
   is unset or set to `auto`.

2. **Whisper device handling.** `faster-whisper` only accepts `cpu` and
   `cuda`. If auto-detect returns `mps`, downgrade Whisper to `cpu` (with
   a log line explaining why). Kokoro can still use MPS independently.

3. **Kokoro device handling.** The `KPipeline` constructor accepts a
   `device` arg. Plumb it from env (`KOKORO_DEVICE=auto|cuda|mps|cpu`)
   so Mac users can pick MPS or CPU explicitly.

4. **README updates.** Add a short macOS section: brew deps, no `cu126`
   index for torch, recommended env values
   (`WHISPER_DEVICE=cpu`, `KOKORO_DEVICE=mps`).

5. **Smoke-test on Mac.** Hit `/v1/audio/speech` (Kokoro) and the WS
   voice path (Whisper). Note first-token latency; if Kokoro is too slow
   on MPS, do step 6.

6. **(Optional) `kokoro-onnx` fast path.** Behind an env flag like
   `KOKORO_BACKEND=torch|onnx`. Keep `torch` as default, `onnx` as the
   Mac-recommended option. Means a thin wrapper around `synthesize()`
   that selects the backend.

7. **(Optional) `mlx-whisper` fast path.** Same shape: env flag, thin
   wrapper. Only if `faster-whisper` on CPU is too slow.

## Open questions to decide later

- Do we want one `.env.example` with all the platform variants commented
  out, or separate `.env.example.linux` / `.env.example.macos`? Probably
  the first — less file sprawl.
- For the openclaw and GT race tracker (downstream consumers of
  `/v1/audio/speech`), nothing changes — same API, just different
  hardware behind it. No client work needed.
- If we later add `kokoro-onnx`: the voice catalog (`KOKORO_VOICES_BY_LANG`)
  may need verification — onnx port may not have every voice.

## What is *not* in scope

- Windows. If someone wants Windows, it's WSL2 + the existing CUDA path
  (or CPU-only).
- Docker. Out of scope per project ethos.
- A separate `M1mac/` folder. Explicitly rejected above.

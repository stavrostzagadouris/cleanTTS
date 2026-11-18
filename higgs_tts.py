"""Higgs TTS 3 (Boson AI, bosonai/higgs-tts-3-4b) on MLX, for cleanTTS.

mlx-audio ships the model but its generate() only returns audio once the whole
take is finished, even with stream=True, and it re-reads the voice's reference
clip into the model on every call. This module drives the same model with its
own loop, adding two things:

1. **A reusable reference prefix.** A cloned voice's prompt is
   [tts][ref_text …][ref_audio][~420 frames of ref codes][text …][audio]. Everything
   before [text] is the same on every request, so its KV cache is computed once
   per voice and copied. Measured on an M1 Max: ~1 s of prefill per
   request for a 17 s clip, now paid once per voice.
2. **Real streaming.** The codec decoder is convolutional and non-causal: a
   frame's samples depend on ~8 frames before it and ~6 after. Audio is decoded
   in sliding windows with that much context either side, so streamed audio
   matches a one-shot decode to within 0.005 peak (inaudible; see LOOKAHEAD).

Everything runs on whatever single thread the caller uses (MLX state is
per-thread), and nothing here is thread-safe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

FRAME_SAMPLES = 960      # codec hop at 24 kHz (25 frames/s)
LEFT_CTX = 12            # frames of left context per decode window (8 suffices)
# Frames held back until their right context exists. Measured seam error vs a
# one-shot decode: 6 -> 0.02 peak, 8 -> 0.005 (-46 dB), 10 -> 2e-6. Each frame
# costs ~21 ms of first-audio latency.
LOOKAHEAD = 8
# Streamed chunk sizes in frames (40 ms each): small first for latency, then
# growing so each chunk is generated (RTF ~0.6) before the last one finishes
# playing. A flat 4-then-12 stalled ~90 ms after the first chunk.
CHUNKS = (4, 6, 9, 12)
# Silence appended to every reference clip before encoding. A clip that ends
# right on its last word primes the model to start the new line mid-speech:
# with one test clip it swallowed the opening "I" of a paragraph in 9/12
# takes; with 0.5 s of padding, 0/12.
REF_PAD_S = 0.5


@dataclass
class _Prefix:
    stamp: tuple
    n_tokens: int
    kv: list = field(default_factory=list)   # per layer: (keys, values)


class HiggsEngine:
    def __init__(self, repo: str):
        from mlx_audio.tts.utils import load_model

        self.repo = repo
        self.model = load_model(repo)
        cfg = self.model.config
        self.n_cb = cfg.audio_num_codebooks
        self.boc = cfg.audio_boc_token_id
        self.eoc = cfg.audio_eoc_token_id
        self.sample_rate = self.model.sample_rate
        self._prefixes: dict[str, _Prefix] = {}

    # -- prompt ------------------------------------------------------------

    def _prefix(self, ref_wav: str | None, ref_text: str | None) -> _Prefix:
        """The KV cache of everything before the target text, cached per
        reference clip (rebuilt if the clip or its transcript changes)."""
        from mlx_audio.lm.models.cache import make_prompt_cache
        from mlx_audio.tts.models.higgs_audio_v3.prompt import ReferenceCodes

        key = ref_wav or ""
        stamp = (
            (os.path.getmtime(ref_wav), os.path.getsize(ref_wav), ref_text)
            if ref_wav else (None,)
        )
        hit = self._prefixes.get(key)
        if hit and hit.stamp == stamp:
            return hit

        m = self.model
        builder = m._prompt_builder
        if ref_wav:
            from mlx_audio.utils import load_audio

            clip = np.asarray(load_audio(ref_wav, sample_rate=self.sample_rate), dtype=np.float32).reshape(-1)
            clip = np.concatenate([clip, np.zeros(int(REF_PAD_S * self.sample_rate), np.float32)])
            codes = m.encode_reference_audio(clip)
            refs = [ReferenceCodes(codes=codes, text=ref_text or None)]
        else:
            refs = []
        # Build the prompt for an empty target and drop its [text][audio] tail.
        full, n = m._build_prompt_embeddings("", refs)
        tail = len(builder.build_prompt("", references=[]).token_ids) - 1  # [text][audio]
        prefix_emb = full[:, : n - tail]
        cache = make_prompt_cache(m)
        m.backbone(mx.zeros((1, prefix_emb.shape[1]), dtype=mx.int32), cache=cache,
                   input_embeddings=prefix_emb)
        kv = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in cache]
        mx.eval([t for pair in kv for t in pair])
        p = _Prefix(stamp=stamp, n_tokens=prefix_emb.shape[1], kv=kv)
        self._prefixes[key] = p
        return p

    def _tail_embeddings(self, text: str) -> mx.array:
        b = self.model._prompt_builder
        ids = [b.text_id, *b.encode_text(text), b.audio_id]
        return self.model._text_embeddings(ids)[None]

    # -- generation --------------------------------------------------------

    def generate(
        self,
        text: str,
        ref_wav: str | None = None,
        ref_text: str | None = None,
        *,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = None,
        max_frames: int = 2048,
        stream: bool = False,
        fade_in_ms: float = 30.0,
        fade_out_ms: float = 15.0,
    ):
        """Yield float32 mono audio chunks at self.sample_rate. With
        stream=False, one chunk once the take is done."""
        from mlx_audio.lm.models.cache import make_prompt_cache
        from mlx_audio.tts.models.higgs_audio_v3.generation import (
            HiggsSamplerState, reverse_delay_pattern, step,
        )

        m = self.model
        prefix = self._prefix(ref_wav, ref_text)
        cache = make_prompt_cache(m)
        for c, (k, v) in zip(cache, prefix.kv):
            c.keys, c.values, c.offset = k, v, prefix.n_tokens
        emb = self._tail_embeddings(text)
        hidden = m.backbone(mx.zeros((1, emb.shape[1]), dtype=mx.int32), cache=cache,
                            input_embeddings=emb)
        last = hidden[:, -1, :]

        state = HiggsSamplerState(num_codebooks=self.n_cb)
        rows: list[mx.array] = []
        emitted = 0          # raw frames already sent
        sent = 0             # chunks already sent
        n_in = int(fade_in_ms * self.sample_rate / 1000)
        n_out = int(fade_out_ms * self.sample_rate / 1000)

        def decode(start: int, end: int, final: bool) -> np.ndarray:
            raw = reverse_delay_pattern(mx.stack(rows, axis=0).astype(mx.int32))
            lo = max(0, start - LEFT_CTX)
            hi = raw.shape[0] if final else min(raw.shape[0], end + LOOKAHEAD)
            audio = np.array(m._codec.decode(raw[lo:hi]).astype(mx.float32)).reshape(-1)
            a = (start - lo) * FRAME_SAMPLES
            out = audio[a:] if final else audio[a : a + (end - start) * FRAME_SAMPLES]
            out = out.copy()
            if start == 0 and n_in and out.size > n_in:
                out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
            if final and n_out and out.size > n_out:
                out[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
            return out

        for _ in range(max_frames):
            logits = m._audio_logits(last)[0]
            codes = step(logits, state, temperature=float(temperature), top_p=top_p,
                         top_k=top_k, boc_id=self.boc, eoc_id=self.eoc)
            rows.append(codes)
            if state.generation_done:
                break
            hidden = m.backbone(mx.zeros((1, 1), dtype=mx.int32), cache=cache,
                                input_embeddings=m._embed_audio_codes(codes)[None])
            last = hidden[:, -1, :]
            if stream:
                ready = len(rows) - (self.n_cb - 1) - LOOKAHEAD   # frames with full right context
                if ready - emitted >= CHUNKS[min(sent, len(CHUNKS) - 1)]:
                    yield decode(emitted, ready, final=False)
                    emitted = ready
                    sent += 1

        if len(rows) >= self.n_cb:
            out = decode(emitted, -1, final=True)
            if out.size:
                yield out

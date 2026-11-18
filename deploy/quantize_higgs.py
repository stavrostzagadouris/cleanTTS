"""Make a quantized MLX copy of Higgs TTS 3 for HIGGS_TTS_MODEL.

    python3 deploy/quantize_higgs.py ~/models/higgs-tts-3-4b-8bit [--bits 8]

mlx-audio's converter quantizes the backbone but silently drops the audio codec
(it lives inside the original checkpoint under a prefix the converter filters
out), and the result then fails to load with "No safetensors with codec prefix
… found". This script converts, then copies the codec tensors back in at full
precision and registers them in the index.

Measured on an M1 Max with a cloned voice: bf16 RTF ~0.83, 8-bit ~0.63, 6-bit
~0.63 (no faster than 8-bit, so not worth it). Peak memory 9.4 / 5.7 / 4.7 GB.
"""
import argparse
import json
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file

SRC = "bosonai/higgs-tts-3-4b"
CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--group-size", type=int, default=64)
a = ap.parse_args()
out = os.path.expanduser(a.out)

from mlx_audio.convert import convert

convert(SRC, mlx_path=out, quantize=True, q_bits=a.bits, q_group_size=a.group_size)

src = snapshot_download(SRC)
weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
by_shard: dict[str, list[str]] = {}
for key, shard in weight_map.items():
    if key.startswith(CODEC_PREFIX):
        by_shard.setdefault(shard, []).append(key)
codec = {}
for shard, keys in by_shard.items():
    with safe_open(os.path.join(src, shard), framework="pt") as f:  # pt: numpy can't read bf16
        for key in keys:
            codec[key] = f.get_tensor(key).float().numpy()
save_file(codec, os.path.join(out, "codec.safetensors"))

index_path = os.path.join(out, "model.safetensors.index.json")
index = json.load(open(index_path))
index["weight_map"].update({key: "codec.safetensors" for key in codec})
json.dump(index, open(index_path, "w"), indent=2)
print(f"Copied {len(codec)} codec tensors into {out}")

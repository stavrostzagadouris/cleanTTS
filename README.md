# Local Voice Assistant with Qwen3-TTS and vLLM

This project is a low-latency local voice assistant designed for a split-machine architecture. It runs transcription and voice generation on your local machine and the heavy language model on a remote server.

## 🏗️ Architecture

1.  **Local Machine (e.g., RTX 3080, 10GB)**: 
    - Runs the **Web App** (FastAPI).
    - Runs **Faster-Whisper** for instant Speech-to-Text.
    - Runs **Qwen3-TTS** via `vllm-omni` for "talking back".
2.  **Remote Machine (e.g., RTX 5090 via Tailscale)**:
    - Runs the **LLM** (Chat Model) via `vLLM`.

---

## 🛠️ Step 1: Remote LLM Setup (5090 Machine)

Start your chat model on the remote server:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 1234 --gpu-memory-utilization 0.9
```

---

## 🎙️ Step 2: Local TTS Setup (3080 Machine)

### 1. Install System Dependencies
Qwen3-TTS requires the CUDA toolkit for model compilation and audio libraries for processing:

```bash
sudo apt update
sudo apt install -y ffmpeg sox nvidia-cuda-toolkit
```

### 2. Install vLLM-Omni
```bash
# Recommendation: Use a virtual environment
pip install -U vllm
pip install git+https://github.com/vllm-project/vllm-omni.git
```

### 3. Start Qwen3-TTS
Since Whisper also uses the 3080, we limit vLLM's memory usage to 50%:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.5
```

---

## 💻 Step 3: Web App Configuration

1.  Install app requirements: `pip install -r requirements.txt`
2.  Configure your `.env` file:
    ```env
    # Remote IP of your 5090
    LM_STUDIO_BASE_URL=http://100.x.y.z:1234/v1

    # Stays localhost because TTS is running on the 3080
    VOXTRAL_TTS_BASE_URL=http://localhost:8000/v1

    WHISPER_MODEL=distil-large-v3
    WHISPER_DEVICE=cuda
    ```
3.  Run the app: `python run_test.py`

---

## ⚠️ Key Learnings & Troubleshooting

- **Microphone Security**: Browsers block mic access on `http://100.x.y.z`. You **must** access the UI via `http://localhost:5000`. If you are using a separate device (like a phone), use an SSH tunnel: `ssh -L 5000:localhost:5000 user@3080-ip`.
- **NVCC Error**: If you see `InductorError: OSError: nvcc`, ensure `nvidia-cuda-toolkit` is installed or use the `--enforce-eager` flag in vLLM.
- **VRAM Conflicts**: If the app crashes on the 3080, lower the `--gpu-memory-utilization` in the TTS command or change `WHISPER_MODEL` to `small` in `.env`.

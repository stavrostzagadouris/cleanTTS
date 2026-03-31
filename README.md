# Local Voice Assistant with Voxtral TTS and LM Studio

This project allows you to build a local voice assistant that runs on your computer or phone. It uses a combination of local models to provide a fast, privacy-preserving, and customizable voice interaction experience.

## Architecture

1. **Speech-to-Text (STT):** `faster-whisper` (running locally on your machine) transcribes your voice.
2. **Language Model (LLM):** Connects to any local model running via **LM Studio**.
3. **Text-to-Speech (TTS):** Streams responses using Mistral's **Voxtral-4B-TTS-2603** running locally via `vllm-omni`.

## Prerequisites

1. An NVIDIA GPU with at least 16GB of VRAM (e.g., RTX 5090) to run both `vllm` and `faster-whisper` concurrently.
2. [Python 3.10+](https://www.python.org/downloads/)
3. [LM Studio](https://lmstudio.ai/) installed.

## Setup Instructions

### 1. Configure LM Studio

1. Open LM Studio.
2. Download a fast conversational model of your choice (e.g., `Llama-3-8B-Instruct` or `Mistral-7B-Instruct`).
3. Start the **Local Server** in LM Studio. Make sure the port is `1234` (the default). The endpoint should be `http://localhost:1234/v1`.

### 2. Start Voxtral TTS via vLLM-Omni

Mistral's Voxtral TTS runs best using `vllm-omni`. You will need to install and run it in a separate terminal.

```bash
# It is highly recommended to do this in a virtual environment
pip install -U vllm
pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade

# Start the vllm server for Voxtral TTS
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8000
```
*Note: The first time you run this, it will download the ~8GB model weights from Hugging Face.*

### 3. Start the Web App

In a new terminal, install the dependencies for this web application and start the backend.

```bash
# Install requirements
pip install -r requirements.txt

# Create or modify the .env file if your ports are different
# LM_STUDIO_BASE_URL=http://localhost:1234/v1
# VOXTRAL_TTS_BASE_URL=http://localhost:8000/v1
# WHISPER_MODEL=distil-large-v3
# WHISPER_DEVICE=cuda

# Run the backend server
uvicorn main:app --host 0.0.0.0 --port 5000
```

### 4. Access the App

1. Open your web browser and go to `http://localhost:5000`.
2. **From your phone:** Connect your phone to the same Wi-Fi network as your computer, and navigate to `http://<YOUR_COMPUTER_IP>:5000`.
3. Select the LLM model from LM Studio.
4. Select the Voxtral TTS voice you want to use.
5. Click the **microphone icon** to start speaking, and click it again to stop and send your audio.

Enjoy your local voice assistant!

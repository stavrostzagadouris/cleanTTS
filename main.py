import os
from fastapi import FastAPI, WebSocket, Request, HTTPException, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import httpx
from faster_whisper import WhisperModel
import json
import logging
import asyncio
import tempfile
import base64
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
VOXTRAL_TTS_BASE_URL = os.getenv("VOXTRAL_TTS_BASE_URL", "http://localhost:8000/v1")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

# Initialize FastAPI
app = FastAPI(title="Local Voice Assistant")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Whisper Model (lazy load if possible, but fast init for realtime)
try:
    logger.info(f"Loading Whisper Model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute_type)
    logger.info("Whisper Model loaded.")
except Exception as e:
    logger.error(f"Failed to load Whisper on {WHISPER_DEVICE}, falling back to CPU and tiny.en. Error: {e}")
    # Fallback for local testing if GPU is unavailable or out of memory
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    logger.info("Whisper Model loaded on CPU.")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/models")
async def get_models():
    """Fetch available models from LM Studio."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{LM_STUDIO_BASE_URL}/models")
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models from LM Studio: {e}")
        return {"models": [], "error": str(e)}

@app.get("/api/tts-models")
async def get_tts_models():
    """Fetch available models from the TTS server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{VOXTRAL_TTS_BASE_URL}/models")
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models from TTS server: {e}")
        return {"models": [], "error": str(e)}

@app.get("/api/voices")
async def get_voices():
    """Return available preset voices for Voxtral TTS."""
    voices = [
        "ar_male", "casual_female", "casual_male", "cheerful_female",
        "de_female", "de_male", "es_female", "es_male", "fr_female",
        "fr_male", "hi_female", "hi_male", "it_female", "it_male",
        "neutral_female", "neutral_male", "nl_female", "nl_male",
        "pt_female", "pt_male"
    ]
    return {"voices": voices}


async def generate_tts(text: str, voice: str, model: str) -> bytes:
    """Send text to Voxtral TTS and return audio bytes."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "input": text,
                "model": model,
                "response_format": "wav",
                "voice": voice,
            }
            # Typically /audio/speech for vllm-omni
            response = await client.post(f"{VOXTRAL_TTS_BASE_URL}/audio/speech", json=payload)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error(f"Error calling Voxtral TTS: {e}")
        return None

def is_sentence_end(text: str) -> bool:
    """Simple check if the chunk ends a sentence."""
    return bool(re.search(r'[.!?]\s*$', text))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    # Store chat history
    messages = [{"role": "system", "content": "You are a helpful and concise voice assistant. Your responses should be short and natural for speech."}]

    try:
        while True:
            # Wait for data from client
            data = await websocket.receive_text()
            payload = json.loads(data)

            if "audio" in payload:
                model_id = payload.get("model_id")
                voice_id = payload.get("voice_id", "casual_female")
                tts_model_id = payload.get("tts_model_id", "mistralai/Voxtral-4B-TTS-2603")

                if not model_id:
                    await websocket.send_text(json.dumps({"type": "error", "message": "No model selected."}))
                    continue

                # 1. Transcribe Audio
                audio_b64 = payload["audio"]
                audio_bytes = base64.b64decode(audio_b64)

                # Write to temp file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                start_time = time.time()
                segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
                transcription = "".join([segment.text for segment in segments]).strip()
                os.unlink(tmp_path)
                logger.info(f"Transcription took {time.time() - start_time:.2f}s: {transcription}")

                if not transcription:
                    continue

                await websocket.send_text(json.dumps({
                    "type": "transcription",
                    "text": transcription
                }))

                messages.append({"role": "user", "content": transcription})

                # 2. Get LLM response stream
                try:
                    async with httpx.AsyncClient() as client:
                        llm_payload = {
                            "model": model_id,
                            "messages": messages,
                            "stream": True,
                            "temperature": 0.7,
                            "max_tokens": 150
                        }

                        logger.info(f"Sending request to LM Studio ({model_id})...")

                        full_response = ""
                        current_chunk = ""

                        async with client.stream("POST", f"{LM_STUDIO_BASE_URL}/chat/completions", json=llm_payload) as response:
                            response.raise_for_status()

                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    line_data = line[6:]
                                    if line_data == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(line_data)
                                        delta = chunk_json["choices"][0]["delta"]
                                        if "content" in delta:
                                            text_chunk = delta["content"]
                                            full_response += text_chunk
                                            current_chunk += text_chunk

                                            # Send text stream to frontend
                                            await websocket.send_text(json.dumps({
                                                "type": "llm_chunk",
                                                "text": text_chunk
                                            }))

                                            # If chunk ends with sentence punctuation, trigger TTS
                                            if is_sentence_end(current_chunk):
                                                tts_text = current_chunk.strip()
                                                if tts_text:
                                                    logger.info(f"Sending to TTS: {tts_text}")
                                                    audio_content = await generate_tts(tts_text, voice_id, tts_model_id)
                                                    if audio_content:
                                                        await websocket.send_text(json.dumps({
                                                            "type": "tts_audio",
                                                            "audio": base64.b64encode(audio_content).decode("utf-8")
                                                        }))
                                                current_chunk = ""
                                    except json.JSONDecodeError:
                                        pass
                                    except Exception as e:
                                        logger.error(f"Error processing LLM chunk: {e}")

                        # Process any remaining text in the buffer
                        if current_chunk.strip():
                            tts_text = current_chunk.strip()
                            logger.info(f"Sending final chunk to TTS: {tts_text}")
                            audio_content = await generate_tts(tts_text, voice_id, tts_model_id)
                            if audio_content:
                                await websocket.send_text(json.dumps({
                                    "type": "tts_audio",
                                    "audio": base64.b64encode(audio_content).decode("utf-8")
                                }))

                        messages.append({"role": "assistant", "content": full_response})

                        # Tell frontend we are done
                        await websocket.send_text(json.dumps({
                            "type": "llm_done"
                        }))

                except Exception as e:
                    logger.error(f"Error calling LLM: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Error communicating with the LLM."
                    }))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket: {e}")

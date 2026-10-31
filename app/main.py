"""
FastAPI Serving Application for Hindi Speech Pipeline.

Endpoints:
- GET  /health      : Service health check, loaded models, GPU memory stats
- POST /transcribe  : Upload audio file (WAV/MP3/FLAC) -> Transcribed Hindi text + RTF
- POST /synthesize  : Send Hindi text (Devanagari) -> 16 kHz synthesized WAV audio
"""

import os
import sys
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import torch
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    VitsModel,
    AutoTokenizer,
)

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.schemas import (
    HealthResponse,
    TranscribeResponse,
    SynthesizeRequest,
    ErrorResponse,
)
from src.eval_asr import normalize_hindi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("HindiSpeechAPI")

# Global model store
models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads models into GPU memory on startup
    and clears them gracefully on application shutdown.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing Hindi Speech Pipeline on hardware: {device.upper()}")

    # 1. Load Fine-Tuned Whisper ASR Model
    asr_path = "models/whisper-small-hindi"
    if not os.path.exists(asr_path):
        logger.warning(f"Fine-tuned model '{asr_path}' not found, falling back to 'openai/whisper-small'")
        asr_path = "openai/whisper-small"

    logger.info(f"Loading Whisper ASR model from '{asr_path}'...")
    asr_processor = WhisperProcessor.from_pretrained(asr_path)
    asr_model = WhisperForConditionalGeneration.from_pretrained(asr_path).to(device)
    asr_model.eval()

    # Get forced prompt IDs for Hindi transcription
    forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="Hindi", task="transcribe")

    models["asr_processor"] = asr_processor
    models["asr_model"] = asr_model
    models["asr_path"] = asr_path
    models["forced_decoder_ids"] = forced_decoder_ids

    # 2. Load MMS-TTS Model
    tts_id = "facebook/mms-tts-hin"
    logger.info(f"Loading MMS-TTS model from '{tts_id}'...")
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_id)
    tts_model = VitsModel.from_pretrained(tts_id).to(device)
    tts_model.eval()

    models["tts_tokenizer"] = tts_tokenizer
    models["tts_model"] = tts_model
    models["tts_id"] = tts_id
    models["device"] = device

    logger.info("All models loaded successfully. Ready to serve requests!")
    yield

    # Clean up on shutdown
    logger.info("Shutting down service and freeing GPU VRAM...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize FastAPI app
app = FastAPI(
    title="Hindi Speech Pipeline API",
    description="Production REST API for Hindi Automatic Speech Recognition (ASR) and Text-to-Speech (TTS).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint providing service overview and documentation links."""
    return {
        "service": "Hindi Speech Pipeline API",
        "documentation": "/docs",
        "endpoints": {
            "health": "GET /health",
            "transcribe": "POST /transcribe (multipart audio upload)",
            "synthesize": "POST /synthesize (JSON text body)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check route returning hardware status, GPU VRAM allocation, and model status.
    """
    device = models.get("device", "cpu")
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = round(torch.cuda.memory_allocated() / (1024**3), 2)

    return HealthResponse(
        status="ok",
        device=device,
        asr_model_loaded="asr_model" in models,
        asr_model_path=models.get("asr_path", "none"),
        tts_model_loaded="tts_model" in models,
        tts_model_id=models.get("tts_id", "none"),
        gpu_memory_allocated_gb=gpu_mem,
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["ASR"],
)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an uploaded audio file into Hindi text.

    Supports WAV, MP3, FLAC, and OGG formats. Automatically decodes and
    resamples to 16 kHz mono float32 for Whisper.
    """
    # 1. Validate file extension or MIME type
    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    if ext not in allowed_extensions and not (file.content_type and "audio" in file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(allowed_extensions))}",
        )

    # 2. Read audio bytes into memory
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {str(e)}",
        )

    # 3. Decode audio bytes into float32 array at 16 kHz
    try:
        with io.BytesIO(audio_bytes) as bio:
            audio_arr, sr = sf.read(bio)
            # If multi-channel, average to mono
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            # Resample to 16 kHz if necessary
            if sr != 16000:
                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            audio_arr = audio_arr.astype("float32")
    except Exception as sf_err:
        # Fallback to librosa decoding
        try:
            with io.BytesIO(audio_bytes) as bio:
                audio_arr, _ = librosa.load(bio, sr=16000, mono=True)
                audio_arr = audio_arr.astype("float32")
        except Exception as lb_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode audio content: {str(sf_err)} / {str(lb_err)}",
            )

    audio_duration = len(audio_arr) / 16000.0

    # 4. Transcribe using fine-tuned Whisper model
    start_time = time.time()
    try:
        processor = models["asr_processor"]
        model = models["asr_model"]
        device = models["device"]
        forced_decoder_ids = models["forced_decoder_ids"]

        # Extract log-Mel features
        inputs = processor.feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception as inf_err:
        logger.error(f"Inference error during transcription: {inf_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ASR inference failed: {str(inf_err)}",
        )

    latency = time.time() - start_time
    rtf = latency / audio_duration if audio_duration > 0 else 0.0
    norm_text = normalize_hindi(transcription)

    return TranscribeResponse(
        transcription=transcription,
        transcription_normalized=norm_text,
        audio_duration_sec=round(audio_duration, 2),
        processing_time_sec=round(latency, 3),
        real_time_factor=round(rtf, 4),
    )


@app.post(
    "/synthesize",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized 16 kHz mono WAV audio file stream.",
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["TTS"],
)
async def synthesize_speech(request: SynthesizeRequest):
    """
    Synthesizes Hindi text (Devanagari script) into a 16 kHz WAV audio file.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace.",
        )

    start_time = time.time()
    try:
        tokenizer = models["tts_tokenizer"]
        model = models["tts_model"]
        device = models["device"]

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform

        audio_arr = output.squeeze().cpu().numpy().astype("float32")
        sampling_rate = model.config.sampling_rate  # 16000 Hz
    except Exception as tts_err:
        logger.error(f"TTS inference failed: {tts_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(tts_err)}",
        )

    latency = time.time() - start_time
    duration = len(audio_arr) / sampling_rate
    rtf = latency / duration if duration > 0 else 0.0

    # Write WAV to memory buffer
    wav_bio = io.BytesIO()
    sf.write(wav_bio, audio_arr, sampling_rate, format="WAV")
    wav_bytes = wav_bio.getvalue()

    headers = {
        "Content-Disposition": 'attachment; filename="synthesized.wav"',
        "X-Audio-Duration": str(round(duration, 2)),
        "X-Processing-Time": str(round(latency, 3)),
        "X-Real-Time-Factor": str(round(rtf, 4)),
    }

    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

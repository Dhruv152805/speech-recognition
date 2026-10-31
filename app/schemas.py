"""
Pydantic Request and Response Schemas for Hindi Speech Pipeline API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status and service metadata."""
    status: str = Field(default="ok", description="Overall service health status")
    device: str = Field(..., description="Compute hardware (cuda or cpu)")
    asr_model_loaded: bool = Field(..., description="Whether fine-tuned Whisper model is loaded")
    asr_model_path: str = Field(..., description="Path or identifier of ASR model")
    tts_model_loaded: bool = Field(..., description="Whether MMS-TTS model is loaded")
    tts_model_id: str = Field(..., description="Identifier of TTS model")
    gpu_memory_allocated_gb: Optional[float] = Field(None, description="Current GPU VRAM allocation in GB")


class TranscribeResponse(BaseModel):
    """Transcription response containing predicted text and performance metrics."""
    transcription: str = Field(..., description="Raw predicted Hindi transcription")
    transcription_normalized: str = Field(..., description="Unicode NFKC normalized Hindi transcription")
    audio_duration_sec: float = Field(..., description="Duration of input audio in seconds")
    processing_time_sec: float = Field(..., description="Wall-clock inference latency in seconds")
    real_time_factor: float = Field(..., description="Real-Time Factor (RTF = latency / duration)")


class SynthesizeRequest(BaseModel):
    """Request payload for text-to-speech synthesis."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description="Hindi text in Devanagari script to synthesize into audio",
        examples=["नमस्ते, यह हिंदी स्पीच पाइपलाइन का एक टेस्ट है।"],
    )


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str = Field(..., description="High-level error category")
    detail: str = Field(..., description="Descriptive error explanation")

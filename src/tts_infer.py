"""
MMS-TTS Inference Script for Hindi (facebook/mms-tts-hin).

Uses Meta's VITS architecture to synthesize Devanagari Hindi text
into 16 kHz mono float32 waveforms and saves to disk as WAV files.
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, List, Dict, Any
import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

# Configure UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class HindiTTS:
    """Wrapper around facebook/mms-tts-hin for simple, robust inference."""

    def __init__(self, model_id: str = "facebook/mms-tts-hin", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading MMS-TTS model '{model_id}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.sampling_rate = self.model.config.sampling_rate  # 16000 Hz

    def synthesize(self, text: str) -> Tuple[np.ndarray, int, float]:
        """
        Synthesizes a single Hindi text string into a float32 numpy audio array.

        Returns:
            audio_arr: 1D numpy array of audio samples
            sampling_rate: 16000
            latency_sec: Wall-clock synthesis duration
        """
        start_time = time.time()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs).waveform

        latency_sec = time.time() - start_time
        # Convert tensor to 1D float32 numpy array on CPU
        audio_arr = output.squeeze().cpu().numpy().astype(np.float32)
        return audio_arr, self.sampling_rate, latency_sec

    def synthesize_and_save(self, text: str, output_wav_path: str) -> Dict[str, Any]:
        """Synthesizes text and writes output directly to a WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
        audio_arr, sr, latency = self.synthesize(text)
        sf.write(output_wav_path, audio_arr, sr)
        audio_duration = len(audio_arr) / sr
        rtf = latency / audio_duration if audio_duration > 0 else 0.0

        return {
            "text": text,
            "wav_path": output_wav_path,
            "duration_sec": round(audio_duration, 2),
            "latency_sec": round(latency, 3),
            "rtf": round(rtf, 4),
        }


def synthesize_fleurs_subset(
    num_samples: int = 50,
    output_dir: str = "samples/tts",
    metadata_json: str = "samples/tts/metadata.json",
):
    """
    Extracts 50 diverse test sentences from Google FLEURS Hindi and synthesizes them.
    """
    # Import FLEURS loader
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data import load_fleurs_hindi

    print(f"\nLoading Google FLEURS Hindi test split for {num_samples} evaluation sentences...")
    test_ds = load_fleurs_hindi(split="test", decode_audio=False)

    tts = HindiTTS()
    os.makedirs(output_dir, exist_ok=True)

    records = []
    print(f"Synthesizing {num_samples} sentences to '{output_dir}'...")

    total_audio_sec = 0.0
    total_synth_time = 0.0

    for i in range(num_samples):
        sample = test_ds[i]
        text = sample["transcription"].strip()
        wav_path = os.path.join(output_dir, f"sample_{i+1:03d}.wav")

        result = tts.synthesize_and_save(text, wav_path)
        result["sample_id"] = sample.get("id", i)
        records.append(result)

        total_audio_sec += result["duration_sec"]
        total_synth_time += result["latency_sec"]

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  [{i+1:02d}/{num_samples}] Synthesized {result['duration_sec']}s audio in {result['latency_sec']}s (RTF: {result['rtf']})")

    avg_rtf = total_synth_time / total_audio_sec if total_audio_sec > 0 else 0.0
    summary = {
        "model": "facebook/mms-tts-hin",
        "num_samples": num_samples,
        "total_audio_seconds": round(total_audio_sec, 2),
        "total_synthesis_seconds": round(total_synth_time, 2),
        "average_rtf": round(avg_rtf, 4),
        "samples": records,
    }

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nAll {num_samples} files synthesized successfully!")
    print(f"Total audio duration: {total_audio_sec:.2f} s | Total synth time: {total_synth_time:.2f} s")
    print(f"Average Real-Time Factor (RTF): {avg_rtf:.4f} (synthesizing {1.0/avg_rtf:.1f}x faster than real-time)")
    print(f"Metadata saved to: {metadata_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMS-TTS Hindi Speech Synthesis")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test sentences to synthesize")
    parser.add_argument("--output_dir", type=str, default="samples/tts", help="Directory for generated WAV files")
    parser.add_argument("--text", type=str, default=None, help="Optional single sentence to synthesize")
    args = parser.parse_args()

    if args.text:
        tts = HindiTTS()
        res = tts.synthesize_and_save(args.text, "samples/custom_tts.wav")
        print(f"Synthesized custom sentence to samples/custom_tts.wav: {res}")
    else:
        synthesize_fleurs_subset(num_samples=args.num_samples, output_dir=args.output_dir)

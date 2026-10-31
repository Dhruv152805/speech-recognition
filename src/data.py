"""
Data loading and preprocessing utilities for the Hindi Speech Pipeline.

Dataset: Google FLEURS (hi_in) - Hindi speech dataset with transcriptions.
Audio decoding strategy:
In datasets >= 4.0 (we have datasets 5.1.0), HuggingFace defaults to 'torchcodec'
for audio decoding, which fails on Windows when torchcodec/ffmpeg is not natively
built. To keep our pipeline robust and portable, we disable automatic decoding in
datasets (`Audio(decode=False)`) and decode raw audio bytes directly with
`soundfile` and `librosa`.
"""

import io
import sys
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio, Dataset

# Ensure standard output can print Devanagari (Hindi) Unicode characters on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_fleurs_hindi(split: str = "train", decode_audio: bool = False) -> Dataset:
    """
    Load a split of the Google FLEURS Hindi (hi_in) dataset.

    Args:
        split: The dataset split to load ('train', 'validation', 'test', or slice).
        decode_audio: Whether to let datasets decode audio (defaults to False
                      to prevent torchcodec dependency errors on Windows).

    Returns:
        The loaded HuggingFace Dataset.
    """
    # Load dataset configuration without automatic decoding
    dataset = load_dataset("google/fleurs", "hi_in", split=split)
    if not decode_audio:
        # Cast the audio column with decode=False so audio comes as raw bytes dict
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def decode_audio_sample(audio_dict: Dict[str, Any], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode raw audio bytes from a dataset item into a NumPy float array.

    Args:
        audio_dict: Dictionary containing 'bytes' and optional 'path'.
        target_sr: Desired sampling rate (default 16000 Hz for Whisper/MMS).

    Returns:
        Tuple of (audio_array, sampling_rate).
    """
    # Read audio from byte stream using soundfile
    audio_bytes = audio_dict["bytes"]
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono if multi-channel
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=-1)

    # Ensure float32 representation
    audio_array = audio_array.astype(np.float32)

    # Resample if sample rate does not match target_sr
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    return audio_array, sample_rate


def get_dataset_summary() -> Dict[str, Dict[str, Any]]:
    """
    Compute example counts and total duration in hours for all FLEURS hi_in splits.

    Returns:
        Dictionary mapping split name to count, duration in seconds, and hours.
    """
    splits = ["train", "validation", "test"]
    summary = {}

    for split in splits:
        # Load metadata only (audio column decode=False)
        ds = load_fleurs_hindi(split=split, decode_audio=False)
        sample_count = len(ds)

        # FLEURS provides num_samples (recorded at 16,000 Hz)
        total_samples = sum(ds["num_samples"])
        duration_seconds = total_samples / 16000.0
        duration_hours = duration_seconds / 3600.0

        summary[split] = {
            "num_examples": sample_count,
            "total_samples": total_samples,
            "duration_seconds": round(duration_seconds, 2),
            "duration_hours": round(duration_hours, 2),
        }

    return summary


if __name__ == "__main__":
    print("=== Google FLEURS (hi_in) Dataset Summary ===")
    stats = get_dataset_summary()
    total_hours = sum(s["duration_hours"] for s in stats.values())
    total_examples = sum(s["num_examples"] for s in stats.values())

    for split_name, s in stats.items():
        print(f"Split: {split_name:10s} | {s['num_examples']:4d} clips | {s['duration_seconds']:9.2f}s | {s['duration_hours']:5.2f} hours")
    print(f"Total:      {total_examples:4d} clips | {total_hours:5.2f} hours")

    # Quick verification on one sample
    print("\n=== Verifying Sample Decoding ===")
    val_ds = load_fleurs_hindi(split="validation", decode_audio=False)
    sample = val_ds[0]
    audio_arr, sr = decode_audio_sample(sample["audio"], target_sr=16000)
    print(f"Sample ID:           {sample['id']}")
    print(f"Transcript:          {sample['transcription']}")
    print(f"Audio array shape:   {audio_arr.shape}")
    print(f"Sampling rate:       {sr} Hz")
    print(f"Calculated duration: {len(audio_arr) / sr:.2f} seconds")

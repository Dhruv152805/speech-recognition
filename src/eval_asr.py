"""
ASR Evaluation script for Hindi Speech Recognition.

Evaluates Whisper models on Google FLEURS Hindi (hi_in).
Computes Word Error Rate (WER) and Character Error Rate (CER) with jiwer.
Normalizes Hindi text consistently (Unicode NFKC, punctuation, whitespace).
Saves empirical metrics and sample predictions to JSON.
"""

import os
import sys
import json
import time
import re
import unicodedata
import argparse
from typing import List, Dict, Any, Tuple
import torch
import jiwer
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_fleurs_hindi, decode_audio_sample

# Configure UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalize_hindi(text: str) -> str:
    """
    Standard text normalizer for Hindi (Devanagari) ASR evaluation.

    Why this is critical:
    1. Unicode equivalence: Devanagari characters can have multiple encodings (e.g.,
       composed vs decomposing nukta). NFKC maps equivalent sequences to canonical forms.
    2. Punctuation: ASR acoustic models predict spoken words; sentence terminators
       (danda '।', double danda '॥', commas, question marks) must be stripped so
       punctuation mismatches do not penalize acoustic transcription accuracy.
    3. Whitespace: Multiple spaces, newlines, and zero-width spaces/joiners (ZWJ/ZWNJ)
       must be removed to prevent false tokenization mismatches.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove Devanagari and Latin punctuation
    # Devanagari danda (\u0964) and double danda (\u0965), plus standard punctuation
    punctuation_pattern = r"[\u0964\u0965\.,!\?;:\'\"/\\`~@#\$%\^&\*\+\=\<\>\|\(\)\[\]\{\}\-_]"
    text = re.sub(punctuation_pattern, " ", text)

    # 3. Remove zero-width joiners and formatting characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 4. Lowercase (for any embedded English words or letters)
    text = text.lower()

    # 5. Collapse repeated whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def evaluate_asr(
    model_name_or_path: str = "openai/whisper-small",
    split: str = "test",
    batch_size: int = 8,
    output_json: str = "results/baseline_asr.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Run batched inference on FLEURS Hindi split and calculate WER and CER.
    """
    print(f"\n==========================================")
    print(f"Evaluating ASR Model: {model_name_or_path}")
    print(f"Dataset Split:        {split}")
    print(f"Batch Size:           {batch_size}")
    print(f"Device:               {device}")
    print(f"==========================================")

    # Load processor and model
    print(f"Loading Whisper processor and model...")
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Load dataset
    print(f"Loading FLEURS hi_in {split} dataset...")
    dataset = load_fleurs_hindi(split=split, decode_audio=False)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_examples = len(dataset)
    print(f"Total evaluation samples: {total_examples}")

    predictions_raw = []
    references_raw = []
    sample_ids = []
    sample_durations = []

    start_eval_time = time.time()

    # Iterate in batches
    for i in tqdm(range(0, total_examples, batch_size), desc="Transcribing"):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, total_examples))]
        audio_arrays = []

        for item in batch_items:
            arr, sr = decode_audio_sample(item["audio"], target_sr=16000)
            audio_arrays.append(arr)
            sample_ids.append(item["id"])
            references_raw.append(item["transcription"])
            sample_durations.append(len(arr) / sr)

        # Preprocess audio batch to 80-channel log-Mel spectrograms
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_features.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)

        # Generate predictions forced into Hindi transcription
        with torch.no_grad():
            generated_ids = model.generate(
                inputs,
                language="hi",
                task="transcribe",
            )

        batch_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions_raw.extend(batch_preds)

    total_eval_time = time.time() - start_eval_time
    total_audio_seconds = sum(sample_durations)
    rtf = total_eval_time / total_audio_seconds if total_audio_seconds > 0 else 0.0

    print("\nNormalizing texts and calculating metrics with jiwer...")
    predictions_norm = [normalize_hindi(p) for p in predictions_raw]
    references_norm = [normalize_hindi(r) for r in references_raw]

    # Calculate WER and CER on normalized text
    wer_norm = jiwer.wer(references_norm, predictions_norm) * 100.0
    cer_norm = jiwer.cer(references_norm, predictions_norm) * 100.0

    # Also compute raw metrics without normalization for comparison
    wer_raw = jiwer.wer(references_raw, predictions_raw) * 100.0
    cer_raw = jiwer.cer(references_raw, predictions_raw) * 100.0

    print("\n=== Empirical Results ===")
    print(f"Normalized WER:  {wer_norm:.2f}%")
    print(f"Normalized CER:  {cer_norm:.2f}%")
    print(f"Raw WER:         {wer_raw:.2f}%")
    print(f"Raw CER:         {cer_raw:.2f}%")
    print(f"Total Audio:     {total_audio_seconds:.2f} s ({total_audio_seconds/3600:.2f} hrs)")
    print(f"Eval Wall Time:  {total_eval_time:.2f} s")
    print(f"Real-Time Factor:{rtf:.4f} (lower is faster; < 1.0 is real-time)")

    # Prepare structured results
    sample_records = []
    for k in range(min(50, total_examples)):
        sample_records.append({
            "id": sample_ids[k],
            "duration_sec": round(sample_durations[k], 2),
            "reference_raw": references_raw[k],
            "prediction_raw": predictions_raw[k],
            "reference_norm": references_norm[k],
            "prediction_norm": predictions_norm[k],
        })

    results_data = {
        "model_name_or_path": model_name_or_path,
        "dataset": "google/fleurs (hi_in)",
        "split": split,
        "num_samples": total_examples,
        "total_audio_seconds": round(total_audio_seconds, 2),
        "evaluation_wall_time_seconds": round(total_eval_time, 2),
        "real_time_factor": round(rtf, 4),
        "metrics": {
            "wer_normalized_pct": round(wer_norm, 2),
            "cer_normalized_pct": round(cer_norm, 2),
            "wer_raw_pct": round(wer_raw, 2),
            "cer_raw_pct": round(cer_raw, 2),
        },
        "sample_predictions": sample_records,
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved empirical results to: {output_json}")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Whisper ASR on Google FLEURS Hindi")
    parser.add_argument("--model", type=str, default="openai/whisper-small", help="Model name or checkpoint directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test', 'validation', 'train')")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--output", type=str, default="results/baseline_asr.json", help="Path to save output JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples evaluated (optional)")
    args = parser.parse_args()

    evaluate_asr(
        model_name_or_path=args.model,
        split=args.split,
        batch_size=args.batch_size,
        output_json=args.output,
        max_samples=args.max_samples,
    )

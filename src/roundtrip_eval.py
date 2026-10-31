"""
Phase 5: Roundtrip Speech-to-Speech Evaluation.

Pipeline:
Reference Hindi Text -> MMS-TTS (facebook/mms-tts-hin) -> Synthesized WAV
Synthesized WAV -> Fine-Tuned Whisper (models/whisper-small-hindi) -> Predicted Hindi Text

Metrics:
- Word Error Rate (WER) and Character Error Rate (CER) between Original Reference and Predicted Text.
- Empirical metrics saved to results/tts_roundtrip.json.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
import torch
import soundfile as sf
import jiwer
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import normalizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval_asr import normalize_hindi


def run_roundtrip_evaluation(
    tts_metadata_path: str = "samples/tts/metadata.json",
    asr_model_dir: str = "models/whisper-small-hindi",
    output_json: str = "results/tts_roundtrip.json",
    device: str = None,
):
    """
    Evaluates TTS synthesized audio using our fine-tuned Whisper Hindi ASR.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==========================================")
    print("Running TTS Roundtrip Evaluation")
    print(f"ASR Model: {asr_model_dir}")
    print(f"TTS Audio Metadata: {tts_metadata_path}")
    print(f"Target Device: {device}")
    print("==========================================")

    if not os.path.exists(tts_metadata_path):
        raise FileNotFoundError(f"TTS metadata file not found at {tts_metadata_path}. Please run src/tts_infer.py first.")

    with open(tts_metadata_path, "r", encoding="utf-8") as f:
        tts_data = json.load(f)

    samples = tts_data.get("samples", [])
    if not samples:
        raise ValueError("No samples found in TTS metadata.")

    print(f"Loading ASR model '{asr_model_dir}'...")
    processor = WhisperProcessor.from_pretrained(asr_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(asr_model_dir).to(device)
    model.eval()

    # Configure forced decoder tokens for Hindi transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Hindi", task="transcribe")

    eval_records = []
    total_audio_sec = 0.0
    start_eval_time = time.time()

    print(f"Transcribing {len(samples)} TTS audio clips...")
    for item in tqdm(samples, desc="Roundtrip Transcribing"):
        wav_path = item["wav_path"]
        ref_text = item["text"]

        if not os.path.exists(wav_path):
            print(f"Warning: Missing audio file {wav_path}, skipping...")
            continue

        # Load audio using soundfile
        audio_arr, sr = sf.read(wav_path)
        if sr != 16000:
            # Resample if necessary
            import librosa
            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_duration = len(audio_arr) / sr
        total_audio_sec += audio_duration

        # Extract features
        inputs = processor.feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        ref_norm = normalize_hindi(ref_text)
        pred_norm = normalize_hindi(transcription)

        try:
            sample_wer = jiwer.wer(ref_norm, pred_norm) * 100
        except Exception:
            sample_wer = 100.0

        try:
            sample_cer = jiwer.cer(ref_norm, pred_norm) * 100
        except Exception:
            sample_cer = 100.0

        eval_records.append({
            "sample_id": item.get("sample_id"),
            "wav_path": wav_path,
            "duration_sec": round(audio_duration, 2),
            "reference_raw": ref_text,
            "prediction_raw": transcription,
            "reference_norm": ref_norm,
            "prediction_norm": pred_norm,
            "wer_pct": round(sample_wer, 2),
            "cer_pct": round(sample_cer, 2),
        })

    eval_wall_time = time.time() - start_eval_time
    rtf = eval_wall_time / total_audio_sec if total_audio_sec > 0 else 0.0

    # Corpus-level WER and CER
    all_refs = [r["reference_norm"] for r in eval_records]
    all_preds = [r["prediction_norm"] for r in eval_records]

    overall_wer = jiwer.wer(all_refs, all_preds) * 100
    overall_cer = jiwer.cer(all_refs, all_preds) * 100

    results = {
        "asr_model": asr_model_dir,
        "tts_model": tts_data.get("model", "facebook/mms-tts-hin"),
        "num_samples_evaluated": len(eval_records),
        "total_audio_seconds": round(total_audio_sec, 2),
        "evaluation_wall_time_seconds": round(eval_wall_time, 2),
        "real_time_factor": round(rtf, 4),
        "metrics": {
            "roundtrip_wer_normalized_pct": round(overall_wer, 2),
            "roundtrip_cer_normalized_pct": round(overall_cer, 2),
        },
        "sample_evaluations": eval_records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Roundtrip Evaluation Empirical Results ===")
    print(f"Total Clips Evaluated: {len(eval_records)}")
    print(f"Total Audio Duration:  {total_audio_sec:.2f} s")
    print(f"ASR Wall-Clock Time:   {eval_wall_time:.2f} s (RTF: {rtf:.4f})")
    print(f"Roundtrip WER:         {overall_wer:.2f}%")
    print(f"Roundtrip CER:         {overall_cer:.2f}%")
    print(f"Saved results to:      {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roundtrip TTS-ASR Evaluation")
    parser.add_argument("--metadata", type=str, default="samples/tts/metadata.json")
    parser.add_argument("--asr_model", type=str, default="models/whisper-small-hindi")
    parser.add_argument("--output", type=str, default="results/tts_roundtrip.json")
    args = parser.parse_args()

    run_roundtrip_evaluation(
        tts_metadata_path=args.metadata,
        asr_model_dir=args.asr_model,
        output_json=args.output,
    )

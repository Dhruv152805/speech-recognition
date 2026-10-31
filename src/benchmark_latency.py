"""
Phase 7: Latency & Real-Time Factor (RTF) Benchmarking.

Compares HuggingFace Transformers Whisper vs CTranslate2 (faster-whisper).
Metrics:
- Real-Time Factor (RTF = wall_clock_seconds / audio_seconds)
- Latency per clip (mean, median, p95)
- Throughput (audio seconds processed per second of compute)
- Peak GPU VRAM allocated
Saves empirical benchmark results to results/latency.json.
"""

import os
import sys
import time
import json
import argparse
import platform
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm
# On Windows, add nvidia CUDA 12 package bin directories to DLL search path
if sys.platform.startswith("win"):
    site_packages = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages")
    for pkg in ["nvidia/cublas/bin", "nvidia/cudnn/bin", "torch/lib"]:
        dll_dir = os.path.abspath(os.path.join(site_packages, pkg))
        if os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
                os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")
            except Exception:
                pass

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from faster_whisper import WhisperModel

# Ensure local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_fleurs_hindi, decode_audio_sample

# Windows Unicode stdout safety
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_hardware_info() -> Dict[str, Any]:
    """Captures exact hardware environment specifications."""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "None (CPU only)"
        info["gpu_vram_total_gb"] = 0.0
        info["cuda_version"] = "None"
    return info


def benchmark_hf_whisper(
    model_path: str,
    audio_clips: List[np.ndarray],
    durations: List[float],
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmarks standard HuggingFace Transformers Whisper inference."""
    print(f"\n--- Benchmarking HuggingFace Transformers ({model_path}) on {device.upper()} ---")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    if device == "cuda":
        model = model.half()  # fp16
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Hindi", task="transcribe")

    # Warmup pass (1 clip) to eliminate CUDA context creation overhead
    warmup_feat = processor.feature_extractor(audio_clips[0], sampling_rate=16000, return_tensors="pt").input_features.to(device)
    if device == "cuda":
        warmup_feat = warmup_feat.half()
    with torch.no_grad():
        _ = model.generate(warmup_feat, forced_decoder_ids=forced_decoder_ids, max_new_tokens=64)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    transcriptions = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for audio_arr in tqdm(audio_clips, desc="HF Transformers"):
        feat = processor.feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        if device == "cuda":
            feat = feat.half()

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            pred_ids = model.generate(feat, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
        latencies.append(elapsed)
        transcriptions.append(text)

    total_audio = sum(durations)
    total_time = sum(latencies)
    rtf = total_time / total_audio if total_audio > 0 else 0.0
    vram_peak_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device == "cuda" else 0.0

    # Free memory
    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "engine": "HuggingFace Transformers (PyTorch fp16)",
        "num_clips": len(audio_clips),
        "total_audio_sec": round(total_audio, 2),
        "total_latency_sec": round(total_time, 3),
        "real_time_factor": round(rtf, 4),
        "throughput_audio_sec_per_sec": round(total_audio / total_time, 2),
        "avg_latency_per_clip_sec": round(float(np.mean(latencies)), 3),
        "median_latency_per_clip_sec": round(float(np.median(latencies)), 3),
        "p95_latency_per_clip_sec": round(float(np.percentile(latencies, 95)), 3),
        "peak_vram_gb": vram_peak_gb,
    }


def benchmark_faster_whisper(
    model_path: str,
    audio_clips: List[np.ndarray],
    durations: List[float],
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmarks CTranslate2 faster-whisper inference."""
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"\n--- Benchmarking faster-whisper (CTranslate2 {compute_type}) on {device.upper()} ---")

    fw_model = WhisperModel(model_path, device=device, compute_type=compute_type)

    # Warmup pass (1 clip)
    _segments, _ = fw_model.transcribe(audio_clips[0], language="hi", task="transcribe", beam_size=1)
    _ = list(_segments)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    transcriptions = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for audio_arr in tqdm(audio_clips, desc="faster-whisper"):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        segments, _info = fw_model.transcribe(
            audio_arr,
            language="hi",
            task="transcribe",
            beam_size=1,  # greedy for fair 1-to-1 comparison with HF generate()
        )
        text = "".join([s.text for s in segments]).strip()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        transcriptions.append(text)

    total_audio = sum(durations)
    total_time = sum(latencies)
    rtf = total_time / total_audio if total_audio > 0 else 0.0
    vram_peak_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if device == "cuda" else 0.0

    del fw_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "engine": f"faster-whisper (CTranslate2 {compute_type})",
        "num_clips": len(audio_clips),
        "total_audio_sec": round(total_audio, 2),
        "total_latency_sec": round(total_time, 3),
        "real_time_factor": round(rtf, 4),
        "throughput_audio_sec_per_sec": round(total_audio / total_time, 2),
        "avg_latency_per_clip_sec": round(float(np.mean(latencies)), 3),
        "median_latency_per_clip_sec": round(float(np.median(latencies)), 3),
        "p95_latency_per_clip_sec": round(float(np.percentile(latencies, 95)), 3),
        "peak_vram_gb": vram_peak_gb,
    }


def run_benchmark(
    num_samples: int = 30,
    hf_model_path: str = "models/whisper-small-hindi",
    ct2_model_path: str = "models/whisper-small-hindi-ct2",
    output_json: str = "results/latency.json",
):
    print("==================================================")
    print("ASR Latency Benchmark: HF Transformers vs faster-whisper")
    print(f"Sample Count: {num_samples} clips")
    print(f"HF Model:     {hf_model_path}")
    print(f"CT2 Model:    {ct2_model_path}")
    print("==================================================")

    # 1. Load test samples
    print(f"Loading {num_samples} test audio clips from Google FLEURS Hindi...")
    raw_test_ds = load_fleurs_hindi(split="test", decode_audio=False)
    samples_to_use = min(num_samples, len(raw_test_ds))

    audio_clips = []
    durations = []
    for i in range(samples_to_use):
        arr, sr = decode_audio_sample(raw_test_ds[i]["audio"], target_sr=16000)
        audio_clips.append(arr)
        durations.append(len(arr) / sr)

    total_audio_sec = sum(durations)
    print(f"Decoded {len(audio_clips)} clips | Total audio duration: {total_audio_sec:.2f} s ({total_audio_sec/60:.2f} mins)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw_info = get_hardware_info()

    # 2. Benchmark HuggingFace Transformers
    hf_results = benchmark_hf_whisper(hf_model_path, audio_clips, durations, device=device)

    # 3. Benchmark faster-whisper (CTranslate2)
    # If custom CT2 model exists, use it; otherwise fallback to 'small'
    ct2_path_to_use = ct2_model_path if os.path.exists(ct2_model_path) else "small"
    fw_results = benchmark_faster_whisper(ct2_path_to_use, audio_clips, durations, device=device)

    # 4. Comparative Speedup Factor
    speedup = round(hf_results["total_latency_sec"] / fw_results["total_latency_sec"], 2) if fw_results["total_latency_sec"] > 0 else 1.0

    benchmark_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "dataset": "Google FLEURS Hindi (hi_in) Test Split",
        "num_clips": len(audio_clips),
        "total_audio_seconds": round(total_audio_sec, 2),
        "speedup_factor_faster_whisper_vs_hf": speedup,
        "engines": {
            "huggingface_transformers": hf_results,
            "faster_whisper_ctranslate2": fw_results,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

    print("\n==================================================")
    print("           LATENCY BENCHMARK RESULTS              ")
    print("==================================================")
    print(f"Hardware: {hw_info['gpu_name']} ({hw_info['gpu_vram_total_gb']} GB VRAM)")
    print(f"Total Audio Evaluated: {total_audio_sec:.2f} seconds ({total_audio_sec/60:.2f} minutes)")
    print("--------------------------------------------------")
    print(f"HF Transformers Wall Time:  {hf_results['total_latency_sec']:.2f} s | RTF: {hf_results['real_time_factor']} ({hf_results['throughput_audio_sec_per_sec']}x real-time)")
    print(f"faster-whisper Wall Time:   {fw_results['total_latency_sec']:.2f} s | RTF: {fw_results['real_time_factor']} ({fw_results['throughput_audio_sec_per_sec']}x real-time)")
    print(f"Speedup with faster-whisper:{speedup}x FASTER than HuggingFace PyTorch")
    print("--------------------------------------------------")
    print(f"Saved benchmark results to: {output_json}")

    return benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Latency Benchmark")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of clips to benchmark")
    parser.add_argument("--hf_model", type=str, default="models/whisper-small-hindi")
    parser.add_argument("--ct2_model", type=str, default="models/whisper-small-hindi-ct2")
    parser.add_argument("--output", type=str, default="results/latency.json")
    args = parser.parse_args()

    run_benchmark(
        num_samples=args.num_samples,
        hf_model_path=args.hf_model,
        ct2_model_path=args.ct2_model,
        output_json=args.output,
    )

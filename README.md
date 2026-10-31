# Hindi Speech Pipeline: End-to-End ASR, TTS & Microservice

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CTranslate2](https://img.shields.io/badge/CTranslate2-faster--whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Hindi speech processing pipeline for Hindi (Devanagari script). The system fine-tunes `openai/whisper-small` on Google FLEURS Hindi, integrates Meta's pretrained Hindi `mms-tts-hin` model, optimizes autoregressive inference via CTranslate2 FP16 model conversion, and serves both models through an asynchronous FastAPI microservice with in-memory audio processing.

---

## ⚡ Key Highlights at a Glance

| 🎯 ASR Accuracy | ⚡ Inference Latency | 🎙️ Speech Synthesis | 💾 Resident Model VRAM |
|:---:|:---:|:---:|:---:|
| **46.22 pp WER Reduction**<br>81.79% $\to$ **35.57%** on FLEURS | **1.97× Faster Latency**<br>0.958s/clip (12.65× real-time) | **72.7× Real-Time TTS**<br>525s audio generated in 7.2s | **1.06 GB PyTorch VRAM**<br>Resident ASR + TTS models |

```
Quick Navigation:
[Key Results] • [Quickstart & Demo] • [Architecture] • [Pillar 1: Model Dev] • [Pillar 2: Empirical Eval] • [Pillar 3: Optimization] • [Reproduction]
```

---

## 📊 Benchmark Summary

All metrics are empirically verified and persisted in the [`results/`](results/) directory.

### 1. Speech Recognition Accuracy (Google FLEURS `hi_in` Test Set — 418 clips, 1.34 hrs)
*Source: [`results/baseline_asr.json`](results/baseline_asr.json) & [`results/finetuned_asr.json`](results/finetuned_asr.json)*

| Model | Normalized WER (%) | Normalized CER (%) | Raw WER (%) | Raw CER (%) | Eval Throughput |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline `whisper-small` (Zero-shot)** | 81.79% | 49.12% | 84.12% | 49.69% | 22.2× real-time (RTF: 0.0450) |
| **Fine-Tuned `whisper-small-hindi`** | **35.57%** | **13.25%** | **36.22%** | **13.42%** | **32.7× real-time** (RTF: 0.0306) |
| **Absolute Improvement** | **46.22 pp reduction** | **35.87 pp reduction** | **47.90 pp reduction** | **36.27 pp reduction** | **32% Faster Decoding** |

### 2. Inference Engine Optimization (30 test clips — 363.84 s audio)
*Source: [`results/latency.json`](results/latency.json)*

| Inference Engine | Runtime & Precision | Avg Latency / Clip | p95 Latency | Real-Time Factor (RTF) | Model Weights Size |
|---|---|:---:|:---:|:---:|:---:|
| **HuggingFace Transformers** | PyTorch FP16 | 1.892 s | 3.065 s | 0.1560 (6.41× real-time) | 966 MB |
| **`faster-whisper` (CTranslate2)** | C++ CUDA Flat Binary FP16 | **0.958 s** | **1.670 s** | **0.0790 (12.65× real-time)** | **483 MB** (50% smaller) |
| **Optimization Factor** | - | **1.97× Speedup** | **1.84× Faster** | **Almost 2× Throughput** | **50% Model Weight Reduction** |

### 3. Speech Synthesis & Roundtrip Intelligibility (50 clips — 525.49 s audio)
*Source: [`results/tts_roundtrip.json`](results/tts_roundtrip.json)*

| Pipeline Component | Metric / Finding | Technical Detail |
|---|:---:|---|
| **Meta MMS-TTS Hindi Synthesis** | **72.7× Real-Time** (RTF: 0.0137) | Synthesized 8.76 minutes of audio in **7.23 seconds** on GPU. |
| **Roundtrip Transcription WER / CER** | **37.34% WER / 13.90% CER** | Within 1.77 percentage points of human speech ASR (35.57%). |
| **Decodability Scope** | Machine Decodability Proxy | Measures machine acoustic decodability through ASR; does not prove human-perceived naturalness or replace subjective listening tests (MOS). |

---

## 🚀 30-Second Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/hindi-speech-pipeline.git
cd hindi-speech-pipeline

# Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate

# Install PyTorch with CUDA 12.1+ support and dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Start the Serving Layer
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: `curl http://127.0.0.1:8000/health` $\to$ Returns GPU VRAM allocation (`1.06 GB`).

### 3. Test Endpoints via cURL

**A. Synthesize Hindi Text to WAV (`POST /synthesize`):**
```bash
curl -s -X POST http://127.0.0.1:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, आईआईआईटी हैदराबाद स्पीच लैब में आपका स्वागत है।"}' \
  --output greeting.wav
```

**B. Transcribe Audio to Text (`POST /transcribe`):**
```bash
curl -s -X POST http://127.0.0.1:8000/transcribe -F "file=@greeting.wav"
```
```json
{
  "transcription": "नमस्ते आईआईआईटी हैदराबाद स्पीच लैब में आपका स्वागत है",
  "transcription_normalized": "नमस्ते आईआईआईटी हैदराबाद स्पीच लैब में आपका स्वागत है",
  "audio_duration_sec": 4.12,
  "processing_time_sec": 0.98,
  "real_time_factor": 0.238
}
```

---

## 🛠️ System Architecture

```mermaid
flowchart TD
    subgraph Data["1. Data Ingestion & Signal Processing"]
        A["Google FLEURS Hindi (hi_in)"] --> B["In-Memory Byte Stream Decoding (soundfile/librosa)"]
        B --> C["Resample to 16 kHz Mono Float32 PCM"]
        C --> D["STFT (25ms Hann Window, 10ms Hop)"]
        D --> E["80-Channel Log-Mel Filterbank Feature Extractor"]
    end

    subgraph Train["2. Model Development & Fine-Tuning"]
        E --> F["Whisper Audio Encoder (12 Layers, 768 Dim)"]
        G["Hindi Transcripts (Devanagari)"] --> H["Whisper Autoregressive Decoder (Teacher Forcing)"]
        F -->|Cross-Attention| H
        H --> I["Cross-Entropy Loss (Labels Masked at -100)"]
        I --> J["AdamW (lr=1e-5, fp16, Grad Accum=8, Grad Checkpointing)"]
    end

    subgraph Optimize["3. Deployment & Engine Optimization"]
        J --> K["Exported Model: models/whisper-small-hindi (966 MB)"]
        K --> L["CTranslate2 Converter (ct2-transformers-converter)"]
        L --> M["CTranslate2 FP16 Model: models/whisper-small-hindi-ct2 (483 MB)"]
    end

    subgraph Serving["4. Serving API & Dual Model Co-location"]
        M --> N["FastAPI Asynchronous Microservice (app/main.py)"]
        O["Meta MMS-TTS VITS (facebook/mms-tts-hin)"] --> N
        N -->|GET /health| P["Health & VRAM Status (1.06 GB Co-located)"]
        N -->|POST /transcribe| Q["Audio Upload -> In-Memory Buffer -> Transcription + RTF"]
        N -->|POST /synthesize| R["Text -> VITS Vocoder -> Streaming 16 kHz WAV"]
    end

    subgraph Eval["5. Empirical Evaluation Harness"]
        Q -.-> S["Test Set Evaluation: 418 clips / 1.34 hrs (results/finetuned_asr.json)"]
        R -.->|Synthesized Audio| T["Roundtrip Evaluation: 50 clips / 525s (results/tts_roundtrip.json)"]
        T -.-> S
    end

    style Data fill:#f8f9fa,stroke:#343a40,stroke-width:1px
    style Train fill:#e8f4f8,stroke:#0288d1,stroke-width:2px
    style Optimize fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Serving fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style Eval fill:#eafaf1,stroke:#27ae60,stroke-width:2px
```

---

## 🔬 Pillar 1: Model Development & Training Recipe

### 1. Model & Data Specifications
- **Base Foundation Model**: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) (242M parameters; 12 encoder layers, 12 decoder layers, 768 hidden dimension, 80 mel channels, pretrained on 680,000 hours of multilingual speech).
- **Target Language & Script**: Hindi (`hi_in`) in native Devanagari script.
- **Training Corpus**: Google FLEURS Hindi train split, partitioned into **1,591 clips (~5.0 hours of speech)**.
- **Hardware Platform**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB dedicated GDDR6 VRAM, Driver 592.82, CUDA 13.1).

### 2. Audio Signal Processing & Data Collation
- **In-Memory Streaming Decoding**: Raw audio byte streams are loaded with HuggingFace Datasets `decode=False` and decoded via in-memory `soundfile.read(io.BytesIO(bytes))` with `librosa` fallback, eliminating temporary disk I/O and bypassing Windows `torchcodec` compilation issues.
- **Feature Extraction**: 16,000 Hz single-channel Float32 PCM transformed into an **80-channel log-Mel spectrogram** (25 ms Hann window, 10 ms hop size, padded/truncated to 3,000 frames).
- **Devanagari BPE Tokenization**: Whisper Byte-Pair Encoding (`vocab_size=51865`) conditioned on language and task tokens (`<|startoftranscript|> <|hi|> <|transcribe|> <|notimestamps|>`).
- **Dynamic Data Collator (`DataCollatorSpeechSeq2SeqWithPadding`)**: Pads acoustic features dynamically and masks target label sequences with `-100` so that `CrossEntropyLoss(ignore_index=-100)` ignores padding tokens during backpropagation.

### 3. Hyperparameter Recipe & Memory Budgeting Under 6 GB VRAM

| Hyperparameter | Value | Architectural Justification |
|---|:---:|---|
| **Per-Device Micro-Batch Size** | `2` | Keeps active tensor allocations under 4.7 GB during forward/backward passes. |
| **Gradient Accumulation Steps** | `8` | Accumulates gradients across 8 micro-batches before calling `optimizer.step()`. |
| **Effective Batch Size** | **`16`** | $\text{Batch (2)} \times \text{Accumulation (8)} = \mathbf{16 \text{ clips/step}}$, stabilizing AdamW updates. |
| **Max Training Steps** | **`100`** | Processes $100 \times 16 = 1,600 \text{ samples}$, equivalent to $\sim 1.0\text{ epoch}$ of the 1,591-clip training subset. |
| **Learning Rate & Schedule** | `1e-5` (Linear) | Linear warmup over first 20 steps (0 to $10^{-5}$), decaying to $1.25 \times 10^{-7}$ to prevent catastrophic forgetting. |
| **Mixed Precision** | `fp16=True` | Halves tensor memory via PyTorch Automatic Mixed Precision (AMP) and utilizes Tensor Cores. |
| **Gradient Checkpointing** | `True` | Discards intermediate activations in forward pass and recomputes them on backward pass, cutting activation VRAM by ~70%. |
| **Peak VRAM Allocated** | **`4.67 GB`** | Strictly below the 6.0 GB hardware ceiling, resulting in **0 CUDA OOM events**. |

### 4. Empirical Training Dynamics & Compute Telemetry
*Source: [`checkpoints/whisper-small-hindi/train.log`](checkpoints/whisper-small-hindi/train.log)*

- **Training Wall Time**: **189.29 seconds (~3.15 minutes)** on RTX 4050 GPU.
- **Throughput**: **4.226 samples/second** | **0.264 steps/second**.
- **Executed Compute**: **$4.591 \times 10^{17}$ Floating Point Operations (FLOPs)**.
- **Loss Trajectory**: Initial training loss of `5.9320` dropped smoothly to `2.6625` (final cumulative loss: `2.8655`).
- **Empirical Validation Loss**: **`0.3514`** evaluated across 50 unseen validation clips post-training.

<details>
<summary><b>🔍 Click to expand step-by-step training log (Steps 10 – 100)</b></summary>

```text
Step   10 | Loss: 5.9320 | Learning Rate: 4.50e-06
Step   20 | Loss: 4.6900 | Learning Rate: 9.50e-06
Step   30 | Loss: 4.1744 | Learning Rate: 8.88e-06
Step   40 | Loss: 3.4457 | Learning Rate: 7.63e-06
Step   50 | Loss: 3.2955 | Learning Rate: 6.38e-06  --> Checkpoint-50 Saved (checkpoints/checkpoint-50/)
Step   60 | Loss: 3.1199 | Learning Rate: 5.13e-06
Step   70 | Loss: 2.8818 | Learning Rate: 3.88e-06
Step   80 | Loss: 2.8998 | Learning Rate: 2.63e-06
Step   90 | Loss: 2.7637 | Learning Rate: 1.38e-06
Step  100 | Loss: 2.6625 | Learning Rate: 1.25e-07  --> Model Weights Exported (models/whisper-small-hindi/)
```
</details>

---

## 📈 Pillar 2: Empirical Evaluation & Categorized Error Analysis

### 1. Test Set Evaluation Methodology
Evaluated across all **418 Google FLEURS Hindi test clips (4,832.22s / 1.34 hrs audio)** using `jiwer`. Text normalization applies:
1. Devanagari Unicode NFKC canonicalization (normalizing nuktas and chandrabindus).
2. Punctuation stripping (`।`, `?`, `!`, `,`, `.`, `-`).
3. Case-folding of embedded Latin characters.

$$\text{WER} = \frac{S + D + I}{N} \times 100\% \quad\quad \text{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%$$

| Evaluation Metric | Baseline Zero-Shot | Fine-Tuned Whisper | Absolute Delta | Relative Gain |
|---|:---:|:---:|:---:|:---:|
| **Normalized WER** | 81.79% | **35.57%** | **46.22 percentage points (pp)** | **-56.5% relative error reduction** |
| **Normalized CER** | 49.12% | **13.25%** | **35.87 pp** | **-73.0% relative error reduction** |
| **Raw WER** | 84.12% | **36.22%** | **47.90 pp** | **-56.9% relative error reduction** |
| **Raw CER** | 49.69% | **13.42%** | **36.27 pp** | **-73.0% relative error reduction** |
| **Evaluation Wall Time** | 217.57 s | **147.98 s** | **-69.59 s** | **32% faster decoding** |
| **Real-Time Factor (RTF)** | 0.0450 | **0.0306** | **-0.0144** | **32.7× real-time throughput** |

### 2. Categorized Error Analysis Taxonomy
*Source: [`results/error_analysis.md`](results/error_analysis.md)*

Sample-level inspection across 418 test clips reveals 5 distinct linguistic error behaviors:

| Error Category | Reference Text | Zero-Shot Baseline | Fine-Tuned Model | Linguistic & Acoustic Analysis |
|---|---|---|---|---|
| **1. Repetitive Hallucination** | *Scientific observation* | `अगर अगर अगर अगर अगर अगर...` (WER: 228.2%) | `हालां कि यह आकेला नहीं है प्रयोग करना...` (WER: 61.5%) | **Not observed in inspected examples**: Cross-attention anchored to Mel spectrogram; repetition loop resolved. |
| **1. Repetitive Hallucination** | *Historical flood statement* | `बाद बाद बाद बाद बाद बाद...` (WER: 457.9%) | `1963 में बाद बनने के बाद मौसमी बाद...` (WER: 26.3%) | **Not observed in inspected examples**: Token repetition loop resolved. |
| **2. English Loanwords** | `हॉट चॉकलेट` | `अद चोकलेत` | `होट चोकलेट` | High-fidelity phonetic transliteration of loanword vowels. |
| **2. English Loanwords** | `एथलीट` | `अठलीड़` | `एठलीड़` | Preserved phoneme accuracy despite out-of-vocabulary spelling. |
| **3. Numerals / ITN** | `1963` | `अगर तो` (hallucination) | `1963` (Exact match) | Learned direct numerical representation (Inverse Text Normalization). |
| **3. Numerals / ITN** | `15 मील` | `पन्डरा मील` (verbalized) | `15 मील` (digit) | Successfully normalizes spoken number into standard digit format. |
| **4. Devanagari Conjuncts** | `वैज्ञानिक` | *Omitted entirely* | `बैज्यानिक` | High phonetic fidelity for complex consonant cluster `ज्ञ` (samyuktakshar). |
| **5. Inflectional Endings** | `सरकारों` (plural) | `सर्कारो` (unmarked) | `सरकारों` (Exact match) | Accurately predicts grammatical plural anusvara/matra. |

---

## ⚡ Pillar 3: Deployment & CTranslate2 Optimization

### 1. CTranslate2 FP16 Engine Benchmark (30 clips — 363.84 s)
*Source: [`results/latency.json`](results/latency.json)*

Converted using `ct2-transformers-converter --quantization float16`, halving model weight size to **483 MB**:

| Inference Engine | Runtime & Precision | Total Latency | Avg Latency / Clip | Median (p50) | Tail (p95) | Real-Time Factor (RTF) | Throughput | Model Weight Size |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HuggingFace Transformers** | PyTorch FP16 | 56.77 s | 1.892 s | 1.733 s | 3.065 s | 0.1560 | 6.41× real-time | 966 MB |
| **`faster-whisper` (CTranslate2)** | C++ CUDA Flat Binary FP16 | **28.75 s** | **0.958 s** | **0.719 s** | **1.670 s** | **0.0790** | **12.65× real-time** | **483 MB** |
| **Optimization Factor** | - | **-28.02 s** | **-0.934 s (1.97×)** | **2.41× faster** | **-1.395 s (1.84×)** | **1.97× Faster** | **Almost 2×** | **50% Model Weight Reduction** |

#### Why CTranslate2 Delivers a 1.97× Speedup:
1. **Dedicated C++ CUDA Execution Engine**: CTranslate2 replaces PyTorch's dynamic autograd runtime with a standalone, compiled C++ inference loop optimized for autoregressive sequence decoding.
2. **Zero Python Dispatch Overhead**: Bypasses Python interpreter GIL and PyTorch tensor dispatch latency on every generated token during decoding.
3. **Optimized Multi-Head Self-Attention**: Leverages optimized matrix multiplication routines and custom memory management for past key-value states.
4. **Memory Allocation & Profiler Transparency**:
   - In `src/benchmark_latency.py`, `torch.cuda.max_memory_allocated()` tracks allocations made through PyTorch's caching allocator (0.56 GB for HF vs 0.01 GB residual for CTranslate2).
   - Because CTranslate2 is an independent C++ library, it allocates memory via native `cudaMalloc` outside PyTorch.
   - Physical GPU allocation is determined by the **483 MB FP16 model weights** plus transient activation scratchpad buffers.

### 2. REST API Architecture (`app/main.py`)
- **FastAPI Lifespan Context Manager**: Loads both ASR (`models/whisper-small-hindi`) and TTS (`facebook/mms-tts-hin`) on server startup. The `/health` endpoint tracks `torch.cuda.memory_allocated()`, reporting 1.06 GB of GPU memory allocated to resident PyTorch model weights.
- **In-Memory Audio Processing**: Audio payloads are parsed and serialized entirely in memory via `io.BytesIO` buffers, avoiding temporary disk writes and I/O bottlenecks.
- **Pydantic v2 Defensive Validation**: Enforces string bounds (`min_length=1`, `max_length=1500`), MIME-type validation, and RFC HTTP 400/422 responses.

---

## 🔁 Reproducing All Experiments from Scratch

To reproduce all benchmarks recorded in this repository:

```bash
# 1. Evaluate baseline zero-shot Whisper on test split
python src/eval_asr.py --model openai/whisper-small --split test --output results/baseline_asr.json

# 2. Fine-tune Whisper-Small on FLEURS Hindi (5 hours / 100 steps on GPU)
python src/finetune_whisper.py --output_dir checkpoints/whisper-small-hindi --final_model_dir models/whisper-small-hindi

# 3. Evaluate fine-tuned model on test split
python src/eval_asr.py --model models/whisper-small-hindi --split test --output results/finetuned_asr.json

# 4. Generate categorized qualitative error report
python src/error_analysis.py

# 5. Synthesize 50 Hindi sentences using MMS-TTS
python src/tts_infer.py --num_samples 50 --output_dir samples/tts

# 6. Execute speech-to-speech roundtrip evaluation
python src/roundtrip_eval.py --metadata samples/tts/metadata.json --asr_model models/whisper-small-hindi

# 7. Convert model to CTranslate2 FP16 & benchmark latency
ct2-transformers-converter --model models/whisper-small-hindi --output_dir models/whisper-small-hindi-ct2 --quantization float16 --force
python src/benchmark_latency.py --num_samples 30 --output results/latency.json
```

## 📁 Repository Structure

```text
hindi-speech-pipeline/
├── README.md                   # Primary documentation
├── README_HYBRID.md            # Two-tier executive & deep-dive portfolio README
├── LEARNING_LOG.md             # Theoretical concepts, interview Q&As, commands
├── requirements.txt            # Pinned Python dependencies & Windows CUDA DLLs
├── .gitignore                  # Clean repository ignore rules (weights, checkpoints)
├── LICENSE                     # MIT License
├── notebooks/
│   └── 01_audio_basics.ipynb  # Waveform, STFT, Mel spectrogram signal processing
├── src/
│   ├── data.py                 # FLEURS data loading & in-memory soundfile decoding
│   ├── eval_asr.py             # Unicode NFKC normalized ASR evaluation & jiwer metrics
│   ├── finetune_whisper.py     # Whisper fine-tuning (Seq2SeqTrainer, fp16, grad accum)
│   ├── error_analysis.py       # Comparative baseline vs fine-tuned error analysis
│   ├── tts_infer.py            # Meta MMS-TTS Hindi VITS synthesis
│   ├── roundtrip_eval.py       # Automated speech-to-speech roundtrip evaluation
│   └── benchmark_latency.py    # HF PyTorch vs CTranslate2 RTF latency benchmark
├── app/
│   ├── main.py                 # FastAPI serving layer (/health, /transcribe, /synthesize)
│   └── schemas.py              # Pydantic request & response validation schemas
├── models/
│   ├── whisper-small-hindi/    # Exported fine-tuned PyTorch safetensors model (966 MB)
│   └── whisper-small-hindi-ct2/# Converted CTranslate2 FP16 model (483 MB)
├── checkpoints/                # Training logs and intermediate checkpoints
├── results/                    # Ground-truth empirical JSON benchmark outputs
│   ├── baseline_asr.json       # Baseline Whisper test set evaluation
│   ├── finetuned_asr.json      # Fine-tuned Whisper test set evaluation
│   ├── tts_roundtrip.json      # Speech-to-speech roundtrip evaluation
│   ├── latency.json            # HF vs CTranslate2 latency benchmark
│   └── error_analysis.md       # Categorized error analysis report
└── samples/                    # Synthesized WAV clips and acoustic visualizations
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

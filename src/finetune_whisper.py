"""
Whisper-Small Fine-Tuning Script for Hindi ASR on Google FLEURS (hi_in).

Adheres to strict GPU memory limits for 6GB VRAM (NVIDIA RTX 4050):
- Model: openai/whisper-small
- Precision: fp16
- Per device train batch size: 2
- Gradient accumulation steps: 8 (effective batch size: 16)
- Gradient checkpointing: True
- Dataloader num workers: 0 (Windows safe)
- Resumable: Saves checkpoints and training logs to disk
"""

import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)

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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads input features and label sequences for Whisper.
    Pads labels with -100 so that cross-entropy loss ignores padding tokens.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract and pad 80-channel log-Mel input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract and pad label token sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token ids with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token was prepended, trim it since Whisper decoder appends it automatically
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class LoggingCallback(TrainerCallback):
    """Logs training metrics to both console and checkpoint log file."""
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            step = state.global_step
            self.logger.info(f"Step {step:4d} | Loss: {loss} | Learning Rate: {lr}")


def prepare_dataset_for_whisper(hf_dataset, processor, max_items=None):
    """
    Preprocess raw audio bytes into Whisper 80-channel Log-Mel spectrograms and tokenized labels.
    """
    items_to_process = len(hf_dataset) if max_items is None else min(max_items, len(hf_dataset))
    processed_records = []

    for i in range(items_to_process):
        sample = hf_dataset[i]
        audio_arr, sr = decode_audio_sample(sample["audio"], target_sr=16000)

        # Extract 80-channel log-Mel spectrogram (padded/truncated to 3000 frames)
        input_feat = processor.feature_extractor(audio_arr, sampling_rate=16000).input_features[0]

        # Tokenize Hindi text
        label_ids = processor.tokenizer(sample["transcription"]).input_ids

        processed_records.append({
            "input_features": input_feat,
            "labels": label_ids,
        })

    return Dataset.from_list(processed_records)


def train(
    output_dir: str = "checkpoints/whisper-small-hindi",
    final_model_dir: str = "models/whisper-small-hindi",
    train_subset_hours: float = 5.0,
    max_steps: int = 100,
    learning_rate: float = 1e-5,
    resume_from_checkpoint: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    # Set up dedicated file logging
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("WhisperFineTune")
    logger.info("==================================================")
    logger.info("Starting Whisper-Small Fine-Tuning for Hindi (hi_in)")
    logger.info(f"Target GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Output Checkpoints: {output_dir}")
    logger.info(f"Final Model Destination: {final_model_dir}")
    logger.info("==================================================")

    model_id = "openai/whisper-small"
    logger.info(f"Loading WhisperProcessor for {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id, language="Hindi", task="transcribe")

    logger.info(f"Loading WhisperForConditionalGeneration for {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Configure model generation config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Required when gradient checkpointing is enabled
    model.gradient_checkpointing_enable()

    # Load dataset: FLEURS train split (~6.66 hrs total; 5 hrs is ~1600 clips)
    logger.info("Loading Google FLEURS Hindi (hi_in) train split...")
    raw_train_ds = load_fleurs_hindi(split="train", decode_audio=False)
    # 5.0 hours corresponds to approx 1600 clips
    num_train_clips = min(int(train_subset_hours / 6.66 * len(raw_train_ds)), len(raw_train_ds))
    logger.info(f"Preparing training subset: {num_train_clips} clips (~{train_subset_hours:.1f} hours of audio)...")
    train_dataset = prepare_dataset_for_whisper(raw_train_ds, processor, max_items=num_train_clips)

    # Load validation split for evaluation
    logger.info("Loading validation split (50 clips)...")
    raw_val_ds = load_fleurs_hindi(split="validation", decode_audio=False)
    eval_dataset = prepare_dataset_for_whisper(raw_val_ds, processor, max_items=50)

    # Initialize speech data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training Arguments strictly adhering to 6GB VRAM guidelines
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,          # Rule 7: Batch size 2
        gradient_accumulation_steps=8,          # Rule 7: 8 gradient accumulation steps
        learning_rate=learning_rate,            # 1e-5 to prevent catastrophic forgetting
        warmup_steps=20,                        # Linear warmup
        max_steps=max_steps,                    # Controlled number of steps
        gradient_checkpointing=True,            # Rule 7: Mandatory for 6GB VRAM
        fp16=True,                              # Rule 7: Mixed precision training
        dataloader_num_workers=0,               # Rule 7: Windows safe
        logging_steps=10,                       # Log every 10 steps
        save_steps=50,                          # Save checkpoint every 50 steps
        save_total_limit=2,                     # Keep last 2 checkpoints
        predict_with_generate=False,            # Faster training (generate during eval phase)
        report_to=["none"],                     # No wandb/external service needed
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback(logger)],
    )

    # Train
    logger.info("Initiating training loop...")
    checkpoint_to_resume = resume_from_checkpoint if (resume_from_checkpoint and os.path.isdir(resume_from_checkpoint)) else None
    if checkpoint_to_resume:
        logger.info(f"Resuming training from checkpoint: {checkpoint_to_resume}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # Save final fine-tuned model and processor
    logger.info(f"Training completed! Saving model weights to {final_model_dir}...")
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    logger.info(f"Peak VRAM allocated during training: {peak_vram_gb:.2f} GB")
    logger.info(f"Training stats: {train_result.metrics}")
    logger.info("Fine-tuning finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-Small on Hindi FLEURS")
    parser.add_argument("--output_dir", type=str, default="checkpoints/whisper-small-hindi")
    parser.add_argument("--final_model_dir", type=str, default="models/whisper-small-hindi")
    parser.add_argument("--train_subset_hours", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        final_model_dir=args.final_model_dir,
        train_subset_hours=args.train_subset_hours,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

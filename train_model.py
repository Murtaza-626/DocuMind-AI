import _fix_win_encoding  # noqa: F401  — must be FIRST import (patches io/pathlib on Windows)

import gc
import os
import ssl

# Fix SSL certificate verification issues on Windows
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

from config import Settings, settings


ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def _format_records(examples: Dict[str, list]) -> Dict[str, list]:
    texts = []
    for instruction, model_input, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(
            ALPACA_TEMPLATE.format(
                instruction=instruction,
                input=model_input,
                output=output,
            )
        )
    return {"text": texts}


def train_lora_adapter(
    train_jsonl_path: str | Path,
    output_dir: str | Path | None = None,
    cfg: Settings = settings,
) -> Dict[str, str]:

    output_path = Path(output_dir) if output_dir else cfg.adapter_output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }

    print(f"Loading model: {cfg.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapter using peft
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    dataset = load_dataset("json", data_files=str(train_jsonl_path), split="train")
    dataset = dataset.map(_format_records, batched=True)

    sft_config = SFTConfig(
        output_dir=str(output_path),
        dataset_text_field="text",
        max_length=cfg.max_length,
        packing=False,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        optim="adamw_torch",
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to="none",
        bf16=False,
        fp16=False,
        use_cpu=True,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    print("Starting training...")
    train_result = trainer.train()
    metrics = dict(train_result.metrics)

    # Save the LoRA adapter and tokenizer
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Explicitly free training objects so RAM is available for inference loading
    del trainer
    del model
    del tokenizer
    del dataset
    gc.collect()

    return {
        "adapter_dir": str(output_path),
        "base_model": cfg.base_model_name,
        "train_runtime_seconds": str(metrics.get("train_runtime", "n/a")),
        "train_samples_per_second": str(metrics.get("train_samples_per_second", "n/a")),
    }


if __name__ == "__main__":
    info = train_lora_adapter(settings.training_data_file, settings.adapter_output_dir, settings)
    print("Training complete")
    for k, v in info.items():
        print(f"{k}: {v}")

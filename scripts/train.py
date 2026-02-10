#!/usr/bin/env python3
"""
SFT Training Script — streams JSON progress to stdout.
Called by the Next.js API route. Expects CLI args for all parameters.

Usage:
  python train.py \
    --base_model "unsloth/Llama-3.2-1B" \
    --dataset /path/to/data.csv \
    --output_dir /path/to/output \
    --epochs 3 --batch_size 4 --learning_rate 2e-4 \
    --max_seq_length 2048 --lora_r 16 --lora_alpha 32 \
    --gradient_accumulation_steps 4 --warmup_ratio 0.03 \
    --weight_decay 0.01 --save_steps 50
"""

import argparse, csv, json, os, sys, time, math, signal, psutil

# ── helpers ---------------------------------------------------------------- #

def emit(obj: dict):
    """Print a JSON line to stdout for the API to consume."""
    print(json.dumps(obj), flush=True)


def detect_device() -> dict:
    """Return device info dict."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        return {"device": "cuda", "name": name, "memory_gb": round(mem, 1)}
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"device": "mps", "name": "Apple Silicon GPU", "memory_gb": 0}
    return {"device": "cpu", "name": "CPU", "memory_gb": 0}


def read_csv_dataset(path: str):
    """Read the Q/A CSV and return list[dict]."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question") or row.get("question") or ""
            a = row.get("Answer") or row.get("answer") or ""
            if q.strip() and a.strip():
                rows.append({"question": q.strip(), "answer": a.strip()})
    return rows


def get_system_metrics():
    """Return CPU/RAM usage.  GPU metrics added when available."""
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    m = {"cpu_percent": cpu, "ram_percent": ram}
    try:
        import torch
        if torch.cuda.is_available():
            m["gpu_util"] = 0  # no pynvml dep – placeholder
            m["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            m["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    except Exception:
        pass
    return m


# ── main ------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_token", default="", help="Optional Hugging Face token for gated models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=50)
    args = parser.parse_args()

    # ── 1. detect device --------------------------------------------------- #
    emit({"type": "status", "message": "Detecting device…"})
    device_info = detect_device()
    emit({"type": "device", **device_info})

    # ── 2. load dataset ---------------------------------------------------- #
    emit({"type": "status", "message": "Loading dataset…"})
    raw_data = read_csv_dataset(args.dataset)
    if not raw_data:
        emit({"type": "error", "message": "Dataset is empty or could not be parsed."})
        sys.exit(1)
    emit({"type": "status", "message": f"Loaded {len(raw_data)} examples."})

    # ── 3. import heavy libs ---------------------------------------------- #
    emit({"type": "status", "message": "Loading libraries (this may take a moment)…"})
    
    # Authenticate with HF token if provided (for gated models)
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            emit({"type": "status", "message": "Authenticated with Hugging Face token."})
        except Exception as e:
            emit({"type": "status", "message": f"Warning: HF token authentication failed: {e}"})
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # ── 4. build HF dataset ------------------------------------------------ #
    emit({"type": "status", "message": "Preparing dataset…"})

    def format_example(ex):
        return {"text": f"<|user|>\n{ex['question']}\n<|assistant|>\n{ex['answer']}"}

    dataset = Dataset.from_list(raw_data).map(format_example)

    # ── 5. load model & tokenizer ----------------------------------------- #
    emit({"type": "status", "message": f"Loading model {args.base_model}…"})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device_info["device"] == "cuda" else torch.float32,
        device_map="auto" if device_info["device"] == "cuda" else None,
        trust_remote_code=True,
    )

    # ── 6. LoRA ----------------------------------------------------------- #
    emit({"type": "status", "message": "Applying LoRA adapter…"})
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    emit({
        "type": "status",
        "message": f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)",
    })

    # ── 7. training args -------------------------------------------------- #
    total_steps = math.ceil(
        len(dataset) / (args.batch_size * args.gradient_accumulation_steps)
    ) * args.epochs

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=(device_info["device"] == "cuda"),
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    # ── 8. custom callback to stream metrics -------------------------------- #
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            elapsed = time.time() - self.start_time
            metrics = get_system_metrics()
            emit({
                "type": "metrics",
                "step": state.global_step,
                "total_steps": state.max_steps or total_steps,
                "epoch": round(state.epoch or 0, 2),
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "elapsed_seconds": round(elapsed, 1),
                "system": metrics,
            })

        def on_train_end(self, _args, state, control, **kwargs):
            emit({"type": "status", "message": "Training complete."})

    # ── 9. train ---------------------------------------------------------- #
    emit({
        "type": "training_start",
        "total_steps": total_steps,
        "total_examples": len(dataset),
        "epochs": args.epochs,
    })

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[ProgressCallback()],
    )
    trainer.train()

    # ── 10. merge LoRA & save --------------------------------------------- #
    emit({"type": "status", "message": "Merging LoRA adapter into base model…"})
    merged_model = trainer.model
    try:
        merged_model = merged_model.to("cpu")
    except Exception:
        pass
    try:
        merged_model = merged_model.merge_and_unload()
    except Exception as e:
        emit({"type": "log", "message": f"⚠ merge_and_unload failed ({e}), saving adapter only."})

    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    emit({"type": "done", "output_dir": args.output_dir})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FastAPI Training Server
Provides endpoints for:
  - GET  /health          – server health check
  - GET  /device          – detect training device (CPU/MPS/CUDA)
  - POST /train           – start training (SSE stream)
  - POST /train/stop      – cancel running training job
"""

import argparse
import asyncio
import csv
import json
import math
import os
import signal
import sys
import time
import threading
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="GATA Training Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ---------------------------------------------------------- #
training_thread: threading.Thread | None = None
cancel_event = threading.Event()
training_active = False


# ── Helpers --------------------------------------------------------------- #

def get_system_ram_gb() -> float:
    """Return total system RAM in GB (cross-platform)."""
    try:
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return 0


def detect_device() -> dict:
    """Return the best available device info dict."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return {"device": "cuda", "name": name, "memory_gb": round(mem, 1)}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import platform
            ram = get_system_ram_gb()
            return {"device": "mps", "name": f"Apple Silicon GPU ({platform.processor()})", "memory_gb": ram}
    except ImportError:
        pass
    return {"device": "cpu", "name": "CPU", "memory_gb": get_system_ram_gb()}


def detect_all_devices() -> list[dict]:
    """Return a list of all available training devices."""
    devices: list[dict] = []
    ram = get_system_ram_gb()

    try:
        import torch

        # CUDA GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                devices.append({
                    "device": "cuda",
                    "device_index": i,
                    "name": name,
                    "memory_gb": round(mem, 1),
                })

        # Apple MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import platform
            devices.append({
                "device": "mps",
                "device_index": 0,
                "name": f"Apple Silicon GPU ({platform.processor()})",
                "memory_gb": ram,  # MPS shares system RAM
            })
    except ImportError:
        pass

    # CPU is always available
    devices.append({
        "device": "cpu",
        "device_index": 0,
        "name": "CPU",
        "memory_gb": ram,
    })

    return devices


def get_system_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    m = {"cpu_percent": cpu, "ram_percent": ram}
    try:
        import torch
        if torch.cuda.is_available():
            m["gpu_util"] = 0
            m["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 2)
            m["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
    except Exception:
        pass
    return m


def read_csv_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question") or row.get("question") or ""
            a = row.get("Answer") or row.get("answer") or ""
            if q.strip() and a.strip():
                rows.append({"question": q.strip(), "answer": a.strip()})
    return rows


# ── Endpoints ------------------------------------------------------------- #

@app.get("/health")
async def health():
    return {"status": "ok", "training_active": training_active}


@app.get("/device")
async def device():
    info = detect_device()
    return info


@app.get("/devices")
async def all_devices():
    """Return all available training devices."""
    devices = detect_all_devices()
    return {"devices": devices}


@app.post("/train/stop")
async def stop_training():
    global training_active
    if training_active:
        cancel_event.set()
        return {"success": True, "message": "Cancel signal sent."}
    return JSONResponse({"error": "No training in progress"}, status_code=404)


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    params: str = Form(...),
):
    global training_active, training_thread

    if training_active:
        return JSONResponse(
            {"error": "A training job is already running."},
            status_code=409,
        )

    config = json.loads(params)

    # Save CSV to temp location
    project_root = Path(__file__).resolve().parent.parent
    tmp_dir = project_root / "training_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_dir / f"dataset_{int(time.time())}.csv"
    content = await file.read()
    csv_path.write_bytes(content)

    output_dir = tmp_dir / f"output_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use an asyncio queue to pass messages from training thread → SSE stream
    msg_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def enqueue(obj: dict):
        """Thread-safe enqueue."""
        loop.call_soon_threadsafe(msg_queue.put_nowait, json.dumps(obj))

    def run_training():
        global training_active
        training_active = True
        cancel_event.clear()
        try:
            _do_training(config, str(csv_path), str(output_dir), enqueue)
        except Exception as e:
            enqueue({"type": "error", "message": str(e)})
        finally:
            training_active = False
            loop.call_soon_threadsafe(msg_queue.put_nowait, None)  # sentinel

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    async def event_generator():
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Core training logic --------------------------------------------------- #

def _make_download_callback(emit):
    """
    Monkey-patch huggingface_hub's file download to emit progress events
    with percentage and speed.
    """
    try:
        import huggingface_hub.file_download as _hf_dl
        import functools

        _original_download = _hf_dl.http_get

        def _patched_http_get(url, temp_file, *, proxies=None, resume_size=0,
                              headers=None, expected_size=None, displayed_filename=None,
                              _nb_retries=5, **kwargs):
            """Wrap the real http_get to stream download progress."""
            import requests as _req

            # Determine the filename being downloaded
            fname = displayed_filename or url.split("/")[-1].split("?")[0]

            # Call original but intercept by wrapping temp_file.write
            _orig_write = temp_file.write
            _state = {"downloaded": resume_size, "last_time": time.time(),
                       "last_bytes": resume_size, "total": expected_size or 0}

            def _write_hook(data):
                n = _orig_write(data)
                _state["downloaded"] += len(data)
                now = time.time()
                dt = now - _state["last_time"]
                if dt >= 0.5:  # emit at most 2x per second
                    speed = (_state["downloaded"] - _state["last_bytes"]) / dt
                    _state["last_time"] = now
                    _state["last_bytes"] = _state["downloaded"]
                    total = _state["total"]
                    pct = round((_state["downloaded"] / total) * 100, 1) if total else 0
                    emit({
                        "type": "download_progress",
                        "filename": fname,
                        "downloaded_bytes": _state["downloaded"],
                        "total_bytes": total,
                        "percent": pct,
                        "speed_bytes_per_sec": round(speed),
                    })
                return n

            temp_file.write = _write_hook
            try:
                return _original_download(
                    url, temp_file, proxies=proxies, resume_size=resume_size,
                    headers=headers, expected_size=expected_size,
                    displayed_filename=displayed_filename,
                    _nb_retries=_nb_retries, **kwargs,
                )
            finally:
                temp_file.write = _orig_write

        _hf_dl.http_get = _patched_http_get
        return _original_download  # so we can restore later
    except Exception:
        return None


def _restore_download_callback(original):
    """Restore the original download function."""
    if original is None:
        return
    try:
        import huggingface_hub.file_download as _hf_dl
        _hf_dl.http_get = original
    except Exception:
        pass


# Training pipeline step IDs (order matters)
PIPELINE_STEPS = [
    "detect_device",
    "load_dataset",
    "load_libraries",
    "prepare_dataset",
    "download_model",
    "apply_lora",
    "training",
    "saving",
]


def _do_training(config: dict, csv_path: str, output_dir: str, emit):
    """Run the actual training. Runs in a background thread."""

    def step_event(step_id: str, status: str, detail: str = ""):
        """Emit a pipeline step status update."""
        emit({"type": "pipeline_step", "step": step_id, "status": status, "detail": detail})

    try:
        # Main training logic wrapped in try-except to catch critical errors
        _training_logic(config, csv_path, output_dir, emit, step_event)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        emit({"type": "log", "message": f"❌ Critical training error:\n{error_trace}"})
        emit({"type": "error", "message": f"Training failed: {str(e)}"})
        # Mark all steps as error
        for step_id in ["detect_device", "load_dataset", "load_libraries", "prepare_dataset",
                        "download_model", "apply_lora", "training", "saving", "convert_gguf"]:
            step_event(step_id, "error", "Training stopped due to error")


def _training_logic(config: dict, csv_path: str, output_dir: str, emit, step_event):

    # 1. resolve device (user-selected or auto-detect)
    step_event("detect_device", "in-progress", "Detecting device…")
    emit({"type": "status", "message": "Detecting device…"})

    selected_device = config.get("trainingDevice")  # e.g. "mps", "cuda", "cpu"
    if selected_device:
        # Validate the selected device is actually available
        all_devs = detect_all_devices()
        device_info = next((d for d in all_devs if d["device"] == selected_device), None)
        if device_info is None:
            emit({"type": "status", "message": f"Selected device '{selected_device}' not available, falling back…"})
            device_info = detect_device()
    else:
        device_info = detect_device()

    emit({"type": "device", **device_info})
    step_event("detect_device", "completed", f"{device_info['device'].upper()} — {device_info['name']}")

    # 2. load dataset
    step_event("load_dataset", "in-progress", "Loading dataset…")
    emit({"type": "status", "message": "Loading dataset…"})
    raw_data = read_csv_dataset(csv_path)
    if not raw_data:
        step_event("load_dataset", "error", "Dataset is empty")
        emit({"type": "error", "message": "Dataset is empty or could not be parsed."})
        return
    step_event("load_dataset", "completed", f"{len(raw_data)} examples loaded")
    emit({"type": "status", "message": f"Loaded {len(raw_data)} examples."})

    if cancel_event.is_set():
        emit({"type": "error", "message": "Training cancelled."})
        return

    # 3. import heavy libs
    step_event("load_libraries", "in-progress", "Importing torch, transformers, peft, trl…")
    emit({"type": "status", "message": "Loading libraries (this may take a moment)…"})
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    step_event("load_libraries", "completed", "All libraries loaded")

    if cancel_event.is_set():
        emit({"type": "error", "message": "Training cancelled."})
        return

    # 4. build HF dataset
    step_event("prepare_dataset", "in-progress", "Formatting dataset…")
    emit({"type": "status", "message": "Preparing dataset…"})

    def format_example(ex):
        return {"text": f"<|user|>\n{ex['question']}\n<|assistant|>\n{ex['answer']}"}

    dataset = Dataset.from_list(raw_data).map(format_example)
    step_event("prepare_dataset", "completed", f"{len(dataset)} formatted examples")

    # 5. load model & tokenizer (with download progress)
    step_event("download_model", "in-progress", f"Downloading {config['baseModel']}…")
    emit({"type": "status", "message": f"Loading model {config['baseModel']}…"})

    # Authenticate with HF token if provided (for gated models)
    hf_token = config.get("hfToken")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            emit({"type": "status", "message": "Authenticated with Hugging Face token."})
        except Exception as e:
            emit({"type": "status", "message": f"Warning: HF token authentication failed: {e}"})

    original_dl = _make_download_callback(emit)

    tokenizer = AutoTokenizer.from_pretrained(config["baseModel"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cancel_event.is_set():
        _restore_download_callback(original_dl)
        emit({"type": "error", "message": "Training cancelled."})
        return

    # Pick dtype & device_map based on selected device
    dev = device_info["device"]
    if dev == "cuda":
        model_dtype = torch.float16
        model_device_map = "auto"
    elif dev == "mps":
        model_dtype = torch.float32  # MPS doesn't fully support fp16 training
        model_device_map = None
    else:
        model_dtype = torch.float32
        model_device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        config["baseModel"],
        torch_dtype=model_dtype,
        device_map=model_device_map,
        trust_remote_code=True,
    )

    # Move model to the correct device (needed for MPS & single-GPU without device_map)
    if dev == "mps":
        model = model.to("mps")
    elif dev == "cuda" and model_device_map is None:
        model = model.to("cuda")

    _restore_download_callback(original_dl)
    # Send a final 100% to close any open download bar
    emit({"type": "download_progress", "filename": "", "downloaded_bytes": 0,
          "total_bytes": 0, "percent": 100, "speed_bytes_per_sec": 0})
    step_event("download_model", "completed", "Model loaded into memory")

    # 6. LoRA
    step_event("apply_lora", "in-progress", "Applying LoRA adapter…")
    emit({"type": "status", "message": "Applying LoRA adapter…"})
    lora_r = int(config.get("loraR", 16))
    lora_alpha = int(config.get("loraAlpha", 32))
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
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
    step_event("apply_lora", "completed", f"{trainable:,} / {total:,} trainable params")

    if cancel_event.is_set():
        emit({"type": "error", "message": "Training cancelled."})
        return

    # 7. training args
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batchSize", 4))
    grad_accum = int(config.get("gradientAccumulationSteps", 4))
    lr = float(config.get("learningRate", 2e-4))
    max_length = int(config.get("maxSeqLength", 2048))
    warmup_ratio = float(config.get("warmupRatio", 0.03))
    weight_decay = float(config.get("weightDecay", 0.01))
    save_steps = int(config.get("saveSteps", 50))

    total_steps = math.ceil(len(dataset) / (batch_size * grad_accum)) * epochs

    # Build device-aware training config
    sft_kwargs: dict = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=2,
        report_to="none",
        max_length=max_length,
        dataset_text_field="text",
    )

    # Set fp16 only for CUDA (SFTConfig doesn't accept no_cuda/use_cpu)
    dev = device_info["device"]
    if dev == "cuda":
        sft_kwargs["fp16"] = True
    else:
        # MPS and CPU: no fp16, device placement already handled via model.to()
        sft_kwargs["fp16"] = False

    training_args = SFTConfig(**sft_kwargs)

    # 8. callback
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()

        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if cancel_event.is_set():
                control.should_training_stop = True
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

    # 9. train
    step_event("training", "in-progress", "Training in progress…")
    emit({
        "type": "training_start",
        "total_steps": total_steps,
        "total_examples": len(dataset),
        "epochs": epochs,
    })

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[ProgressCallback()],
    )

    trainer.train()

    if cancel_event.is_set():
        step_event("training", "error", "Cancelled by user")
        emit({"type": "error", "message": "Training cancelled."})
        return

    step_event("training", "completed", f"Completed {total_steps} steps")

    # 10. merge LoRA adapter and save full model
    step_event("saving", "in-progress", "Merging LoRA adapter into base model…")
    emit({"type": "status", "message": "Merging LoRA adapter into base model…"})

    # Move model to CPU for merging (avoids MPS / half-precision issues)
    merged_model = trainer.model
    try:
        merged_model = merged_model.to("cpu")
    except Exception:
        pass

    try:
        merged_model = merged_model.merge_and_unload()
        emit({"type": "status", "message": "LoRA adapter merged successfully."})
    except Exception as e:
        emit({"type": "log", "message": f"⚠ merge_and_unload failed ({e}), saving adapter only."})

    emit({"type": "status", "message": "Saving merged model to disk…"})
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    # 11. Convert to GGUF for Ollama compatibility (optional)
    step_event("convert_gguf", "in-progress", "Converting model to GGUF format…")
    emit({"type": "status", "message": "Converting model to GGUF format for Ollama…"})
    
    try:
        import subprocess
        import sys
        import shutil
        
        # Try to find llama.cpp convert script in common locations
        gguf_output = os.path.join(output_dir, "model.gguf")
        convert_script = None
        
        # Check if convert-hf-to-gguf.py exists in PATH or common locations
        possible_scripts = [
            "convert-hf-to-gguf.py",
            "convert_hf_to_gguf.py",
            shutil.which("convert-hf-to-gguf.py"),
        ]
        
        # Also check in llama.cpp installation if it exists
        llama_cpp_paths = [
            "/usr/local/bin/convert-hf-to-gguf.py",
            os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
            os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        ]
        possible_scripts.extend(llama_cpp_paths)
        
        for script_path in possible_scripts:
            if script_path and os.path.exists(str(script_path)):
                convert_script = script_path
                break
        
        if convert_script:
            convert_cmd = [
                sys.executable, convert_script,
                output_dir,
                "--outfile", gguf_output,
                "--outtype", "f16",
            ]
            
            emit({"type": "log", "message": f"Running GGUF conversion: {' '.join(convert_cmd)}"})
            
            result = subprocess.run(
                convert_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=output_dir
            )
            
            if result.returncode == 0 and os.path.exists(gguf_output):
                emit({"type": "status", "message": "✓ Model successfully converted to GGUF format."})
                step_event("convert_gguf", "completed", f"GGUF saved ({os.path.getsize(gguf_output) // (1024**2)} MB)")
            else:
                error_msg = result.stderr or result.stdout or "Conversion failed"
                emit({"type": "log", "message": f"⚠ GGUF conversion failed: {error_msg[:500]}"})
                emit({"type": "log", "message": "Model saved as safetensors. You can convert manually or try a different model."})
                step_event("convert_gguf", "error", "Conversion failed - safetensors available")
        else:
            emit({"type": "log", "message": "⚠ llama.cpp conversion script not found."})
            emit({"type": "log", "message": "Model saved as safetensors. Install llama.cpp for GGUF conversion, or use compatible models with Ollama."})
            emit({"type": "log", "message": "Info: Some models work with Ollama's safetensors import (e.g., Llama, Mistral, Qwen2). Qwen3 may need GGUF."})
            step_event("convert_gguf", "error", "Conversion tool not available")
            
    except subprocess.TimeoutExpired:
        emit({"type": "log", "message": "⚠ GGUF conversion timed out after 10 minutes."})
        emit({"type": "log", "message": "Model saved as safetensors only."})
        step_event("convert_gguf", "error", "Conversion timeout")
    except Exception as e:
        emit({"type": "log", "message": f"⚠ GGUF conversion error: {e}"})
        emit({"type": "log", "message": "Model saved as safetensors only."})
        step_event("convert_gguf", "error", f"Error: {str(e)[:100]}")
    
    step_event("saving", "completed", f"Saved to {output_dir}")
    emit({"type": "done", "output_dir": output_dir, "base_model": config.get("baseModel", "")})


# ── Entry point ----------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"🚀 Training server starting on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

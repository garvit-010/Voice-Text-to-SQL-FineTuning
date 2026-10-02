"""Copy v2 adapter to garvit-010 - streams the large file directly."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env", override=True)

SRC_REPO  = "hari-krishna-ai/qwen3-8b-text2sql-qlora-v2"
DEST_REPO = "garvit-010/qwen3-8b-text2sql-qlora-v3"
TOKEN     = os.getenv("HF_WRITE_TOKEN", "").strip()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

api = HfApi(token=TOKEN)
who = api.whoami()
print(f"Authenticated as: {who['name']}")
print(f"Source : {SRC_REPO}")
print(f"Dest   : {DEST_REPO}")

print("\nCreating destination repo...")
api.create_repo(DEST_REPO, repo_type="model", exist_ok=True)
print("Repo ready.")

# Small files first (fast)
SMALL_FILES = [
    "adapter_config.json",
    "tokenizer.json", 
    "tokenizer_config.json",
    "chat_template.jinja",
    "training_config.json",
    "training_metrics.json",
]

print("\nCopying small files...")
for fname in SMALL_FILES:
    try:
        local = hf_hub_download(repo_id=SRC_REPO, filename=fname, token=TOKEN)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=fname,
            repo_id=DEST_REPO,
            repo_type="model",
            commit_message=f"Add {fname}",
        )
        size = Path(local).stat().st_size / 1024
        print(f"  OK {fname} ({size:.0f} KB)")
    except Exception as e:
        print(f"  SKIP {fname}: {e}")

# Big file
print("\nCopying adapter_model.safetensors (~167 MB, please wait)...")
try:
    local = hf_hub_download(
        repo_id=SRC_REPO,
        filename="adapter_model.safetensors",
        token=TOKEN,
    )
    size_mb = Path(local).stat().st_size / 1024 / 1024
    print(f"  Downloaded: {size_mb:.1f} MB")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo="adapter_model.safetensors",
        repo_id=DEST_REPO,
        repo_type="model",
        commit_message="Add adapter weights",
    )
    print("  OK adapter_model.safetensors")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Model card
CARD = """---
language: en
license: apache-2.0
base_model: Qwen/Qwen3-8B
base_model_relation: finetune
tags:
  - text-to-sql
  - qlora
  - peft
  - qwen3
  - postgresql
  - voice-to-sql
---

# Enterprise Text-to-SQL + Voice -- QLoRA fine-tuned Qwen3-8B

**Strict execution accuracy: 43.71% to 70.86%** on 453 held-out questions.

## By

**Garvit Audichya** - [HuggingFace](https://huggingface.co/garvit-010)

## Features

- Voice input -- speak your question, Whisper transcribes it
- Fine-tuned Qwen3-8B -- QLoRA r=16, 4-bit NF4
- Self-correction loop -- feeds DB errors back to model, retries once
- Static SQL validation -- sqlglot catches bad queries before execution
- PostgreSQL execution -- read-only, statement timeout, no writes possible

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE    = "Qwen/Qwen3-8B"
ADAPTER = "garvit-010/qwen3-8b-text2sql-qlora-v3"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)
base  = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"":0})
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
```

IMPORTANT: Always use enable_thinking=False in apply_chat_template.

## Results

| Configuration | Strict accuracy |
|---|---:|
| Base Qwen3-8B | 43.71% |
| Fine-tuned (this adapter) | 68.43% |
| Fine-tuned + self-correction | **70.86%** |

## Training

- Base: Qwen/Qwen3-8B @ b968826d9c46, 4-bit NF4
- LoRA r=16, alpha=32, 43.6M trainable params (0.917%)
- Completion-only loss on SQL tokens only
- Kaggle T4, 1 epoch, 135 steps, ~7 hours

## License

Apache 2.0 - all training data is synthetic
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
    f.write(CARD)
    tmp = f.name

api.upload_file(
    path_or_fileobj=tmp,
    path_in_repo="README.md",
    repo_id=DEST_REPO,
    repo_type="model",
    commit_message="Add model card - Garvit Audichya",
)
print("OK README.md")

print()
print("=" * 60)
print(f"DONE! https://huggingface.co/{DEST_REPO}")
print("=" * 60)

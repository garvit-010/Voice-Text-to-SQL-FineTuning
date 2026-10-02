"""Push the Space files to garvit-010/enterprise-text-to-sql on HuggingFace.

Run:
    env\Scripts\python.exe scripts/push_space.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env", override=True)

SPACE_REPO = "garvit-010/enterprise-text-to-sql"
SPACE_DIR  = Path(__file__).parents[1] / "deploy" / "space"
TOKEN      = os.getenv("HF_WRITE_TOKEN", "").strip()

if not TOKEN or TOKEN.startswith("<<"):
    print("[error] HF_WRITE_TOKEN not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi

api = HfApi(token=TOKEN)

# Verify
try:
    who = api.whoami()
    print(f"Authenticated as: {who['name']}")
except Exception as e:
    print(f"[error] Token rejected: {e}")
    sys.exit(1)

print(f"\nPushing Space files to: {SPACE_REPO}")
print(f"Source directory: {SPACE_DIR}")
print()

files = list(SPACE_DIR.iterdir())
for f in sorted(files):
    if f.is_file():
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

print()

# Upload all files in the space directory
api.upload_folder(
    folder_path=str(SPACE_DIR),
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy Enterprise Text-to-SQL Space — garvit-010",
    ignore_patterns=["*.pyc", "__pycache__", ".gitattributes"],
)

print()
print("=" * 60)
print(f"DONE! Space live at:")
print(f"  https://huggingface.co/spaces/{SPACE_REPO}")
print("=" * 60)

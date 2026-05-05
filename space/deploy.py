"""Create the HF Space and upload this directory to it."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "kevinreyesds/signlang"
SPACE_DIR = Path(__file__).parent


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN env var.")
        return 1
    api = HfApi(token=token)
    print(f"Creating Space {REPO_ID} (skipped if it exists)...")
    api.create_repo(
        repo_id=REPO_ID, repo_type="space", space_sdk="gradio",
        private=False, exist_ok=True,
    )
    print(f"Uploading {SPACE_DIR} ...")
    api.upload_folder(
        folder_path=str(SPACE_DIR), repo_id=REPO_ID, repo_type="space",
        commit_message="Deploy signlang demo",
        ignore_patterns=["deploy.py", "__pycache__/*", "*.pyc"],
    )
    print(f"\nDone.\n  Space: https://huggingface.co/spaces/{REPO_ID}\n  Embed: https://{REPO_ID.replace('/', '-')}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())

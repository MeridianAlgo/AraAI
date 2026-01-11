#!/usr/bin/env python3
"""
Setup and Cleanup script for Ara AI Hugging Face Repository.
Run this to reset or initialize the repository structure.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path for imports
root = str(Path(__file__).parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from huggingface_hub import HfApi
    from scripts.hf_manager import HFManager, generate_model_card
except ImportError:
    print(
        "Error: Required dependencies not found. Please run: pip install huggingface_hub python-dotenv"
    )
    sys.exit(1)

# Load environment variables
load_dotenv()


def setup_huggingface():
    print("=== Ara AI Hugging Face Setup & Cleanup ===")

    token = os.getenv("HF_TOKEN")
    repo_id = "MeridianAlgo/ARA.AI"

    if not token:
        print("✗ Error: HF_TOKEN not found in .env file.")
        print("Please add HF_TOKEN=your_token_here to your .env file.")
        return

    manager = HFManager(repo_id=repo_id, token=token)
    api = HfApi(token=token)

    # 1. Ask for confirmation to clean up (delete existing models)
    confirm = input(
        f"Do you want to delete ALL existing models in {repo_id} to start fresh? (y/N): "
    )
    if confirm.lower() == "y":
        print(f"Cleaning all files in {repo_id}...")
        try:
            files = api.list_repo_files(repo_id=repo_id)
            # Filter for files in models/ directory
            to_delete = [f for f in files if f.startswith("models/")]

            for file_path in to_delete:
                print(f"  Deleting {file_path}...")
                api.delete_file(
                    path_in_repo=file_path, repo_id=repo_id, repo_type="model"
                )
            print("✓ Repository models cleaned.")
        except Exception as e:
            print(f"✗ Failed to clean repository: {e}")
    else:
        print("Skipping full cleanup.")

    # 2. Setup Initial Model Card
    print("\nSetting up initial model card...")
    initial_card = generate_model_card(
        symbol="Initialization", accuracy="N/A", loss="N/A", trained_on_count="0"
    )

    if manager.update_model_card(initial_card):
        print("✓ Model card initialized.")

    # 3. Create .gitattributes for LFS (Required for model files)
    print("\nSetting up .gitattributes for LFS...")
    lfs_config = "*.pt filter=lfs diff=lfs merge=lfs -text\n*.bin filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text"

    try:
        temp_attr = Path("temp_gitattributes")
        temp_attr.write_text(lfs_config)
        api.upload_file(
            path_or_fileobj=str(temp_attr),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="model",
        )
        temp_attr.unlink()
        print("✓ LFS configured (.gitattributes uploaded).")
    except Exception as e:
        print(f"✗ Failed to configure LFS: {e}")

    print("\n=== Setup Complete! ===")
    print(f"Repository {repo_id} is now ready for continuous training.")
    print("The GitHub Actions workflow will now be able to push and pull models.")


if __name__ == "__main__":
    setup_huggingface()

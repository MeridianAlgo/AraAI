#!/usr/bin/env python3
"""Safe cleanup script for the AraAI project.

This script removes unnecessary files and directories while logging what
has been deleted. It is deliberately conservative – it only deletes items
that are known to be safe to remove in a typical development environment.

Usage:
    python scripts/cleanup_project.py
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging – prints to console and writes a log file in the project root.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleanup_log.txt", mode="a", encoding="utf-8"),
    ],
)

# ---------------------------------------------------------------------------
# Define patterns / paths that are safe to delete.
# ---------------------------------------------------------------------------
DELETE_PATHS = [
    # Development artefacts
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.bak",
    "*.tmp",
    "*.log",
    "*.DS_Store",
    # Jupyter notebook checkpoints
    ".ipynb_checkpoints",
    # Credential files that are no longer needed (the project now uses HF_TOKEN)
    "credentials",
    # Large data dumps that are not required for training (replace with your own path if needed)
    "data/raw",
    "data/processed",
]

def should_delete(path: Path) -> bool:
    """Return True if *path* matches any of the DELETE_PATHS patterns.
    The function works for both files and directories.
    """
    for pattern in DELETE_PATHS:
        if path.match(pattern):
            return True
        # For directories we also allow exact name matches
        if path.is_dir() and path.name == pattern:
            return True
    return False

def safe_remove(path: Path) -> None:
    """Delete *path* safely, logging the action.
    Directories are removed recursively.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path)
            logging.info(f"Removed directory: {path}")
        else:
            path.unlink()
            logging.info(f"Removed file: {path}")
    except Exception as e:
        logging.error(f"Failed to remove {path}: {e}")

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    logging.info(f"Starting cleanup in {project_root}")

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        # Check directories first – we may modify the list in‑place to avoid walking deleted dirs.
        for d in list(dirs):
            dir_path = root_path / d
            if should_delete(dir_path):
                safe_remove(dir_path)
                # Remove from walk list so os.walk does not descend into it.
                dirs.remove(d)
        # Then check files.
        for f in files:
            file_path = root_path / f
            if should_delete(file_path):
                safe_remove(file_path)

    logging.info("Cleanup completed.")

if __name__ == "__main__":
    main()

# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Permission helpers for files created by Torch Tweak."""

from pathlib import Path

SECURE_DIR_MODE = 0o700
SECURE_FILE_MODE = 0o600


def secure_mkdir(path: Path) -> None:
    """Create a directory and prevent permissive mode inheritance."""
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(SECURE_DIR_MODE)


def secure_file(path: Path) -> None:
    """Restrict permissions on a created file."""
    if path.exists():
        path.chmod(SECURE_FILE_MODE)


def secure_tree(path: Path) -> None:
    """Restrict permissions on a created directory tree."""
    if path.is_file():
        secure_file(path)
        return

    for child in path.rglob("*"):
        if child.is_dir():
            child.chmod(SECURE_DIR_MODE)
        elif child.is_file():
            secure_file(child)
    path.chmod(SECURE_DIR_MODE)

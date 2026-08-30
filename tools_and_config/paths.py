"""Repository and runtime path helpers.

All defaults are relative to the checkout. Deployments can relocate writable
state by setting ``KIKIFAST_RUNTIME_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(
    os.environ.get("KIKIFAST_RUNTIME_DIR", REPO_ROOT / "runtime")
).expanduser().resolve()


def repo_path(value: str | os.PathLike[str]) -> Path:
    """Resolve an absolute path or a path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def runtime_path(*parts: str) -> Path:
    """Return a path below the configured writable runtime directory."""
    return RUNTIME_DIR.joinpath(*parts)

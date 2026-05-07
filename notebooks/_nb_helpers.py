"""Tiny helpers used inside notebooks (path setup, plot styling).

Importing /src from a notebook needs the project root on sys.path. Doing it
here keeps every notebook's first cell clean.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_repo_to_path() -> Path:
    here = Path.cwd()
    for candidate in (here, here.parent):
        if (candidate / "src").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Could not locate repo root from notebook cwd")

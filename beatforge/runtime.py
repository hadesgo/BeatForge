from __future__ import annotations

import gc
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def command(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def probe(file: Path) -> dict[str, Any]:
    result = command([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(file),
    ], capture=True)
    return json.loads(result.stdout)


def duration(file: Path) -> float:
    return float(probe(file).get("format", {}).get("duration", 0))


def require_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"PATH 中缺少: {', '.join(missing)}")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def release_gpu(*objects: object) -> None:
    del objects
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


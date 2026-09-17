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


def require_usable_ai_device(requested: str, resolved: str) -> None:
    """Fail early when a CUDA environment is present but its driver is unusable."""
    if requested == "cpu":
        return
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("配置要求 CUDA，但当前环境没有安装 PyTorch") from None
        return
    if torch.cuda.is_available():
        return
    cuda_build = getattr(torch.version, "cuda", None)
    if requested == "cuda" or cuda_build:
        build = f"CUDA {cuda_build}" if cuda_build else "CUDA"
        raise RuntimeError(
            f"已安装 {build} 版 PyTorch，但 CUDA 当前不可用。请升级 NVIDIA 驱动并重启后运行 "
            "`uv run beatforge doctor`；若确实要使用CPU，请安装 ai-cpu profile 并设置 device = \"cpu\"。"
        )


def optimize_torch_runtime(device: str) -> None:
    """Enable safe inference-oriented CUDA fast paths without changing model outputs materially."""
    if device != "cuda":
        return
    try:
        import torch
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except (ImportError, AttributeError):
        pass


def release_gpu(*objects: object) -> None:
    del objects
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, AttributeError, RuntimeError, OSError):
        # Best-effort cleanup, and it must never be the thing that fails a stage. A
        # torch that is present but unusable - a half-finished install, a build with no
        # ``cuda`` attribute, a driver that refuses to initialise - is exactly the state
        # this is most likely to run in, and it is not worth reporting.
        pass

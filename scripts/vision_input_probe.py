"""Measure what normalising the vision inputs actually buys.

The Qwen-VL processors resize every image to their own pixel budget before the
vision tower sees anything, so the original resolution only ever buys a bigger
decode - and the reranker pays it once per shortlist candidate, because the same
asset shows up as several candidates.

This measures both paths on the same oversized photos. Each path runs in its own
process (``--worker``) so the peak reading belongs to that path alone.

Run: uv run python scripts/vision_input_probe.py
     uv run python scripts/vision_input_probe.py --size 6000x4000 --count 8
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.models.vision_index import DEFAULT_INPUT_PIXELS, _fit_within


def peak_bytes() -> int:
    """Peak working set of this process, on Windows as well as Unix.

    The ``argtypes``/``restype`` declarations are not optional: without them ctypes
    guesses, and ``GetProcessMemoryInfo`` reports zero instead of failing.
    """
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        query(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters), ctypes.sizeof(counters),
        )
        return int(counters.PeakWorkingSetSize)
    import resource
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak * 1024 if sys.platform.startswith("linux") else peak


def worker(mode: str, budget: int, files: list[Path]) -> int:
    """Load the files the way one of the two paths would, then report the cost."""
    started = time.perf_counter()
    pixels = 0
    for name in files:
        with Image.open(name) as source:
            if mode == "normalised":
                target = _fit_within(*source.size, budget)
                # The free 1/2, 1/4, 1/8 JPEG decode is a large part of the win.
                source.draft("RGB", target)
            image = source.convert("RGB")
            if mode == "normalised" and image.size != target:
                image = image.resize(target, Image.LANCZOS)
            pixels += image.width * image.height
            del image
    print(json.dumps({
        "seconds": time.perf_counter() - started,
        "peak_bytes": peak_bytes(),
        "pixels": pixels,
    }))
    return 0


def make_photos(directory: Path, count: int, size: tuple[int, int]) -> list[Path]:
    """Synthetic photos with a 1/f-ish spectrum.

    Flat colour would compress to nothing and flatter both paths equally; real
    photographs are the case this exists for.
    """
    rng = np.random.default_rng(4)
    width, height = size
    coarse = (max(8, width // 24), max(8, height // 24))
    files = []
    for index in range(count):
        low = rng.normal(0, 1, (coarse[1], coarse[0], 3)).astype(np.float32)
        low = (low - low.min()) / max(float(np.ptp(low)), 1e-6)
        base = np.asarray(
            Image.fromarray((low * 200 + 28).clip(0, 255).astype(np.uint8))
            .resize((width, height), Image.BICUBIC),
            dtype=np.float32,
        )
        base += rng.normal(0, 7, base.shape)
        file = directory / f"photo-{index}.jpg"
        Image.fromarray(base.clip(0, 255).astype(np.uint8)).save(file, quality=90)
        files.append(file)
    return files


def measure(mode: str, budget: int, files: list[Path]) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode, str(budget),
         *map(str, files)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", default="6000x4000")
    parser.add_argument("--budget", type=int, default=DEFAULT_INPUT_PIXELS)
    parser.add_argument("--out", type=Path, default=Path(".probe/vision-input"))
    parser.add_argument("--worker", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        mode, budget, *rest = args.worker
        return worker(mode, int(budget), [Path(name) for name in rest])

    width, height = (int(value) for value in args.size.lower().split("x"))
    args.out.mkdir(parents=True, exist_ok=True)
    files = make_photos(args.out, args.count, (width, height))
    source_mb = sum(file.stat().st_size for file in files) / 1e6
    target = _fit_within(width, height, args.budget)

    raw = measure("raw", args.budget, files)
    normalised = measure("normalised", args.budget, files)

    print(f"{args.count} 张 {width}x{height} 照片（源文件共 {source_mb:.1f} MB）")
    print(f"编码器像素预算 {args.budget} → 每张压到 {target[0]}x{target[1]}")
    print()
    print(f"{'':<10} {'耗时':>9} {'峰值内存':>11} {'送进模型的像素':>16}")
    for label, data in (("原样喂", raw), ("归一化后", normalised)):
        print(f"{label:<10} {data['seconds']:>8.2f}s {data['peak_bytes'] / 2**20:>9.0f} MiB "
              f"{data['pixels'] / 1e6:>14.2f} MP")

    speedup = raw["seconds"] / max(normalised["seconds"], 1e-6)
    memory = raw["peak_bytes"] / max(normalised["peak_bytes"], 1)
    pixels = raw["pixels"] / max(normalised["pixels"], 1)
    print()
    print(f"快 {speedup:.1f}x · 峰值内存低 {memory:.1f}x · 模型看到的像素少 {pixels:.1f}x")
    print()
    print("说明：这里只量到「素材预处理」这一层。真正的显存收益还要算上视觉编码器——"
          "它按像素数决定 patch 数量，而重排阶段会对每个候选重复支付一次。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare zoompan against sub-pixel alternatives for slow camera moves.

Renders a Gaussian blob through three implementations of the same move and
recovers its sub-pixel x position from the intensity centroid. A smooth move
produces near-constant steps; quantisation shows up as steps that hold at zero and
then jump.

Run: uv run python scripts/zoompan_probe.py
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT, FPS, FRAMES = 1920, 1080, 30, 120
TRAVEL = 40.0
BLOB_X, BLOB_Y, BLOB_SIGMA = 960.0, 540.0, 14.0
ZOOM = 1.2
DURATION = FRAMES / FPS


def blob_image(path: Path) -> None:
    x = np.arange(WIDTH)[None, :]
    y = np.arange(HEIGHT)[:, None]
    value = 20 + 220 * np.exp(
        -(((x - BLOB_X) ** 2 + (y - BLOB_Y) ** 2) / (2 * BLOB_SIGMA**2))
    )
    Image.fromarray(np.clip(value, 0, 255).astype(np.uint8)).save(path)


def measure(clip: Path) -> tuple[np.ndarray, float]:
    """Sub-pixel x position steps per frame, plus a sharpness score."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
    frames = frames.reshape(-1, HEIGHT, WIDTH)

    columns = np.arange(WIDTH)
    positions = []
    for frame in frames:
        profile = frame.mean(axis=0)
        weights = np.clip(profile - np.median(profile), 0, None)
        positions.append(float(np.dot(weights, columns) / weights.sum()))

    # Edge width of the blob is a proxy for resampling sharpness: a blurrier
    # render spreads the peak over more pixels, so the second moment grows.
    profile = frames[len(frames) // 2].mean(axis=0)
    weights = np.clip(profile - np.median(profile), 0, None)
    centre = np.dot(weights, columns) / weights.sum()
    spread = float(np.sqrt(np.dot(weights, (columns - centre) ** 2) / weights.sum()))
    return np.diff(np.array(positions)), spread


def render(filters: str, source: Path, output: Path) -> float:
    started = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS),
         "-i", str(source), "-t", f"{DURATION:.4f}", "-filter_complex", filters,
         "-map", "[vout]", "-an", "-r", str(FPS), "-crf", "14",
         "-pix_fmt", "yuv420p", str(output)],
        check=True,
    )
    return time.perf_counter() - started


def report(label: str, steps: np.ndarray, spread: float, seconds: float) -> float:
    from scipy.ndimage import uniform_filter1d

    deviation = steps - uniform_filter1d(steps, size=15, mode="nearest")
    print(f"{label}")
    print(f"  步长 均值 {steps.mean():+.4f} px/帧 · 范围 {steps.min():+.3f} … {steps.max():+.3f}")
    print(f"  完全静止帧（<0.02px） {int(np.sum(np.abs(steps) < 0.02)):>3} / {len(steps)}")
    print(f"  抖动幅度 max {np.abs(deviation).max():.4f} px · mean {np.abs(deviation).mean():.4f} px")
    print(f"  模糊度（峰值二阶矩） {spread:.3f} px · 渲染 {seconds:.2f}s")
    return float(np.abs(deviation).max())


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="zoompan-probe-"))
    source = work / "blob.png"
    blob_image(source)

    # Both filters expose `on` (output frame index) under eval=frame; neither
    # exposes `t`, so the linear ramp is written in frames.
    zoompan_x = f"({TRAVEL:g}*on/{FRAMES - 1})"
    perspective_x = f"({TRAVEL:g}*on/{FRAMES - 1})"
    print(f"实验：缩放固定 {ZOOM}，crop 窗口在 {FRAMES} 帧内匀速平移 {TRAVEL:g}px"
          f"（{TRAVEL / (FRAMES - 1):.3f} 输入像素/帧）\n")

    crop_w, crop_h = WIDTH / ZOOM, HEIGHT / ZOOM
    centre_y = (HEIGHT - crop_h) / 2

    results = {}

    plain = work / "zoompan.mp4"
    seconds = render(
        f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1[adapted];"
        f"[adapted]zoompan=z='{ZOOM}':x='{zoompan_x}':y='{centre_y:g}':d=1:"
        f"s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p[vout]",
        source, plain,
    )
    steps, spread = measure(plain)
    results["zoompan"] = report("A. 当前实现：zoompan（x/y 被截断为整数）",
                                steps, spread, seconds)

    perspective = work / "perspective.mp4"
    seconds = render(
        f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1[adapted];"
        f"[adapted]perspective="
        f"x0='{perspective_x}':y0='{centre_y:g}':"
        f"x1='{perspective_x}+{crop_w:g}':y1='{centre_y:g}':"
        f"x2='{perspective_x}':y2='{centre_y + crop_h:g}':"
        f"x3='{perspective_x}+{crop_w:g}':y3='{centre_y + crop_h:g}':"
        f"interpolation=cubic:sense=source:eval=frame,format=yuv420p[vout]",
        source, perspective,
    )
    steps, spread = measure(perspective)
    results["perspective"] = report("B. perspective（eval=frame，三次插值）",
                                    steps, spread, seconds)

    factor = 4
    big = work / "supersample.mp4"
    seconds = render(
        f"[0:v]scale={WIDTH * factor}:{HEIGHT * factor},setsar=1[adapted];"
        f"[adapted]zoompan=z='{ZOOM}':x='{zoompan_x}*{factor}':y='{centre_y * factor:g}':d=1:"
        f"s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p[vout]",
        source, big,
    )
    steps, spread = measure(big)
    results[f"supersample x{factor}"] = report(
        f"C. 先放大 {factor}x 再 zoompan（保持 s=目标尺寸）", steps, spread, seconds)

    print()
    best = min(results, key=lambda key: results[key])
    print(f"抖动最小：{best}（{results[best]:.4f} px，相对 zoompan 改善 "
          f"{results['zoompan'] / max(results[best], 1e-9):.1f}x）")
    print(f"（中间文件在 {work}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

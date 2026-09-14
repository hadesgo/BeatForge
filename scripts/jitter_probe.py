"""Measure frame-to-frame global displacement in a rendered clip.

Uses FFT phase correlation with parabolic sub-pixel refinement. The renderer adds
per-frame film grain, which is pure high-frequency energy that changes every frame
and would dominate the correlation, so the frames are low-passed first: the camera
move is a rigid transform of one composited image, so nothing is lost.

A smooth move produces near-constant steps. Jitter shows up as a step-and-hold
pattern where the displacement stays put for several frames and then jumps.

**Check the coherence number before believing anything else.** The measurement is
only valid while the frames are rigid transforms of one another; the renderer's
vignette, grain and colour grade are fixed in frame coordinates, so on a finished
clip the per-frame steps can cancel out and every other number becomes noise. When
coherence is low, measure a clip rendered with those effects off — that is what
``camera_move_probe.py --controlled`` is for.

Run: uv run python scripts/jitter_probe.py <clip.mp4>
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    width, height = out.split("x")
    return int(width), int(height)


def read_gray_frames(path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frame_bytes = width * height
    count = len(raw) // frame_bytes
    return np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8).reshape(count, height, width)


def _parabolic(values: np.ndarray, index: int) -> float:
    """Sub-sample peak offset in [-.5, .5] from three samples around ``index``."""
    left = values[index - 1]
    centre = values[index]
    right = values[index + 1]
    denominator = left - 2 * centre + right
    if denominator == 0:
        return 0.0
    return float(np.clip(.5 * (left - right) / denominator, -.5, .5))


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return the (dy, dx) translation that maps ``a`` onto ``b``."""
    height, width = a.shape
    cross = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1
    correlation = np.fft.irfft2(cross / magnitude, s=(height, width))

    flat = int(np.argmax(correlation))
    peak_y, peak_x = divmod(flat, width)
    dy = peak_y - height if peak_y > height // 2 else peak_y
    dx = peak_x - width if peak_x > width // 2 else peak_x

    row = np.take(correlation[:, peak_x], [peak_y - 1, peak_y, peak_y + 1], mode="wrap")
    col = np.take(correlation[peak_y, :], [peak_x - 1, peak_x, peak_x + 1], mode="wrap")
    return -(dy + _parabolic(row, 1)), -(dx + _parabolic(col, 1))


def measure_steps(
    clip: Path, width: int | None = None, height: int | None = None,
    fps: float = 30.0, sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (dy, dx) steps and the cumulative position for every frame."""
    from scipy.ndimage import gaussian_filter

    if width is None or height is None:
        width, height = probe_size(clip)
    frames = read_gray_frames(clip, width, height)
    smooth = gaussian_filter(frames.astype(np.float32), sigma=(0, sigma, sigma))
    steps = np.array([phase_shift(smooth[i], smooth[i + 1]) for i in range(len(smooth) - 1)])
    return steps, np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="grain-suppression low-pass sigma in pixels")
    args = parser.parse_args()

    width, height = probe_size(args.clip)
    steps, positions = measure_steps(args.clip, width, height, args.fps, args.sigma)
    print(f"{args.clip.name}: {width}x{height}, {len(steps) + 1} frames @ {args.fps:g}fps")
    if len(steps) < 2:
        print("帧数太少，无法测量")
        return 1

    axis = 1 if np.abs(steps[:, 1]).sum() >= np.abs(steps[:, 0]).sum() else 0
    label = "dx（水平）" if axis == 1 else "dy（垂直）"
    step = steps[:, axis]

    from scipy.ndimage import uniform_filter1d

    deviation = step - uniform_filter1d(step, size=15, mode="nearest")

    print(f"\n主导轴 {label} · 累计位移 {positions[-1, axis]:+.2f}px "
          f"· 平均 {step.mean():+.4f}px/帧（{step.mean() * args.fps:+.2f}px/s）")
    print()
    print(f"{'frame':>6}{'step':>10}{'position':>12}")
    for index, value in enumerate(step):
        print(f"{index:>6}{value:>10.3f}{positions[index + 1, axis]:>12.3f}")

    print()
    print("诊断")
    travelled = float(np.abs(step).sum())
    coherence = abs(float(step.sum())) / travelled if travelled else 0.0
    print(f"  步长范围          {step.min():+.3f} … {step.max():+.3f} px")
    print(f"  完全静止帧        {int(np.sum(np.abs(step) < 0.02))} / {len(step)}")
    print(f"  抖动幅度          max {np.abs(deviation).max():.4f} px "
          f"/ mean {np.abs(deviation).mean():.4f} px")
    print(f"  一致性            {coherence:.3f}（|累计| / Σ|单帧|）")

    if coherence < 0.5:
        # The steps cancel out, so they are not measuring a camera move at all.
        # Reporting the wobble anyway would invent a stutter that may not exist.
        print("  结论              测量不可信：帧间不是刚体变换，步长正负相消。")
        print("                    暗角、颗粒、调色、字幕都固定在画面坐标上，会破坏这一前提。")
        print("                    请改用 camera_move_probe.py --controlled 测受控片段。")
        return 2

    if np.abs(deviation).max() > max(4 * abs(step.mean()), .5):
        print("  结论              存在整数像素量化的顿挫：步长在「几乎不动」和「跳一整格」之间摆动")
        print("                    图片运镜应使用 perspective + eval=frame，不要用 zoompan")
    else:
        print("  结论              步长连续，未检出量化顿挫")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

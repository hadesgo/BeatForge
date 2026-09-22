"""Render one shot from an existing plan.json and measure the camera-move jitter.

Reuses the real renderer entry point, so it exercises the production filter graph
rather than a reconstruction of it.

With ``--controlled`` the shot's source is replaced by a single off-centre blob on a
flat field and the frame is made a rigid copy of it (foreground scaled to fill, no
blurred backdrop, no vignette, no grain). The blob's intensity centroid then gives
the camera position to a few hundredths of a pixel, which is what makes the
step-and-hold pattern of a quantised move visible.

Run: uv run python scripts/camera_move_probe.py --shot 2 --controlled
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import RenderConfig
from beatforge.director import ArtDirection
from beatforge.planner import Shot
from beatforge.renderer import _render_shot

BLOB_X, BLOB_Y, BLOB_SIGMA = 1150.0, 430.0, 22.0


def build(plan_path: Path) -> tuple[list[Shot], ArtDirection, RenderConfig]:
    plan = json.loads(plan_path.read_text("utf-8"))
    shots = [Shot(**item) for item in plan["shots"]]
    art = ArtDirection(**plan["art_direction"])
    config = RenderConfig(**{k: v for k, v in plan["render"].items()
                             if k in RenderConfig.model_fields})
    return shots, art, config


def blob_source(path: Path, width: int, height: int) -> None:
    """A single bright blob on a mid-grey field.

    The background must stay mid-grey: the renderer's grade filter applies
    ``eq=contrast=1.07:brightness=-.025``, which crushes a near-black field to zero
    and takes the measurement's dynamic range with it.
    """
    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]
    value = 120 + 115 * np.exp(
        -(((x - BLOB_X) ** 2 + (y - BLOB_Y) ** 2) / (2 * BLOB_SIGMA**2))
    )
    Image.fromarray(np.clip(value, 0, 255).astype(np.uint8)).convert("RGB").save(path)


def measure_centroid(clip: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (dy, dx) steps from the intensity centroid of the blob."""
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).astype(np.float64).reshape(-1, height, width)
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)

    centres = []
    for frame in frames:
        mask = np.clip(frame - np.median(frame), 0, None)
        total = mask.sum()
        centres.append((float((mask.sum(axis=1) * rows).sum() / total),
                        float((mask.sum(axis=0) * columns).sum() / total)))
    centres = np.array(centres)
    return np.diff(centres, axis=0), centres


def report(steps: np.ndarray, centres: np.ndarray, fps: float) -> None:
    from scipy.ndimage import uniform_filter1d

    axis = 1 if np.abs(steps[:, 1]).sum() >= np.abs(steps[:, 0]).sum() else 0
    label = "dx（水平）" if axis == 1 else "dy（垂直）"
    step = steps[:, axis]
    deviation = step - uniform_filter1d(step, size=15, mode="nearest")
    travel = centres[-1, axis] - centres[0, axis]

    print(f"主导轴 {label} · 总位移 {travel:+.2f}px · 平均步长 {step.mean():+.4f}px/帧"
          f"（{step.mean() * fps:+.2f}px/s）")
    print(f"  步长范围 {step.min():+.3f} … {step.max():+.3f}px")
    print(f"  完全静止帧（<0.02px） {int(np.sum(np.abs(step) < 0.02))} / {len(step)}")
    print(f"  抖动幅度 max {np.abs(deviation).max():.4f}px · mean {np.abs(deviation).mean():.4f}px")
    print(f"  抖动 / 单帧步长 比值 {np.abs(deviation).max() / max(abs(step.mean()), 1e-9):.1f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("demo/.beatforge/plan.json"))
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--controlled", action="store_true",
                        help="use a blob source and a rigid, effect-free frame")
    args = parser.parse_args()

    shots, art, config = build(args.plan)
    section_count = max(shot.section_index for shot in shots) + 1
    shot = shots[args.shot]

    work = Path(tempfile.mkdtemp(prefix="camera-move-"))
    if args.controlled:
        source = work / "blob.png"
        blob_source(source, config.width, config.height)
        shot.file = str(source)
        shot.source_width, shot.source_height = config.width, config.height
        shot.layers = []
        shot.source_color = [128, 128, 128]
        config.shot_match_strength = 0.0
        config.look_strength = 0.0
        config.visual_effects = False
        art.vignette = False
        art.grain = 0.0

    print(f"镜头 {shot.index}: effect={shot.image_effect} motion={shot.motion} "
          f"duration={shot.duration:.3f}s intent={shot.edit_intent} "
          f"melody={shot.melody:.4f} media={shot.media_id} "
          f"source={Path(shot.file).name}")

    output = work / f"{shot.index:05}.mp4"
    _render_shot(shot, output, config, art, shot.duration, section_count)

    steps, centres = measure_centroid(output, config.width, config.height)
    print(f"\n输出 {output.name}: {len(steps) + 1} 帧")
    report(steps, centres, config.fps)
    print(f"\n（中间文件在 {work}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

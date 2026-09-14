"""Compare the rendered look of the camera-move shots before and after the change.

Renders each image shot through the current filter graph and reports the frame at
the start, middle and end, so a wrong crop rectangle (a punch-in instead of a slow
drift) is obvious rather than hidden behind an average.

Run: uv run python scripts/camera_look_check.py --out .probe/new
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import RenderConfig  # noqa: E402
from beatforge.director import ArtDirection  # noqa: E402
from beatforge.planner import Shot  # noqa: E402
from beatforge.renderer import _render_shot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("demo/.beatforge/plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", default="1,2,3")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text("utf-8"))
    shots = [Shot(**item) for item in plan["shots"]]
    art = ArtDirection(**plan["art_direction"])
    config = RenderConfig(**{k: v for k, v in plan["render"].items()
                             if k in RenderConfig.model_fields})
    section_count = max(shot.section_index for shot in shots) + 1
    args.out.mkdir(parents=True, exist_ok=True)

    for index in (int(value) for value in args.shots.split(",")):
        shot = shots[index]
        output = args.out / f"{index:05}.mp4"
        _render_shot(shot, output, config, art, shot.duration, section_count)
        frames = _sample_frames(output, config.width, config.height)
        print(f"镜头 {index} {shot.image_effect:14} {shot.motion:8} "
              f"duration {shot.duration:.2f}s")
        for name, frame in frames.items():
            print(f"    {name:6} min {frame.min():3} max {frame.max():3} "
                  f"mean {frame.mean():7.2f}  中心 3x3 亮度 {_centre(frame):6.1f}")
    return 0


def _sample_frames(clip: Path, width: int, height: int) -> dict[str, np.ndarray]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width)
    return {"first": frames[0], "mid": frames[len(frames) // 2], "last": frames[-1]}


def _centre(frame: np.ndarray) -> float:
    height, width = frame.shape
    return float(frame[height // 2 - 1: height // 2 + 2, width // 2 - 1: width // 2 + 2].mean())


if __name__ == "__main__":
    raise SystemExit(main())

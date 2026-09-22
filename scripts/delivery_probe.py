"""Delivery metadata, and the grain-conditioned size drop (R-06/R-12).

Two things a "cheap" delivery gets wrong that a player can read from the container alone:
the colour range the frame is tagged with, and how much of the bitrate is spent on grain
the source already carried. This probe renders a short clip, reads the delivery metadata
back with ``ffprobe``, and compares the encoded size of a clean source (grain added)
against a noisy one (grain withheld) at the *same* CRF - so the size drop R-12 attributes
to the grain gate cannot be confused with a quality knob being turned down.

Run: uv run python scripts/delivery_probe.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.planner import Shot
from beatforge.renderer import _grain, _video_encode_args, render
from beatforge.runtime import command

SHOT_SECONDS = 2.0


def _still(path: Path, width: int, height: int) -> None:
    """A softly lit frame with a little texture, so grain has something to sit on."""
    vertical = np.linspace(.15, .8, height, dtype=np.float32)
    column = np.stack([120 + 80 * vertical, 130 + 70 * vertical, 150 + 60 * vertical], axis=1)
    base = np.repeat(column[:, None, :], width, axis=1)
    rng = np.random.default_rng(3)
    base += rng.normal(0, 3.0, base.shape)
    Image.fromarray(base.clip(0, 255).astype(np.uint8)).save(path, quality=95)


def _config(width: int, height: int, fps: int, grain: float) -> RenderConfig:
    return RenderConfig(
        width=width, height=height, fps=fps, crf=19, preset="ultrafast",
        film_grain=grain, vignette=False, professional_transitions=False,
    )


def _probe(path: Path) -> dict[str, object]:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(raw)
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    return {
        "range": video.get("color_range"),
        "pix_fmt": video.get("pix_fmt"),
        "bytes": int(data["format"]["size"]),
        "kbps": round(int(data["format"]["bit_rate"]) / 1000, 1),
    }


def _render(tmp: Path, name: str, cfg: RenderConfig, art, noise: float) -> Path:
    shot = Shot(
        0, 0.0, SHOT_SECONDS, SHOT_SECONDS, 0, str(tmp / "still.jpg"), "image", 0.0,
        "", .6, "steady", "cut", .8, melody=.5, noise_score=noise,
    )
    output = tmp / f"{name}.mp4"
    command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "anullsrc=r=22050:cl=mono", "-t", f"{SHOT_SECONDS:.3f}", str(tmp / "music.wav")])
    render([shot], [], tmp / "music.wav", output, tmp / f"cache-{name}", cfg, art)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".probe/delivery"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--grain", type=float, default=6.0)
    args = parser.parse_args()

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise SystemExit(f"PATH 中缺少 {binary}")
    args.out = args.out.resolve()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    cfg = _config(args.width, args.height, args.fps, args.grain)
    _still(args.out / "still.jpg", args.width, args.height)
    analysis = AudioAnalysis(
        duration=SHOT_SECONDS, bpm=120, beats=[0.0, 1.0, 2.0], sections=[0.0, SHOT_SECONDS],
        energy_times=[0], energy_values=[.5], average_energy=.5, brightness=.5,
        mood="cinematic", mood_scores={"cinematic": 1},
    )
    art = create_art_direction(analysis, [], cfg)

    print("交付编码参数（cfr 段）")
    print("  ", " ".join(_video_encode_args(cfg, intermediate=False)))
    print(f"  crf={cfg.crf}  film_grain={cfg.film_grain}")
    print()

    clean = _probe(_render(args.out, "clean-source", cfg, art, noise=0.0))
    noisy = _probe(_render(args.out, "noisy-source", cfg, art, noise=0.9))
    clean_shot = Shot(0, 0, 2, 2, 0, "x.jpg", "image", 0, "", .6, "steady", "cut", .8, noise_score=0.0)
    noisy_shot = Shot(0, 0, 2, 2, 0, "x.jpg", "image", 0, "", .6, "steady", "cut", .8, noise_score=0.9)

    header = f"{'源':<14}{'range':>8}{'pix_fmt':>10}{'字节':>10}{'kbps':>9}{'grain':>10}"
    print(header)
    print("-" * len(header))
    for name, meta, shot in (("clean", clean, clean_shot), ("noisy", noisy, noisy_shot)):
        grain = _grain(shot, art)
        print(f"{name:<14}{meta['range']!s:>8}{meta['pix_fmt']!s:>10}"
              f"{meta['bytes']:>10}{meta['kbps']:>9}{('有' if grain else '无'):>10}")

    drop = 1 - noisy["bytes"] / max(clean["bytes"], 1)
    print()
    print(f"同 CRF={cfg.crf} 下，含噪源体积比干净源小 {drop:.0%}（颗粒被条件化，未再为噪点编码）")
    print(f"输出目录：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

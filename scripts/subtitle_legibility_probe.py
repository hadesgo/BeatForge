"""Measure how far the lyric glyphs stand out from the picture behind them (R-05).

The subtitle pass is supposed to keep a line readable wherever it lands: the outline
thickens on a bright backdrop and, where the backdrop is very bright, the picture behind
the letters is dimmed. Whether that actually works is a pixel question, not a string one,
so this probe renders the *same* shot twice - once with the lyric and once without - over a
controlled bright and a controlled dark backdrop, and reports the ``|luma|`` difference
over the glyph mask (the pixels the subtitle changed).

``P10`` is the reading that matters: it is the tenth-percentile of the change, so a single
faint edge does not hide the fact that most of the glyph stands out. ``min`` is the worst
pixel. A well-legible line keeps both well clear of zero.

Run: uv run python scripts/subtitle_legibility_probe.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.fonts import stage_fonts
from beatforge.lyrics import LyricLine, LyricToken
from beatforge.planner import Shot
from beatforge.renderer import _subtitle_filter, render
from beatforge.runtime import command

#: A glyph pixel is one the lyric script lights up; below this luma the pixel is the
#: black the mask was drawn on.
_GLYPH_FLOOR = 60

#: Backdrops chosen so the two ends of the adaptive ramp are both exercised: a lightly
#: textured bright field (the hard case) and a plainly dark one (the easy case).
BACKDROPS: dict[str, tuple[int, int, int]] = {
    "bright": (232, 232, 232),
    "dark": (34, 34, 40),
}

SHOT_SECONDS = 2.0


def build(tmp: Path, backdrop: tuple[int, int, int], width: int, height: int, fps: int) -> RenderConfig:
    """A single-photo project with the free lyric layout and nothing high-frequency."""
    pixels = np.zeros((height, width, 3), dtype=np.float32)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = backdrop
    # A hair of texture so the outline has something to bite on, but no grain the dim
    # would be confounded by.
    rng = np.random.default_rng(7)
    pixels += rng.normal(0, 2.0, pixels.shape)
    Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(tmp / "backdrop.jpg", quality=95)

    command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
             "-t", f"{SHOT_SECONDS:.3f}", str(tmp / "music.wav")])

    return RenderConfig(
        width=width, height=height, fps=fps, crf=28, preset="ultrafast",
        subtitle_size=round(height * .07), subtitle_margin=round(height * .09),
        subtitle_layout="free", subtitle_fill="solid", subtitle_outline=1.1,
        film_grain=0, vignette=False, subtitle_font="preset:cinematic",
    )


def shots(tmp: Path, cfg: RenderConfig) -> list[Shot]:
    return [Shot(
        0, 0.0, SHOT_SECONDS, SHOT_SECONDS, 0, str(tmp / "backdrop.jpg"), "image", 0.0,
        "", .6, "steady", "cut", .8, melody=.5, focus_point=(.5, .5),
        image_effect="cinematic_depth",
    )]


def lyrics() -> list[LyricLine]:
    return [LyricLine(
        0.0, SHOT_SECONDS, "现场听着还好", tokens=[
            LyricToken("现场", 0.0, .7), LyricToken("听着", 1.0, 1.5), LyricToken("还好", 1.5, 1.9),
        ],
    )]


def _frame(video: Path, tmp: Path, at: float) -> np.ndarray:
    command(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(video),
             "-frames:v", "1", str(tmp / "frame.png")])
    with Image.open(tmp / "frame.png") as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def _glyph_mask(subtitle: Path, tmp: Path, cfg: RenderConfig, at: float) -> np.ndarray:
    """The pixels the lyric lights up, read by drawing the script alone on black.

    Drawing the same script over black at the same instant gives the glyph shape directly,
    so the legibility of the type can be read as its contrast against the picture rather
    than against a threshold chosen by eye.
    """
    out = tmp / "mask.png"
    fonts_dir = stage_fonts(cfg.subtitle_fonts_dir)
    command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}:d={at + .5:.3f}",
             "-ss", f"{at:.3f}", "-vf", _subtitle_filter(subtitle, cfg, fonts_dir),
             "-frames:v", "1", str(out)])
    with Image.open(out) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) > _GLYPH_FLOOR


def _masked_delta(
    with_frame: np.ndarray, without_frame: np.ndarray, mask: np.ndarray,
) -> tuple[float, float, float, int]:
    if not mask.any():
        return 0.0, 0.0, 0.0, 0
    values = np.abs(with_frame - without_frame)[mask]
    return (
        float(np.percentile(values, 10)), float(values.min()),
        float(values.mean()), int(mask.sum()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".probe/subtitle-legibility"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("PATH 中缺少 ffmpeg")
    # The clips are concatenated through a text file of paths, and ffmpeg's concat demuxer
    # resolves a relative entry against the concat file's own directory - so the working
    # directory has to be absolute or the single-clip join doubles the path.
    args.out = args.out.resolve()
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    lyrics_list = lyrics()
    analysis = AudioAnalysis(
        duration=SHOT_SECONDS, bpm=120, beats=[0.0, 1.0, 2.0], sections=[0.0, SHOT_SECONDS],
        energy_times=[0], energy_values=[.5], average_energy=.5, brightness=.5,
        mood="melancholic", mood_scores={"melancholic": 1},
    )

    print(f"字形可读性探针 · {args.width}x{args.height} @ {args.fps}fps · 取 t={SHOT_SECONDS * .6:.2f}s")
    print(f"{'底片':<8}{'P10':>8}{'min':>8}{'mean':>8}{'mask px':>10}")
    for name, colour in BACKDROPS.items():
        tmp = args.out / name
        tmp.mkdir(parents=True, exist_ok=True)
        cfg = build(tmp, colour, args.width, args.height, args.fps)
        shot_list = shots(tmp, cfg)
        art = create_art_direction(analysis, lyrics_list, cfg)

        with_subtitles = tmp / "with.mp4"
        render(shot_list, lyrics_list, tmp / "music.wav", with_subtitles, tmp / "cache", cfg, art)
        without_subtitles = tmp / "without.mp4"
        render(shot_list, [], tmp / "music.wav", without_subtitles, tmp / "cache-no-sub", cfg, art)

        at = SHOT_SECONDS * .6
        mask = _glyph_mask(tmp / "cache" / "lyrics.ass", tmp, cfg, at)
        p10, minimum, mean, count = _masked_delta(
            _frame(with_subtitles, tmp, at), _frame(without_subtitles, tmp, at), mask,
        )
        print(f"{name:<8}{p10:>8.1f}{minimum:>8.1f}{mean:>8.1f}{count:>10d}")

    print(f"\n输出目录：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

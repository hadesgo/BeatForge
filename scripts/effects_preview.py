"""Render every still-image effect into one contact sheet.

Tests can prove a camera move stays inside the frame and that an iris opens, but they
cannot tell you whether a move *reads* as the shot it was meant to be. This renders
each effect on a real photo and lays the start / middle / end frame side by side, so a
change to the vocabulary can be looked at in one image instead of inferred.

Run: uv run python scripts/effects_preview.py
     uv run python scripts/effects_preview.py --media demo/media/xxx.jpg --out .probe/effects
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import RenderConfig  # noqa: E402
from beatforge.director import ArtDirection  # noqa: E402
from beatforge.planner import Shot, ShotLayer  # noqa: E402
from beatforge.renderer import _CAMERA_MOVES, _render_shot  # noqa: E402
from beatforge.runtime import command, duration  # noqa: E402

FRAMING = ["film_bars", "iris", "parallax"]
COMPOSITES = ["split_screen", "photo_stack", "double_exposure", "beat_montage"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# A neutral art direction: the preview is about geometry and compositing, so nothing
# here should be tinting the result and hiding a wrong crop.
ART = ArtDirection(
    concept="", narrative_arc="", visual_style="", color_arc=[], motifs=[],
    mood="cinematic", font="sans", highlight_color="&H0000D7FF",
    base_subtitle_effect="", line_effects=[], grade_filter="",
    camera_intensity=1.0, transition_tone="neutral", grain=0.0, vignette=False,
)


def _find_media(directory: Path) -> list[Path]:
    """Media files, most textured first.

    Single-image effects are rendered from ``media[0]``. A flat frame — a plain sky, a
    smooth gradient — hides the crop window completely. Global contrast is the wrong
    metric for that (a bright sky over dark sea scores high while carrying no detail),
    so order by edge energy instead: the busiest source goes first and the calmest ends
    up as a composite layer, where its only job is to be a second picture.
    """
    if directory.is_file():
        return [directory]
    files = [
        path for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"{directory} 里没有图片素材")
    return sorted(files, key=lambda path: -_edge_energy(path))


def _edge_energy(path: Path) -> float:
    with Image.open(path) as image:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return float(np.asarray(edges, dtype=float).mean())


def _render(effect: str, media: list[Path], cfg: RenderConfig, seconds: float, out: Path) -> Path:
    layers = [
        ShotLayer(index, str(media[index % len(media)]))
        for index in range(1, min(len(media), 3))
    ] if effect in COMPOSITES else []
    shot = Shot(
        0, 0, seconds, seconds, 0, str(media[0]), "image", 0, "", .7,
        "steady", "none", .8, melody=.5, image_effect=effect, layers=layers,
    )
    target = out / f"{effect}.mp4"
    _render_shot(shot, target, cfg, ART, seconds, 1)
    return target


def _grab(clip: Path, frame: int, target: Path) -> Image.Image:
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(clip),
        "-vf", rf"select=eq(n\,{frame})", "-vsync", "0", "-frames:v", "1", str(target),
    ])
    return Image.open(target).convert("RGB")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 has no size argument
        return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=Path, default=Path("demo/media"))
    parser.add_argument("--out", type=Path, default=Path(".probe/effects"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=1.4)
    parser.add_argument("--keep-clips", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("PATH 中缺少 ffmpeg")

    media = _find_media(args.media)
    if args.out.exists():
        shutil.rmtree(args.out)
    clips = args.out / "clips"
    frames_dir = args.out / "frames"
    clips.mkdir(parents=True)
    frames_dir.mkdir(parents=True)

    cfg = RenderConfig(
        width=args.width, height=args.height, fps=args.fps, crf=30, preset="ultrafast",
        film_grain=0, vignette=False, subtitle_size=20,
    )

    groups = [
        ("camera moves", sorted(_CAMERA_MOVES)),
        ("framing", FRAMING),
        ("multi-image", COMPOSITES),
    ]
    columns = 3
    tile_width, tile_height = cfg.width, cfg.height
    label_width, header_height = 168, 34
    rows = sum(len(effects) for _, effects in groups)
    sheet = Image.new("RGB", (label_width + columns * tile_width, header_height + rows * tile_height), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    label_font, header_font = _font(17), _font(21)

    y = header_height
    for title, effects in groups:
        draw.text((10, y - header_height + 6), title, fill=(230, 200, 120), font=header_font)
        for effect in effects:
            clip = _render(effect, media, cfg, args.seconds, clips)
            frames = max(1, round(duration(clip) * cfg.fps))
            picks = sorted({0, frames // 2, frames - 1})
            for column, frame in enumerate(picks):
                image = _grab(clip, frame, frames_dir / f"{effect}-{frame}.png")
                sheet.paste(image, (label_width + column * tile_width, y))
            draw.text((10, y + tile_height // 2 - 10), effect, fill=(225, 225, 230), font=label_font)
            draw.text(
                (10, y + tile_height // 2 + 12),
                f"start / mid / end · {frames}f",
                fill=(130, 130, 140), font=label_font,
            )
            y += tile_height
            print(f"\r已渲染 {effect:<18}", end="", flush=True)
    print()

    sheet_path = args.out / "effects-preview.png"
    sheet.save(sheet_path)
    print(f"接触表 {sheet_path}  ({sheet.width}x{sheet.height})")
    if not args.keep_clips:
        shutil.rmtree(clips)
        shutil.rmtree(frames_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

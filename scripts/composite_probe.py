"""Render the multi-image layouts and lay the frames out to be looked at.

Four source shapes on purpose - a 16:9, a 3:4, a 1:1 and a 9:16 - because a panel only
gets cropped into a shape it does not have, and that is the path a layout has to survive.
Run this after touching `_composite_graph`.

Run: uv run python scripts/composite_probe.py
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.lyrics import LyricLine

from beatforge.planner import IMAGE_COMPOSITES, Shot, ShotLayer
from beatforge.renderer import _render_shot
from beatforge.runtime import command

OUT = Path(".probe/composite")
W, H, FPS, SECONDS = 1280, 720, 12, 3.0

EFFECTS = tuple(IMAGE_COMPOSITES)
SHAPES = ((1920, 1080), (900, 1200), (1080, 1080), (720, 1280))


def picture(path: Path, size: tuple[int, int], hue: tuple[int, int, int], label: str) -> Path:
    """Sources in four shapes, so every panel has to crop into something it is not."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        (hue[0] * (xx / w) + 40), (hue[1] * (yy / h) + 30), (hue[2] * (1 - xx / w) + 50),
    ], axis=2)
    image = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle([w * .08, h * .08, w * .92, h * .92], outline=(255, 255, 255), width=max(2, w // 120))
    draw.text((w * .12, h * .45), label, fill=(255, 255, 255))
    image.save(path, quality=95)
    return path


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    hues = ((200, 120, 90), (90, 150, 210), (150, 200, 110), (210, 110, 170))
    sources = [
        picture(OUT / f"source-{index}.jpg", size, hue, f"SOURCE {index + 1}")
        for index, (size, hue) in enumerate(zip(SHAPES, hues))
    ]

    sample_rate = 22050
    import soundfile as sf
    t = np.arange(int(sample_rate * SECONDS)) / sample_rate
    sf.write(OUT / "music.wav", (.06 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)

    cfg = RenderConfig(width=W, height=H, fps=FPS, crf=28, preset="ultrafast",
                       film_grain=0, vignette=False, image_background_blur=26.0)
    analysis = AudioAnalysis(
        duration=SECONDS, bpm=120, beats=[0.0, 1.0, 2.0, 3.0], sections=[0.0, SECONDS],
        energy_times=[0], energy_values=[.5], average_energy=.5, brightness=.5,
        mood="cinematic", mood_scores={"cinematic": 1},
    )
    art = create_art_direction(analysis, [LyricLine(0, SECONDS, "测试")], cfg)

    tiles = []
    for effect in EFFECTS:
        count = IMAGE_COMPOSITES[effect][0]
        layers = [
            ShotLayer(index, str(sources[index % len(sources)]), "image", "secondary",
                      source_width=SHAPES[index % len(SHAPES)][0],
                      source_height=SHAPES[index % len(SHAPES)][1])
            for index in range(1, count)
        ]
        shot = Shot(0, 0.0, SECONDS, SECONDS, 0, str(sources[0]), "image", 0, "", .6, "steady",
                    "cut", .8, image_effect=effect, source_width=SHAPES[0][0],
                    source_height=SHAPES[0][1], layers=layers)
        video = OUT / f"{effect}.mp4"
        _render_shot(shot, video, cfg, art, SECONDS, 1)
        frame = OUT / f"{effect}.png"
        command(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                 "-vf", rf"select=eq(n\,{int(FPS * 1.6)})", "-frames:v", "1", str(frame)])
        with Image.open(frame) as image:
            tiles.append(image.convert("RGB").resize((640, 360), Image.LANCZOS))

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    board = Image.new("RGB", (640 * columns, 360 * rows), (12, 12, 14))
    for index, tile in enumerate(tiles):
        board.paste(tile, ((index % columns) * 640, (index // columns) * 360))
    board.save(OUT / "composites.png")
    print(f"顺序：{', '.join(EFFECTS)}")
    print(f"→ {OUT / 'composites.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

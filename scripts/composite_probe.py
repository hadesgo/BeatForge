"""Render the multi-image composites and lay the frames out to be looked at.

Two source shapes on purpose - a 16:9 and a 3:4 - because the letterbox path only shows
up when an image does not match the canvas, and that is the path the composites must not
take. Run this after touching `_image_filter_graph`.

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
from beatforge.media import MediaAsset
from beatforge.planner import Shot, ShotLayer
from beatforge.renderer import _render_shot
from beatforge.runtime import command

OUT = Path(".probe/composite")
W, H, FPS, SECONDS = 1280, 720, 12, 3.0

EFFECTS = ("split_screen", "photo_stack", "double_exposure", "beat_montage")


def picture(path: Path, size: tuple[int, int], hue: tuple[int, int, int], label: str) -> Path:
    """A landscape and a portrait source, so the letterbox path is exercised."""
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
    # 16:9 and 3:4 - the two shapes that force the letterbox treatment.
    wide = picture(OUT / "wide.jpg", (1920, 1080), (200, 120, 90), "WIDE 16:9")
    tall = picture(OUT / "tall.jpg", (900, 1200), (90, 150, 210), "TALL 3:4")

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
        shot = Shot(0, 0.0, SECONDS, SECONDS, 0, str(wide), "image", 0, "", .6, "steady",
                    "cut", .8, image_effect=effect, source_width=1920, source_height=1080,
                    layers=[ShotLayer(1, str(tall), "image", "secondary",
                                      source_width=900, source_height=1200)])
        video = OUT / f"{effect}.mp4"
        _render_shot(shot, video, cfg, art, SECONDS, 1)
        frame = OUT / f"{effect}.png"
        command(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                 "-vf", rf"select=eq(n\,{int(FPS * 1.6)})", "-frames:v", "1", str(frame)])
        with Image.open(frame) as image:
            tiles.append(image.convert("RGB").resize((640, 360), Image.LANCZOS))

    board = Image.new("RGB", (640 * 2, 360 * 2), (12, 12, 14))
    for index, tile in enumerate(tiles):
        board.paste(tile, ((index % 2) * 640, (index // 2) * 360))
    board.save(OUT / "composites.png")
    print(f"顺序：{', '.join(EFFECTS)}")
    print(f"→ {OUT / 'composites.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

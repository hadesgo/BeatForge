"""Render the lyric layouts to a contact sheet, so a change to them can be looked at.

The free layout depends on data the LRC path does not carry - where the singer pauses
inside a line - so it cannot be exercised from a demo project. This builds lyrics with
word timings directly and renders them through the real ``render()``, then lays the
frames out so the placement of every fragment can be checked at a glance.

Run: uv run python scripts/subtitle_layout_probe.py
     uv run python scripts/subtitle_layout_probe.py --layout band
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.lyrics import LyricLine, LyricToken
from beatforge.planner import Shot
from beatforge.renderer import render
from beatforge.runtime import command

# Ten lines, because the free layout rotates through ten patterns and a shorter sheet
# would show only part of the vocabulary. The split paths are covered too: breaths, a
# comma, and one line with neither, which stays whole.
LINES: list[tuple[str, list[tuple[str, float, float]]]] = [
    ("你反正不会再担心", [("你反正", 0.0, 1.1), ("不会再", 1.6, 2.5), ("担心", 2.5, 3.4)]),
    ("我隐隐作疼的心脏", [("我隐隐", 0.0, 1.2), ("作疼的", 1.7, 2.6), ("心脏", 2.6, 3.6)]),
    ("那天的天气，难得放晴", [("那天的天气", 0.0, 1.9), ("难得放晴", 2.3, 4.0)]),
    ("城市亮起灯光", [("城市亮起灯光", 0.0, 2.6)]),
    ("不爱我的非要上", [("不爱我的", 0.0, 1.4), ("非要上", 1.9, 3.1)]),
    ("那么硬的南墙非要撞", [("那么硬的", 0.0, 1.5), ("南墙非要撞", 2.0, 3.5)]),
    ("你说的 放晴", [("你说的", 0.0, 1.2), ("放晴", 1.8, 2.9)]),
    ("劫后余生好难呼吸", [("劫后余生", 0.0, 1.6), ("好难呼吸", 2.1, 3.4)]),
    ("一场疯狂", [("一场疯狂", 0.0, 2.4)]),
    ("你曾给我", [("你曾给我", 0.0, 1.3), ("给我", 1.9, 3.0)]),
]
LINE_SECONDS = 4.0


def build(tmp: Path, layout: str, fill: str, width: int, height: int, fps: int) -> RenderConfig:
    """A minimal project: one photo, four lyric lines, no AI anywhere."""
    # A dark, low-contrast frame is what the reference style is built for: soft sky,
    # one bright area the text has to avoid, and nothing high-frequency to fight the
    # type. Noise would make the sheet impossible to judge.
    vertical = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    base = np.stack([
        26 + 70 * vertical[:, :, 0],
        34 + 96 * vertical[:, :, 0],
        52 + 120 * vertical[:, :, 0],
    ], axis=2).repeat(width, axis=1)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    glow = np.exp(-(((xx / width - .5) / .22) ** 2 + ((yy / height - .45) / .2) ** 2))
    base += (glow * 96)[:, :, None]
    rng = np.random.default_rng(5)
    base += rng.normal(0, 2.5, base.shape)
    photo = tmp / "backdrop.jpg"
    Image.fromarray(base.clip(0, 255).astype(np.uint8)).save(photo, quality=94)

    music = tmp / "music.wav"
    sample_rate = 22_050
    total = LINE_SECONDS * len(LINES)
    time = np.arange(int(sample_rate * total)) / sample_rate
    import soundfile as sf
    sf.write(music, (.08 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)

    return RenderConfig(
        width=width, height=height, fps=fps, crf=30, preset="ultrafast",
        subtitle_size=round(height * .062), subtitle_margin=round(height * .1),
        subtitle_layout=layout, subtitle_fill=fill, subtitle_outline=1.1,
        film_grain=0, vignette=False, subtitle_font="preset:cinematic",
        image_background_blur=18,
    )


def shots(tmp: Path, cfg: RenderConfig) -> list[Shot]:
    """One shot per line, with the subject deliberately moved around.

    The focus point is what the free layout steers the text away from, so the four
    shots cover a centred subject (the patterns apply), a high one (text drops) and a
    low one (text rises).
    """
    # A mix of centred subjects (where the patterns apply unmodified), high and low
    # ones (where the layout slides), and off-centre ones (where the rotation has to
    # skip a pattern to find one that clears).
    focuses = [
        (.50, .50), (.50, .50), (.50, .28), (.50, .74), (.50, .50),
        (.30, .40), (.70, .62), (.50, .50), (.50, .86), (.50, .14),
    ]
    return [
        Shot(
            index, index * LINE_SECONDS, (index + 1) * LINE_SECONDS, LINE_SECONDS,
            index, str(tmp / "backdrop.jpg"), "image", 0, "", .6, "steady", "cut", .8,
            melody=.5, focus_point=focuses[index],
            image_effect=("cinematic_depth", "drift", "dolly_in", "arc")[index % 4],
        )
        for index in range(len(LINES))
    ]


def lyrics() -> list[LyricLine]:
    lines = []
    for index, (text, tokens) in enumerate(LINES):
        start = index * LINE_SECONDS
        lines.append(LyricLine(
            start, start + LINE_SECONDS, text,
            tokens=[LyricToken(t, start + a, start + b) for t, a, b in tokens],
        ))
    return lines


def sheet(video: Path, tmp: Path, columns: int = 4) -> Image.Image:
    """One row per lyric line, sampled across its own span."""
    frames_dir = tmp / "frames"
    frames_dir.mkdir(exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()
    command(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", "fps=4",
             str(frames_dir / "%04d.png")])
    frames = sorted(frames_dir.glob("*.png"))
    picks = [round((index + fraction) * LINE_SECONDS * 4)
             for index in range(len(LINES)) for fraction in (.30, .60, .90)]
    tiles = []
    for index in picks:
        if index >= len(frames):
            continue
        with Image.open(frames[index]) as image:
            tiles.append(image.convert("RGB").resize((480, 270), Image.LANCZOS))
    rows = (len(tiles) + columns - 1) // columns
    board = Image.new("RGB", (columns * 480, rows * 270), (14, 14, 16))
    for position, tile in enumerate(tiles):
        board.paste(tile, ((position % columns) * 480, (position // columns) * 270))
    return board


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="free", choices=("free", "band"))
    parser.add_argument("--fill", default="solid", choices=("solid", "knockout"))
    parser.add_argument("--out", type=Path, default=Path(".probe/subtitle-layout"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("PATH 中缺少 ffmpeg")
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    cfg = build(args.out, args.layout, args.fill, args.width, args.height, args.fps)
    shots_list = shots(args.out, cfg)
    lyrics_list = lyrics()
    analysis = AudioAnalysis(
        duration=LINE_SECONDS * len(LINES), bpm=120,
        beats=[float(x) for x in range(0, int(LINE_SECONDS * len(LINES)) + 1)],
        sections=[0.0, LINE_SECONDS * len(LINES)],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.35, mood="melancholic", mood_scores={"melancholic": 1},
    )
    art = create_art_direction(analysis, lyrics_list, cfg)
    art.font = "Source Han Serif SC" if args.layout == "free" else art.font

    video = args.out / "layout.mp4"
    render(shots_list, lyrics_list, args.out / "music.wav", video, args.out / "cache", cfg, art)

    board = sheet(video, args.out)
    target = args.out / f"subtitle-{args.layout}-{args.fill}.png"
    board.save(target)
    print(f"{args.layout} 版式 / {args.fill} 填充 · 每个镜头取 3 帧 · "
          f"{target} ({board.width}x{board.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

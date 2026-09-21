"""Render one line per font preset and report which face actually reaches the screen.

libass fails at fonts **silently**. An unreachable directory, a family name that does not
exist, and a variable font whose weight axis is never applied all produce a perfectly
valid frame - just in the wrong typeface. Nothing downstream can tell the difference
between "the configured font" and "some fallback", which is why this measures the ink of
the rendered line instead of trusting the config.

It measures two things, because they fail independently:

* **per preset** - which family and weight each preset resolves to, and whether that
  family is one libass can actually find;
* **per weight** - the ink of one family at ``\\b300``, ``\\b400`` and ``\\b700``. A family
  whose ink does not move when the weight does is a family whose weight never arrives.

Run: uv run python scripts/font_probe.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import RenderConfig  # noqa: E402
from beatforge.fonts import FONT_PRESETS, resolve_subtitle_font, stage_fonts  # noqa: E402
from beatforge.lyrics import LyricLine, write_ass  # noqa: E402
from beatforge.renderer import _subtitle_filter  # noqa: E402
from beatforge.runtime import command  # noqa: E402

OUT = Path(".probe/font")
LINE = "黎明照亮天空"
WIDTH, HEIGHT, FPS = 1280, 720, 10


def background() -> Path:
    path = OUT / "background.jpg"
    Image.new("RGB", (WIDTH, HEIGHT), (15, 15, 20)).save(path)
    return path


def ink(background_path: Path, script: Path, fonts_dir: Path | None) -> int:
    """Non-background pixels of the line, sampled in the middle of its fade-in."""
    clip, frame = OUT / "probe.mp4", OUT / "probe.png"
    frame.unlink(missing_ok=True)
    command([
        "ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS),
        "-i", str(background_path), "-vf", _subtitle_filter(script, _cfg(), fonts_dir),
        "-t", "1", "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", str(clip),
    ])
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(clip),
        "-vf", rf"select=eq(n\,{int(FPS * .5)})", "-fps_mode", "passthrough",
        "-frames:v", "1", str(frame),
    ])
    grey = np.asarray(Image.open(frame).convert("L"), dtype=float)
    return int((grey > 25).sum())


def _cfg() -> RenderConfig:
    # Points at a directory that does not exist, which is what the shipped templates do
    # in every project folder - the shipped library still has to be found.
    return RenderConfig(
        width=WIDTH, height=HEIGHT, fps=FPS, crf=28, preset="ultrafast",
        subtitle_fonts_dir=OUT / "project-fonts-do-not-exist",
        subtitle_effect="cinematic", subtitle_outline=0.0,
    )


def write(script: Path, family: str, weight: int | None) -> None:
    write_ass(
        [LyricLine(0, 1, LINE)], script, width=WIDTH, height=HEIGHT, font=family,
        size=64, weight=weight, margin=48, effect="cinematic", placements=[], outline=0.0,
    )


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    background_path = background()
    cfg = _cfg()
    fonts_dir = stage_fonts(cfg.subtitle_fonts_dir)
    if fonts_dir is None:
        raise SystemExit("没有可用的字体目录")

    staged = sorted(path.name for path in fonts_dir.iterdir() if path.suffix in (".ttf", ".otf"))
    print(f"字体目录：{fonts_dir}")
    print(f"  共 {len(staged)} 个字体文件；可变字体已展开为静态字重")
    print(f"  例：{', '.join(name for name in staged if 'Noto' in name)}")
    print()

    print("每个预设实际用到的face（墨量 = 渲染出的字形像素数）")
    script = OUT / "preset.ass"
    for preset in FONT_PRESETS:
        choice = resolve_subtitle_font(f"preset:{preset}", cfg.subtitle_fonts_dir)
        write(script, choice.family, choice.weight)
        count = ink(background_path, script, fonts_dir)
        weight = f"{choice.weight}" if choice.weight else "默认(400)"
        print(f"  {preset:10s} {choice.family:22s} {weight:10s} ink={count:6d}")

    print()
    print("同一个家族在不同字重下的墨量——不随字重变化就说明字重没送达画面")
    for family in ("Noto Sans SC", "Noto Serif SC"):
        row = []
        for weight in (300, 400, 700):
            write(script, family, weight)
            row.append(ink(background_path, script, fonts_dir))
        spread = max(row) - min(row)
        mark = "  <-- 字重完全无效" if spread == 0 else ""
        print(f"  {family:16s} 300={row[0]:6d} 400={row[1]:6d} 700={row[2]:6d} 差={spread}{mark}")

    print()
    print("同一行在有/没有 fontsdir 时的差别（自带字体是否真的用上了）")
    for family in ("Ma Shan Zheng", "LXGW WenKai"):
        write(script, family, None)
        with_dir = ink(background_path, script, fonts_dir)
        without = ink(background_path, script, None)
        same = "  <-- 完全一样，说明字体没起作用" if with_dir == without else ""
        print(f"  {family:16s} 有 fontsdir={with_dir:6d} 无={without:6d}{same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

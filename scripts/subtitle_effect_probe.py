"""Render every subtitle effect and check that it actually animates.

libass ignores override tags it does not understand, without failing, so a typo or an
unsupported tag produces a clip that renders perfectly and does nothing. The only way
to catch that is to look at consecutive frames and ask whether anything changed - and
separately, whether the line was drawn at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.lyrics import SUBTITLE_EFFECTS, LyricLine, LyricToken, write_ass
from beatforge.runtime import command, duration

WIDTH, HEIGHT, FPS, SECONDS = 480, 270, 12, 2.4
TEXT = "夜色渐浓"
LABEL_WIDTH, HEADER = 190, 30


def render(effect: str, tmp: Path) -> Path:
    line = LyricLine(
        0, SECONDS, TEXT,
        tokens=[
            LyricToken(TEXT[0], 0.0, 0.4), LyricToken(TEXT[1], 0.4, 0.9),
            LyricToken(TEXT[2], 1.2, 1.7), LyricToken(TEXT[3], 1.7, 2.3),
        ],
    )
    ass = tmp / f"{effect}.ass"
    write_ass(
        [line], ass, width=WIDTH, height=HEIGHT, font="Microsoft YaHei",
        size=38, margin=30, effect=effect, line_effects=[effect],
    )
    clip = tmp / f"{effect}.mp4"
    escaped = ass.resolve().as_posix().replace(":", r"\:")
    command([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=black:s={WIDTH}x{HEIGHT}:r={FPS}:d={SECONDS}",
        "-vf", f"ass='{escaped}'", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "28", "-pix_fmt", "yuv420p", str(clip),
    ])
    return clip


def frames(clip: Path, tmp: Path) -> np.ndarray:
    """Every frame as a small greyscale array."""
    out = tmp / "frames"
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(clip),
        "-vf", "scale=240:135", str(out / "%04d.png"),
    ])
    return np.stack([
        np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        for path in sorted(out.glob("*.png"))
    ])


def contact_sheet(effect: str, stack: np.ndarray, sheet: "Image.Image", row: int) -> None:
    """Paste the entrance, middle and end frame of one effect into the sheet."""
    from PIL import ImageDraw, ImageFont

    picks = [1, len(stack) // 2, len(stack) - 2]
    for column, index in enumerate(picks):
        tile = Image.fromarray(stack[index].astype(np.uint8)).convert("RGB")
        tile = tile.resize((tile.width * 2, tile.height * 2), Image.NEAREST)
        sheet.paste(tile, (LABEL_WIDTH + column * tile.width, HEADER + row * tile.height))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((10, HEADER + row * 270 + 120), effect, fill=(225, 225, 230), font=font)


def main() -> int:
    import argparse

    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", type=Path, default=None,
                        help="also write a contact sheet of every effect")
    args = parser.parse_args()

    tmp = Path(".probe/subtitle")
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"{'effect':<14} {'frames':>7} {'ink':>6} {'motion':>8} {'peak step':>10}")
    dead, blank = [], []
    sheet = None
    if args.sheet:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (LABEL_WIDTH + 3 * 480, HEADER + len(SUBTITLE_EFFECTS) * 270),
                          (18, 18, 20))
    for row, effect in enumerate(SUBTITLE_EFFECTS):
        clip = render(effect, tmp)
        stack = frames(clip, tmp)
        ink = float((stack > 40).mean())
        steps = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
        motion = float(steps.max())
        # A static effect still shows ink but no step; a broken one shows neither.
        if ink < 0.002:
            blank.append(effect)
        if motion < 0.4:
            dead.append(effect)
        flag = ""
        if effect in blank:
            flag = "  <-- nothing drawn"
        elif effect in dead:
            flag = "  <-- never animates"
        print(f"{effect:<14} {len(stack):>7} {ink:>6.3f} {motion:>8.2f} "
              f"{float(steps.max()):>10.2f}{flag}")
        if sheet is not None:
            contact_sheet(effect, stack, sheet, row)
    if sheet is not None:
        sheet.save(args.sheet)
        print(f"\ncontact sheet {args.sheet} ({sheet.width}x{sheet.height})")
    print()
    print("drew nothing:", blank or "none")
    print("never animated:", dead or "none")
    return 1 if (blank or dead) else 0


if __name__ == "__main__":
    raise SystemExit(main())

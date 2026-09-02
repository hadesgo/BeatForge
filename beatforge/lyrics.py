from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

STAMP = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")


@dataclass(slots=True)
class LyricLine:
    start: float
    end: float
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def parse_lrc(text: str, total_duration: float) -> list[LyricLine]:
    timed: list[tuple[float, str]] = []
    for raw in text.lstrip("\ufeff").splitlines():
        stamps = list(STAMP.finditer(raw))
        lyric = STAMP.sub("", raw).strip()
        if not stamps or not lyric or re.match(r"^\w+\s*:", lyric):
            continue
        for stamp in stamps:
            fraction = (stamp.group(3) or "0").ljust(3, "0")[:3]
            start = int(stamp.group(1)) * 60 + int(stamp.group(2)) + int(fraction) / 1000
            if start < total_duration:
                timed.append((start, lyric))
    timed.sort()
    return [
        LyricLine(start, min(total_duration, timed[i + 1][0] if i + 1 < len(timed) else start + 5), lyric)
        for i, (start, lyric) in enumerate(timed)
        if start < total_duration
    ]


def read_lrc(file: Path, total_duration: float) -> list[LyricLine]:
    return parse_lrc(file.read_text("utf-8"), total_duration)


def srt_timestamp(value: float) -> str:
    millis = max(0, round(value * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(lines: list[LyricLine], file: Path) -> None:
    blocks = [
        f"{i}\n{srt_timestamp(line.start)} --> {srt_timestamp(line.end)}\n{line.text}\n"
        for i, line in enumerate(lines, 1)
    ]
    file.write_text("\n".join(blocks), "utf-8")


def ass_timestamp(value: float) -> str:
    centis = max(0, round(value * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    seconds, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centis:02}"


def write_ass(
    lines: list[LyricLine],
    file: Path,
    *,
    width: int,
    height: int,
    font: str,
    size: int,
    margin: int,
    effect: Literal["karaoke", "cinematic", "bounce"] = "karaoke",
    highlight_color: str = "&H0000D7FF",
) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,{font},{size},{highlight_color},&H00FFFFFF,&H90000000,&H50000000,-1,0,0,0,100,100,1,0,1,2.2,0,2,48,48,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for line in lines:
        text = _ass_text(line.text)
        if effect == "karaoke":
            text = _karaoke_text(text, line.end - line.start)
            prefix = r"{\fad(160,220)\blur0.4\fscx92\fscy92\t(0,200,\fscx100\fscy100\blur0)}"
        elif effect == "bounce":
            prefix = r"{\fad(90,180)\fscx76\fscy76\t(0,130,\fscx108\fscy108)\t(130,240,\fscx100\fscy100)}"
        else:
            prefix = r"{\fad(360,460)\blur1.2\t(0,360,\blur0)}"
        events.append(
            f"Dialogue: 0,{ass_timestamp(line.start)},{ass_timestamp(line.end)},Lyric,,0,0,0,,{prefix}{text}"
        )
    file.write_text(header + "\n".join(events) + "\n", "utf-8-sig")


def _ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _karaoke_text(text: str, duration: float) -> str:
    characters = list(text)
    if not characters:
        return text
    total = max(len(characters), round(duration * 100))
    base, remainder = divmod(total, len(characters))
    return "".join(f"{{\\kf{base + (1 if i < remainder else 0)}}}{character}" for i, character in enumerate(characters))

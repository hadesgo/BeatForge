from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

STAMP = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")


@dataclass(slots=True)
class LyricToken:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class LyricLine:
    start: float
    end: float
    text: str
    tokens: list[LyricToken] = field(default_factory=list)

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
    effect: Literal["karaoke", "cinematic", "bounce", "float", "glow", "typewriter"] = "karaoke",
    highlight_color: str = "&H0000D7FF",
    line_effects: list[str] | None = None,
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
    for index, line in enumerate(lines):
        text = _ass_text(line.text)
        line_effect = line_effects[index] if line_effects and index < len(line_effects) else effect
        if line_effect == "karaoke":
            text = _karaoke_line(line)
            prefix = r"{\fad(160,220)\blur0.4\fscx92\fscy92\t(0,200,\fscx100\fscy100\blur0)}"
        elif line_effect == "bounce":
            prefix = r"{\fad(90,180)\fscx76\fscy76\t(0,130,\fscx108\fscy108)\t(130,240,\fscx100\fscy100)}"
        elif line_effect == "float":
            prefix = rf"{{\fad(260,360)\1c&HFFFFFF&\move({width // 2},{height - margin + 14},{width // 2},{height - margin},0,500)\blur0.5}}"
        elif line_effect == "glow":
            prefix = r"{\fad(220,300)\blur3\bord3\t(0,320,\blur0.5\bord2.2)}"
        elif line_effect == "typewriter":
            text = _typewriter_text(text, line.end - line.start)
            prefix = r"{\fad(80,240)\1c&HFFFFFF&}"
        else:
            prefix = r"{\fad(360,460)\1c&HFFFFFF&\blur1.2\t(0,360,\blur0)}"
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


def _karaoke_line(line: LyricLine) -> str:
    if not line.tokens:
        return _karaoke_text(_ass_text(line.text), line.end - line.start)
    output = []
    for index, token in enumerate(line.tokens):
        next_start = line.tokens[index + 1].start if index + 1 < len(line.tokens) else line.end
        duration = max(.01, next_start - token.start)
        output.append(f"{{\\kf{round(duration * 100)}}}{_ass_text(token.text)}")
    return "".join(output)


def _typewriter_text(text: str, duration: float) -> str:
    characters = list(text)
    step = max(30, round(duration * 1000 / max(len(characters), 1)))
    return "".join(
        f"{{\\alpha&HFF&\\t({index * step},{index * step + 60},\\alpha&H00&)}}{character}"
        for index, character in enumerate(characters)
    )

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

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

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAMP = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")

#: The weight a family is asked for when the resolution named none. 400 is the face a
#: static family ships as "Regular", and ASS's Bold field is only a switch above it.
_REGULAR_WEIGHT = 400


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


def merge_short_lines(lines: list[LyricLine], *, minimum: float = .6) -> list[LyricLine]:
    """Fold lines that are on screen for less than ``minimum`` into a neighbour.

    A caption that lives for a quarter of a second cannot be read; it just flickers. The
    words are not worth dropping, so a short line is appended to the line before it - or,
    for a short line at the very start where there is nothing before it, to the line
    after - and the merged line keeps the union of the two spans and both token lists.
    Returns a new list; the input is left untouched.
    """
    merged: list[LyricLine] = []
    for line in lines:
        if merged and line.end - line.start < minimum:
            previous = merged[-1]
            merged[-1] = LyricLine(
                previous.start, max(previous.end, line.end),
                f"{previous.text}{line.text}", previous.tokens + line.tokens,
            )
        else:
            merged.append(line)
    if len(merged) >= 2 and merged[0].end - merged[0].start < minimum:
        first, second = merged[0], merged[1]
        merged[0:2] = [LyricLine(
            first.start, max(first.end, second.end),
            f"{first.text}{second.text}", first.tokens + second.tokens,
        )]
    return merged


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


# Every lyric animation BeatForge can build out of ASS override tags. They fall into
# the three groups an editor actually thinks in: an entrance, an exit, and something
# that runs the whole time the line is on screen.
SUBTITLE_EFFECTS: tuple[str, ...] = (
    # entrances
    "cinematic", "bounce", "typewriter", "punch", "slide", "flip_in",
    # sustained
    "karaoke", "float", "glow", "neon", "neon_flicker", "shake", "wave",
    "rainbow", "spotlight",
    # the odd one out: fringed and stuttering, for a broken-signal moment
    "glitch",
)

# Colour is written ``&HAABBGGRR&`` in ASS, which is why these do not read like RGB.
_WHITE = "&H00FFFFFF&"
_MAGENTA = "&H00FF00FF&"
_CYAN = "&H00FFFF00&"
_NEON_BLUE = "&H00FFD700&"
_NEON_CYAN = "&H00FFFFD7&"
_DIM = "&H00303030&"


def write_ass(
    lines: list[LyricLine],
    file: Path,
    *,
    width: int,
    height: int,
    font: str,
    size: int,
    margin: int,
    effect: str = "karaoke",
    highlight_color: str = "&H0000D7FF",
    line_effects: list[str] | None = None,
    line_outlines: list[float] | None = None,
    weight: int | None = None,
    outline: float = 2.2,
    shadow: float = 0.0,
    mask_only: bool = False,
) -> None:
    """Write the lyric script.

    ``weight`` is the weight of the resolved family, and it drives both the style's Bold
    field and a ``\\b`` tag on every event. Neither alone is enough: the style is a
    single on/off bit, so a family that can be anything from 100 to 900 needs the number
    on the event - but the events only override the style, never the other way round, so
    a style left on Bold would make every preset that asks for no weight bold.

    Every line is one centred line in the bottom band. ``mask_only`` drops the fill
    colour to white and the outline to nothing, which is what the knockout compositing
    needs to read the text as a clean alpha channel rather than as a picture of some
    letters.

    ``line_outlines`` gives each line its own outline width, so a line over a bright
    backdrop can be thickened and a line over a dark one kept thin (R-05). A line with no
    entry keeps the single ``outline`` value. The width is emitted as a base ``\\bord``
    before the effect's own tags, so an effect that sets its border on purpose (``glow``,
    ``neon``) still owns it.
    """
    primary = "&H00FFFFFF" if mask_only else highlight_color
    outline_colour = "&H00000000" if mask_only else "&H90000000"
    border = 0.0 if mask_only else outline
    drop = 0.0 if mask_only else shadow
    # A preset that names no weight wants the family's regular face, not a synthetic
    # bold: libass thickens a family with no bold of its own, which on top of a light
    # face reads as neither.
    bold = -1 if (weight or _REGULAR_WEIGHT) >= 600 else 0
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,{font},{size},{primary},&H00FFFFFF,{outline_colour},&H50000000,{bold},0,0,0,100,100,1,0,1,{border},{drop},2,48,48,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    weight_tag = f"{{\\b{weight}}}" if weight else ""
    events = []
    for index, line in enumerate(lines):
        fit = _fit_font_size(line.text, width, size)
        line_effect = line_effects[index] if line_effects and index < len(line_effects) else effect
        if line_effect not in SUBTITLE_EFFECTS:
            # An unknown name - a hand-edited plan, or a director field this build
            # does not know. Fall back to the plain fade rather than to karaoke, so a
            # line without word timings still renders sensibly.
            line_effect = "cinematic"
        line_outline = (
            line_outlines[index]
            if line_outlines and index < len(line_outlines) else None
        )
        prefix, text = _subtitle_effect(
            line_effect, line, width=width, height=height, margin=margin,
            mask_only=mask_only, outline=line_outline,
        )
        events.append(
            f"Dialogue: 0,{ass_timestamp(line.start)},{ass_timestamp(line.end)},Lyric,,0,0,0,,{fit}{weight_tag}{prefix}{text}"
        )
    file.write_text(header + "\n".join(events) + "\n", "utf-8-sig")


def _subtitle_effect(
    effect: str, line: LyricLine, *, width: int, height: int, margin: int,
    mask_only: bool = False,
    outline: float | None = None,
) -> tuple[str, str]:
    """Return the override prefix and the (possibly rewritten) text for one line."""
    duration = max(.01, line.end - line.start)
    base_y = height - margin
    centre_x = width // 2
    text = _ass_text(line.text)
    tags = ""

    if effect == "karaoke":
        tags = r"\fad(160,220)\blur0.4\fscx92\fscy92\t(0,200,\fscx100\fscy100\blur0)"
        text = _karaoke_line(line)
    elif effect == "bounce":
        tags = r"\fad(90,180)\fscx76\fscy76\t(0,130,\fscx108\fscy108)\t(130,240,\fscx100\fscy100)"
    elif effect == "float":
        tags = (r"\fad(260,360)"
                + rf"\move({centre_x},{base_y + 14},{centre_x},{base_y},0,500)"
                + r"\blur0.5")
    elif effect == "glow":
        tags = r"\fad(220,300)\blur3\bord3\t(0,320,\blur0.5\bord2.2)"
    elif effect == "typewriter":
        tags = r"\fad(80,240)"
        text = _typewriter_text(line.text, duration)
    elif effect == "punch":
        # Arrives far too large and slams into place. The blur is what sells the
        # speed - without it the frame just looks like a bad scale.
        tags = r"\fad(50,130)\fscx185\fscy185\blur5\t(0,110,\fscx100\fscy100\blur0)"
    elif effect == "slide":
        travel = round(width * .18)
        tags = (r"\fad(120,200)"
                + rf"\move({centre_x - travel},{base_y},{centre_x},{base_y},0,260)")
    elif effect == "flip_in":
        # Each character flips up into place, so the line assembles itself.
        tags = r"\fad(0,200)"
        text = _per_character(line.text, lambda index, count, total: (
            rf"{{\fry90\fscx55\fscy55\alpha&HFF&"
            rf"\t({index * 45},{index * 45 + 230},\fry0\fscx100\fscy100\alpha&H00&)}}"
        ))
    elif effect == "neon":
        # A bright core (the configured highlight colour) inside a coloured halo that
        # breathes. The core colour is left to the style so the project's own colour shows
        # through; only the halo, which defines the effect, is set here.
        tags = (rf"\fad(160,260)\3c{_NEON_CYAN}\bord3\blur5"
                r"\t(0,650,\blur10\bord4)\t(650,1300,\blur5\bord3)")
    elif effect == "neon_flicker":
        # A neon sign that has not warmed up: mostly lit, with two stutters.
        tags = (rf"\fad(60,180)\3c{_NEON_CYAN}\bord3\blur6"
                r"\t(0,70,\alpha&H30&)\t(70,120,\alpha&H00&)"
                r"\t(120,170,\alpha&H55&)\t(170,230,\alpha&H00&)"
                r"\t(700,780,\alpha&H35&)\t(780,850,\alpha&H00&)")
    elif effect == "shake":
        # ``\jitter`` is a libass extension, so the rotation chain is the guarantee:
        # if a build ignores the jitter, the line still moves.
        tags = (r"\fad(120,220)\bord2.6\jitter(5,4,55,2)"
                r"\t(0,90,\frz-2)\t(90,180,\frz2)\t(180,270,\frz-2)\t(270,360,\frz0)")
    elif effect == "wave":
        tags = r"\fad(140,220)"
        text = _per_character(line.text, lambda index, count, total: (
            rf"{{\fry0\fscx100"
            rf"\t({index * 55},{index * 55 + 140},\fry-72\fscx78)"
            rf"\t({index * 55 + 140},{index * 55 + 280},\fry0\fscx100)}}"
        ))
    elif effect == "rainbow":
        # Six hues over two seconds, then back to the first so the loop is seamless. The
        # opening hue is set through an instant ``\t`` rather than a static ``\1c`` so the
        # line never carries a static colour other than the configured highlight (R-05);
        # every colour the audience sees is produced by the animation.
        steps = 6
        span = max(1, round(duration * 1000 / steps))
        cycle = ("&H000000FF&", "&H0000FFFF&", "&H0000FF00&",
                 "&H00FFFF00&", "&H00FF0000&", "&H00FF00FF&")
        tags = r"\fad(200,300)" + rf"\t(0,1,\1c{cycle[0]})"
        for step in range(steps):
            tags += f"\\t({step * span},{(step + 1) * span},\\1c{cycle[(step + 1) % steps]})"
    elif effect == "spotlight":
        # Starts dark and dim, as if the light has not found it yet. The dark start is
        # itself an animation step (an instant ``\t`` at t=0) so the event keeps a single
        # static base colour - the configured highlight - and the effect is defined purely
        # by its colour animation (R-05).
        tags = (r"\fad(0,240)\blur3\fscx96\fscy96"
                rf"\t(0,1,\1c{_DIM})"
                rf"\t(1,520,\1c{_WHITE}\blur0\fscx100\fscy100)")
    elif effect == "glitch":
        # Chromatic fringing plus an alpha stutter: the two things a dropped signal
        # does to a caption.
        tags = (rf"\fad(60,180)\3c{_MAGENTA}\4c{_CYAN}\bord3\shad4\blur0.6"
                rf"\t(0,90,\shad8\blur1.6)\t(90,180,\shad4\blur0.6)"
                rf"\t(400,470,\alpha&H45&)\t(470,540,\alpha&H00&)"
                rf"\t(900,960,\alpha&H35&)\t(960,1030,\alpha&H00&)")
    else:
        tags = r"\fad(360,460)\blur1.2\t(0,360,\blur0)"

    # The adaptive outline is a base width, so it goes in before the effect's own tags:
    # an effect that sets its border on purpose (glow, neon) still wins, everything else
    # gets the width the backdrop asked for.
    if outline is not None:
        tags = f"\\bord{outline:g}" + tags
    if mask_only:
        # The mask has to trace the same shape the audience sees, so it keeps every
        # geometric and alpha tag above. Only the fill and the outline change: a mask
        # that is not solid white would read as a half-transparent letter.
        tags = _strip_fill(tags)
    return f"{{{tags}}}", text


def _strip_fill(tags: str) -> str:
    """Drop the colour and outline tags, keeping the geometry and the timing.

    Used for the knockout mask: the text's shape and its animation are what matter,
    not which colour it happens to be drawn in.
    """
    return re.sub(r"\\(?:1c|2c|3c|4c|bord|shad|blur)[^\\}]*", "", tags)


def _per_character(text: str, tag_for) -> str:
    """Wrap every character in its own override block, for effects that stagger.

    ``tag_for`` receives the character's index, the line's length and the line's
    duration in milliseconds, so a wave can space itself across however many
    characters it has to cross.
    """
    characters = list(text)
    total = max(1, round(len(characters) * 55))
    return "".join(
        f"{tag_for(index, len(characters), total)}{_ass_text(character)}"
        for index, character in enumerate(characters)
    )


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
    cursor = line.start
    for token in line.tokens:
        gap = max(0.0, token.start - cursor)
        if gap >= .01:
            output.append(f"{{\\k{round(gap * 100)}}}")
        duration = max(.01, token.end - token.start)
        output.append(f"{{\\kf{round(duration * 100)}}}{_ass_text(token.text)}")
        cursor = token.end
    return "".join(output)


def _typewriter_text(text: str, duration: float) -> str:
    characters = list(text)
    step = max(30, round(duration * 1000 / max(len(characters), 1)))
    return "".join(
        f"{{\\alpha&HFF&\\t({index * step},{index * step + 60},\\alpha&H00&)}}{_ass_text(character)}"
        for index, character in enumerate(characters)
    )


def _visual_units(text: str) -> float:
    """Rough width of a string in font-size units: CJK is full width, Latin is not."""
    return sum(1.0 if ord(char) > 255 else .58 for char in text if char not in "\r\n")


def _fit_font_size(text: str, width: int, size: int) -> str:
    """Shrink unusually long lines while keeping normal lyrics at the chosen size."""
    available = width * .84
    estimated = _visual_units(text) * size
    if estimated <= available or estimated <= 0:
        return ""
    fitted = max(round(size * .68), min(size, round(size * available / estimated)))
    return rf"{{\fs{fitted}}}"

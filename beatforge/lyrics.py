from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class Placement:
    """One drawn fragment of a lyric line, in script coordinates.

    ``align`` is ASS numpad alignment, so 4 anchors the fragment's left edge at ``x``
    with its vertical centre at ``y``, and 5 centres it on both. ``tokens`` carries the
    karaoke timings this fragment owns, and ``start`` is when the fragment is sung -
    the free layout fades each one in on its own time, which is what makes the lyric
    assemble itself across the frame instead of appearing all at once.
    """

    text: str
    x: int
    y: int
    align: int = 4
    tokens: tuple[LyricToken, ...] = ()
    start: float = 0.0


# Free-layout anchors, as fractions of the frame, for a subject sitting in the middle.
# Lifted from the reference lyric video, which never uses a subtitle band: it splits
# the line and puts the halves where the subject is not, so the type frames the
# picture instead of sitting underneath it.
_FREE_PATTERNS: dict[str, tuple[tuple[float, float, int], tuple[float, float, int]]] = {
    # Anchors are (x, y, alignment) as fractions of the frame, and the alignment is ASS
    # numpad: 4 puts the fragment's left edge at x, 5 centres it, 6 puts its right edge
    # there. Mixing justification is most of what makes a set of layouts read as ten
    # compositions rather than as one composition in ten places.
    #
    # The second anchor of each pair is kept clear of the frame's centre by enough that
    # a centred subject does not sit under it - a layout that cannot manage that is still
    # usable, it just gets skipped when the subject is in the middle.
    #
    # One half above, one below, subject between them.
    "sandwich":     ((.50, .19, 5), (.50, .75, 5)),
    # A descending diagonal, upper left to lower right.
    "diagonal":     ((.09, .31, 4), (.62, .54, 4)),
    # The same idea climbing instead of falling.
    "diagonal_up":  ((.09, .58, 4), (.62, .33, 4)),
    # One baseline with a deliberate gap, the way a phrase breaks in speech.
    "gap":          ((.10, .63, 4), (.62, .63, 4)),
    # Both fragments left, stacked tight against the left edge.
    "stack_left":   ((.09, .38, 4), (.09, .56, 4)),
    # The same against the right edge, justified the other way.
    "stack_right":  ((.91, .31, 6), (.91, .50, 6)),
    # Both centred and close together - the quietest of the ten, and the one that needs
    # the subject to be off to one side.
    "centre_stack": ((.50, .28, 5), (.50, .45, 5)),
    # Opposite corners, as far apart as the frame allows.
    "corners":      ((.09, .23, 4), (.91, .79, 6)),
    # Pinned to the two side edges at different heights, the subject between them.
    "edges":        ((.08, .44, 4), (.92, .70, 6)),
    # Low and centred, the second half hanging under the first.
    "offset":       ((.50, .65, 5), (.50, .80, 5)),
}

# A step through the table rather than a written-out cycle. The step is coprime with the
# table size, so the rotation visits all ten before repeating, and because the table is
# ordered roughly by how much room each pattern leaves in the middle, consecutive lines
# land on layouts that look nothing alike.
_FREE_STEP = 3

# How far the layout slides away from a subject that is not in the middle. Shifting the
# chosen pattern keeps its shape; swapping in a fixed off-centre layout instead collapsed
# every line of an off-centre shot onto the same two positions, which is what made the
# free layout feel like one layout.
_FREE_SHIFT = .16
_SUBJECT_HIGH = .42
_SUBJECT_LOW = .58

# How much room a fragment has to leave around the subject, as fractions of the frame.
# Both axes matter: the reference style puts type beside the subject as often as above
# or below it, and a vertical-only rule rejects exactly those layouts. The thresholds
# describe a person-sized subject rather than its bounding box, which is all
# ``focus_point`` gives us.
_CLEAR_X = .11
_CLEAR_Y = .10
#: A hair more than the threshold, so rounding to whole pixels cannot land a fragment
#: back inside the box the threshold was meant to clear.
_CLEAR_MARGIN = .005
#: A hair more than the threshold, so rounding to whole pixels cannot land a fragment
#: back inside the box the threshold was meant to clear.
_CLEAR_MARGIN = .005
#: How many layouts the rotation will pass over looking for one that clears the subject.
_FREE_TRIES = 4

# A small deterministic drift so the same pattern never lands pixel-identically twice in
# one song. Big enough to read as placed by hand, small enough that it never looks like a
# mistake, and derived from the line index so a given plan always lays out the same way.
_DRIFT_X = .010
_DRIFT_Y = .008

# How long a silence between two sung words has to last before the free layout treats
# it as a phrase break. Below this the singer is just articulating; above it they
# breathed, and that is where the reference style puts the break.
_MIN_BREAK_SECONDS = 0.22

# How long each fragment takes to appear once it is sung.
_FRAGMENT_FADE_MS = 260


def plan_placements(
    lines: list[LyricLine],
    focus_points: list[tuple[float, float] | None],
    *,
    width: int,
    height: int,
    margin: int,
    size: int,
) -> list[list[Placement]]:
    """Lay each line out freely, in the parts of the frame its subject is not using.

    Ten patterns rotate through the song. Each candidate is slid as a whole to clear its
    subject, and the rotation passes over any that still will not fit, so the variety
    survives an off-centre subject instead of collapsing onto one safe layout.

    ``focus_points`` is the subject centre per line, so the text can be steered away from
    it. Everything is a fixed rotation off the line index - no random numbers - so a
    given plan always lays out the same way.
    """
    names = list(_FREE_PATTERNS)
    inset = margin / max(height, 1)
    placements: list[list[Placement]] = []
    for index, line in enumerate(lines):
        focus = focus_points[index] if index < len(focus_points) else None
        anchors = _choose_layout(names, index, (focus[0], focus[1]) if focus else (.5, .5), inset)
        fragments = _split_line(line)
        if len(fragments) == 1:
            # A line with no detectable pause keeps one fragment, at its pattern's first
            # anchor so it still lands somewhere different each time.
            (text, tokens, start), (fx, fy, _align) = fragments[0], anchors[0]
            placements.append([Placement(
                text, round(width * fx), round(height * fy), 5, tokens, start,
            )])
            continue
        row: list[Placement] = []
        for (text, tokens, start), (fx, fy, align) in zip(fragments, anchors):
            row.append(Placement(
                text, round(width * fx), round(height * fy), align, tokens, start,
            ))
        placements.append(row)
    return placements


def _choose_layout(
    names: list[str], index: int, focus: tuple[float, float], inset: float,
) -> list[tuple[float, float, int]]:
    """The first layout in this line's rotation that does not land on the subject.

    Looking a few candidates ahead is what keeps the variety. The alternative - always
    falling back to one safe layout when the subject is off-centre - is exactly what made
    the free layout feel like a single layout repeated.
    """
    fallback: list[tuple[float, float, int]] | None = None
    for step in range(_FREE_TRIES):
        name = names[(index * _FREE_STEP + step) % len(names)]
        # The drift is part of the candidate, so clearance is checked against the
        # positions that actually get drawn rather than the pre-drift ones.
        placed = _fit(_drift(_FREE_PATTERNS[name], index), focus[1], inset)
        if fallback is None:
            fallback = placed
        if _clears(placed, focus):
            return placed
    assert fallback is not None
    return fallback


def _fit(
    anchors: tuple[tuple[float, float, int], ...], focus_y: float, inset: float,
) -> list[tuple[float, float, int]]:
    """Slide a layout vertically until it clears the subject, then keep it in frame.

    The group moves as a whole and slides back if that took it out of frame, so the shape
    of the pattern survives the move. Clamping each anchor on its own is only the last
    resort, for a layout taller than the band it has to fit in.
    """
    ys = [y for _, y, _ in anchors]
    gap = _CLEAR_Y + _CLEAR_MARGIN
    if focus_y < _SUBJECT_HIGH:
        shift = (focus_y + gap) - min(ys)
    elif focus_y > _SUBJECT_LOW:
        shift = (focus_y - gap) - max(ys)
    else:
        shift = 0.0
    low, high = inset, 1 - inset
    top, bottom = min(ys) + shift, max(ys) + shift
    if top < low:
        shift += low - top
    if bottom > high:
        shift -= bottom - high
    return [(x, min(max(y + shift, low), high), align) for x, y, align in anchors]


def _clears(anchors: list[tuple[float, float, int]], focus: tuple[float, float]) -> bool:
    """Does any fragment land on the subject?

    A box test, not a distance: a fragment beside the subject is as clear as one above
    it. The anchor stands in for the fragment's body, which understates how far a
    left-justified fragment reaches and overstates it for a right-justified one - close
    enough for a threshold, and the reference style works the same way.
    """
    return all(
        abs(x - focus[0]) >= _CLEAR_X or abs(y - focus[1]) >= _CLEAR_Y
        for x, y, _ in anchors
    )


def _drift(
    anchors: tuple[tuple[float, float, int], ...], index: int,
) -> tuple[tuple[float, float, int], ...]:
    """Nudge a layout by a hair, deterministically, so it never lands twice the same.

    Without this the ten patterns are still ten *fixed* compositions, and a song long
    enough to come back round to one of them repeats it exactly. The offsets are derived
    from the line index rather than drawn at random, so a plan stays reproducible.
    """
    dx = ((index * 7) % 5 - 2) * _DRIFT_X
    dy = ((index * 11) % 5 - 2) * _DRIFT_Y
    return tuple((x + dx, y + dy, align) for x, y, align in anchors)


def _split_line(line: LyricLine) -> list[tuple[str, tuple[LyricToken, ...], float]]:
    """Break a line into the fragments a free layout draws separately.

    Returns ``(text, tokens, start)`` per fragment. The break goes at the singer's
    longest pause, because that is where a break reads as phrasing instead of as a
    mistake - splitting a line down the middle cuts words in half ("黎明照 / 亮天空"
    breaks 照亮). Without word timings it falls back to punctuation, and without
    either the line stays whole and simply gets placed freely.
    """
    text = line.text.strip()
    tokens = list(line.tokens)
    if len(tokens) >= 2:
        gap, cut = max(
            (tokens[index + 1].start - tokens[index].end, index + 1)
            for index in range(len(tokens) - 1)
        )
        if gap >= _MIN_BREAK_SECONDS:
            head = "".join(token.text for token in tokens[:cut]).strip()
            tail = "".join(token.text for token in tokens[cut:]).strip()
            if head and tail:
                return [
                    (head, tuple(tokens[:cut]), tokens[0].start),
                    (tail, tuple(tokens[cut:]), tokens[cut].start),
                ]
    for mark in "，。！？、；：,.!?;: ":
        position = text.find(mark)
        if 0 < position < len(text) - 1:
            head, tail = text[:position].strip(), text[position + 1:].strip()
            if head and tail:
                # Punctuation gives the break but not the timing, so the two halves
                # share the line's span evenly.
                return [
                    (head, (), line.start),
                    (tail, (), line.start + (line.end - line.start) / 2),
                ]
    return [(text, tuple(tokens), line.start)]


def _visual_units(text: str) -> float:
    """Rough width of a string in font-size units: CJK is full width, Latin is not."""
    return sum(1.0 if ord(char) > 255 else .58 for char in text if char not in "\r\n")


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
    weight: int | None = None,
    placements: list[list[Placement]] | None = None,
    outline: float = 2.2,
    shadow: float = 0.0,
    mask_only: bool = False,
) -> None:
    """Write the lyric script.

    ``weight`` asks for a specific weight on a variable font. The ASS style's own Bold
    field only offers on or off, so a family that can be anything from 100 to 900 needs
    the number on every event - which is the whole reason the bundled Noto families are
    shipped as variable fonts rather than as five static ones.

    ``placements`` switches the layout: without it every line is one centred line in
    the bottom band, and with it each line is drawn as the freely positioned fragments
    ``plan_placements`` chose. ``mask_only`` drops the fill colour to white and the
    outline to nothing, which is what the knockout compositing needs to read the text
    as a clean alpha channel rather than as a picture of some letters.
    """
    primary = "&H00FFFFFF" if mask_only else highlight_color
    outline_colour = "&H00000000" if mask_only else "&H90000000"
    border = 0.0 if mask_only else outline
    drop = 0.0 if mask_only else shadow
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,{font},{size},{primary},&H00FFFFFF,{outline_colour},&H50000000,-1,0,0,0,100,100,1,0,1,{border},{drop},2,48,48,{margin},1

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
        fragments = placements[index] if placements and index < len(placements) else None
        for fragment in fragments or [None]:
            prefix, text = _subtitle_effect(
                line_effect, line, width=width, height=height, margin=margin,
                placement=fragment, mask_only=mask_only,
            )
            events.append(
                f"Dialogue: 0,{ass_timestamp(line.start)},{ass_timestamp(line.end)},Lyric,,0,0,0,,{fit}{weight_tag}{prefix}{text}"
            )
    file.write_text(header + "\n".join(events) + "\n", "utf-8-sig")


def _anchor(placement: Placement | None, *, move_from: tuple[int, int] | None = None, ms: int = 0) -> str:
    """Pin a fragment where the free layout put it.

    ``\\pos`` and ``\\move`` are mutually exclusive in ASS - the later one wins - so a
    fragment that slides in gets ``\\move`` alone, with its destination at the layout
    position rather than at the frame's centre.
    """
    if placement is None:
        return ""
    if move_from is None:
        return rf"\pos({placement.x},{placement.y})\an{placement.align}"
    return (
        rf"\move({placement.x + move_from[0]},{placement.y + move_from[1]},"
        rf"{placement.x},{placement.y},0,{ms})\an{placement.align}"
    )


def _subtitle_effect(
    effect: str, line: LyricLine, *, width: int, height: int, margin: int,
    placement: Placement | None = None, mask_only: bool = False,
) -> tuple[str, str]:
    """Return the override prefix and the (possibly rewritten) text for one fragment.

    A fragment carries its own slice of the line, so every effect works on that slice
    and the anchor is prepended once at the end.
    """
    duration = max(.01, line.end - line.start)
    base_y = height - margin
    centre_x = width // 2
    fragment = placement.text if placement else line.text
    text = _ass_text(fragment)
    tags = ""

    if effect == "karaoke":
        tags = r"\fad(160,220)\blur0.4\fscx92\fscy92\t(0,200,\fscx100\fscy100\blur0)"
        text = _karaoke_fragment(line, placement)
    elif effect == "bounce":
        tags = r"\fad(90,180)\fscx76\fscy76\t(0,130,\fscx108\fscy108)\t(130,240,\fscx100\fscy100)"
    elif effect == "float":
        tags = (rf"\fad(260,360)\1c{_WHITE}"
                + (_anchor(placement, move_from=(0, 14), ms=500)
                   or rf"\move({centre_x},{base_y + 14},{centre_x},{base_y},0,500)")
                + r"\blur0.5")
    elif effect == "glow":
        tags = r"\fad(220,300)\blur3\bord3\t(0,320,\blur0.5\bord2.2)"
    elif effect == "typewriter":
        tags = rf"\fad(80,240)\1c{_WHITE}"
        text = _typewriter_text(fragment, duration)
    elif effect == "punch":
        # Arrives far too large and slams into place. The blur is what sells the
        # speed - without it the frame just looks like a bad scale.
        tags = r"\fad(50,130)\fscx185\fscy185\blur5\t(0,110,\fscx100\fscy100\blur0)"
    elif effect == "slide":
        travel = round(width * .18)
        tags = (r"\fad(120,200)"
                + (_anchor(placement, move_from=(-travel, 0), ms=260)
                   or rf"\move({centre_x - travel},{base_y},{centre_x},{base_y},0,260)"))
    elif effect == "flip_in":
        # Each character flips up into place, so the line assembles itself.
        tags = r"\fad(0,200)"
        text = _per_character(fragment, lambda index, count, total: (
            rf"{{\fry90\fscx55\fscy55\alpha&HFF&"
            rf"\t({index * 45},{index * 45 + 230},\fry0\fscx100\fscy100\alpha&H00&)}}"
        ))
    elif effect == "neon":
        # A white core inside a coloured halo that breathes.
        tags = (rf"\fad(160,260)\1c{_WHITE}\3c{_NEON_CYAN}\bord3\blur5"
                rf"\t(0,650,\blur10\bord4)\t(650,1300,\blur5\bord3)")
    elif effect == "neon_flicker":
        # A neon sign that has not warmed up: mostly lit, with two stutters.
        tags = (rf"\fad(60,180)\1c{_WHITE}\3c{_NEON_CYAN}\bord3\blur6"
                rf"\t(0,70,\alpha&H30&)\t(70,120,\alpha&H00&)"
                rf"\t(120,170,\alpha&H55&)\t(170,230,\alpha&H00&)"
                rf"\t(700,780,\alpha&H35&)\t(780,850,\alpha&H00&)")
    elif effect == "shake":
        # ``\jitter`` is a libass extension, so the rotation chain is the guarantee:
        # if a build ignores the jitter, the line still moves.
        tags = (r"\fad(120,220)\bord2.6\jitter(5,4,55,2)"
                r"\t(0,90,\frz-2)\t(90,180,\frz2)\t(180,270,\frz-2)\t(270,360,\frz0)")
    elif effect == "wave":
        tags = r"\fad(140,220)"
        text = _per_character(fragment, lambda index, count, total: (
            rf"{{\fry0\fscx100"
            rf"\t({index * 55},{index * 55 + 140},\fry-72\fscx78)"
            rf"\t({index * 55 + 140},{index * 55 + 280},\fry0\fscx100)}}"
        ))
    elif effect == "rainbow":
        # Six hues over two seconds, then back to the first so the loop is seamless.
        steps = 6
        span = max(1, round(duration * 1000 / steps))
        cycle = ("&H000000FF&", "&H0000FFFF&", "&H0000FF00&",
                 "&H00FFFF00&", "&H00FF0000&", "&H00FF00FF&")
        tags = rf"\fad(200,300)\1c{cycle[0]}"
        for step in range(steps):
            tags += f"\\t({step * span},{(step + 1) * span},\\1c{cycle[(step + 1) % steps]})"
    elif effect == "spotlight":
        # Starts dark and dim, as if the light has not found it yet.
        tags = (rf"\fad(0,240)\1c{_DIM}\blur3\fscx96\fscy96"
                rf"\t(0,520,\1c{_WHITE}\blur0\fscx100\fscy100)")
    elif effect == "glitch":
        # Chromatic fringing plus an alpha stutter: the two things a dropped signal
        # does to a caption.
        tags = (rf"\fad(60,180)\3c{_MAGENTA}\4c{_CYAN}\bord3\shad4\blur0.6"
                rf"\t(0,90,\shad8\blur1.6)\t(90,180,\shad4\blur0.6)"
                rf"\t(400,470,\alpha&H45&)\t(470,540,\alpha&H00&)"
                rf"\t(900,960,\alpha&H35&)\t(960,1030,\alpha&H00&)")
    else:
        tags = rf"\fad(360,460)\1c{_WHITE}\blur1.2\t(0,360,\blur0)"

    if mask_only:
        # The mask has to trace the same shape the audience sees, so it keeps every
        # geometric and alpha tag above. Only the fill and the outline change: a mask
        # that is not solid white would read as a half-transparent letter.
        tags = _strip_fill(tags)
    tags = _fragment_entrance(tags, line, placement)
    return f"{{{_anchor(placement)}{tags}}}", text


def _fragment_entrance(tags: str, line: LyricLine, placement: Placement | None) -> str:
    """Hold a fragment back until it is sung, then fade it in.

    This is what makes the free layout read the way the reference does: the line
    assembles itself across the frame over its own duration instead of appearing all
    at once. A fragment that starts with its line keeps whatever entrance the effect
    gave it, which is why the delay has to be measured rather than assumed.
    """
    if placement is None:
        return tags
    delay = round(max(0.0, placement.start - line.start) * 1000)
    if delay <= 40:
        return tags
    # The effect's own fade-in would fight this one for the alpha channel, so that half
    # is dropped and only the fade-out at the end of the line is kept. Appending rather
    # than prepending matters: libass applies ``\t`` chains in order, so the later one
    # wins for the property they share.
    tags = re.sub(r"\\fad\(\s*[\d.]+", r"\\fad(0", tags)
    return tags + rf"\alpha&HFF&\t({delay},{delay + _FRAGMENT_FADE_MS},\alpha&H00&)"


def _strip_fill(tags: str) -> str:
    """Drop the colour and outline tags, keeping the geometry and the timing.

    Used for the knockout mask: the text's shape and its animation are what matter,
    not which colour it happens to be drawn in.
    """
    return re.sub(r"\\(?:1c|2c|3c|4c|bord|shad|blur)[^\\}]*", "", tags)


def _karaoke_fragment(line: LyricLine, placement: Placement | None) -> str:
    """Karaoke markup for one fragment.

    Under the free layout the fragment *is* the karaoke: it fades in at the moment it
    is sung, so a character sweep on top would say the same thing twice. The plain text
    is returned and ``_fragment_entrance`` supplies the timing. The band layout keeps
    the sweep, which is what a single centred line has to work with.
    """
    if placement is None:
        return _karaoke_line(line)
    return _ass_text(placement.text)


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


def _fit_font_size(text: str, width: int, size: int) -> str:
    """Shrink unusually long lines while keeping normal lyrics at the chosen size."""
    available = width * .84
    estimated = _visual_units(text) * size
    if estimated <= available or estimated <= 0:
        return ""
    fitted = max(round(size * .68), min(size, round(size * available / estimated)))
    return rf"{{\fs{fitted}}}"

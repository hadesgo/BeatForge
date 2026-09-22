import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from beatforge.legibility import adaptive_outline, estimate_region_luma, needs_local_dim
from beatforge.lyrics import (
    _CLEAR_X,
    _CLEAR_Y,
    SUBTITLE_EFFECTS,
    LyricLine,
    LyricToken,
    _fragment_half_box,
    _overlap_fraction,
    merge_short_lines,
    parse_lrc,
    plan_placements,
    srt_timestamp,
    write_ass,
)


def test_parse_lrc() -> None:
    lines = parse_lrc("[00:01.20]日落海边\n[00:04.50][00:08.00]一起奔跑", 10)
    assert [(x.start, x.end, x.text) for x in lines] == [
        (1.2, 4.5, "日落海边"), (4.5, 8.0, "一起奔跑"), (8.0, 10, "一起奔跑")
    ]
    assert srt_timestamp(61.234) == "00:01:01,234"


def test_ass_karaoke_contains_timing_and_animation(tmp_path: Path) -> None:
    target = tmp_path / "lyrics.ass"
    write_ass(
        parse_lrc("[00:00.00]星光\n[00:02.00]远方", 4), target,
        width=1920, height=1080, font="Microsoft YaHei", size=46,
        margin=72, effect="karaoke",
    )
    content = target.read_text("utf-8-sig")
    assert "PlayResX: 1920" in content
    assert r"\kf" in content
    assert r"\fscx92" in content
    assert "Dialogue: 0,0:00:00.00,0:00:02.00" in content


def test_ass_alternative_effects(tmp_path: Path) -> None:
    lines = parse_lrc("[00:00.00]星光", 2)
    for effect, expected in (("cinematic", r"\blur1.2"), ("bounce", r"\fscx76")):
        target = tmp_path / f"{effect}.ass"
        write_ass(
            lines, target, width=1280, height=720, font="Arial", size=40,
            margin=60, effect=effect,
        )
        assert expected in target.read_text("utf-8-sig")


def test_ass_uses_forced_alignment_token_timing(tmp_path: Path) -> None:
    line = LyricLine(1, 3, "星光", [LyricToken("星", 1, 1.4), LyricToken("光", 1.4, 3)])
    target = tmp_path / "aligned.ass"
    write_ass([line], target, width=1280, height=720, font="Arial", size=40, margin=60)
    content = target.read_text("utf-8-sig")
    assert r"{\kf40}星" in content
    assert r"{\kf160}光" in content


def test_ass_karaoke_preserves_a_real_pause_between_words(tmp_path: Path) -> None:
    line = LyricLine(0, 2, "星光", [LyricToken("星", 0, .4), LyricToken("光", 1, 1.4)])
    target = tmp_path / "paused.ass"
    write_ass([line], target, width=1280, height=720, font="Arial", size=40, margin=60)
    content = target.read_text("utf-8-sig")
    assert r"{\kf40}星{\k60}{\kf40}光" in content


def test_richer_ass_effects(tmp_path: Path) -> None:
    lines = [LyricLine(0, 2, "旋律")]
    for effect, expected in (("float", r"\move"), ("glow", r"\blur3"), ("typewriter", r"\alpha&HFF&")):
        target = tmp_path / f"{effect}.ass"
        write_ass(lines, target, width=1280, height=720, font="Arial", size=40, margin=60, effect=effect)
        assert expected in target.read_text("utf-8-sig")


def test_ass_shrinks_long_lines_and_typewriter_escapes_markup(tmp_path: Path) -> None:
    target = tmp_path / "long.ass"
    write_ass(
        [LyricLine(0, 3, "这是一句非常非常长的歌词用于验证字幕不会超出画面安全区域{心声}")],
        target, width=640, height=360, font="Arial", size=40, margin=40,
        effect="typewriter",
    )
    content = target.read_text("utf-8-sig")
    assert r"{\fs27}" in content
    assert r"\{" in content and r"\}" in content


def _breathing_line() -> LyricLine:
    """One line whose longest silence sits off-centre, the way sung phrasing does."""
    return LyricLine(0, 4, "你反正不会再担心", tokens=[
        LyricToken("你反正", 0.0, 1.1),
        LyricToken("不会再", 1.6, 2.5),
        LyricToken("担心", 2.5, 3.4),
    ])


def test_the_free_layout_breaks_the_line_at_the_singers_pause() -> None:
    """Not at the midpoint. Splitting down the middle cuts words in half.

    "黎明照亮天空" halves to "黎明照 / 亮天空", which breaks 照亮 - a break has to land
    where the singer breathed, or the layout reads as a bug rather than as phrasing.
    """
    placements = plan_placements(
        [_breathing_line()], [(.5, .5)], width=1280, height=720, margin=72, size=45,
    )[0]

    assert [fragment.text for fragment in placements] == ["你反正", "不会再担心"]
    assert placements[1].start == 1.6, "the second fragment starts when it is sung"


def test_the_free_layout_keeps_lines_without_a_break_whole() -> None:
    """No pause and no punctuation means no split, however long the line is."""
    line = LyricLine(0, 4, "城市亮起灯光", tokens=[
        LyricToken("城市", 0.0, 1.0), LyricToken("亮起", 1.0, 2.0),
        LyricToken("灯光", 2.0, 3.0),
    ])
    placements = plan_placements(
        [line], [(.5, .5)], width=1280, height=720, margin=72, size=45,
    )[0]

    assert len(placements) == 1
    assert placements[0].text == "城市亮起灯光"


def test_the_free_layout_splits_on_punctuation_when_there_are_no_word_timings() -> None:
    placements = plan_placements(
        [LyricLine(0, 4, "那天的天气，难得放晴")], [(.5, .5)],
        width=1280, height=720, margin=72, size=45,
    )[0]

    assert [fragment.text for fragment in placements] == ["那天的天气", "难得放晴"]


@pytest.mark.parametrize(
    "focus", [(.5, .5), (.5, .25), (.5, .75), (.3, .4), (.7, .6), (.5, .12), (.5, .88)],
)
def test_the_free_layout_keeps_the_type_off_the_subject(focus: tuple[float, float]) -> None:
    """The whole point: the words share the frame with the subject, not sit under it.

    The check is the box the subject occupies rather than a vertical ordering. The
    reference style puts type *beside* the subject as often as above or below it, so a
    layout that only ever moved up and down would be avoiding the wrong thing - and an
    earlier version did exactly that, which collapsed every off-centre shot onto the
    same two positions.

    R-10 tightens this from the anchor point to the *area* the fragment and the subject
    cover: the axis clearance still has to hold, and on top of it no more than 5% of a
    fragment's own area may sit on the subject's box.
    """
    placements = plan_placements(
        [_breathing_line()], [focus], width=1280, height=720, margin=72, size=45,
    )[0]

    assert placements
    for fragment in placements:
        dx = abs(fragment.x / 1280 - focus[0])
        dy = abs(fragment.y / 720 - focus[1])
        assert dx >= _CLEAR_X or dy >= _CLEAR_Y, (focus, fragment)
        half = _fragment_half_box(fragment.text, 1280, 720, 45)
        overlap = _overlap_fraction(fragment.x / 1280, fragment.y / 720, half, focus)
        assert overlap <= .05 + 1e-9, (focus, fragment.text, overlap)


def test_the_free_layout_does_not_repeat_itself() -> None:
    """Ten patterns that all land in the same place are one pattern.

    This is what "too rigid" looked like: three anchors, and an off-centre subject
    collapsing even those onto two fixed positions.
    """
    lines = [
        LyricLine(i * 4, i * 4 + 4, f"第{i}句歌词要断成两半", tokens=[
            LyricToken("第i句", i * 4, i * 4 + 1.5),
            LyricToken("歌词要", i * 4 + 2.0, i * 4 + 3.0),
            LyricToken("断成两半", i * 4 + 3.0, i * 4 + 4.0),
        ])
        for i in range(10)
    ]
    placed = plan_placements(
        lines, [(.5, .5)] * len(lines), width=1280, height=720, margin=72, size=45,
    )

    shapes = [tuple((f.x, f.y, f.align) for f in row) for row in placed]
    assert len(set(shapes)) >= 8, f"only {len(set(shapes))} distinct layouts in ten lines"
    assert all(
        shapes[index] != shapes[index - 1] for index in range(1, len(shapes))
    ), "two consecutive lines landed on the same layout"


def test_a_fragment_stays_hidden_until_it_is_sung(tmp_path: Path) -> None:
    """The free layout assembles the line across the frame over its own duration."""
    line = _breathing_line()
    placements = plan_placements(
        [line], [(.5, .5)], width=1280, height=720, margin=72, size=45,
    )
    target = tmp_path / "placed.ass"
    write_ass([line], target, width=1280, height=720, font="sans", size=45, margin=72,
              effect="cinematic", placements=placements)

    events = [row for row in target.read_text("utf-8-sig").splitlines()
              if row.startswith("Dialogue")]
    assert len(events) == 2, "one event per fragment"
    assert "\\pos(" in events[0] and "\\pos(" in events[1]
    # The first fragment comes in with the line; the second waits for its own moment.
    assert "\\alpha&HFF&" not in events[0]
    assert "\\alpha&HFF&\\t(1600,1860,\\alpha&H00&)" in events[1]


def test_the_band_layout_is_unchanged_without_placements(tmp_path: Path) -> None:
    """The classic bottom band stays available, and stays one centred line."""
    line = _breathing_line()
    target = tmp_path / "band.ass"
    write_ass([line], target, width=1280, height=720, font="sans", size=45, margin=72,
              effect="karaoke")

    events = [row for row in target.read_text("utf-8-sig").splitlines()
              if row.startswith("Dialogue")]
    assert len(events) == 1
    assert "\\pos(" not in events[0]
    assert "\\kf" in events[0], "the band layout keeps the character sweep"


def test_the_style_is_only_bold_when_a_bold_weight_was_asked_for(tmp_path: Path) -> None:
    """The style's Bold is a switch, so it has to agree with the weight on the events.

    It used to be on unconditionally, which put every preset that asks for no weight -
    ``modern``, ``minimal``, ``cinematic`` and the rest - into a synthetic bold. On a
    family with no bold of its own libass thickens the glyphs itself, and synthetic bold
    over a hairline face reads as neither one thing nor the other.
    """
    line = _breathing_line()
    rows = {}
    for weight in (None, 300, 400, 700):
        target = tmp_path / f"weight-{weight}.ass"
        write_ass([line], target, width=1280, height=720, font="sans", size=45,
                  margin=72, effect="cinematic", weight=weight)
        rows = target.read_text("utf-8-sig").splitlines()
        style = next(row for row in rows if row.startswith("Style: Lyric"))
        event = next(row for row in rows if row.startswith("Dialogue"))
        bold = style.split(",")[7]
        expected = "-1" if (weight or 400) >= 600 else "0"
        assert bold == expected, (weight, style)
        # The number has to travel on the event as well: the style is one bit, and a
        # family that can be anything from 100 to 900 needs the value.
        assert (f"\\b{weight}" in event) == (weight is not None)


def test_the_outline_is_configurable_for_type_inside_the_picture(tmp_path: Path) -> None:
    """A caption needs an outline; type that shares the frame with the picture does not."""
    line = _breathing_line()
    for outline, expected in ((2.2, ",2.2,0.0,2,"), (0.0, ",0.0,0.0,2,")):
        target = tmp_path / f"outline-{outline}.ass"
        write_ass([line], target, width=1280, height=720, font="sans", size=45,
                  margin=72, effect="cinematic", outline=outline)
        style = next(row for row in target.read_text("utf-8-sig").splitlines()
                     if row.startswith("Style: Lyric"))
        assert expected in style, style


def _static_fill_colours(content: str) -> set[str]:
    """The ``\\1c`` fills in the events, minus any that live inside a ``\\t`` animation."""
    events = "\n".join(
        row for row in content.splitlines() if row.startswith("Dialogue")
    )
    without_animations = re.sub(r"\\t\([^)]*\)", "", events)
    return set(re.findall(r"\\1c(&H[0-9A-Fa-f]{8}&)", without_animations))


@pytest.mark.parametrize("effect", SUBTITLE_EFFECTS)
def test_the_static_fill_colour_is_always_the_configured_highlight(
    tmp_path: Path, effect: str,
) -> None:
    """R-05: only the *static* base colour is pinned; colour animation is still allowed.

    Every preset used to hard-code a white (or dim) base fill, which silently overrode the
    project's own highlight colour - the same class of bug R-02 names. Now the base is left
    to the style, and an effect that is defined by colour moves it only inside a ``\\t``.
    """
    highlight = "&H0000D7FF"
    target = tmp_path / f"{effect}.ass"
    write_ass(
        [LyricLine(0, 2, "旋律")], target, width=1280, height=720, font="sans", size=45,
        margin=72, effect=effect, highlight_color=highlight,
    )
    content = target.read_text("utf-8-sig")
    assert _static_fill_colours(content) <= {highlight}, effect


@pytest.mark.parametrize("effect", ["rainbow", "spotlight"])
def test_colour_effects_still_animate_their_colour(tmp_path: Path, effect: str) -> None:
    """Pinning the static base must not have flattened the effects defined by colour."""
    target = tmp_path / f"{effect}.ass"
    write_ass(
        [LyricLine(0, 3, "旋律")], target, width=1280, height=720, font="sans", size=45,
        margin=72, effect=effect,
    )
    assert re.search(r"\\t\([^)]*\\1c", target.read_text("utf-8-sig")), effect


def test_short_lines_are_merged_into_their_neighbour() -> None:
    """A caption on screen for a quarter of a second flickers; it cannot be read."""
    lines = [
        LyricLine(0, 3, "第一句"),
        LyricLine(3.0, 3.4, "嗯"),
        LyricLine(3.4, 6, "第二句"),
    ]
    merged = merge_short_lines(lines, minimum=.6)
    assert len(merged) == 2
    assert merged[0].text == "第一句嗯"
    assert merged[0].start == 0 and merged[0].end == 3.4
    assert merged[1].text == "第二句"


def test_a_short_first_line_merges_into_the_second() -> None:
    """The first line has nothing before it, so it folds forward instead."""
    merged = merge_short_lines(
        [LyricLine(0, .3, "啊"), LyricLine(.3, 3, "你好")], minimum=.6,
    )
    assert len(merged) == 1
    assert merged[0].text == "啊你好"


def test_a_tail_shorter_than_the_floor_is_not_split_into_its_own_event() -> None:
    """R-10: a fragment that would flash for less than the floor is folded back."""
    line = LyricLine(0, 1.0, "星光", tokens=[
        LyricToken("星", 0.0, .4), LyricToken("光", .8, .95),
    ])
    kept = plan_placements(
        [line], [(.5, .5)], width=1280, height=720, margin=72, size=45,
        min_fragment_seconds=.6,
    )[0]
    assert len(kept) == 1, "a 0.2s tail was given its own event"

    split = plan_placements(
        [line], [(.5, .5)], width=1280, height=720, margin=72, size=45,
        min_fragment_seconds=.0,
    )[0]
    assert len(split) == 2, "the split must still be available when each half lives long enough"


def test_write_ass_emits_a_per_line_outline(tmp_path: Path) -> None:
    """R-05: the outline is adaptive per line, so it travels on each event."""
    lines = [LyricLine(0, 2, "第一句"), LyricLine(2, 4, "第二句")]
    target = tmp_path / "perline.ass"
    write_ass(
        lines, target, width=1280, height=720, font="sans", size=45, margin=72,
        effect="cinematic", outline=1.0, line_outlines=[1.5, 3.0],
    )
    events = [row for row in target.read_text("utf-8-sig").splitlines()
              if row.startswith("Dialogue")]
    assert r"\bord1.5" in events[0]
    assert r"\bord3" in events[1]


def test_the_outline_thickens_as_the_backdrop_brightens() -> None:
    """A bright backdrop needs weight; a dark one keeps the thin reference edge."""
    assert adaptive_outline(.2, 1.0, max_outline=2.4) == 1.0
    assert adaptive_outline(1.0, 1.0, max_outline=2.4) == 2.4
    assert adaptive_outline(.2, 1.0) < adaptive_outline(.8, 1.0)
    # A project that wants no outline keeps none on a dark scene.
    assert adaptive_outline(0.0, 0.0, max_outline=2.4) == 0.0


def test_the_local_dim_only_fires_on_a_bright_backdrop() -> None:
    assert not needs_local_dim(.4)
    assert not needs_local_dim(.6)
    assert needs_local_dim(.8)


def test_legibility_reads_a_real_backdrop(tmp_path: Path) -> None:
    """The reading comes from the source picture and distinguishes bright from dark."""
    bright = tmp_path / "bright.png"
    dark = tmp_path / "dark.png"
    Image.new("RGB", (64, 64), (250, 250, 250)).save(bright)
    Image.new("RGB", (64, 64), (5, 5, 5)).save(dark)
    cache = tmp_path / "cache"
    region = (0.0, 0.0, 1.0, 1.0)
    assert estimate_region_luma(bright, "image", None, region, cache) > .9
    assert estimate_region_luma(dark, "image", None, region, cache) < .1
    # A file that cannot be read returns neutral mid-grey rather than raising.
    assert estimate_region_luma(
        tmp_path / "missing.png", "image", None, region, cache,
    ) == .5


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_legibility_accepts_a_str_path_and_a_video(tmp_path: Path) -> None:
    """``Shot.file`` is a plain str, and videos go through ffmpeg frame extraction.

    The video branch used to raise ``AttributeError: 'str' object has no attribute
    'stem'``. Images survived a str because ``Image.open`` takes one too - which is
    exactly why a stills-only project never hit it and my-mv did, after 234 shots had
    already rendered. Both branches must accept what the renderer actually passes.
    """
    bright = tmp_path / "bright.png"
    Image.new("RGB", (64, 64), (250, 250, 250)).save(bright)
    cache = tmp_path / "cache"
    region = (0.0, 0.0, 1.0, 1.0)
    assert estimate_region_luma(str(bright), "image", None, region, cache) > .9

    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=white:s=64x64:d=0.5", "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True,
    )
    assert estimate_region_luma(str(clip), "video", None, region, cache, at=0.0) > .9
    # The extracted frame is cached per source, so a second line does not re-decode.
    assert (cache / "clip.png").exists()


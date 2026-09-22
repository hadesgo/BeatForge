import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from beatforge.legibility import adaptive_outline, estimate_region_luma, needs_local_dim
from beatforge.lyrics import (
    SUBTITLE_EFFECTS,
    LyricLine,
    LyricToken,
    merge_short_lines,
    parse_lrc,
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


def test_every_line_is_one_centred_event_in_the_bottom_band(tmp_path: Path) -> None:
    """The band layout is the only layout: one event per line, pinned by the style.

    The free layout - ten rotated patterns scattering fragments across the frame - was
    removed on purpose: it read as jumpy and unfocused, a line's pieces appearing in
    different places at different times. Band placement belongs to the ASS style
    (``Alignment 2`` + ``MarginV``), so no event may carry its own ``\\pos`` or
    ``\\move`` - that is what keeps every line exactly where a bottom band sits.
    """
    lines = [
        _breathing_line(),
        LyricLine(4, 8, "城市亮起灯光"),
        LyricLine(8, 12, "那天的天气，难得放晴"),
    ]
    target = tmp_path / "band.ass"
    write_ass(lines, target, width=1280, height=720, font="sans", size=45, margin=72,
              effect="cinematic")

    events = [row for row in target.read_text("utf-8-sig").splitlines()
              if row.startswith("Dialogue")]
    assert len(events) == 3, "one event per line - the line is never split into fragments"
    for event in events:
        assert "\\pos(" not in event, "a band event must not pin its own position"
        assert "\\move(" not in event, "a band event must not carry its own motion"
        assert "\\alpha&HFF&" not in event, "no fragment-timing hold may return"
    assert "Alignment" not in "\n".join(events)
    style = next(row for row in target.read_text("utf-8-sig").splitlines()
                 if row.startswith("Style:"))
    assert style.endswith(",2,48,48,72,1") or ",2,48,48," in style, (
        "the style must anchor centred bottom (numpad 2) with the project margin"
    )


def test_karaoke_still_sweeps_in_the_band(tmp_path: Path) -> None:
    """The band keeps the per-character sweep: one centred line is what karaoke is for."""
    line = _breathing_line()
    target = tmp_path / "karaoke.ass"
    write_ass([line], target, width=1280, height=720, font="sans", size=45, margin=72,
              effect="karaoke")

    events = [row for row in target.read_text("utf-8-sig").splitlines()
              if row.startswith("Dialogue")]
    assert len(events) == 1
    assert "\\k" in events[0], "the karaoke sweep was lost with the free layout"


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


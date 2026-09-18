from pathlib import Path

import pytest

from beatforge.lyrics import (
    _CLEAR_X,
    _CLEAR_Y,
    LyricLine,
    LyricToken,
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
    """
    placements = plan_placements(
        [_breathing_line()], [focus], width=1280, height=720, margin=72, size=45,
    )[0]

    assert placements
    for fragment in placements:
        dx = abs(fragment.x / 1280 - focus[0])
        dy = abs(fragment.y / 720 - focus[1])
        assert dx >= _CLEAR_X or dy >= _CLEAR_Y, (focus, fragment)


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


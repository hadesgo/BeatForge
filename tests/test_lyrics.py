from pathlib import Path

from beatforge.lyrics import LyricLine, LyricToken, parse_lrc, srt_timestamp, write_ass


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


def test_richer_ass_effects(tmp_path: Path) -> None:
    lines = [LyricLine(0, 2, "旋律")]
    for effect, expected in (("float", r"\move"), ("glow", r"\blur3"), ("typewriter", r"\alpha&HFF&")):
        target = tmp_path / f"{effect}.ass"
        write_ass(lines, target, width=1280, height=720, font="Arial", size=40, margin=60, effect=effect)
        assert expected in target.read_text("utf-8-sig")

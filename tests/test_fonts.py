import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beatforge import fonts
from beatforge.config import RenderConfig

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "fonts"


def test_explicit_font_is_preserved() -> None:
    assert fonts.resolve_subtitle_font("My MV Font") == fonts.FontChoice("My MV Font")


def test_preset_selects_first_available_family(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"Noto Sans CJK SC"})
    assert fonts.resolve_subtitle_font("preset:modern", tmp_path) == fonts.FontChoice("Noto Sans CJK SC")


def test_unknown_preset_uses_modern_fallback(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"Source Han Sans SC"})
    assert fonts.resolve_subtitle_font("preset:not-real") == fonts.FontChoice("Source Han Sans SC")


# ------------------------------------------------------------------ the bundle


def test_bundled_fonts_are_real_font_files() -> None:
    """A clone made without git-lfs leaves a pointer file where a font should be.

    Catching that here beats discovering it as tofu boxes halfway through a render, and
    the magic bytes are the only thing that tells the two apart - a pointer file is
    perfectly readable text.
    """
    files = sorted(BUNDLED.glob("*.ttf"))
    assert files, "no bundled fonts found"

    for path in files:
        assert path.read_bytes()[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), (
            f"{path.name} is not a font - is git-lfs installed?"
        )
        assert fonts._font_families(path), path.name


def test_every_bundled_font_is_declared_in_the_presets() -> None:
    """A font nobody can select is weight in the repository and nothing else."""
    declared = {
        fonts._parse_entry(entry).family
        for entries in fonts.FONT_PRESETS.values()
        for entry in entries
    }
    for path in sorted(BUNDLED.glob("*.ttf")):
        names = fonts._font_families(path)
        assert names & declared, f"{path.name} {sorted(names)} is bundled but no preset names it"


def test_every_bundled_font_is_reachable_from_some_preset(monkeypatch, tmp_path: Path) -> None:
    """Being *declared* in a preset is not enough - it has to be the one that gets picked.

    A font listed behind another bundled family is never selected on a machine that has
    the bundle, so it would sit in the repository unreachable. Checking the resolved set
    rather than the declared one is what makes this test worth having.
    """
    available = fonts._custom_font_families(BUNDLED)
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: available)

    picked = {
        fonts.resolve_subtitle_font(f"preset:{preset}", tmp_path).family
        for preset in fonts.FONT_PRESETS
    }

    for path in sorted(BUNDLED.glob("*.ttf")):
        names = fonts._font_families(path)
        assert names & picked, f"{path.name} {sorted(names)} is bundled but no preset picks it"


def test_a_font_with_a_chinese_default_name_is_still_discoverable() -> None:
    """PIL decodes a Chinese default family name to "?????", which makes it unfindable.

    ZCOOL XiaoWei is exactly that font: its default-language family is 站酷小薇体, which
    PIL cannot decode, while fontconfig renders it happily under "ZCOOL XiaoWei". Going
    through PIL alone left a bundled font that no preset could ever select. The name
    table is read directly now, so every platform and language record is collected.
    """
    names = fonts._font_families(BUNDLED / "ZCOOLXiaoWei-Regular.ttf")

    assert "ZCOOL XiaoWei" in names
    assert not any("?" in name for name in names)
    assert names, "the fallback did not produce a name either"


def test_every_preset_resolves_to_a_bundled_family(monkeypatch, tmp_path: Path) -> None:
    """On a machine with none of the system fonts, every preset still has to work.

    That is the whole point of shipping the fonts: without them a bare Linux box or a
    stripped Windows install renders every preset as the same fallback, or as tofu.
    """
    bundled = fonts._custom_font_families(BUNDLED)
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: bundled)

    resolved = {
        preset: fonts.resolve_subtitle_font(f"preset:{preset}", tmp_path)
        for preset in fonts.FONT_PRESETS
    }

    for preset, choice in resolved.items():
        assert choice.family in bundled, f"{preset} fell through to {choice.family}"
    # And they must not all land on the same face, or the presets would be decorative.
    assert len({choice.family for choice in resolved.values()}) >= 3, resolved


def test_a_variable_family_carries_the_weight_it_was_asked_for(monkeypatch, tmp_path: Path) -> None:
    """fontconfig does not expose named instances, so the weight travels separately.

    "Noto Sans SC Light" resolves to nothing and silently falls back to a system font.
    The bundled Noto families are variable, so the weight goes out as an ASS ``\\b`` tag
    instead - which is the only reason one file can cover 100 through 900.
    """
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"Noto Sans SC"})

    assert fonts.resolve_subtitle_font("preset:dreamy", tmp_path) == fonts.FontChoice("Noto Sans SC", 300)
    assert fonts.resolve_subtitle_font("preset:dark", tmp_path) == fonts.FontChoice("Noto Sans SC", 700)
    assert fonts.resolve_subtitle_font("preset:modern", tmp_path) == fonts.FontChoice("Noto Sans SC")


def test_a_family_without_a_weight_suffix_keeps_none() -> None:
    assert fonts._parse_entry("Source Han Sans SC") == fonts.FontChoice("Source Han Sans SC")
    assert fonts._parse_entry("Noto Sans SC@700") == fonts.FontChoice("Noto Sans SC", 700)


# ------------------------------------------------------- the fonts libass may read


def test_the_shipped_library_is_always_a_search_path() -> None:
    """``subtitle_fonts_dir`` is resolved against the project folder, not the repository.

    The template ships ``subtitle_fonts_dir = "fonts"``, every project folder has its own
    - usually empty - ``fonts/``, and neither of the two shipped projects points anywhere
    useful. Searching only that directory made the whole bundled library invisible to
    libass, which is what "the font is not being used" looks like from the outside.
    """
    assert fonts.font_directories(None) == [fonts.PACKAGED_FONTS_DIR]
    assert fonts.font_directories(Path("does-not-exist")) == [fonts.PACKAGED_FONTS_DIR]


def test_a_project_font_directory_comes_first(tmp_path: Path) -> None:
    assert fonts.font_directories(tmp_path) == [tmp_path, fonts.PACKAGED_FONTS_DIR]


def test_a_fuzzy_match_writes_the_spelling_the_font_has(monkeypatch, tmp_path: Path) -> None:
    """A candidate that merely *contains* an installed family must not name itself.

    ``Kaiti SC`` matched the installed ``KaiTi`` and then went into the lyric script as
    ``Kaiti SC`` - a family nothing can resolve, so libass drew the line in whatever
    fallback it had, not in the 楷体 the preset asked for. A name that does not exist
    and a font that cannot be found look exactly alike on screen.
    """
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"KaiTi"})

    assert fonts.resolve_subtitle_font("preset:lyrical", tmp_path) == fonts.FontChoice("KaiTi")


def test_the_bundled_variable_fonts_default_to_a_light_instance() -> None:
    """Which is the whole reason they cannot be handed to libass as they are.

    libass does not apply the ``wght`` axis: it asks the provider for a 700 face, gets
    the font's *default* instance, and - seeing the family already has a bold - draws no
    synthetic bold either. So the default instance is what lands on screen, and these
    two are Thin (100) and ExtraLight (200).
    """
    assert fonts._weight_axis(BUNDLED / "NotoSansSC-VF.ttf") == (100.0, 100.0, 900.0)
    assert fonts._weight_axis(BUNDLED / "NotoSerifSC-VF.ttf") == (200.0, 200.0, 900.0)
    assert fonts._weight_axis(BUNDLED / "MaShanZheng-Regular.ttf") is None


def test_staging_carves_variable_fonts_into_static_faces() -> None:
    """Staging has to end up with real faces, not with the variable file renamed."""
    plan = {name: (source.name, weight) for name, source, weight in fonts._staging_plan(
        fonts.font_directories(None), fonts.STAGED_WEIGHTS,
    )}

    assert "NotoSansSC-VF.ttf" not in plan
    assert plan["NotoSansSC-Regular.ttf"] == ("NotoSansSC-VF.ttf", 400)
    assert plan["NotoSansSC-Bold.ttf"] == ("NotoSansSC-VF.ttf", 700)
    assert plan["NotoSansSC-Light.ttf"] == ("NotoSansSC-VF.ttf", 300)
    # Static families travel through untouched, and cheaply - they are hardlinked.
    assert plan["MaShanZheng-Regular.ttf"] == ("MaShanZheng-Regular.ttf", None)


def test_the_staged_library_is_named_after_what_it_was_built_from(tmp_path: Path) -> None:
    """The cache key is the content, so an unchanged library is never rebuilt."""

    def key(directories: list[Path], weights: tuple[int, ...] = fonts.STAGED_WEIGHTS) -> str:
        return fonts._digest(fonts._manifest(fonts._staging_plan(directories, weights)))

    first = key(fonts.font_directories(None))

    assert first == key(fonts.font_directories(None)), "the same inputs named it differently"
    assert first != key(fonts.font_directories(None), (400,)), "a weight set was ignored"

    # An *empty* extra directory changes nothing, which is the point: it is what every
    # project folder looks like, and it must not cost a rebuild.
    assert first == key(fonts.font_directories(tmp_path))

    # A font the project actually brings does change it - including when it takes the
    # place of a bundled file by keeping the bundled file's name.
    shutil.copy(BUNDLED / "ZCOOLKuaiLe-Regular.ttf", tmp_path / "ZCOOLKuaiLe-Regular.ttf")
    assert first != key(fonts.font_directories(tmp_path)), "a new font reused the old library"


def test_a_staged_library_is_reused_only_while_its_files_are_there(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "one.ttf").write_bytes(b"\x00\x01\x00\x00")
    manifest = [["one.ttf", "one.ttf", 4, 0, None]]

    assert not fonts._stage_is_usable(stage, manifest), "no stamp, but it was trusted"
    (stage / ".staged.json").write_text(
        json.dumps({"manifest": [["one.ttf", "one.ttf", 999, 0, None]]}), "utf-8"
    )
    assert not fonts._stage_is_usable(stage, manifest), "a stamp from another build was trusted"
    (stage / ".staged.json").write_text(json.dumps({"manifest": manifest}), "utf-8")
    assert fonts._stage_is_usable(stage, manifest)
    (stage / "one.ttf").unlink()
    assert not fonts._stage_is_usable(stage, manifest), "a missing font was reused"
    (stage / "one.ttf").write_bytes(b"\x00\x01\x00\x00")
    (stage / ".staged.json").write_text(
        json.dumps({"manifest": [["other.ttf", "other.ttf", 4, 0, None]]}), "utf-8"
    )
    assert not fonts._stage_is_usable(stage, manifest), "a stale stamp was trusted"


def test_the_font_cache_can_be_moved_with_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BEATFORGE_FONT_CACHE", str(tmp_path))

    assert fonts._stage_root() == tmp_path


def test_an_instance_keeps_its_family_and_claims_its_weight(tmp_path: Path) -> None:
    """An instance has to be findable under the family the presets name, and be honest
    about its weight - a face that says Regular while 700 is asked for is exactly the
    confusion that made the variable font useless."""
    pytest.importorskip("fontTools")
    target = tmp_path / "instance.ttf"

    fonts._write_instance(BUNDLED / "NotoSansSC-VF.ttf", 700, target)

    records = fonts._name_records(target)
    assert "Noto Sans SC" in records[1], records
    assert records[2] == ["Bold"]
    assert fonts._weight_axis(target) is None, "the instance is still variable"

    from fontTools.ttLib import TTFont

    font = TTFont(str(target))
    assert font["OS/2"].usWeightClass == 700
    assert font["OS/2"].fsSelection & (1 << 5), "the bold bit is not set"
    assert font["head"].macStyle & 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_a_requested_weight_actually_reaches_the_rendered_frame(tmp_path: Path) -> None:
    """The failure this guards is invisible: a weight that never arrives still renders.

    ``\\b300``, ``\\b400`` and ``\\b700`` came out pixel for pixel identical, because
    libass handed the question to a font provider that answered with the variable font's
    default instance every time. Weight is only credible if the ink moves with it.
    """
    stage = fonts.stage_fonts()
    assert stage is not None
    background = tmp_path / "background.jpg"
    Image.new("RGB", (1280, 720), (15, 15, 20)).save(background)

    inks = []
    for weight in (300, 400, 700):
        inks.append(_rendered_ink(tmp_path, background, stage, weight))

    assert inks[0] < inks[1] < inks[2], f"the weight never reached the frame: {inks}"


def _rendered_ink(tmp_path: Path, background: Path, stage: Path, weight: int) -> int:
    """Non-background pixels of one line, sampled after the fade-in has finished."""
    from beatforge.lyrics import LyricLine, write_ass
    from beatforge.renderer import _subtitle_filter
    from beatforge.runtime import command

    script = tmp_path / f"weight-{weight}.ass"
    write_ass(
        [LyricLine(0, 1, "黎明照亮天空")], script, width=1280, height=720,
        font="Noto Sans SC", size=64, weight=weight, margin=48,
        effect="cinematic", placements=[], outline=0.0,
    )
    clip, frame = tmp_path / f"weight-{weight}.mp4", tmp_path / f"weight-{weight}.png"
    cfg = RenderConfig(width=1280, height=720, fps=10, crf=28, preset="ultrafast")
    command([
        "ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "10",
        "-i", str(background), "-vf", _subtitle_filter(script, cfg, stage),
        "-t", "1", "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", str(clip),
    ])
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(clip),
        "-vf", r"select=eq(n\,5)", "-fps_mode", "passthrough", "-frames:v", "1", str(frame),
    ])
    grey = np.asarray(Image.open(frame).convert("L"), dtype=float)
    return int((grey > 25).sum())



@pytest.mark.parametrize("name", sorted(BUNDLED.glob("OFL-*.txt")), ids=lambda p: p.name)
def test_bundled_fonts_ship_their_licence(name: Path) -> None:
    """Redistribution requires the licence to travel with the binary."""
    assert "SIL OPEN FONT LICENSE" in name.read_text("utf-8", errors="ignore")

from pathlib import Path

import pytest
from PIL import ImageFont

from beatforge import fonts

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
        family, _ = ImageFont.truetype(str(path), 12).getname()
        assert family, path.name


def test_every_bundled_font_is_declared_in_the_presets() -> None:
    """A font nobody can select is weight in the repository and nothing else."""
    declared = {
        fonts._parse_entry(entry).family
        for entries in fonts.FONT_PRESETS.values()
        for entry in entries
    }
    for path in sorted(BUNDLED.glob("*.ttf")):
        family, _ = ImageFont.truetype(str(path), 12).getname()
        assert family in declared, f"{path.name} ({family}) is bundled but no preset can pick it"


def test_every_preset_resolves_to_a_bundled_family(monkeypatch, tmp_path: Path) -> None:
    """On a machine with none of the system fonts, every preset still has to work.

    That is the whole point of shipping the fonts: without them a bare Linux box or a
    stripped Windows install renders every preset as the same fallback, or as tofu.
    """
    bundled = {
        ImageFont.truetype(str(path), 12).getname()[0] for path in BUNDLED.glob("*.ttf")
    }
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


@pytest.mark.parametrize("name", sorted(BUNDLED.glob("OFL-*.txt")), ids=lambda p: p.name)
def test_bundled_fonts_ship_their_licence(name: Path) -> None:
    """Redistribution requires the licence to travel with the binary."""
    assert "SIL OPEN FONT LICENSE" in name.read_text("utf-8", errors="ignore")

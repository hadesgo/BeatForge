from pathlib import Path

from beatforge import fonts


def test_explicit_font_is_preserved() -> None:
    assert fonts.resolve_subtitle_font("My MV Font") == "My MV Font"


def test_preset_selects_first_available_family(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"Noto Sans CJK SC"})
    assert fonts.resolve_subtitle_font("preset:modern", tmp_path) == "Noto Sans CJK SC"


def test_unknown_preset_uses_modern_fallback(monkeypatch) -> None:
    monkeypatch.setattr(fonts, "_available_font_families", lambda _: {"Source Han Sans SC"})
    assert fonts.resolve_subtitle_font("preset:not-real") == "Source Han Sans SC"

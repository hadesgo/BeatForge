from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FontChoice:
    """A resolved font family, and the weight to ask for when the family can vary.

    The bundled Noto families are **variable** fonts: one file covers 100..900 and the
    weight is picked with an ASS ``\\b`` tag rather than by naming a separate family.
    That is why the weight travels beside the family instead of inside it - fontconfig
    does not expose the named instances, so "Noto Sans SC Light" resolves to nothing and
    silently falls back to a system font.
    """

    family: str
    weight: int | None = None


#: A preset entry may pin a weight: ``"Noto Sans SC@300"``. Only the bundled variable
#: families use it; anything without the suffix is a plain family name.
_WEIGHT_SUFFIX = re.compile(r"^(?P<family>.+?)@(?P<weight>\d{3})$")


def _parse_entry(entry: str) -> FontChoice:
    match = _WEIGHT_SUFFIX.match(entry)
    if match:
        return FontChoice(match["family"], int(match["weight"]))
    return FontChoice(entry)


# Presets list the bundled open-source families first, then common system fonts, so a
# machine with none of them installed still gets the bundled face rather than tofu.
# Everything under ``fonts/`` is SIL OFL 1.1 and redistributable; see fonts/README.md.
FONT_PRESETS: dict[str, tuple[str, ...]] = {
    "modern": (
        "Noto Sans SC", "Source Han Sans SC", "Noto Sans CJK SC", "MiSans",
        "HarmonyOS Sans SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei",
    ),
    "cinematic": (
        "Noto Serif SC", "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC",
        "SimSun", "Source Han Sans SC", "Microsoft YaHei",
    ),
    "lyrical": (
        "LXGW WenKai", "Kaiti SC", "STKaiti", "KaiTi", "FangSong",
        "Noto Serif SC", "Source Han Serif SC", "Noto Serif CJK SC",
    ),
    "energetic": (
        "Smiley Sans", "Noto Sans SC@700", "Alimama ShuHeiTi",
        "Source Han Sans SC Heavy", "Noto Sans CJK SC Black",
        "Microsoft YaHei UI", "SimHei",
    ),
    "dreamy": (
        "Noto Sans SC@300", "Source Han Sans SC Light", "Noto Sans CJK SC Light",
        "MiSans Light", "Microsoft YaHei Light", "PingFang SC Light", "LXGW WenKai",
    ),
    "minimal": (
        "Noto Sans SC", "MiSans", "HarmonyOS Sans SC", "Source Han Sans SC",
        "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC",
    ),
    "dark": (
        "Smiley Sans", "Noto Sans SC@700", "Source Han Sans SC Heavy",
        "Noto Sans CJK SC Black", "Microsoft YaHei UI", "SimHei",
    ),
}


def resolve_subtitle_font(requested: str, fonts_dir: Path | None = None) -> FontChoice:
    """Resolve ``preset:name`` to an installed or bundled font family."""
    if not requested.startswith("preset:"):
        return FontChoice(requested)
    preset = requested.partition(":")[2].strip().lower()
    candidates = FONT_PRESETS.get(preset, FONT_PRESETS["modern"])
    available = _available_font_families(fonts_dir)
    if available:
        normalized = {_normalize_font(name): name for name in available}
        for candidate in candidates:
            choice = _parse_entry(candidate)
            key = _normalize_font(choice.family)
            if key in normalized:
                return choice
            if any(key in installed or installed in key for installed in normalized):
                return choice
    # Font discovery is not guaranteed on every FFmpeg build. A platform-native
    # Chinese family is a safer fallback than Arial, which may render tofu boxes.
    if platform.system() == "Windows":
        return FontChoice("Microsoft YaHei")
    if platform.system() == "Darwin":
        return FontChoice("PingFang SC")
    return FontChoice("Noto Sans CJK SC")


def _available_font_families(fonts_dir: Path | None) -> set[str]:
    families = _custom_font_families(fonts_dir)
    if platform.system() == "Windows":
        families.update(_windows_font_families())
    else:
        families.update(_fontconfig_families())
    return families


def _custom_font_families(fonts_dir: Path | None) -> set[str]:
    if fonts_dir is None or not fonts_dir.is_dir():
        return set()
    try:
        from PIL import ImageFont
    except ImportError:
        return set()
    families: set[str] = set()
    for file in [*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf"), *fonts_dir.glob("*.ttc")]:
        try:
            family, _ = ImageFont.truetype(str(file), 12).getname()
            if family:
                families.add(family)
        except OSError:
            continue
    return families


def _fontconfig_families() -> set[str]:
    executable = shutil.which("fc-list")
    if not executable:
        return set()
    try:
        result = subprocess.run(
            [executable, ":", "family"], capture_output=True, text=True,
            encoding="utf-8", errors="ignore", timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {
        family.strip()
        for line in result.stdout.splitlines()
        for family in line.split(",")
        if family.strip()
    }


def _windows_font_families() -> set[str]:
    try:
        import winreg
    except ImportError:
        return set()
    families: set[str] = set()
    key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, key_path) as key:
                index = 0
                while True:
                    try:
                        name, _, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    families.add(name.split("(", 1)[0].strip())
                    index += 1
        except OSError:
            continue
    return families


def _normalize_font(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())

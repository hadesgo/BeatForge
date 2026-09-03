from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


# Presets list open-source families first, then common system fonts. BeatForge does
# not redistribute font binaries; users can put any matching TTF/OTF in fonts_dir.
FONT_PRESETS: dict[str, tuple[str, ...]] = {
    "modern": (
        "Source Han Sans SC", "Noto Sans CJK SC", "MiSans", "HarmonyOS Sans SC",
        "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei",
    ),
    "cinematic": (
        "Source Han Serif SC", "Noto Serif CJK SC", "Songti SC", "SimSun",
        "Source Han Sans SC", "Microsoft YaHei",
    ),
    "lyrical": (
        "LXGW WenKai", "Kaiti SC", "STKaiti", "KaiTi", "FangSong",
        "Source Han Serif SC", "Noto Serif CJK SC",
    ),
    "energetic": (
        "Smiley Sans", "Alimama ShuHeiTi", "Source Han Sans SC Heavy",
        "Noto Sans CJK SC Black", "Microsoft YaHei UI", "SimHei",
    ),
    "dreamy": (
        "Source Han Sans SC Light", "Noto Sans CJK SC Light", "MiSans Light",
        "Microsoft YaHei Light", "PingFang SC Light", "LXGW WenKai Light",
    ),
    "minimal": (
        "MiSans", "HarmonyOS Sans SC", "Source Han Sans SC", "Noto Sans CJK SC",
        "Microsoft YaHei", "PingFang SC",
    ),
    "dark": (
        "Source Han Sans SC Heavy", "Noto Sans CJK SC Black", "Smiley Sans",
        "Microsoft YaHei UI", "SimHei",
    ),
}


def resolve_subtitle_font(requested: str, fonts_dir: Path | None = None) -> str:
    """Resolve ``preset:name`` to an installed/custom font family."""
    if not requested.startswith("preset:"):
        return requested
    preset = requested.partition(":")[2].strip().lower()
    candidates = FONT_PRESETS.get(preset, FONT_PRESETS["modern"])
    available = _available_font_families(fonts_dir)
    if available:
        normalized = {_normalize_font(name): name for name in available}
        for candidate in candidates:
            key = _normalize_font(candidate)
            if key in normalized:
                return candidate
            if any(key in installed or installed in key for installed in normalized):
                return candidate
    # Font discovery is not guaranteed on every FFmpeg build. A platform-native
    # Chinese family is a safer fallback than Arial, which may render tofu boxes.
    if platform.system() == "Windows":
        return "Microsoft YaHei"
    if platform.system() == "Darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


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

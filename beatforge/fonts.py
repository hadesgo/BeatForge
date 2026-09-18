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
    # --- art faces. None of these is auto-selected by mood: a brush or a poster face on
    # the wrong song is worse than a plain one, so they are opt-in via
    # ``subtitle_font = "preset:brush"`` and friends.
    "brush": (
        "Ma Shan Zheng", "Zhi Mang Xing", "Kaiti SC", "STKaiti", "KaiTi",
    ),
    "script": (
        "Zhi Mang Xing", "Ma Shan Zheng", "STXingkai", "Xingkai SC", "KaiTi",
    ),
    "playful": (
        "ZCOOL KuaiLe", "ZCOOL QingKe HuangYou", "Microsoft YaHei UI", "SimHei",
    ),
    "poster": (
        "ZCOOL QingKe HuangYou", "ZCOOL KuaiLe", "Smiley Sans",
        "Source Han Sans SC Heavy", "SimHei",
    ),
    "elegant": (
        "ZCOOL XiaoWei", "Noto Serif SC", "Songti SC", "SimSun",
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
    families: set[str] = set()
    for file in [*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf"), *fonts_dir.glob("*.ttc")]:
        families |= _font_families(file)
    return families


def _font_families(path: Path) -> set[str]:
    """Every family name a font file answers to.

    ``PIL.ImageFont.getname()`` returns the font's **default-language** family, and for a
    font whose default name is Chinese it decodes to ``"?????"`` - which makes a
    perfectly usable font undiscoverable, because fontconfig will happily render it under
    its English name. Chinese fonts routinely have a Chinese default name, so the name
    table is read directly and every platform and language record is collected.

    Falls back to PIL if the table cannot be parsed, since a wrong-but-present name still
    beats no name at all.
    """
    names = _name_table_families(path)
    if names:
        return names
    try:
        from PIL import ImageFont

        family, _ = ImageFont.truetype(str(path), 12).getname()
        return {family} if family else set()
    except (ImportError, OSError):
        return set()


def _name_table_families(path: Path) -> set[str]:
    """Family (name ID 1) and typographic family (ID 16) records, all languages."""
    import struct

    try:
        data = path.read_bytes()
        table_count = struct.unpack(">H", data[4:6])[0]
    except (OSError, struct.error, IndexError):
        return set()
    offset = None
    for index in range(table_count):
        base = 12 + index * 16
        if data[base:base + 4] == b"name":
            try:
                offset = struct.unpack(">I", data[base + 8:base + 12])[0]
            except struct.error:
                return set()
            break
    if offset is None:
        return set()
    try:
        count, string_offset = struct.unpack(">HH", data[offset + 2:offset + 6])
    except struct.error:
        return set()

    families: set[str] = set()
    for index in range(count):
        base = offset + 6 + index * 12
        try:
            platform, _encoding, _language, name_id, length, string_at = struct.unpack(
                ">HHHHHH", data[base:base + 12]
            )
        except struct.error:
            break
        if name_id not in (1, 16):
            continue
        start = offset + string_offset + string_at
        raw = data[start:start + length]
        # Unicode platforms are UTF-16BE; the Mac platform is whatever its encoding
        # byte says, which in practice means trying the plausible ones in order.
        codecs = ("utf-16-be",) if platform in (0, 3) else ("utf-8", "gb18030", "latin-1")
        for codec in codecs:
            try:
                value = raw.decode(codec).replace("\x00", "").strip()
            except (UnicodeDecodeError, LookupError):
                continue
            if value and "?" not in value:
                families.add(value)
                break
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

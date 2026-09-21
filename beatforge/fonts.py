from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FontChoice:
    """A resolved font family, and the weight to ask for when the family can vary.

    ``weight`` travels beside the family rather than inside it because ASS addresses a
    family, not a named instance: "Noto Sans SC Light" is not a family any provider will
    resolve. The weight goes out as an ``\\b`` tag on every event instead - see
    ``stage_fonts`` for why the bundled families are shipped as static instances rather
    than as the variable files they are downloaded as.
    """

    family: str
    weight: int | None = None


#: The fonts BeatForge ships. A project's ``subtitle_fonts_dir`` is resolved against the
#: *project* directory, so leaving the template's ``"fonts"`` in place points at the
#: project's own - usually empty - folder. The shipped library therefore has to be a
#: search path of its own, or nothing that ships with the tool is ever reachable.
PACKAGED_FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts"

FONT_SUFFIXES = (".ttf", ".otf", ".ttc")

#: Weights carved out of the bundled variable fonts when staging. 400 and 700 are the
#: faces everything else is expressed against, 300 is what the ``dreamy`` preset asks
#: for, and nothing in the presets names a weight outside that set.
STAGED_WEIGHTS = (300, 400, 700)

#: Bumped when the staged names or the instancing change, so an old cache rebuilds.
STAGE_VERSION = 1

_WEIGHT_NAMES = {
    100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular",
    500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black",
}

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


def font_directories(configured: Path | Sequence[Path] | None = None) -> list[Path]:
    """Every directory a font may come from, the project's own first.

    The shipped library is always in the list. Bundling fonts is only worth anything if
    a machine with none of them installed still renders every preset, and that is the
    opposite of what happened while only the project's own directory was searched.
    """
    candidates: list[Path] = []
    if configured is None:
        pass
    elif isinstance(configured, (str, Path)):
        candidates.append(Path(configured))
    else:
        candidates.extend(Path(item) for item in configured)
    candidates.append(PACKAGED_FONTS_DIR)
    directories: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def resolve_subtitle_font(
    requested: str, fonts_dir: Path | Sequence[Path] | None = None,
) -> FontChoice:
    """Resolve ``preset:name`` to an installed or bundled font family.

    Any fuzzy match resolves to the spelling the font actually has. Writing the
    candidate's own spelling instead put names like ``Kaiti SC`` into the lyric script -
    a family that exists only as a longer spelling of the installed ``KaiTi`` - and
    libass answers a family it cannot find with a silent fallback, so the line was drawn
    in some other face entirely. A wrong name and a missing font look exactly alike on
    screen, which is why the script has to name something that is really there.
    """
    if not requested.startswith("preset:"):
        return _parse_entry(requested)
    preset = requested.partition(":")[2].strip().lower()
    candidates = FONT_PRESETS.get(preset, FONT_PRESETS["modern"])
    available = _available_font_families(fonts_dir)
    if available:
        normalized = {_normalize_font(name): name for name in available}
        for candidate in candidates:
            choice = _parse_entry(candidate)
            key = _normalize_font(choice.family)
            if key in normalized:
                return FontChoice(normalized[key], choice.weight)
            for installed_key, installed in sorted(normalized.items()):
                if key in installed_key or installed_key in key:
                    return FontChoice(installed, choice.weight)
    # Font discovery is not guaranteed on every FFmpeg build. A platform-native
    # Chinese family is a safer fallback than Arial, which may render tofu boxes.
    if platform.system() == "Windows":
        return FontChoice("Microsoft YaHei")
    if platform.system() == "Darwin":
        return FontChoice("PingFang SC")
    return FontChoice("Noto Sans CJK SC")


def _available_font_families(fonts_dir: Path | Sequence[Path] | None) -> set[str]:
    families = _custom_font_families(fonts_dir)
    if platform.system() == "Windows":
        families.update(_windows_font_families())
    else:
        families.update(_fontconfig_families())
    return families


def _custom_font_families(fonts_dir: Path | Sequence[Path] | None) -> set[str]:
    families: set[str] = set()
    for directory in font_directories(fonts_dir):
        for file in sorted(directory.iterdir()):
            if file.suffix.casefold() in FONT_SUFFIXES:
                families |= _font_families(file)
    return families


def _instancing_available() -> bool:
    return importlib.util.find_spec("fontTools") is not None


def stage_fonts(
    configured: Path | Sequence[Path] | None = None,
    *,
    weights: Sequence[int] = STAGED_WEIGHTS,
    root: Path | None = None,
) -> Path | None:
    """Collect every usable font into one directory, flattening the variable ones.

    libass takes exactly **one** ``fontsdir``: hand it a second path - separated by
    ``;`` or ``:`` - and it silently finds no fonts at all, which is worse than either
    directory on its own. So the project's fonts and the shipped ones have to end up
    side by side in a single directory, and this builds that directory.

    Variable fonts are replaced by static instances, because libass does not apply the
    ``wght`` axis. It asks the font provider for a 700 face, the provider answers with
    the variable font's *default* instance, and libass - seeing that the family already
    has a bold - draws no synthetic bold either. The bundled Noto files default to Thin
    (100) and ExtraLight (200), so every subtitle came out hairline whatever the config
    said and ``族名@字重`` did nothing at all. A static instance has real faces, and then
    ``\\b`` selects one.

    The result lives in a per-machine cache keyed by what it was built from, not in the
    project's cache: carving a CJK instance takes seconds and six of them take about a
    minute, and the work depends on nothing but the fonts, so every project on the
    machine shares one build. ``BEATFORGE_FONT_CACHE`` moves that cache elsewhere.

    Returns the directory to hand libass, or ``None`` when there is nothing to stage.
    """
    sources = font_directories(configured)
    if not sources:
        return None
    if not _instancing_available() and any(
        _weight_axis(file) is not None for directory in sources
        for file in directory.iterdir() if file.suffix.casefold() in FONT_SUFFIXES
    ):
        print("警告：未安装 fontTools，可变字体将按原样使用默认字重（字重设置不会生效）")
    planned = _staging_plan(sources, weights)
    manifest = _manifest(planned)
    stage = (root or _stage_root()) / _digest(manifest)
    if _stage_is_usable(stage, manifest):
        return stage
    parent = stage.parent
    parent.mkdir(parents=True, exist_ok=True)
    building = parent / f".{stage.name}.building"
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    print("正在准备字体（首次运行需要从可变字体生成静态字重，之后会复用缓存）")
    for name, source, weight in planned:
        if weight is None:
            _place(source, building / name)
        else:
            _write_instance(source, weight, building / name)
    (building / ".staged.json").write_text(
        json.dumps({"version": STAGE_VERSION, "manifest": manifest}, indent=2),
        encoding="utf-8",
    )
    # Replace in one step so a second run can never read a half-built directory.
    shutil.rmtree(stage, ignore_errors=True)
    try:
        os.replace(building, stage)
    except OSError:
        shutil.rmtree(building, ignore_errors=True)
    return stage


def _stage_root() -> Path:
    """Where staged font libraries live, one per machine."""
    override = os.environ.get("BEATFORGE_FONT_CACHE")
    if override:
        return Path(override)
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "beatforge" / "fonts"


def _stage_is_usable(stage: Path, manifest: list[list[object]]) -> bool:
    """A staged library is reused only if every file it promises is still there."""
    stamp = stage / ".staged.json"
    if not stamp.is_file():
        return False
    try:
        if json.loads(stamp.read_text("utf-8")).get("manifest") != manifest:
            return False
    except (OSError, ValueError):
        return False
    return all((stage / str(entry[0])).is_file() for entry in manifest)


def _digest(manifest: Sequence[Sequence[object]]) -> str:
    """A name for this exact library: what goes in, and how it is built."""
    payload = json.dumps([list(entry) for entry in manifest], ensure_ascii=False)
    return hashlib.sha256(f"{STAGE_VERSION}|{payload}".encode()).hexdigest()[:16]


def _staging_plan(
    sources: Sequence[Path], weights: Sequence[int],
) -> list[tuple[str, Path, int | None]]:
    """(target name, source file, weight to carve) for the whole staged directory.

    The shipped library goes in first and the project's own last, so a project can
    override a bundled file by using the same file name.
    """
    plan: dict[str, tuple[str, Path, int | None]] = {}
    for directory in reversed(list(sources)):
        for file in sorted(directory.iterdir()):
            if file.suffix.casefold() not in FONT_SUFFIXES:
                continue
            carved = _variable_weights(file, weights)
            if not carved:
                plan[file.name] = (file.name, file, None)
                continue
            family = (_typographic_family(file) or file.stem).replace(" ", "")
            for weight in carved:
                name = f"{family}-{_weight_name(weight)}.ttf"
                plan[name] = (name, file, weight)
    return [plan[name] for name in sorted(plan)]


def _manifest(planned: Sequence[tuple[str, Path, int | None]]) -> list[list[object]]:
    """What the staged directory was built from, so an unchanged library is reused.

    Size and timestamp are in here, not just the file name: a project that overrides a
    bundled font keeps the file name, and caching on the name alone would then hand back
    a library built from the file it replaced.
    """
    manifest: list[list[object]] = []
    for name, source, weight in planned:
        try:
            stat = source.stat()
        except OSError:
            manifest.append([name, source.name, 0, 0, weight])
            continue
        manifest.append([name, source.name, stat.st_size, int(stat.st_mtime), weight])
    return manifest


def _variable_weights(path: Path, weights: Sequence[int] = STAGED_WEIGHTS) -> tuple[int, ...]:
    """The weights to carve out of a variable font, or ``()`` for a static one.

    Also ``()`` when fontTools is missing: then the variable file is staged as it is,
    which renders at its default instance and ignores the weight tags.
    """
    if not _instancing_available():
        return ()
    axis = _weight_axis(path)
    if axis is None:
        return ()
    low, _default, high = axis
    return tuple(weight for weight in weights if low <= weight <= high)


def _write_instance(source: Path, weight: int, target: Path) -> None:
    """Save one static instance of a variable font, under the family it belongs to."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = instancer.instantiateVariableFont(
        TTFont(str(source)), {"wght": weight}, inplace=False, updateFontNames=False,
    )
    family = _typographic_family(source) or source.stem
    font["OS/2"].usWeightClass = weight
    # The provider matches on these. A face that claims to be Regular while 700 is
    # asked for is exactly the confusion that made the variable font useless.
    if weight >= 600:
        font["OS/2"].fsSelection = (font["OS/2"].fsSelection | (1 << 5)) & ~(1 << 6)
        font["head"].macStyle |= 1
    else:
        font["OS/2"].fsSelection = (font["OS/2"].fsSelection | (1 << 6)) & ~(1 << 5)
        font["head"].macStyle &= ~1
    _rename(font, family, _weight_name(weight))
    target.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(target))


def _rename(font: object, family: str, subfamily: str) -> None:
    """Re-point a font's family and subfamily names at the instance's own weight.

    Only the naming records are touched. Copyright, licence and version records stay
    with the font, which is what the OFL asks of a derived file.
    """
    name = font["name"]  # type: ignore[index]
    for record in list(name.names):
        if record.nameID in (1, 2, 4, 6, 16, 17):
            name.removeNames(record.nameID, record.platformID, record.platEncID, record.langID)
    for name_id, value in (
        (1, family), (2, subfamily), (4, f"{family} {subfamily}"),
        (6, f"{family.replace(' ', '')}-{subfamily}"), (16, family), (17, subfamily),
    ):
        name.setName(value, name_id, 3, 1, 0x409)


def _place(source: Path, target: Path) -> None:
    """Hardlink the font into the cache when the volume allows it, else copy it.

    The library is tens of megabytes; linking keeps a per-project staging directory from
    costing that again for every project on the same disk.
    """
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _weight_name(weight: int) -> str:
    return _WEIGHT_NAMES.get(weight, "Regular")


def _weight_axis(path: Path) -> tuple[float, float, float] | None:
    """``(min, default, max)`` of the ``wght`` axis, or ``None`` for a static font.

    Read straight out of the ``fvar`` table: the header is major/minor/axesArrayOffset/
    reserved/axisCount/axisSize, and every axis record is tag/min/default/max/flags/
    nameID. Getting either layout off by one field yields a version number where an
    offset belongs, which reads as "no axis at all" and quietly skips the instancing.
    """
    data = _table_bytes(path, b"fvar")
    if data is None:
        return None
    try:
        _major, _minor, axes_at, _reserved, axis_count, axis_size = struct.unpack(
            ">HHHHHH", data[:12]
        )
    except struct.error:
        return None
    for index in range(axis_count):
        at = axes_at + index * axis_size
        if data[at:at + 4] != b"wght":
            continue
        try:
            low, default, high = struct.unpack(">iii", data[at + 4:at + 16])
        except struct.error:
            return None
        return low / 65536, default / 65536, high / 65536
    return None


def _typographic_family(path: Path) -> str | None:
    """The family a copy should answer to: typographic (ID 16) before plain (ID 1).

    A variable font's ID 1 names its *default* instance - the bundled Noto files say
    "Noto Sans SC Thin" and "Noto Serif SC ExtraLight" - while ID 16 names the family
    the instances belong to. An instance that kept ID 1 would make the preset name the
    thin default and point every weight at the same file.
    """
    records = _name_records(path)
    for name_id in (16, 1):
        for value in records.get(name_id, ()):
            return value
    return None



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
    records = _name_records(path)
    names = set(records.get(1, ())) | set(records.get(16, ()))
    if names:
        return names
    try:
        from PIL import ImageFont

        family, _ = ImageFont.truetype(str(path), 12).getname()
        return {family} if family else set()
    except (ImportError, OSError):
        return set()


def _name_records(path: Path) -> dict[int, list[str]]:
    """Decoded name-table strings, keyed by name ID, across every platform and language."""
    data = _table_bytes(path, b"name")
    if data is None:
        return {}
    try:
        count, string_offset = struct.unpack(">HH", data[2:6])
    except struct.error:
        return {}

    records: dict[int, list[str]] = {}
    for index in range(count):
        base = 6 + index * 12
        try:
            platform, _encoding, _language, name_id, length, string_at = struct.unpack(
                ">HHHHHH", data[base:base + 12]
            )
        except struct.error:
            break
        if name_id not in (1, 2, 4, 6, 16, 17):
            continue
        raw = data[string_offset + string_at:string_offset + string_at + length]
        # Unicode platforms are UTF-16BE; the Mac platform is whatever its encoding
        # byte says, which in practice means trying the plausible ones in order.
        codecs = ("utf-16-be",) if platform in (0, 3) else ("utf-8", "gb18030", "latin-1")
        for codec in codecs:
            try:
                value = raw.decode(codec).replace("\x00", "").strip()
            except (UnicodeDecodeError, LookupError):
                continue
            if value and "?" not in value:
                records.setdefault(name_id, []).append(value)
                break
    return records


def _table_bytes(path: Path, tag: bytes) -> bytes | None:
    """One top-level table's bytes, without reading the rest of the font.

    A bundled CJK font is 17 MiB and the tables read here - ``name``, ``fvar`` - are a
    few kilobytes. Loading the whole file to parse them cost a few hundred megabytes of
    reads per render, because the staging path looks at every font it can see.

    Assumes a single-font file, which is what the presets name; a ``.ttc`` collection
    has a different layout and yields nothing, as it did before.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) < 12:
                return None
            count = struct.unpack(">H", header[4:6])[0]
            directory = handle.read(count * 16)
            for index in range(count):
                base = index * 16
                if directory[base:base + 4] != tag:
                    continue
                offset, length = struct.unpack(">II", directory[base + 8:base + 16])
                handle.seek(offset)
                return handle.read(length)
    except (OSError, struct.error):
        return None
    return None




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

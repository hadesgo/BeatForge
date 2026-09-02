from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from beatforge.runtime import probe

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEOS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass(slots=True)
class MediaAsset:
    id: int
    file: Path
    kind: str
    duration: float
    width: int
    height: int
    tags: list[str] = field(default_factory=list)
    description: str = ""
    mood: str = "neutral"

    def as_dict(self) -> dict:
        data = asdict(self)
        data["file"] = str(self.file)
        return data


def discover_media(directory: Path) -> list[MediaAsset]:
    assets: list[MediaAsset] = []
    for file in sorted(directory.iterdir()):
        suffix = file.suffix.lower()
        kind = "image" if suffix in IMAGES else "video" if suffix in VIDEOS else None
        if not file.is_file() or kind is None:
            continue
        info = probe(file)
        stream = next((x for x in info.get("streams", []) if x.get("codec_type") == "video"), {})
        sidecar = _sidecar(file)
        assets.append(MediaAsset(
            id=len(assets), file=file.resolve(), kind=kind,
            duration=float(info.get("format", {}).get("duration", 0)) if kind == "video" else float("inf"),
            width=int(stream.get("width", 0)), height=int(stream.get("height", 0)),
            tags=sidecar.get("tags", []) + _filename_tags(file.stem),
            description=sidecar.get("description", file.stem), mood=sidecar.get("mood", "neutral"),
        ))
    if not assets:
        raise RuntimeError(f"素材目录中没有受支持的图片或视频: {directory}")
    return assets


def _sidecar(file: Path) -> dict:
    try:
        return json.loads(Path(f"{file}.json").read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _filename_tags(stem: str) -> list[str]:
    return [item for item in stem.replace("-", "_").split("_") if item]

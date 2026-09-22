from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from beatforge.runtime import probe

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEOS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

#: How much of each edge counts as the "edge" for the vignette gate. The vignette only
#: helps a picture whose border is bright enough to be pulled down without going muddy.
_EDGE_BAND = 0.14


@dataclass(slots=True)
class VisualMetrics:
    """Everything the still-image pass measures in one decode.

    Kept together because they share a single ``Image.open`` + ``thumbnail``: opening a
    6000×4000 photo six times to read six numbers would be six full decodes.
    """

    quality: float
    color: list[int]
    focus: list[float]
    #: Mean luminance of the frame, 0..1. Used by the readability pass.
    luma: float
    #: Mean luminance of the outer band, 0..1 - the vignette gate reads this.
    edge_luma: float
    #: Normalised high-frequency energy. A source that already carries noise (a
    #: double-compressed JPEG) scores high and must not be given more grain.
    noise_score: float


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
    quality_score: float = .5
    dominant_color: list[int] = field(default_factory=lambda: [128, 128, 128])
    shot_size: str = "unknown"
    focus_point: list[float] = field(default_factory=lambda: [.5, .5])
    luma: float = 0.5
    edge_luma: float = 0.5
    noise_score: float = 0.0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["file"] = str(self.file)
        return data


def discover_media(directory: Path, audit=None) -> list[MediaAsset]:
    assets: list[MediaAsset] = []
    for file in sorted(directory.iterdir()):
        suffix = file.suffix.lower()
        kind = "image" if suffix in IMAGES else "video" if suffix in VIDEOS else None
        if not file.is_file() or kind is None:
            continue
        info = probe(file)
        stream = next((x for x in info.get("streams", []) if x.get("codec_type") == "video"), {})
        sidecar = _sidecar(file)
        if audit is not None and "camera_motion" in sidecar:
            # ``camera_motion`` was a dead field: nothing filled it, so every asset read
            # "unknown" while the planner scored on it and the director brief carried it.
            # It is gone from the model (R-11); a sidecar that still names it is noted so
            # the user is not left believing it does something.
            audit.record(
                "camera_motion", requested=sidecar["camera_motion"], effective=None,
                overridden_by="deprecated-key",
                reason=f"{file.name} 的 sidecar 携带已移除字段，已忽略",
            )
        metrics = _visual_quality(
            file, kind, int(stream.get("width", 0)), int(stream.get("height", 0)),
        )
        assets.append(MediaAsset(
            id=len(assets), file=file.resolve(), kind=kind,
            duration=float(info.get("format", {}).get("duration", 0)) if kind == "video" else float("inf"),
            width=int(stream.get("width", 0)), height=int(stream.get("height", 0)),
            tags=sidecar.get("tags", []) + _filename_tags(file.stem),
            description=sidecar.get("description", file.stem), mood=sidecar.get("mood", "neutral"),
            quality_score=float(sidecar.get("quality_score", metrics.quality)),
            dominant_color=list(sidecar.get("dominant_color", metrics.color)),
            shot_size=sidecar.get("shot_size", "unknown"),
            focus_point=_validated_focus(sidecar.get("focus_point", metrics.focus)),
            luma=float(sidecar.get("luma", metrics.luma)),
            edge_luma=float(sidecar.get("edge_luma", metrics.edge_luma)),
            noise_score=float(sidecar.get("noise_score", metrics.noise_score)),
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


def _visual_quality(
    file: Path, kind: str, width: int, height: int,
) -> VisualMetrics:
    resolution = min(1.0, math.sqrt(max(width * height, 1) / (1920 * 1080)))
    if kind == "video":
        # A video frame is not decoded here - that costs a seek per clip and the still
        # pass already runs once per asset. The readability pass samples a real frame
        # only for the shots that actually carry subtitles (see ``legibility.py``).
        return VisualMetrics(round(.45 + resolution * .4, 4), [128, 128, 128], [.5, .5], .5, .5, 0.0)
    try:
        image = Image.open(file).convert("RGB")
        image.thumbnail((256, 256))
        pixels = np.asarray(image, dtype=np.float32)
        luminance = pixels.mean(axis=2)
        exposure = 1 - min(1.0, abs(float(luminance.mean()) - 127.5) / 127.5)
        contrast = min(1.0, float(luminance.std()) / 64)
        detail = (float(np.abs(np.diff(luminance, axis=0)).mean())
                  + float(np.abs(np.diff(luminance, axis=1)).mean())) / 2
        sharpness = min(1.0, detail / 24)
        quality = .3 * resolution + .2 * exposure + .2 * contrast + .3 * sharpness
        color = np.median(pixels.reshape(-1, 3), axis=0).astype(int).tolist()
        luma = float(luminance.mean()) / 255
        edge_luma = _edge_luminance(luminance)
        noise_score = float(np.clip(detail / 32, 0, 1))
        return VisualMetrics(
            round(float(quality), 4), color, estimate_focus_point(image),
            round(float(np.clip(luma, 0, 1)), 4),
            round(float(np.clip(edge_luma, 0, 1)), 4),
            round(noise_score, 4),
        )
    except (OSError, ValueError):
        return VisualMetrics(round(.4 + resolution * .3, 4), [128, 128, 128], [.5, .5], .5, .5, 0.0)


def _edge_luminance(luminance: np.ndarray) -> float:
    """Mean brightness of the outer band of the frame, 0..1.

    The vignette gate wants to know how bright the *corners* are: a vignette on a picture
    that is already dark at the edges only crushes it, while on a bright one it brings the
    subject forward. Reading the border rather than the whole frame is what makes the two
    cases distinguishable - a dark subject on a bright wall should still get its vignette.
    """
    height, width = luminance.shape
    if height < 3 or width < 3:
        return float(luminance.mean()) / 255
    band_y = max(1, round(height * _EDGE_BAND))
    band_x = max(1, round(width * _EDGE_BAND))
    border = np.concatenate([
        luminance[:band_y, :].ravel(),
        luminance[-band_y:, :].ravel(),
        luminance[:, :band_x].ravel(),
        luminance[:, -band_x:].ravel(),
    ])
    return float(border.mean()) / 255


def estimate_focus_point(image: Image.Image) -> list[float]:
    """Estimate a conservative visual center from detail, contrast and color saliency."""
    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((192, 192))
    pixels = np.asarray(thumbnail, dtype=np.float32) / 255
    if pixels.size == 0:
        return [.5, .5]
    luma = pixels[..., 0] * .2126 + pixels[..., 1] * .7152 + pixels[..., 2] * .0722
    grad_x = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    grad_y = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    saturation_edges = (
        np.abs(np.diff(saturation, axis=1, prepend=saturation[:, :1]))
        + np.abs(np.diff(saturation, axis=0, prepend=saturation[:1, :]))
    )
    saliency = grad_x + grad_y + saturation_edges * .12
    height, width = saliency.shape
    yy, xx = np.mgrid[0:height, 0:width]
    center_prior = np.exp(-(((xx / max(width - 1, 1) - .5) / .48) ** 2 + ((yy / max(height - 1, 1) - .46) / .52) ** 2))
    saliency *= .45 + .55 * center_prior
    threshold = float(np.quantile(saliency, .72))
    weights = np.where(saliency >= threshold, saliency, 0)
    total = float(weights.sum())
    if total < 1e-6:
        return [.5, .5]
    focus_x = float((weights * xx).sum() / total / max(width - 1, 1))
    focus_y = float((weights * yy).sum() / total / max(height - 1, 1))
    return [round(float(np.clip(focus_x, .2, .8)), 4), round(float(np.clip(focus_y, .18, .82)), 4)]


def _validated_focus(value: object) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [round(float(np.clip(float(value[0]), 0, 1)), 4), round(float(np.clip(float(value[1]), 0, 1)), 4)]
        except (TypeError, ValueError):
            pass
    return [.5, .5]

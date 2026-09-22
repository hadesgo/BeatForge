"""How bright the picture is exactly where a line of lyrics is about to land.

A caption is only ever as readable as the backdrop behind it: white type on a bright
beach dissolves, the same type on a night scene needs almost no help. These helpers read
that backdrop straight from the *source* media - a plain average luminance over the
region a line occupies - so the subtitle pass can thicken a line's outline where the
picture is bright and leave it thin where the picture is dark, without ever loading a
model. Everything here is deterministic and CPU-only.

The region is measured *after* the line's placement is known, because the answer depends
on where the words land; the caller lays the line out first and asks this module how
bright the picture is under it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

#: The long edge the picture is reduced to before it is averaged. A backdrop average
#: does not need more resolution, and decoding a 24MP still just to average it is waste.
_SAMPLE_EDGE = 320

#: Returned when the backdrop cannot be read at all (a missing file, a broken video).
#: Mid-grey is the honest answer: it neither adds nor removes outline.
_UNKNOWN_LUMA = .5


def estimate_region_luma(
    file: Path, kind: str, focus: tuple[float, float] | None,
    region: tuple[float, float, float, float], cache: Path, *,
    at: float = 0.0,
) -> float:
    """Average luminance (0..1) of ``region`` in the picture at ``file``.

    ``region`` is a normalised bounding box ``(x0, y0, x1, y1)`` in ``[0, 1]`` - the band
    a caption sits in, or the box a free-layout line occupies. ``kind`` is the media kind
    (``"image"`` or ``"video"``); a video is reduced to a single frame at ``at`` seconds
    first, once per source, so a long clip is not decoded for every lyric line.

    ``focus`` is accepted for symmetry with the media pass and reserved for a future
    subject-weighted reading; the current reading is a flat average over the region.
    Any failure - an unreadable file, a missing codec - returns a neutral mid-grey rather
    than raising, because a lyric has to render even when its backdrop cannot be measured.
    """
    del focus  # the reading is flat over the region; focus is a placeholder for now
    # ``Shot.file`` is a plain str, and the video branch needs a real Path (``.stem``,
    # ``/``). The image branch would survive a str - which is exactly why this stayed
    # unnoticed until a project with video assets hit it.
    file = Path(file)
    frame = _frame_for(file, kind, cache, at)
    if frame is None:
        return _UNKNOWN_LUMA
    try:
        with Image.open(frame) as image:
            return _region_luma(image, region)
    except (OSError, ValueError):
        return _UNKNOWN_LUMA


def adaptive_outline(
    luma: float, base: float, *, bright: float = .55, max_outline: float = 2.4,
) -> float:
    """The outline width a backdrop of this luminance wants.

    The width ramps from ``base`` at ``bright`` and below up to ``max_outline`` on pure
    white: a night scene keeps the thin, unobtrusive edge the reference style has, while a
    blown-out sky gets enough weight to hold the letters apart from it. ``base`` is the
    project's chosen outline, so a project that wants no outline at all (``base == 0``)
    still gets none on a dark scene - and only the bright end of the ramp rescues it.
    """
    floor = max(0.0, float(base))
    ceiling = max(floor, float(max_outline))
    if luma <= bright:
        return round(floor, 2)
    span = max(1e-6, 1.0 - bright)
    weight = min(1.0, (luma - bright) / span)
    return round(floor + (ceiling - floor) * weight, 2)


def needs_local_dim(luma: float, *, threshold: float = .66) -> bool:
    """Is the backdrop bright enough that an outline alone will not carry the line?

    On a bright backdrop the outline helps but the letters can still sit in a wash of
    light; the caller dims the glyph-shaped patch of picture under such a line. The
    threshold is deliberately high - dimming is the exception, not the rule, and the
    reference look keeps the picture untouched wherever it can.
    """
    return luma >= threshold


def _frame_for(file: Path, kind: str, cache: Path, at: float) -> Path | None:
    """The still to measure: the image itself, or one frame pulled from a video.

    A video frame is written to ``cache`` keyed by the source's name and reused, so the
    same clip is decoded once no matter how many lines land on it.
    """
    if kind != "video":
        return file
    cache.mkdir(parents=True, exist_ok=True)
    frame = cache / f"{file.stem}.png"
    if frame.exists():
        return frame
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, at):.3f}",
             "-i", str(file), "-frames:v", "1", str(frame)],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return frame if frame.exists() else None


def _region_luma(image: Image.Image, region: tuple[float, float, float, float]) -> float:
    """Average a normalised box out of an opened image, on a reduced copy."""
    gray = image.convert("L")
    gray.thumbnail((_SAMPLE_EDGE, _SAMPLE_EDGE))
    width, height = gray.size
    if width <= 0 or height <= 0:
        return _UNKNOWN_LUMA
    x0, y0, x1, y1 = (min(max(float(value), 0.0), 1.0) for value in region)
    left, right = sorted((round(x0 * width), round(x1 * width)))
    top, bottom = sorted((round(y0 * height), round(y1 * height)))
    left, right = min(left, width - 1), max(right, left + 1)
    top, bottom = min(top, height - 1), max(bottom, top + 1)
    values = np.asarray(gray.crop((left, top, right, bottom)), dtype=np.float32)
    if values.size == 0:
        return _UNKNOWN_LUMA
    return float(values.mean() / 255.0)

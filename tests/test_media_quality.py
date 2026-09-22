from pathlib import Path

from PIL import Image

from beatforge.media import _visual_quality, estimate_focus_point


def test_visual_quality_extracts_color_and_score(tmp_path: Path) -> None:
    image = tmp_path / "red.jpg"
    Image.new("RGB", (1920, 1080), (200, 20, 20)).save(image)
    metrics = _visual_quality(image, "image", 1920, 1080)
    assert 0 <= metrics.quality <= 1
    assert metrics.color[0] > metrics.color[1]
    assert metrics.focus == [.5, .5]


def test_focus_estimation_stays_conservative_for_off_center_detail() -> None:
    image = Image.new("RGB", (400, 200), (20, 20, 20))
    for x in range(280, 350):
        for y in range(40, 160):
            image.putpixel((x, y), (235, 220, 60))
    focus = estimate_focus_point(image)
    assert .5 < focus[0] <= .8
    assert .18 <= focus[1] <= .82


def test_edge_luminance_separates_a_bright_border_from_a_dark_one(tmp_path: Path) -> None:
    """The vignette gate reads the *border*, not the whole frame.

    A vignette darkens the corners. On a frame whose border is already dark - a subject
    lit against black - pulling the edges down further only crushes them, while the same
    move on a bright wall brings the subject forward. The two cases have to be told apart
    by the edge brightness, so a bright middle over a dark border must score low.
    """
    dark = tmp_path / "dark-border.jpg"
    image = Image.new("RGB", (256, 256), (10, 10, 10))
    for x in range(80, 176):
        for y in range(80, 176):
            image.putpixel((x, y), (240, 240, 240))
    image.save(dark)
    bright = tmp_path / "bright-border.jpg"
    Image.new("RGB", (256, 256), (235, 235, 235)).save(bright)

    dark_metrics = _visual_quality(dark, "image", 256, 256)
    bright_metrics = _visual_quality(bright, "image", 256, 256)

    assert dark_metrics.edge_luma < .2 < bright_metrics.edge_luma
    # The whole-frame luma is dominated by the middle, so it must not be the gate.
    assert dark_metrics.luma > dark_metrics.edge_luma


def test_a_flat_frame_reads_as_cleaner_than_a_noisy_one(tmp_path: Path) -> None:
    """``noise_score`` is the grain gate: a source that already carries noise is left alone.

    A smooth gradient has no high-frequency energy, so it can take the film grain the
    look asks for. A frame already full of compression noise scores high and must not be
    given more - stacking grain on grain is most of what made double-compressed photos
    read as "dirty".
    """
    import numpy as np

    smooth = tmp_path / "smooth.jpg"
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    Image.fromarray(np.stack([ramp] * 3, axis=2)).save(smooth, quality=100)
    noisy = tmp_path / "noisy.jpg"
    rng = np.random.default_rng(0)
    speckle = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    Image.fromarray(np.stack([speckle] * 3, axis=2)).save(noisy, quality=100)

    smooth_score = _visual_quality(smooth, "image", 256, 256).noise_score
    noisy_score = _visual_quality(noisy, "image", 256, 256).noise_score

    assert smooth_score < noisy_score
    assert noisy_score > .5, "a frame of noise should read as noisy"


def test_video_assets_get_safe_default_metrics(tmp_path: Path) -> None:
    """A video is not decoded for the still pass; its metrics must stay neutral, not crash."""
    metrics = _visual_quality(tmp_path / "clip.mp4", "video", 1920, 1080)
    assert 0 <= metrics.quality <= 1
    assert metrics.luma == .5 and metrics.edge_luma == .5 and metrics.noise_score == 0.0

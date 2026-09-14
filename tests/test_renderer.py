from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image
import pytest
import soundfile as sf

from beatforge.config import RenderConfig
from beatforge.audio import AudioAnalysis
from beatforge.director import create_art_direction
from beatforge.lyrics import LyricLine
from beatforge.planner import Shot, ShotLayer
from beatforge.renderer import (
    _image_filter_graph,
    _render_shot,
    _section_color_filter,
    _shot_match_filter,
    _video_encode_args,
    render,
)
from beatforge.runtime import duration


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_end_to_end_renderer_without_models(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 360), (30, 80, 140)).save(image)
    image_2 = tmp_path / "frame-2.jpg"
    Image.new("RGB", (640, 360), (180, 90, 40)).save(image_2)
    image_3 = tmp_path / "frame-3.jpg"
    Image.new("RGB", (640, 360), (50, 150, 80)).save(image_3)
    music = tmp_path / "music.wav"
    sample_rate = 22_050
    time = np.arange(sample_rate * 3) / sample_rate
    sf.write(music, (.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)
    shots = [
        Shot(0, 0, 1, 1, 0, str(image), "image", 0, "测试", .5, "steady", "cut", .8),
        Shot(1, 1, 2, 1, 1, str(image_2), "image", 0, "字幕", .7, "dynamic", "dissolve", .7),
        Shot(2, 2, 3, 1, 2, str(image_3), "image", 0, "成片", .4, "gentle", "none", .6),
    ]
    output = tmp_path / "output.mp4"
    config = RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast", subtitle_size=20)
    lyrics = [LyricLine(0, 1, "测试"), LyricLine(1, 2, "字幕"), LyricLine(2, 3, "成片")]
    analysis = AudioAnalysis(
        duration=3, bpm=100, beats=[0, 1, 2, 3], sections=[0, 3],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="uplifting", mood_scores={"uplifting": 1},
    )
    art = create_art_direction(analysis, lyrics, config)
    render(shots, lyrics, music, output, tmp_path / "cache", config, art)
    assert output.exists()
    assert (tmp_path / "cache" / "lyrics.ass").exists()
    assert 2.9 <= duration(output) <= 3.1


def test_director_color_arc_and_shot_matching_become_filters() -> None:
    shot = Shot(
        0, 0, 2, 2, 0, "frame.jpg", "image", 0, "", .5, "steady", "cut", .5,
        section_index=1, source_color=[210, 130, 70],
    )
    analysis = AudioAnalysis(
        duration=2, bpm=90, beats=[], sections=[0, 2], energy_times=[0],
        energy_values=[.4], average_energy=.4, brightness=.5,
        mood="cinematic", mood_scores={"cinematic": 1},
    )
    art = create_art_direction(analysis, [], RenderConfig())
    art.color_arc = ["cold blue", "warm amber"]

    assert "eq=brightness=" in _shot_match_filter(shot, .3)
    assert "colorbalance=" in _section_color_filter(shot, art, 2, .72)


def test_intermediate_encoding_uses_higher_quality_crf() -> None:
    cfg = RenderConfig(crf=19, intermediate_crf=13, encoder_tune="film")
    args = _video_encode_args(cfg, intermediate=True)
    assert args[args.index("-crf") + 1] == "13"
    assert args[args.index("-tune") + 1] == "film"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.parametrize("effect", [
    "cinematic_depth", "focus_pull", "pan_reveal", "split_screen",
    "photo_stack", "double_exposure", "beat_montage",
])
def test_all_still_image_effects_render(effect: str, tmp_path: Path) -> None:
    files = []
    for index, size in enumerate(((240, 360), (420, 180), (240, 240))):
        file = tmp_path / f"image-{index}.jpg"
        Image.new("RGB", size, (40 + index * 70, 80, 150 - index * 40)).save(file)
        files.append(file)
    layers = [
        ShotLayer(index, str(files[index]))
        for index in range(1, 3)
    ] if effect in {"split_screen", "photo_stack", "double_exposure", "beat_montage"} else []
    shot = Shot(
        0, 0, .5, .5, 0, str(files[0]), "image", 0, "", .75,
        "dynamic", "none", .8, melody=.7, image_effect=effect, layers=layers,
    )
    cfg = RenderConfig(
        width=160, height=90, fps=10, crf=35, preset="ultrafast",
        image_background_blur=4, film_grain=0, vignette=False,
    )
    analysis = AudioAnalysis(
        duration=.5, bpm=120, beats=[0, .5], sections=[0, .5],
        energy_times=[0], energy_values=[.7], average_energy=.7,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
    )
    output = tmp_path / f"{effect}.mp4"

    _render_shot(shot, output, cfg, create_art_direction(analysis, [], cfg), .5, 1)

    assert output.exists()
    assert .4 <= duration(output) <= .6


def _camera_move(shot: Shot, cfg: RenderConfig) -> str:
    art = create_art_direction(
        AudioAnalysis(
            duration=4, bpm=100, beats=[0, 1, 2, 3, 4], sections=[0, 4],
            energy_times=[0], energy_values=[.5], average_energy=.5,
            brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        ),
        [], cfg,
    )
    filters, _ = _image_filter_graph(shot, cfg, art, shot.duration, 1)
    return next(item for item in filters if "perspective" in item or "zoompan" in item)


def _evaluate_quad(filter_string: str, cfg: RenderConfig, frame: int) -> dict[str, float]:
    """Evaluate the eight crop coordinates the way FFmpeg's eval would."""
    points = {name: expr for name, expr in re.findall(r"([xy][0-3])='([^']*)'", filter_string)}
    assert len(points) == 8, filter_string

    def clip(value, low, high):
        return max(low, min(high, value))

    namespace = {"W": cfg.width, "H": cfg.height, "on": frame, "clip": clip}
    return {name: float(eval(expr, {"__builtins__": {"max": max, "min": min}}, namespace))
            for name, expr in points.items()}


@pytest.mark.parametrize("effect,intent", [
    ("pan_reveal", "breathe"),
    ("focus_pull", "continuity"),
    ("cinematic_depth", "impact"),
])
def test_camera_move_crops_the_full_frame_without_quantising(effect: str, intent: str) -> None:
    """The zoom must crop ``W/zoom``, not the ``W - W/zoom`` travel range.

    Confusing the two silently turns a 5% drift into a hard punch-in, and
    ``zoompan`` (which truncates the crop origin to whole pixels) would make a
    sub-pixel-per-frame move stutter instead.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 3, 3, 0, "frame.jpg", "image", 0, "", .6, "dynamic", "none", .8,
        melody=.5, edit_intent=intent, image_effect=effect,
    )

    filter_string = _camera_move(shot, cfg)

    assert "zoompan" not in filter_string, "zoompan snaps the crop origin to whole pixels"
    assert "sense=source" in filter_string
    assert "eval=frame" in filter_string

    intensity = max(.35, create_art_direction(
        AudioAnalysis(
            duration=3, bpm=100, beats=[], sections=[0, 3], energy_times=[0],
            energy_values=[.5], average_energy=.5, brightness=.5,
            mood="cinematic", mood_scores={"cinematic": 1},
        ), [], cfg).camera_intensity) * (1 + .5 * .18)
    amount = ({"pan_reveal": .075, "focus_pull": .045}.get(effect, .055)
              * intensity * {"dynamic": 1.35, "gentle": .7}.get("dynamic", 1.0))

    frames = round(3 * cfg.fps)
    for frame in (0, frames // 2, frames):
        quad = _evaluate_quad(filter_string, cfg, frame)
        width = quad["x1"] - quad["x0"]
        height = quad["y2"] - quad["y0"]
        # A rectangle, not a trapezoid.
        assert quad["x3"] - quad["x2"] == pytest.approx(width, abs=1e-6)
        assert quad["y3"] - quad["y1"] == pytest.approx(height, abs=1e-6)
        # Never wider than the frame, never smaller than the deepest zoom.
        assert cfg.width / (1 + amount) - .5 <= width <= cfg.width + 1e-6
        assert cfg.height / (1 + amount) - .5 <= height <= cfg.height + 1e-6
        # Always inside the frame, so nothing is sampled out of bounds.
        assert -1e-6 <= quad["x0"] and quad["x3"] <= cfg.width + 1e-6
        assert -1e-6 <= quad["y0"] and quad["y3"] <= cfg.height + 1e-6
        # The crop must keep the frame's aspect ratio or the image is distorted.
        assert width / height == pytest.approx(cfg.width / cfg.height, rel=1e-3)


def test_camera_move_never_jumps_more_than_the_intended_step() -> None:
    """Adjacent frames must move by a small, smooth amount.

    ``zoompan`` produced steps of a whole input pixel while the intended move was
    a tenth of that, which is what made stills visibly shake. Guard the invariant
    directly: no single frame may move more than a few times the average step.
    """
    cfg = RenderConfig(width=1920, height=1080, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 4, 4, 0, "frame.jpg", "image", 0, "", .5, "dynamic", "none", .8,
        melody=.5, edit_intent="breathe", image_effect="pan_reveal",
    )
    filter_string = _camera_move(shot, cfg)

    frames = round(4 * cfg.fps)
    centres = []
    for frame in range(frames + 1):
        quad = _evaluate_quad(filter_string, cfg, frame)
        centres.append(((quad["x0"] + quad["x3"]) / 2, (quad["y0"] + quad["y3"]) / 2))
    steps = np.abs(np.diff(np.array(centres), axis=0))
    average = steps.mean()
    assert average > 0, "the shot should actually move"
    assert steps.max() <= max(average * 6, .5), (
        f"largest step {steps.max():.3f}px vs average {average:.3f}px"
    )

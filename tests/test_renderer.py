from pathlib import Path
import math
import re
import shutil
import subprocess

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
    _CAMERA_MOVES,
    _MIN_ZOOM,
    _TRANSITION_LIBRARY,
    _image_filter_graph,
    _render_shot,
    _section_color_filter,
    _shot_match_filter,
    _transition_spec,
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
@pytest.mark.parametrize("effect", sorted(set(_CAMERA_MOVES) | {
    "split_screen", "photo_stack", "double_exposure", "beat_montage",
    "film_bars", "iris", "parallax",
}))
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
    # Every effect has to deliver the shot's full frame count. The iris composites
    # through generated planes, and letting one of those be the shortest stream
    # silently costs the shot its last frame - one frame short of every other effect.
    assert round(duration(output) * cfg.fps) == round(.5 * cfg.fps)


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
    """Evaluate the eight crop coordinates the way FFmpeg's eval would.

    The namespace has to carry FFmpeg's own helpers (``pow``, ``cos``, ``PI``), or
    an eased or breathing move cannot be evaluated at all.
    """
    points = {name: expr for name, expr in re.findall(r"([xy][0-3])='([^']*)'", filter_string)}
    assert len(points) == 8, filter_string

    def clip(value, low, high):
        return max(low, min(high, value))

    namespace = {
        "W": cfg.width, "H": cfg.height, "on": frame, "clip": clip,
        "pow": pow, "cos": math.cos, "sqrt": math.sqrt, "PI": math.pi,
    }
    builtins = {"max": max, "min": min, "abs": abs}
    return {name: float(eval(expr, {"__builtins__": builtins}, namespace))
            for name, expr in points.items()}


def _crop_geometry(filter_string: str, cfg: RenderConfig, frames: int) -> dict[str, np.ndarray]:
    """Per-frame crop centre and size, read back from the filter's own expressions.

    ``perspective`` numbers its frames from 1 (``on`` is ``frame_count_in + 1``), so
    a shot of ``frames`` frames is sampled at ``on = 1 .. frames``. Sampling from
    ``on = 0`` would evaluate one frame the renderer never emits and hide exactly the
    off-by-one this helper exists to catch.
    """
    centres_x, centres_y, widths, heights = [], [], [], []
    for frame in range(1, frames + 1):
        quad = _evaluate_quad(filter_string, cfg, frame)
        centres_x.append((quad["x0"] + quad["x3"]) / 2)
        centres_y.append((quad["y0"] + quad["y3"]) / 2)
        widths.append(quad["x1"] - quad["x0"])
        heights.append(quad["y2"] - quad["y0"])
    return {
        "x": np.array(centres_x), "y": np.array(centres_y),
        "width": np.array(widths), "height": np.array(heights),
    }


@pytest.mark.parametrize("effect", sorted(_CAMERA_MOVES))
def test_every_camera_move_crops_the_full_frame_without_quantising(effect: str) -> None:
    """The zoom must crop ``W/zoom``, not the ``W - W/zoom`` travel range.

    Confusing the two silently turns a 5% drift into a hard punch-in, and
    ``zoompan`` (which truncates the crop origin to whole pixels) would make a
    sub-pixel-per-frame move stutter instead. Every move in the table has to hold
    both invariants, not just the three that happened to exist first.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 3, 3, 0, "frame.jpg", "image", 0, "", .6, "dynamic", "none", .8,
        melody=.5, edit_intent="continuity", image_effect=effect,
    )

    filter_string = _camera_move(shot, cfg)

    assert "zoompan" not in filter_string, "zoompan snaps the crop origin to whole pixels"
    assert "sense=source" in filter_string
    assert "eval=frame" in filter_string

    move = _CAMERA_MOVES[effect]
    # ``amount`` is intensity times motion scale, and both stay well under 1.5.
    # Every move also carries the renderer's ``_MIN_ZOOM`` lift, so the deepest crop
    # is one hair tighter than the move's own magnification suggests.
    deepest = _MIN_ZOOM + max(move.zoom_from, move.zoom_to) * 1.5
    frames = round(3 * cfg.fps)
    geometry = _crop_geometry(filter_string, cfg, frames)

    for frame in (1, frames // 2, frames):
        quad = _evaluate_quad(filter_string, cfg, frame)
        width = quad["x1"] - quad["x0"]
        height = quad["y2"] - quad["y0"]
        # A rectangle, not a trapezoid.
        assert quad["x3"] - quad["x2"] == pytest.approx(width, abs=1e-6)
        assert quad["y3"] - quad["y1"] == pytest.approx(height, abs=1e-6)
        # Never wider than the frame, never smaller than the deepest zoom.
        assert cfg.width / (1 + deepest) - .5 <= width <= cfg.width + 1e-6
        assert cfg.height / (1 + deepest) - .5 <= height <= cfg.height + 1e-6
        # Always inside the frame, so nothing is sampled out of bounds.
        assert -1e-6 <= quad["x0"] and quad["x3"] <= cfg.width + 1e-6
        assert -1e-6 <= quad["y0"] and quad["y3"] <= cfg.height + 1e-6
        # The crop must keep the frame's aspect ratio or the image is distorted.
        assert width / height == pytest.approx(cfg.width / cfg.height, rel=1e-3)

    # ... and it has to change over the shot, or the "move" is a still frame.
    travel = max(np.ptp(geometry[name]) for name in ("x", "y", "width"))
    assert travel > 1, f"{effect} never moves"


@pytest.mark.parametrize("effect", sorted(_CAMERA_MOVES))
def test_every_camera_move_is_smooth(effect: str) -> None:
    """Adjacent frames must move by a small, smooth amount.

    ``zoompan`` produced steps of a whole input pixel while the intended move was a
    tenth of that, which is what made stills visibly shake. Guard the invariant on
    every channel that actually moves: no single frame may travel more than a few
    times the average step.
    """
    cfg = RenderConfig(width=1920, height=1080, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 4, 4, 0, "frame.jpg", "image", 0, "", .5, "dynamic", "none", .8,
        melody=.5, edit_intent="continuity", image_effect=effect,
    )
    filter_string = _camera_move(shot, cfg)
    geometry = _crop_geometry(filter_string, cfg, round(4 * cfg.fps))

    moved = False
    for name, series in geometry.items():
        steps = np.abs(np.diff(series))
        if steps.sum() <= .5:
            continue
        moved = True
        average = steps.mean()
        assert steps.max() <= max(average * 6, .5), (
            f"{effect}.{name}: largest step {steps.max():.3f}px vs average {average:.3f}px"
        )
    assert moved, f"{effect} does not move at all"


def test_a_breathe_move_returns_to_where_it_started() -> None:
    """A breathe swells and comes back; a push does not.

    The shape is the whole point of the move - if it only ramps one way it is just
    a slower dolly, and the release at the end of the phrase disappears.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 4, 4, 0, "frame.jpg", "image", 0, "", .5, "steady", "none", .8,
        melody=.5, edit_intent="continuity", image_effect="breathe",
    )
    geometry = _crop_geometry(_camera_move(shot, cfg), cfg, round(4 * cfg.fps))
    widths = geometry["width"]

    assert widths[0] == pytest.approx(widths[-1], abs=.5)
    # The shape is the point: it leaves the framing, peaks mid-shot and returns.
    # Which way it goes is the move's business, so measure the excursion from the
    # endpoints rather than assuming the frame widens.
    excursion = np.abs(widths - widths[0])
    assert excursion.max() > 5, "the frame should visibly swell"
    assert excursion.argmax() not in {0, len(widths) - 1}, "the peak belongs mid-shot"


@pytest.mark.parametrize("effect", sorted(_CAMERA_MOVES))
def test_every_camera_move_holds_its_framing_once_the_shot_is_over(effect: str) -> None:
    """``on`` counts from 1 *and* keeps counting past the end of the shot.

    ``perspective`` numbers frames as ``outl->frame_count_in + 1``, and a still is fed
    from ``-loop 1``, so filters further down the chain (``fps``, ``trim``) pull frames
    that lie beyond the shot to settle their own timestamps. An unclamped
    ``on/(frames-1)`` therefore walks the curve straight past its end - and for a move
    that finishes at 1.0 magnification those extra frames push the zoom under 1, which
    turns ``W - W/zoom`` negative and makes FFmpeg reject the frame outright. Holding
    the last value is free: those frames are trimmed away.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 3, 3, 0, "frame.jpg", "image", 0, "", .6, "dynamic", "none", .8,
        melody=.5, edit_intent="continuity", image_effect=effect,
    )
    filter_string = _camera_move(shot, cfg)
    frames = round(3 * cfg.fps)

    last = _evaluate_quad(filter_string, cfg, frames)
    for frame in (frames + 1, frames + 12, frames * 3):
        quad = _evaluate_quad(filter_string, cfg, frame)
        for name, value in last.items():
            assert quad[name] == pytest.approx(value, abs=1e-6), (
                f"{effect}.{name} keeps moving after the shot has ended"
            )

    # The other end of the same clamp: nothing may be evaluated before frame one.
    first = _evaluate_quad(filter_string, cfg, 1)
    for name, value in first.items():
        assert _evaluate_quad(filter_string, cfg, 0)[name] == pytest.approx(value, abs=1e-6)

    # At every one of those frames the crop is still a rectangle inside the frame.
    for frame in (1, frames // 2, frames, frames * 3):
        quad = _evaluate_quad(filter_string, cfg, frame)
        assert quad["x0"] >= -1e-6 and quad["x3"] <= cfg.width + 1e-6
        assert quad["y0"] >= -1e-6 and quad["y3"] <= cfg.height + 1e-6


def _frame_brightness(video: Path, frame: int, tmp_path: Path) -> float:
    """Mean luminance of one rendered frame, addressed by frame number."""
    target = tmp_path / f"grab-{frame:04d}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vf", rf"select=eq(n\,{frame})", "-vsync", "0", "-frames:v", "1", str(target)],
        check=True, capture_output=True,
    )
    return float(np.asarray(Image.open(target).convert("L"), dtype=float).mean())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_the_iris_mask_opens_from_a_keyhole_to_the_whole_frame(tmp_path: Path) -> None:
    """The mask has to travel - a still circle would just be a vignette.

    It is also built at an eighth of the canvas and scaled up, which is the only
    reason the effect is affordable: ``geq`` evaluates per pixel, per frame.
    """
    image = tmp_path / "flat.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(image)
    shot = Shot(
        0, 0, .8, .8, 0, str(image), "image", 0, "", .6, "steady", "none", .8,
        melody=.5, image_effect="iris",
    )
    cfg = RenderConfig(
        width=320, height=180, fps=10, crf=28, preset="ultrafast",
        image_background_blur=0, image_foreground_scale=1.0,
        film_grain=0, vignette=False,
    )
    analysis = AudioAnalysis(
        duration=.8, bpm=100, beats=[0, .8], sections=[0, .8], energy_times=[0],
        energy_values=[.5], average_energy=.5, brightness=.5,
        mood="cinematic", mood_scores={"cinematic": 1},
    )
    output = tmp_path / "iris.mp4"
    _render_shot(shot, output, cfg, create_art_direction(analysis, [], cfg), .8, 1)

    frames = round(duration(output) * cfg.fps)
    start = _frame_brightness(output, 0, tmp_path)
    middle = _frame_brightness(output, frames // 2, tmp_path)
    end = _frame_brightness(output, frames - 1, tmp_path)

    assert start < middle < end, f"the iris is not opening: {start:.1f} {middle:.1f} {end:.1f}"
    assert start < 40, "the first frame should still be a keyhole"
    assert end > 150, "the last frame should be clear of the mask"


def _transition_art(tone: str = "neutral"):
    analysis = AudioAnalysis(
        duration=8, bpm=100, beats=[0, 2, 4, 6, 8], sections=[0, 8], energy_times=[0],
        energy_values=[.5], average_energy=.5, brightness=.5,
        mood="cinematic", mood_scores={"cinematic": 1},
    )
    art = create_art_direction(analysis, [], RenderConfig())
    art.transition_tone = tone
    return art


def _transition_shots(family: str, media_id: int = 5, section_index: int = 2) -> tuple[Shot, Shot]:
    shot = Shot(3, 0, 2, 2, media_id, "frame.jpg", "image", 0, "", .8, "dynamic", family, .7,
                section_index=section_index)
    following = Shot(4, 2, 4, 2, media_id + 1, "frame.jpg", "image", 0, "", .8, "dynamic", "cut", .7,
                     section_index=section_index)
    return shot, following


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_every_transition_name_is_a_real_xfade_transition() -> None:
    """A typo in the library fails minutes into a render, on a graph nobody reads."""
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "filter=xfade"],
        capture_output=True, text=True, check=True,
    ).stdout
    supported = set(re.findall(r"^\s+(\w+)\s+-?\d+\s", listing, re.MULTILINE))
    assert {"dissolve", "wipeleft", "zoomin"} <= supported, listing
    for family, names in _TRANSITION_LIBRARY.items():
        for name in names:
            assert name in supported, f"{family}: {name!r} is not an xfade transition"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_every_transition_in_the_library_actually_composes(tmp_path: Path) -> None:
    """A name has to survive a real ``xfade``, not merely appear in ``-h``.

    This library is the entire vocabulary the planner can reach for, and a transition
    that only breaks once it meets two real streams would not show up until a full
    render was already minutes in.
    """
    clips = []
    for name, colour in (("a", "red"), ("b", "blue")):
        file = tmp_path / f"{name}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"color=c={colour}:s=64x36:r=10:d=0.6", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "40", "-pix_fmt", "yuv420p", str(file)],
            check=True, capture_output=True,
        )
        clips.append(file)

    for family, names in _TRANSITION_LIBRARY.items():
        for name in names:
            output = tmp_path / f"{name}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(clips[0]), "-i", str(clips[1]),
                 "-filter_complex",
                 f"[0:v][1:v]xfade=transition={name}:duration=0.3:offset=0.3[v]",
                 "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40",
                 "-pix_fmt", "yuv420p", str(output)],
                check=True, capture_output=True,
            )
            assert output.exists() and duration(output) > 0, f"{family}: {name}"


@pytest.mark.parametrize("family", sorted({*_TRANSITION_LIBRARY, "cut"}))
def test_transition_families_resolve_within_the_configured_bounds(family: str) -> None:
    cfg = RenderConfig(transition_min_seconds=.16, transition_max_seconds=.55)
    shot, following = _transition_shots(family)
    name, seconds = _transition_spec(shot, following, _transition_art(), cfg)

    if family == "cut":
        assert (name, seconds) == ("cut", 0.0)
        return
    assert name in _TRANSITION_LIBRARY[family]
    assert cfg.transition_min_seconds <= seconds <= cfg.transition_max_seconds


@pytest.mark.parametrize("tone,expected", [("bright", "fadewhite"), ("dark", "fadeblack")])
def test_a_dip_takes_its_colour_from_the_tone(tone: str, expected: str) -> None:
    """A dip to white and a dip to black are different editorial statements."""
    cfg = RenderConfig()
    shot, following = _transition_shots("dip")
    name, _ = _transition_spec(shot, following, _transition_art(tone=tone), cfg)
    assert name == expected


def test_directional_transitions_follow_the_shots_own_drift() -> None:
    """A wipe that fights the camera move reads as a mistake, not as a flourish."""
    cfg = RenderConfig()
    art = _transition_art()
    names = {
        _transition_spec(*_transition_shots("wipe", media_id=media_id), art, cfg)[0]
        for media_id in range(4)
    }
    assert names == {"wipeleft", "wiperight"}

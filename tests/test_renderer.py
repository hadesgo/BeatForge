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
    _EFFECT_TRANSITIONS,
    _KNOCKOUT_DIM,
    _KNOCKOUT_LIFT,
    _MIN_ZOOM,
    _TRANSITION_LIBRARY,
    _image_filter_graph,
    _knockout_graph,
    _render_shot,
    _section_color_filter,
    _shot_match_filter,
    _transition_effect_filters,
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
def test_end_to_end_renderer_composes_effect_transitions(tmp_path: Path) -> None:
    """The effect transitions have to survive the *composer*, not just the shot.

    They carry no handle, so the timeline butts the two shots straight together
    instead of overlapping them. A graph that still assumed an overlap would either
    fail outright or quietly swallow a shot, and the total length is what shows it.
    """
    files = []
    for index in range(3):
        image = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (320, 180), (30 + index * 60, 80, 150 - index * 40)).save(image)
        files.append(str(image))
    music = tmp_path / "music.wav"
    sample_rate = 22_050
    time = np.arange(sample_rate * 3) / sample_rate
    sf.write(music, (.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)
    shots = [
        Shot(0, 0, 1, 1, 0, files[0], "image", 0, "测试", .9, "dynamic", "glitch", .8,
             melody=.7, image_effect="whip_pan"),
        Shot(1, 1, 2, 1, 1, files[1], "image", 0, "字幕", .9, "dynamic", "film_burn", .7,
             melody=.7, image_effect="tilt3d_back"),
        Shot(2, 2, 3, 1, 2, files[2], "image", 0, "成片", .4, "gentle", "none", .6,
             melody=.5, image_effect="spiral_in"),
    ]
    cfg = RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast",
                       film_grain=0, vignette=False)
    analysis = AudioAnalysis(
        duration=3, bpm=100, beats=[0, 1, 2, 3], sections=[0, 3],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="uplifting", mood_scores={"uplifting": 1},
    )
    lyrics = [LyricLine(0, 1, "测试"), LyricLine(1, 2, "字幕"), LyricLine(2, 3, "成片")]
    output = tmp_path / "output.mp4"
    render(shots, lyrics, music, output, tmp_path / "cache", cfg,
           create_art_direction(analysis, lyrics, cfg))

    assert output.exists()
    assert 2.8 <= duration(output) <= 3.2, "the effect transitions shifted the timeline"


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


def _image_filters(shot: Shot, cfg: RenderConfig) -> list[str]:
    art = create_art_direction(
        AudioAnalysis(
            duration=4, bpm=100, beats=[0, 1, 2, 3, 4], sections=[0, 4],
            energy_times=[0], energy_values=[.5], average_energy=.5,
            brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        ),
        [], cfg,
    )
    filters, _ = _image_filter_graph(shot, cfg, art, shot.duration, 1)
    return filters


def _camera_move(shot: Shot, cfg: RenderConfig) -> str:
    return next(item for item in _image_filters(shot, cfg)
                if "perspective" in item or "zoompan" in item)


def _evaluate_quad(filter_string: str, cfg: RenderConfig, frame: int) -> dict[str, float]:
    """Evaluate the eight crop coordinates the way FFmpeg's eval would.

    The namespace has to carry FFmpeg's own helpers (``pow``, ``cos``, ``sin``,
    ``PI``), or an eased, breathing, rolling or handheld move cannot be evaluated at
    all.
    """
    points = {name: expr for name, expr in re.findall(r"([xy][0-3])='([^']*)'", filter_string)}
    assert len(points) == 8, filter_string

    def clip(value, low, high):
        return max(low, min(high, value))

    namespace = {
        "W": cfg.width, "H": cfg.height, "on": frame, "clip": clip,
        "pow": pow, "cos": math.cos, "sin": math.sin, "sqrt": math.sqrt, "PI": math.pi,
    }
    builtins = {"max": max, "min": min, "abs": abs}
    return {name: float(eval(expr, {"__builtins__": builtins}, namespace))
            for name, expr in points.items()}


def _corners(quad: dict[str, float]) -> list[tuple[float, float]]:
    """The quad's four corners in *cycle* order: top-left, top-right, bottom-right, bottom-left.

    The filter names them ``x0..x3`` as top-left, top-right, bottom-left, bottom-right,
    which is a bow tie when read in index order - so the last two are swapped here.
    Area and turn direction are only meaningful on a proper cycle.
    """
    return [
        (quad["x0"], quad["y0"]),
        (quad["x1"], quad["y1"]),
        (quad["x3"], quad["y3"]),
        (quad["x2"], quad["y2"]),
    ]


def _quad_area(corners: list[tuple[float, float]]) -> float:
    """Shoelace area, signed so a self-crossing quad shows up as near zero."""
    return sum(
        corners[i][0] * corners[(i + 1) % 4][1] - corners[(i + 1) % 4][0] * corners[i][1]
        for i in range(4)
    ) / 2


def _turn(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Cross product at ``b``: positive, negative or zero for left, right or straight."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


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


def test_the_whip_pan_travels_far_enough_to_be_a_whip() -> None:
    """A "whip" that moves eight pixels is a drift, and a drift needs no smear.

    A pan can only travel ``W - W/zoom``, so the distance available is set by how
    deeply the shot is cropped - which is why this move crops hard. Both halves
    matter: the distance is what makes it a throw, and the ``tmix`` drag is what keeps
    that speed from reading as a jump cut. Sizing it from a shallow crop is the
    mistake this guards against, and nothing else in the graph would complain.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    shot = Shot(
        0, 0, 2, 2, 0, "frame.jpg", "image", 0, "", .8, "dynamic", "none", .8,
        melody=.7, image_effect="whip_pan",
    )
    filters = _image_filters(shot, cfg)
    assert any("tmix" in item for item in filters), "the whip pan lost its drag"

    geometry = _crop_geometry(
        next(item for item in filters if "perspective" in item), cfg, round(2 * cfg.fps)
    )
    travel = float(np.ptp(geometry["x"]))
    assert travel > cfg.width * .05, f"the whip only travels {travel:.1f}px"


def test_a_drag_is_only_asked_for_by_moves_that_need_one() -> None:
    """``tmix`` costs a frame buffer and softens whatever it touches.

    Only a move that crosses several pixels in a single frame earns it; putting it on
    a slow push would just make the shot look out of focus.
    """
    cfg = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
    dragged = set()
    for effect in sorted(_CAMERA_MOVES):
        shot = Shot(
            0, 0, 2, 2, 0, "frame.jpg", "image", 0, "", .6, "dynamic", "none", .8,
            melody=.5, image_effect=effect,
        )
        if any("tmix" in item for item in _image_filters(shot, cfg)):
            dragged.add(effect)
    assert dragged == {"whip_pan"}, f"unexpected drag on {sorted(dragged - {'whip_pan'})}"


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
    frames = round(3 * cfg.fps)
    geometry = _crop_geometry(filter_string, cfg, frames)

    for frame in (1, frames // 2, frames):
        quad = _evaluate_quad(filter_string, cfg, frame)
        corners = _corners(quad)
        # A real quadrilateral, not a bow tie: every turn has to go the same way.
        area = _quad_area(corners)
        assert abs(area) > cfg.width * cfg.height * .2, f"{effect} collapses to nothing"
        turns = [
            _turn(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4])
            for i in range(4)
        ]
        assert all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns), (
            f"{effect} folds over itself"
        )
        # Always inside the frame, so nothing is sampled out of bounds. A roll or a
        # keystone swings the corners away from the crop box, so this is the check
        # that the travel range was sized from the quad's bounding box and not from
        # W/zoom.
        for x, y in corners:
            assert -1e-6 <= x <= cfg.width + 1e-6, f"{effect} samples outside the frame"
            assert -1e-6 <= y <= cfg.height + 1e-6, f"{effect} samples outside the frame"

        if move.roll:
            # Rolling keeps the rectangle, so measure its own edges: the axis-aligned
            # extents of a turned rectangle say nothing about its shape.
            width = (math.dist(corners[0], corners[1]) + math.dist(corners[3], corners[2])) / 2
            height = (math.dist(corners[0], corners[3]) + math.dist(corners[1], corners[2])) / 2
        else:
            # Without a roll the quad stays axis-aligned and a keystone is a trapezoid,
            # so average the opposite edges - that recovers W/zoom and H/zoom exactly.
            width = ((quad["x1"] - quad["x0"]) + (quad["x3"] - quad["x2"])) / 2
            height = ((quad["y2"] - quad["y0"]) + (quad["y3"] - quad["y1"])) / 2

        # Never wider than the frame, never so tight that the move stops reading as a
        # move rather than as a crop.
        assert cfg.width * .45 <= width <= cfg.width + 1e-6
        assert cfg.height * .45 <= height <= cfg.height + 1e-6
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


@pytest.mark.parametrize("family", sorted({*_TRANSITION_LIBRARY, *_EFFECT_TRANSITIONS, "cut"}))
def test_transition_families_resolve_within_the_configured_bounds(family: str) -> None:
    cfg = RenderConfig(transition_min_seconds=.16, transition_max_seconds=.55)
    shot, following = _transition_shots(family)
    transition = _transition_spec(shot, following, _transition_art(), cfg)

    if family == "cut":
        assert (transition.kind, transition.seconds) == ("cut", 0.0)
        return
    if family in _EFFECT_TRANSITIONS:
        # An effect transition carries no footage of its own - it is burned into the
        # frames each shot already has - so it must never ask for a handle.
        assert transition.kind == "effect"
        assert transition.name == family
        assert transition.handle == 0.0
    else:
        assert transition.kind == "xfade"
        assert transition.name in _TRANSITION_LIBRARY[family]
    assert cfg.transition_min_seconds <= transition.seconds <= cfg.transition_max_seconds


@pytest.mark.parametrize("family", sorted(_EFFECT_TRANSITIONS))
def test_effect_transitions_burn_into_both_sides_of_the_cut(family: str) -> None:
    """A flash or a glitch has to live inside the shots, gated to the window.

    Both ends of the cut need the treatment - one side alone reads as a mistake
    rather than as a transition - and every filter has to be timeline-gated, or the
    effect would tint the whole shot instead of the moment around the cut.
    """
    cfg = RenderConfig(width=640, height=360, fps=30)
    shot, following = _transition_shots(family)
    transition = _transition_spec(shot, following, _transition_art(), cfg)

    outgoing = _transition_effect_filters(transition, "out", shot.duration, cfg)
    incoming = _transition_effect_filters(transition, "in", shot.duration, cfg)

    assert outgoing and incoming, f"{family} only touches one side of the cut"
    assert any("fade=t=out" in item for item in outgoing)
    assert any("fade=t=in" in item for item in incoming)
    for item in (*outgoing, *incoming):
        assert "enable=" in item or "fade=t=" in item, item
    # The dip must sit at the end of the outgoing shot, not at its start.
    assert f"st={shot.duration - transition.seconds:.4f}" in " ".join(outgoing) or "st=" in " ".join(outgoing)


def _frame_stats(clip: Path, frame: int, tmp_path: Path) -> tuple[float, float, float]:
    """Mean luminance, spatial deviation and warmth (red minus blue) of one frame.

    Mean alone is the wrong instrument: sensor noise and a channel split barely move
    the average, so a glitch would look like it did nothing.
    """
    target = tmp_path / f"stat-{frame:04d}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(clip),
         "-vf", rf"select=eq(n\,{frame})", "-vsync", "0", "-frames:v", "1", str(target)],
        check=True, capture_output=True,
    )
    pixels = np.asarray(Image.open(target).convert("RGB"), dtype=float)
    grey = pixels.mean(axis=2)
    return float(grey.mean()), float(grey.std()), float(pixels[:, :, 0].mean() - pixels[:, :, 2].mean())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.parametrize("family", sorted(_EFFECT_TRANSITIONS))
def test_effect_transitions_actually_reach_the_boundary_frame(family: str, tmp_path: Path) -> None:
    """The dip has to *arrive* on the frame at the cut, not merely be configured.

    ``fade`` reaches its colour at ``st + d``, so a window that stops at the clip end
    leaves the last frame only part of the way there - and a flash that peaks at 30%
    is just a slight brightening. Nothing else in the graph would notice.
    """
    image = tmp_path / "flat.jpg"
    Image.new("RGB", (320, 180), (120, 120, 120)).save(image)
    cfg = RenderConfig(width=320, height=180, fps=15, crf=30, preset="ultrafast",
                       film_grain=0, vignette=False)
    art = _transition_art()
    outgoing, following = _transition_shots(family)
    outgoing.file = str(image)
    transition = _transition_spec(outgoing, following, art, cfg)

    clip = tmp_path / f"{family}.mp4"
    _render_shot(outgoing, clip, cfg, art, outgoing.duration, 2, outgoing=transition)

    frames = round(duration(clip) * cfg.fps)
    rest = _frame_stats(clip, frames // 4, tmp_path)
    at_cut = _frame_stats(clip, frames - 1, tmp_path)
    assert (
        abs(at_cut[0] - rest[0]) > 8 or at_cut[1] > rest[1] + 4 or at_cut[2] > rest[2] + 8
    ), f"{family} leaves the frame at the cut untouched: {rest} -> {at_cut}"


def test_a_cut_and_an_xfade_ask_for_different_handles() -> None:
    """Only an overlap needs extra footage; getting this wrong shifts the timeline."""
    cfg = RenderConfig()
    shot, following = _transition_shots("dissolve")
    dissolve = _transition_spec(shot, following, _transition_art(), cfg)
    assert dissolve.handle == dissolve.seconds > 0

    cut_shot, cut_following = _transition_shots("cut")
    assert _transition_spec(cut_shot, cut_following, _transition_art(), cfg).handle == 0.0


@pytest.mark.parametrize("tone,expected", [("bright", "fadewhite"), ("dark", "fadeblack")])
def test_a_dip_takes_its_colour_from_the_tone(tone: str, expected: str) -> None:
    """A dip to white and a dip to black are different editorial statements."""
    cfg = RenderConfig()
    shot, following = _transition_shots("dip")
    transition = _transition_spec(shot, following, _transition_art(tone=tone), cfg)
    assert transition.name == expected


def test_directional_transitions_follow_the_shots_own_drift() -> None:
    """A wipe that fights the camera move reads as a mistake, not as a flourish."""
    cfg = RenderConfig()
    art = _transition_art()
    names = {
        _transition_spec(*_transition_shots("wipe", media_id=media_id), art, cfg).name
        for media_id in range(4)
    }
    assert names == {"wipeleft", "wiperight"}


def test_the_knockout_fill_guarantees_its_own_contrast() -> None:
    """The letters have to read wherever the layout puts them.

    Lifting the fill alone leaves the type invisible on a smooth dark patch, and the
    free layout deliberately puts the type where the frame is empty. Pulling the rest
    of the picture down at the same time makes the contrast a property of the graph
    rather than a property of whatever happened to be behind the text.
    """
    cfg = RenderConfig(width=1280, height=720, fps=24)
    graph = _knockout_graph(Path("lyrics.ass"), 12.5, cfg)

    assert "split=2[base][treat]" in graph
    assert f"brightness={_KNOCKOUT_LIFT:.3f}" in graph, "the letters are not lifted"
    assert f"brightness={_KNOCKOUT_DIM:.3f}" in graph, "the picture is not pulled down"
    assert _KNOCKOUT_LIFT - _KNOCKOUT_DIM > .2, "the two have to add up to real contrast"
    assert "format=gray[mask]" in graph and "alphamerge[hole]" in graph
    assert "[dim][hole]overlay" in graph, "the hole must sit in the dimmed picture"
    assert "ass='" in graph, "the mask has to come from the lyric script"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_the_knockout_fill_renders_and_differs_from_solid(tmp_path: Path) -> None:
    """A graph this fiddly is worth running once: alphamerge silently takes an empty
    input for a black frame, which would look like the subtitles simply vanished."""
    image = tmp_path / "flat.jpg"
    Image.new("RGB", (320, 180), (70, 80, 110)).save(image)
    music = tmp_path / "music.wav"
    sample_rate = 22_050
    time = np.arange(sample_rate * 2) / sample_rate
    sf.write(music, (.05 * np.sin(2 * np.pi * 200 * time)).astype(np.float32), sample_rate)
    shot = Shot(0, 0, 2, 2, 0, str(image), "image", 0, "测试字幕", .5, "steady", "none", .8,
                melody=.5, image_effect="cinematic_depth")
    line = LyricLine(0, 2, "测试字幕")

    frames = {}
    for fill in ("solid", "knockout"):
        cfg = RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast",
                           film_grain=0, vignette=False, subtitle_fill=fill,
                           subtitle_layout="band", subtitle_size=28, subtitle_margin=18)
        output = tmp_path / f"{fill}.mp4"
        render([shot], [line], music, output, tmp_path / f"cache-{fill}", cfg,
               _transition_art())
        frames[fill] = _frame_stats(output, 12, tmp_path)[1]
        assert output.exists()

    assert frames["knockout"] != frames["solid"], "the knockout drew the same frame"
    assert frames["knockout"] > 0, "the knockout drew nothing"


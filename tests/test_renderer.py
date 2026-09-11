from pathlib import Path
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
from beatforge.renderer import _render_shot, _section_color_filter, _shot_match_filter, _video_encode_args, render
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

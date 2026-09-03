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
from beatforge.planner import Shot
from beatforge.renderer import _section_color_filter, _shot_match_filter, _video_encode_args, render
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

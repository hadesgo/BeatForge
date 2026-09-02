from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import soundfile as sf

from beatforge.config import RenderConfig
from beatforge.lyrics import LyricLine
from beatforge.planner import Shot
from beatforge.renderer import render
from beatforge.runtime import duration


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_end_to_end_renderer_without_models(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 360), (30, 80, 140)).save(image)
    music = tmp_path / "music.wav"
    sample_rate = 22_050
    time = np.arange(sample_rate * 2) / sample_rate
    sf.write(music, (.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sample_rate)
    shots = [Shot(0, 0, 2, 2, 0, str(image), "image", 0, "测试字幕", .5, "steady", "fade", .8)]
    output = tmp_path / "output.mp4"
    config = RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast", subtitle_size=20)
    render(shots, [LyricLine(0, 2, "测试字幕")], music, output, tmp_path / "cache", config)
    assert output.exists()
    assert 1.9 <= duration(output) <= 2.1

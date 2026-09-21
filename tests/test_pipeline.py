"""The pipeline's main path, run end to end without loading a single model.

``run_project`` is the function the CLI and every real invocation go through, but its
``--no-ai`` branch had no coverage beyond the vocal-separation helper. This drives it on
synthetic media so the stage orchestration - analysis, style, planning and the plan
handoff - is exercised for real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from beatforge.config import AIConfig, ProjectConfig, RenderConfig
from beatforge.pipeline import run_project


def _write_music(path: Path, seconds: float = 12.0) -> None:
    """Two detuned sines with tremolo, so the analysis has onsets to find."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}:sample_rate=22050",
            "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}:sample_rate=22050",
            "-filter_complex",
            "[0:a]volume=.2,tremolo=f=2:d=.7[a0];[1:a]volume=.1,tremolo=f=3:d=.6[a1];"
            "[a0][a1]amix=inputs=2",
            str(path),
        ],
        check=True, capture_output=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_run_project_plan_only_without_models(tmp_path: Path) -> None:
    """``--plan-only --no-ai`` has to reach a written plan without touching a model.

    This is the smoke test the CLI's own path leans on: an existing music file, an LRC
    and a folder of stills are enough to produce shots, so a regression that only shows
    up when the stages are wired together - a missing print, a bad key in the plan, a
    style that crashes on the analysis - is caught here rather than on a user's machine.
    """
    media = tmp_path / "media"
    media.mkdir()
    for index in range(3):
        Image.new("RGB", (320, 180), (30 + index * 60, 80, 150 - index * 40)).save(
            media / f"frame-{index}.jpg"
        )
    music = tmp_path / "music.wav"
    _write_music(music)
    lyrics = tmp_path / "lyrics.lrc"
    lyrics.write_text(
        "[00:00.00]黎明照亮天空\n[00:03.00]我们奔向希望\n[00:06.00]像海浪追逐着梦\n"
        "[00:09.00]城市亮起灯光\n",
        "utf-8",
    )
    project = ProjectConfig(
        root=tmp_path, music=music, media_dir=media, output=tmp_path / "output.mp4",
        lyrics=lyrics, cache_dir=tmp_path / ".beatforge", ai=AIConfig(),
        render=RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast"),
    )

    plan_file = run_project(project, plan_only=True, no_ai=True)

    assert plan_file == project.cache_dir / "plan.json"
    plan = json.loads(plan_file.read_text("utf-8"))
    assert plan["shots"], "the run produced no shots"
    assert plan["lyrics"], "the LRC was not read into the plan"
    assert all(shot["kind"] == "image" for shot in plan["shots"]), "still frames became videos"
    # With no AI the model slots have to be empty, or the plan would advertise a model
    # that never ran.
    assert plan["models"]["asr"] is None
    assert plan["models"]["vision"] is None
    assert plan["models"]["director"] is None

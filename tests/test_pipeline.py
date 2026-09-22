"""The pipeline's main path, run end to end without loading a single model.

``run_project`` is the function the CLI and every real invocation go through, but its
``--no-ai`` branch had no coverage beyond the vocal-separation helper. This drives it on
synthetic media so the stage orchestration - analysis, style, planning and the plan
handoff - is exercised for real.
"""

from __future__ import annotations

import json
import logging
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


def _build_project(tmp_path: Path, *, render: RenderConfig | None = None) -> ProjectConfig:
    """A minimal ``--no-ai`` project: three stills, a tune and four lyric lines."""
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
    return ProjectConfig(
        root=tmp_path, music=music, media_dir=media, output=tmp_path / "output.mp4",
        lyrics=lyrics, cache_dir=tmp_path / ".beatforge", ai=AIConfig(),
        render=render or RenderConfig(width=320, height=180, fps=12, crf=30, preset="ultrafast"),
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_run_project_plan_only_without_models(tmp_path: Path) -> None:
    """``--plan-only --no-ai`` has to reach a written plan without touching a model.

    This is the smoke test the CLI's own path leans on: an existing music file, an LRC
    and a folder of stills are enough to produce shots, so a regression that only shows
    up when the stages are wired together - a missing print, a bad key in the plan, a
    style that crashes on the analysis - is caught here rather than on a user's machine.
    """
    project = _build_project(tmp_path)

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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_the_plan_is_version_4_with_an_audit_and_motif_counts(tmp_path: Path) -> None:
    """R-02/R-03/R-11: the plan carries the audit, the motif counts, and no camera_motion.

    Downstream readers key off ``version``; a v4 plan has to advertise itself, list the
    overrides it made, count how often each motif recurred, and carry the new media metrics
    while dropping ``camera_motion`` everywhere.
    """
    plan = json.loads(
        run_project(_build_project(tmp_path), plan_only=True, no_ai=True).read_text("utf-8")
    )

    assert plan["version"] == 4
    assert isinstance(plan["config_audit"], list)
    assert isinstance(plan["motif_appearances"], dict)
    assert all("camera_motion" not in shot for shot in plan["shots"])
    assert all("camera_motion" not in asset for asset in plan["media"])
    assert all("is_motif" in shot for shot in plan["shots"])
    assert all(
        {"luma", "edge_luma", "noise_score"} <= set(asset) for asset in plan["media"]
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_an_explicit_shot_length_is_not_silently_overridden(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """R-02: the user *chose* the shot length, so the style defers - and does not warn.

    ``beat`` wants a 0.55-1.9s window; the user picked 2.2-6.0. The explicit values win,
    the disagreement is recorded against ``explicit-config``, and no WARNING is emitted
    because nothing was actually changed. The values are deliberately non-default: the
    template writes 1.8-5.5 into every project it creates, and boilerplate must not
    count as a decision (see ``RenderConfig.explicit``).
    """
    render = RenderConfig(
        width=320, height=180, fps=12, crf=30, preset="ultrafast",
        edit_style="beat", min_shot_seconds=2.2, max_shot_seconds=6.0,
    )
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        plan = json.loads(
            run_project(_build_project(tmp_path, render=render), plan_only=True, no_ai=True)
            .read_text("utf-8")
        )

    entries = {item["key"]: item for item in plan["config_audit"]}
    for key in ("min_shot_seconds", "max_shot_seconds"):
        assert entries[key]["effective"] == entries[key]["requested"], key
        assert entries[key]["overridden_by"] == "explicit-config"
        assert entries[key]["explicit"] is True
    assert "min_shot_seconds" not in caplog.text


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_a_style_takeover_is_recorded_and_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """R-02: when the user stays silent the style takes over - loudly and in the plan.

    The default window is 1.8-5.5; ``beat`` takes it down to 0.55-1.9. That is a real
    change, so it is recorded with a requested/effective pair and it emits a WARNING.
    """
    render = RenderConfig(
        width=320, height=180, fps=12, crf=30, preset="ultrafast", edit_style="beat",
    )
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        plan = json.loads(
            run_project(_build_project(tmp_path, render=render), plan_only=True, no_ai=True)
            .read_text("utf-8")
        )

    entries = {item["key"]: item for item in plan["config_audit"]}
    entry = entries["max_shot_seconds"]
    assert entry["requested"] != entry["effective"]
    assert entry["overridden_by"] == "edit_style:卡点快剪"
    assert entry["explicit"] is False
    assert "max_shot_seconds" in caplog.text

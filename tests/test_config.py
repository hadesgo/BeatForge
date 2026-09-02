from pathlib import Path

from beatforge.config import PROJECT_TEMPLATE, load_project


def test_project_paths_and_cpu_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text(PROJECT_TEMPLATE, "utf-8")
    config = load_project(project)
    assert config.music == (tmp_path / "music.mp3").resolve()
    assert config.output == (tmp_path / "output.mp4").resolve()
    assert config.ai.whisper_model == "small"
    assert config.ai.whisper_compute_type == "int8"
    assert config.render.subtitle_effect == "karaoke"
    assert config.render.visual_effects is True

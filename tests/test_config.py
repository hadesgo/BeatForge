from pathlib import Path

from beatforge.config import PROJECT_TEMPLATE, load_project


def test_project_paths_and_cpu_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text(PROJECT_TEMPLATE, "utf-8")
    config = load_project(project)
    assert config.music == (tmp_path / "music.mp3").resolve()
    assert config.output == (tmp_path / "output.mp4").resolve()
    assert config.ai.asr_backend == "qwen3"
    assert config.ai.qwen_asr_model == "Qwen/Qwen3-ASR-1.7B-hf"
    assert config.ai.vision_backend == "qwen3-vl-embedding"
    assert config.ai.director_enabled is True
    assert config.ai.vision_model == "Qwen/Qwen3-VL-Embedding-8B"
    assert config.ai.vision_quantization == "nf4"
    assert config.ai.vision_batch_size == 4
    assert config.ai.director_model == "Qwen/Qwen3.5-9B"
    assert config.ai.director_quantization == "nf4"
    assert config.render.subtitle_effect == "auto"
    assert config.render.subtitle_font == "auto"
    assert config.render.visual_effects is True

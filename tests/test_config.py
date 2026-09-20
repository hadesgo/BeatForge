from pathlib import Path
import tomllib

from beatforge.config import PROJECT_TEMPLATE, load_project


def test_project_paths_and_cpu_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text(PROJECT_TEMPLATE, "utf-8")
    config = load_project(project)
    assert config.music == (tmp_path / "music.mp3").resolve()
    assert config.output == (tmp_path / "output.mp4").resolve()
    assert config.ai.qwen_asr_model == "Qwen/Qwen3-ASR-1.7B-hf"
    assert config.ai.vision_backend == "wemm-embedding"
    assert config.ai.director_enabled is True
    assert config.ai.vision_model == "tencent/WeMM-Embedding-2B"
    assert config.ai.vision_batch_size == 4
    assert config.ai.frame_samples == 8
    assert config.ai.director_model == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert config.ai.director_gguf_file == "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    assert config.render.subtitle_effect == "auto"
    assert config.render.subtitle_font == "auto"
    assert config.render.subtitle_fonts["energetic"] == "preset:energetic"
    assert config.render.visual_effects is True
    assert config.render.intermediate_crf < config.render.crf


def test_the_template_only_names_real_config_fields() -> None:
    """A stale key in the template is invisible at runtime.

    Pydantic ignores unknown keys by default, so a renamed or removed option would sit
    in every generated project looking authoritative while doing nothing at all.
    """
    from beatforge.config import AIConfig, RenderConfig

    template = tomllib.loads(PROJECT_TEMPLATE)

    assert set(template["ai"]) <= set(AIConfig.model_fields)
    assert set(template["render"]) <= set(RenderConfig.model_fields)


def test_the_shipped_project_files_only_name_real_config_fields() -> None:
    """The demo and working projects are templates too, and the same silent-typo risk
    applies: an unknown key is ignored, so a removed option would sit there looking
    authoritative while doing nothing - including the old director engine switch."""
    from beatforge.config import AIConfig, RenderConfig

    root = Path(__file__).resolve().parents[1]
    projects = [
        path for path in (root / "demo" / "project.toml", root / "my-mv" / "project.toml")
        if path.is_file()
    ]

    assert projects, "no shipped project file found to check"
    for path in projects:
        data = tomllib.loads(path.read_text("utf-8"))
        assert set(data["ai"]) <= set(AIConfig.model_fields), path
        assert set(data["render"]) <= set(RenderConfig.model_fields), path


def test_intermediate_crf_cannot_be_worse_than_delivery() -> None:
    from beatforge.config import RenderConfig

    assert RenderConfig(crf=16, intermediate_crf=24).intermediate_crf == 16


def test_a_blank_optional_path_means_unset_not_the_current_directory() -> None:
    """``""`` 必须表示"没配"，而不是 ``Path(".")``。

    真机踩到：``director_llama_server = ""`` 被 pathlib 折叠成 ``Path(".")``，于是走了
    "显式路径"分支，在 PATH 与 ``<cache>/bin`` 查找之前就报「指向的文件不存在：.」——
    模板里每一个"留空即自动查找"的选项都被这一条静默废掉。
    """
    from beatforge.config import AIConfig, RenderConfig

    ai = AIConfig(director_llama_server="", director_gguf="")

    assert ai.director_llama_server is None
    assert ai.director_gguf is None
    assert RenderConfig(subtitle_fonts_dir="").subtitle_fonts_dir is None
    # 真的给了路径就要保留，别把自动查找变成唯一选项。
    assert AIConfig(director_llama_server="bin/llama-server.exe").director_llama_server == (
        Path("bin/llama-server.exe")
    )
    # 只把空白和 "." 视为未配置；"./fonts" 这类相对路径照旧。
    assert RenderConfig(subtitle_fonts_dir="./fonts").subtitle_fonts_dir == Path("./fonts")


def test_a_blank_lyrics_path_is_unset_too(tmp_path: Path) -> None:
    """``lyrics = ""`` 是文档里"改由 ASR 转写"的写法。

    折叠成项目根目录更糟：它是个存在的目录，于是 pipeline 里"已经有歌词了吗"的检查答"有"，
    接着把一个目录当 LRC 读。
    """
    project = tmp_path / "project.toml"
    project.write_text(
        PROJECT_TEMPLATE.replace('lyrics = "lyrics.lrc"', 'lyrics = ""'), "utf-8",
    )

    config = load_project(project)

    assert config.lyrics is None

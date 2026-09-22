import tomllib
from pathlib import Path

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


def test_explicit_distinguishes_a_choice_from_template_boilerplate() -> None:
    """显式 = 用户**选了**一个不等于默认值的取值，而不是"键出现过"。

    模板会把 ``min_shot_seconds = 1.8``、``max_shot_seconds = 5.5``、
    ``subtitle_layout``、``transition_density``、``image_composite_ratio`` 写进每个
    ``init`` 生成的工程。若"键出现过"就算表态，这些样板行会永久顶掉 edit_style 的
    节奏接管——``edit_style = "beat"`` 在所有新工程上都只改转场、不改镜长，而且静默。
    实测（my-mv）：raw ``model_fields_set`` 判定下风格镜长接管完全失效。
    所以判据是「写过 **且** 不等于默认值」。真想压过风格的用户用
    ``edit_style = "manual"``；风格接管到的每一项都经 audit 留痕，不静默。
    """
    from beatforge.config import RenderConfig

    chosen = RenderConfig(min_shot_seconds=2.5, max_shot_seconds=6.0, subtitle_layout="band")
    assert "min_shot_seconds" in chosen.explicit()
    assert "max_shot_seconds" in chosen.explicit()
    assert "subtitle_layout" in chosen.explicit()  # band != 默认的 free

    boilerplate = RenderConfig(min_shot_seconds=1.8, max_shot_seconds=5.5)
    assert "min_shot_seconds" not in boilerplate.explicit()
    assert "max_shot_seconds" not in boilerplate.explicit()
    # 没写过的键不在集合里，仍可被 style 接管。
    assert "subtitle_layout" not in boilerplate.explicit()
    assert "transition_density" not in boilerplate.explicit()


def test_no_silent_config_override(caplog) -> None:
    """R-02：程序覆盖用户配置必须留痕；但真正的显式选择下风格只能"不认同"，不许改。

    三个 fixture 覆盖三个方向：
    (a) 用户**选了**非默认值 → 生效值 == 用户值，且**没有** WARNING；
    (b) 模板样板形状（键出现、值等于默认值）→ 风格接管，但每一项都要留痕并 WARNING
        ——这正是模板陷阱的修复点：接管发生了，但绝不静默；
    (c) 用户什么都没写 → style 接管五项，每项都要有 requested/effective/overridden_by，
        且 caplog 里出现对应 WARNING。
    """
    import logging

    from beatforge.audit import ConfigAudit, apply_style
    from beatforge.config import RenderConfig
    from beatforge.editing import EDIT_STYLES

    style = EDIT_STYLES["beat"]

    # (a) 显式优先：用户选了非默认值，风格一律不改，只记录"风格不认同"。
    explicit_render = RenderConfig(
        min_shot_seconds=2.5, max_shot_seconds=6.0, subtitle_layout="band",
        transition_density=0.4, image_composite_ratio=0.4,
    )
    audit_a = ConfigAudit()
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        effective_a = apply_style(explicit_render, style, audit_a)

    entries = {item["key"]: item for item in audit_a.as_list()}
    for key in ("min_shot_seconds", "max_shot_seconds", "subtitle_layout"):
        assert key in entries, f"{key} 的「风格不认同」没有被记录"
        assert entries[key]["effective"] == entries[key]["requested"], (key, entries[key])
        assert entries[key]["explicit"] is True
    assert effective_a.min_shot_seconds == 2.5
    assert effective_a.subtitle_layout == "band"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (
        "显式配置没有被覆盖，就不该出现任何覆盖 WARNING"
    )

    # (b) 模板样板：键出现但值等于默认值 → 风格接管，且接管必须留痕（不静默）。
    caplog.clear()
    audit_b = ConfigAudit()
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        effective_b = apply_style(
            RenderConfig(min_shot_seconds=1.8, max_shot_seconds=5.5), style, audit_b,
        )

    entries = {item["key"]: item for item in audit_b.as_list()}
    for key in ("min_shot_seconds", "max_shot_seconds"):
        item = entries[key]
        assert item["overridden_by"] == "edit_style:卡点快剪", item
        assert item["effective"] == getattr(style, "shot_min" if key == "min_shot_seconds" else "shot_max"), item
        assert item["explicit"] is False, item
    assert effective_b.min_shot_seconds == style.shot_min
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) >= 2, "模板样板被接管也必须 WARNING，不许静默"

    # (c) 未表态：五个键全部由 style 接管，每一项都要留痕并 WARNING。
    caplog.clear()
    audit_c = ConfigAudit()
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        effective_c = apply_style(RenderConfig(), EDIT_STYLES["cinematic"], audit_c)

    entries = {item["key"]: item for item in audit_c.as_list()}
    for key in ("min_shot_seconds", "max_shot_seconds", "transition_density",
                "image_composite_ratio", "subtitle_layout"):
        item = entries[key]
        assert item["overridden_by"] == "edit_style:电影感长镜", item
        assert item["effective"] != item["requested"], item
        assert item["explicit"] is False
    assert effective_c.min_shot_seconds == 3.0
    assert effective_c.subtitle_layout == "band"
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) >= 5, "每个接管项都要有 WARNING"


def test_audit_is_the_only_place_an_override_is_logged(caplog) -> None:
    """没有值变化时只记账、不报警，避免把"风格不认同"误报成覆盖。"""
    import logging

    from beatforge.audit import ConfigAudit

    audit = ConfigAudit()
    with caplog.at_level(logging.WARNING, logger="beatforge.audit"):
        audit.record("subtitle_layout", requested="band", effective="band",
                     overridden_by="explicit-config", explicit=True)
        audit.record("theme", requested="A", effective="B", overridden_by="edit_style:x")

    assert len(audit) == 2
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    assert audit.as_list()[1]["effective"] == "B"


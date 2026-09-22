from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

from beatforge.editing import style_choices
from beatforge.lyrics import SUBTITLE_EFFECTS

# One source of truth: the renderer builds these, so the config validates against
# whatever it can actually draw rather than against a second hand-kept list.
SubtitleEffect = Literal[*SUBTITLE_EFFECTS]
SubtitleEffectChoice = Literal["auto", *SUBTITLE_EFFECTS]


def _blank_path_is_none(value: object) -> object:
    """Read a blank optional path as "not configured" instead of "the current directory".

    ``pathlib`` folds ``""`` (and ``"."``) into ``Path(".")``, so a template that ships
    ``director_llama_server = ""`` arrives as a *real* explicit path. That silently
    breaks every "留空即自动查找" option: the lookup takes the explicit-path branch and
    fails with ``指向的文件不存在：.`` before PATH or the cache directory is ever
    consulted. Neither value can name a file, so both mean unset.
    """
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.strip() in {"", "."}:
        return None
    return value


class RenderConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crf: int = Field(default=19, ge=0, le=51)
    intermediate_crf: int = Field(default=14, ge=0, le=35)
    preset: str = "medium"
    encoder_tune: Literal["none", "film", "grain", "animation"] = "film"
    min_shot_seconds: float = 1.8
    max_shot_seconds: float = 5.5
    subtitle_font: str = "auto"
    subtitle_fonts_dir: Path | None = None

    @field_validator("subtitle_fonts_dir", mode="before")
    @classmethod
    def _blank_fonts_dir_is_unset(cls, value: object) -> object:
        return _blank_path_is_none(value)

    subtitle_fonts: dict[str, str] = Field(
        default_factory=lambda: {
            "energetic": "preset:energetic",
            "uplifting": "preset:modern",
            "melancholic": "preset:cinematic",
            "dreamy": "preset:dreamy",
            "romantic": "preset:lyrical",
            "dark": "preset:dark",
            "cinematic": "preset:cinematic",
        }
    )
    subtitle_size: int = 46
    subtitle_effect: SubtitleEffectChoice = "auto"
    #: Ceiling for the background-adaptive lyric outline (R-05). The outline is thickened
    #: on bright backgrounds and thinned on dark ones; this is how far it may grow.
    subtitle_max_outline: float = Field(default=2.4, ge=0, le=8)

    @field_validator("edit_style")
    @classmethod
    def _known_style(cls, value: str) -> str:
        if value not in style_choices():
            raise ValueError(
                f"未知剪辑风格 {value!r}；可选：{'/'.join(style_choices())}"
            )
        return value

    subtitle_margin: int = 72
    #: Which editing craft to apply. See beatforge/editing.py for what each one means.
    edit_style: str = "auto"
    subtitle_fill: Literal["solid", "knockout"] = "solid"
    subtitle_outline: float = Field(default=1.1, ge=0, le=6)
    subtitle_highlight_color: str = "&H0000D7FF"
    visual_effects: bool = True
    image_composites: bool = True
    image_composite_ratio: float = Field(default=0.24, ge=0, le=1)
    max_composite_images: int = Field(default=3, ge=2, le=4)
    avoid_asset_repeats: bool = True
    vignette: bool = True
    film_grain: float = Field(default=1.6, ge=0, le=8)
    look_strength: float = Field(default=0.72, ge=0, le=1)
    shot_match_strength: float = Field(default=0.3, ge=0, le=1)
    professional_transitions: bool = True
    transition_min_seconds: float = Field(default=0.16, ge=0.05, le=1.0)
    transition_max_seconds: float = Field(default=0.55, ge=0.1, le=1.5)
    transition_density: float = Field(default=0.35, ge=0, le=1)

    #: Unknown keys are a hard error, not a shrug. The first quality rework kept the
    #: single-image keys around as inert compat, and that only worked because nothing
    #: could tell "accepted for compat" from "typed by mistake". A project carrying a
    #: removed key now fails here, naming the key - which is the honest reading of
    #: "this option no longer exists".
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def keep_intermediates_high_quality(self) -> RenderConfig:
        # In x264 a lower CRF means higher quality. Intermediate generations must
        # never be more compressed than the delivery encode.
        self.intermediate_crf = min(self.intermediate_crf, self.crf)
        return self

    def explicit(self) -> frozenset[str]:
        """The keys the user actually chose, as far as this program can tell.

        ``model_fields_set`` alone is not enough, and this used to be the whole answer.
        The catch is the shipped template: ``init`` writes the style-owned keys
        (``min_shot_seconds``、``max_shot_seconds``、``transition_density``、
        ``image_composite_ratio``) into every ``project.toml`` it creates, so for a
        template-generated project "the key is present" means "boilerplate", not "the
        user picked it". Treating it as a decision permanently disables the pacing a
        style is supposed to own - ``edit_style = "beat"`` would change transitions but
        never shot length, silently, on every fresh project.

        So a key counts as spoken only when its value differs from the field default.
        A user who genuinely wants the default to outrank a style sets
        ``edit_style = "manual"``; and whatever a style does take over is logged
        through the audit, so no takeover is silent.
        """
        spoken: set[str] = set()
        for key in self.model_fields_set:
            info = type(self).model_fields.get(key)
            if info is None:
                spoken.add(key)
                continue
            default = info.get_default(call_default_factory=True)
            if default is PydanticUndefined or getattr(self, key) != default:
                spoken.add(key)
        return frozenset(spoken)


class AIConfig(BaseModel):
    enabled: bool = True
    device: Literal["auto", "cuda", "cpu"] = "auto"
    offline: bool = False
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B-hf"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    #: Separate the vocal before transcribing. The recogniser is being asked to hear a
    #: voice through a drum kit otherwise, and its failure mode is a confident
    #: transcript with the wrong words rather than an empty one.
    separate_vocals: bool = True
    separation_model: str = "vocals_mel_band_roformer.ckpt"
    clap_model: str = "laion/clap-htsat-fused"
    music_structure_backend: Literal["librosa", "beat-this", "allin1"] = "allin1"
    vision_backend: Literal["wemm-embedding", "qwen3-vl-embedding", "siglip2"] = (
        "wemm-embedding"
    )
    vision_model: str = "tencent/WeMM-Embedding-2B"
    vision_reranker_model: str | None = "Qwen/Qwen3-VL-Reranker-2B"
    vision_batch_size: int = Field(default=4, ge=1, le=32)
    vision_rerank_top_k: int = Field(default=8, ge=0, le=20)
    vision_input_pixels: int = Field(default=1280 * 28 * 28, ge=224 * 224, le=4_000_000)
    frame_samples: int = Field(default=8, ge=1, le=12)
    director_enabled: bool = True
    director_model: str = "prism-ml/Ternary-Bonsai-2-27B-gguf"
    #: Which quantisation to fetch. The repository holds both packings - PTQ1_0 at
    #: 5.95 GB and PQ2_0 at 7.21 GB - so a snapshot download would cost 13 GB to get
    #: one usable file. PQ2_0 decodes faster on Hopper and Blackwell; PTQ1_0 is the
    #: better pick on Ada and when memory is tightest.
    director_gguf_file: str = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    #: The GGUF file to serve. Blank means "find the configured quant under the cache
    #: directory", which is where ``download-models`` puts it.
    director_gguf: Path | None = None
    #: The llama-server executable. Blank means "look on PATH, then in the cache".
    director_llama_server: Path | None = None
    #: Point at an already-running server instead of starting one. Any llama.cpp build
    #: serving the same checkpoint will do, which is how you avoid a second copy of a
    #: 7 GB model in memory.
    director_llama_url: str | None = None
    #: Passed straight through to llama-server. ``-ngl 99`` offloads every layer; the
    #: context has to cover the prompt *and* the answer, which is why it is bigger than
    #: ``director_prompt_tokens``.
    director_llama_args: list[str] = Field(
        default_factory=lambda: ["-ngl", "99", "-fa", "on", "-c", "24576"]
    )
    director_llama_port: int = Field(default=8917, ge=1024, le=65535)
    director_temperature: float = Field(default=0.7, ge=0, le=1.5)
    director_max_new_tokens: int = Field(default=4096, ge=256, le=32768)
    #: Prompt ceiling, and only an *upper* bound: the real budget is clamped to whatever
    #: the server's ``-c`` has left after ``director_max_new_tokens``, because the prompt
    #: and the answer share one context window.
    director_prompt_tokens: int = Field(default=24576, ge=256, le=131072)

    @field_validator("director_gguf", "director_llama_server", mode="before")
    @classmethod
    def _blank_paths_are_unset(cls, value: object) -> object:
        return _blank_path_is_none(value)


class ProjectConfig(BaseModel):
    root: Path
    music: Path
    media_dir: Path
    output: Path
    lyrics: Path | None = None
    cache_dir: Path
    ai: AIConfig = AIConfig()
    render: RenderConfig = RenderConfig()

    @field_validator("lyrics", mode="before")
    @classmethod
    def _blank_lyrics_is_unset(cls, value: object) -> object:
        # ``lyrics = ""`` is a documented way to say "transcribe instead" (see the
        # template). Without this it becomes the project root, which resolves to an
        # existing directory - so the "do we already have lyrics?" checks below answer
        # yes and then try to read a directory as an LRC file.
        return _blank_path_is_none(value)

    @model_validator(mode="after")
    def resolve_paths(self) -> ProjectConfig:
        for name in ("music", "media_dir", "output", "cache_dir", "lyrics"):
            value = getattr(self, name)
            if value is not None and not value.is_absolute():
                setattr(self, name, (self.root / value).resolve())
        if (
            self.render.subtitle_fonts_dir
            and not self.render.subtitle_fonts_dir.is_absolute()
        ):
            self.render.subtitle_fonts_dir = (
                self.root / self.render.subtitle_fonts_dir
            ).resolve()
        return self


def load_project(file: Path) -> ProjectConfig:
    project_file = file.resolve()
    with project_file.open("rb") as handle:
        data = tomllib.load(handle)
    data["root"] = project_file.parent
    data.setdefault("cache_dir", ".beatforge")
    return ProjectConfig.model_validate(data)


PROJECT_TEMPLATE = """music = "music.mp3"
lyrics = "lyrics.lrc" # 可删除；缺失时由 Qwen3-ASR 自动转写
media_dir = "media"
output = "output.mp4"
cache_dir = ".beatforge"

[ai]
enabled = true
device = "auto"
offline = false
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
separate_vocals = true # 转录前先做 MelBand-RoFormer 人声分离；需要 uv sync --extra ai --extra separation
separation_model = "vocals_mel_band_roformer.ckpt" # 换更强的分离模型只改这一行
clap_model = "laion/clap-htsat-fused"
music_structure_backend = "allin1" # 需要 music-ai extra；也可用 beat-this 或 librosa
vision_backend = "wemm-embedding"
vision_model = "tencent/WeMM-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
vision_batch_size = 4 # 更大显存可提高到 8
vision_rerank_top_k = 8
vision_input_pixels = 1003520 # 每张图送进编码器前的像素上限（默认 = 1280*28*28）
frame_samples = 8 # WeMM 默认增加长视频覆盖率
director_enabled = true # BeatForge 分阶段加载并释放模型；失败会回退规则导演
director_model = "prism-ml/Ternary-Bonsai-2-27B-gguf"
director_gguf = "" # 留空则自动找 cache 目录下 download-models 放好的量化文件
director_gguf_file = "Ternary-Bonsai-2-27B-PQ2_0.gguf" # PTQ1_0 更省内存，PQ2_0 在较新显卡上更快
director_llama_server = "" # 留空则先在 PATH 找 llama-server，再找 cache/bin
director_llama_url = "" # 指向已启动的 llama-server 可省掉一次 7GB 加载
director_llama_args = ["-ngl", "99", "-fa", "on", "-c", "24576"] # -ngl 0 为纯 CPU；-c 同时要装下提示词和回复，所以大于 director_prompt_tokens
director_llama_port = 8917
director_temperature = 0.7
director_max_new_tokens = 4096
director_prompt_tokens = 24576 # 提示词上限，只是上界；实际会被 -c 减去回复长度后的余量压低

[render]
width = 1920
height = 1080
fps = 30
crf = 19
intermediate_crf = 14 # 中间文件使用更高质量，减少多次编码造成的细节损失
preset = "medium"
encoder_tune = "film" # 也可用 grain/animation/none
# 下面这些键由 edit_style 接管（见 README「剪辑风格」）。保持注释 = 让风格决定节奏；
# 一旦取消注释，取值不等于默认值时会被当成你的显式选择，风格对它失效（并写入 config_audit）。
# min_shot_seconds = 1.8
# max_shot_seconds = 5.5
subtitle_font = "auto"
subtitle_fonts_dir = "fonts" # 项目自己的额外字体目录；仓库自带的九个中文字体始终可用
subtitle_size = 46
edit_style = "auto" # auto 按歌曲情绪自动选；也可固定为某个风格名；manual = 用下面的手工值
subtitle_effect = "auto" # 也可固定为某个特效名；可选值见 README「字幕特效」
subtitle_margin = 72
subtitle_fill = "solid" # knockout = 文字从画面里镂空，字中透出提亮虚化的同一帧
subtitle_outline = 1.1 # 描边宽度；0 为无描边（更融入画面，但需要画面本身够暗）
subtitle_max_outline = 2.4 # 亮背景自适应描边的上限；描边随背景变亮而加粗、变暗而收细
subtitle_highlight_color = "&H0000D7FF" # ASS 的金黄色（BGR）
visual_effects = true
image_composites = true # AI 按段落自动选择多图版式：分屏、斜切、三联、主副网格、节拍蒙太奇
# image_composite_ratio = 0.24 # 多图镜头占比；副歌会适当提高（不写则由 edit_style 决定）
max_composite_images = 3 # 2 只够双栏；3 可做三联和主副网格；4 仅供节拍蒙太奇
avoid_asset_repeats = true # 非母题素材充足时每个镜头用不同素材；多图合成只消耗富余素材
vignette = true
film_grain = 1.6
look_strength = 0.72 # AI 导演色彩弧的应用强度
shot_match_strength = 0.3 # 不同来源素材的轻量曝光/饱和度匹配
professional_transitions = true
transition_min_seconds = 0.16
transition_max_seconds = 0.55
# transition_density = 0.35 # 段落内部使用可见转场的比例；0 = 只在段落切换时转场（不写则由 edit_style 决定）

[render.subtitle_fonts]
energetic = "preset:energetic"
uplifting = "preset:modern"
melancholic = "preset:cinematic"
dreamy = "preset:dreamy"
romantic = "preset:lyrical"
dark = "preset:dark"
cinematic = "preset:cinematic"
"""

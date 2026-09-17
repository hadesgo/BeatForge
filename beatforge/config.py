from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from beatforge.editing import style_choices

from beatforge.lyrics import SUBTITLE_EFFECTS

# One source of truth: the renderer builds these, so the config validates against
# whatever it can actually draw rather than against a second hand-kept list.
SubtitleEffect = Literal[*SUBTITLE_EFFECTS]
SubtitleEffectChoice = Literal["auto", *SUBTITLE_EFFECTS]


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

    @field_validator("edit_style")
    @classmethod
    def _known_style(cls, value: str) -> str:
        if value not in style_choices():
            raise ValueError(f"未知剪辑风格 {value!r}；可选：{'/'.join(style_choices())}")
        return value
    subtitle_margin: int = 72
    #: Which editing craft to apply. See beatforge/editing.py for what each one means.
    edit_style: str = "auto"
    subtitle_layout: Literal["band", "free"] = "free"
    subtitle_fill: Literal["solid", "knockout"] = "solid"
    subtitle_outline: float = Field(default=1.1, ge=0, le=6)
    subtitle_highlight_color: str = "&H0000D7FF"
    visual_effects: bool = True
    image_composites: bool = True
    image_composite_ratio: float = Field(default=0.24, ge=0, le=1)
    max_composite_images: int = Field(default=3, ge=2, le=4)
    avoid_asset_repeats: bool = True
    blurred_image_background: bool = True
    image_background_blur: float = Field(default=26.0, ge=0, le=80)
    image_foreground_scale: float = Field(default=0.92, ge=0.55, le=1.0)
    vignette: bool = True
    film_grain: float = Field(default=1.6, ge=0, le=8)
    look_strength: float = Field(default=0.72, ge=0, le=1)
    shot_match_strength: float = Field(default=0.3, ge=0, le=1)
    professional_transitions: bool = True
    transition_min_seconds: float = Field(default=0.16, ge=0.05, le=1.0)
    transition_max_seconds: float = Field(default=0.55, ge=0.1, le=1.5)
    transition_density: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def keep_intermediates_high_quality(self) -> "RenderConfig":
        # In x264 a lower CRF means higher quality. Intermediate generations must
        # never be more compressed than the delivery encode.
        self.intermediate_crf = min(self.intermediate_crf, self.crf)
        return self


class AIConfig(BaseModel):
    enabled: bool = True
    device: Literal["auto", "cuda", "cpu"] = "auto"
    offline: bool = False
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B-hf"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
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
    director_model: str = "XHToken/Spark-X2.5-4B"
    director_backend: Literal["text", "multimodal"] = "text"
    director_temperature: float = Field(default=0.18, ge=0, le=1.5)
    director_max_new_tokens: int = Field(default=3072, ge=256, le=8192)
    director_gpu_memory_gb: float = Field(default=9.0, ge=1, le=80)
    director_cpu_memory_gb: float = Field(default=20.0, ge=4, le=256)
    director_offload: bool = True
    director_contact_sheet_assets: int = Field(default=0, ge=0, le=48)
    director_prompt_tokens: int = Field(default=2600, ge=256, le=32000)


class ProjectConfig(BaseModel):
    root: Path
    music: Path
    media_dir: Path
    output: Path
    lyrics: Path | None = None
    cache_dir: Path
    ai: AIConfig = AIConfig()
    render: RenderConfig = RenderConfig()

    @model_validator(mode="after")
    def resolve_paths(self) -> "ProjectConfig":
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
director_model = "XHToken/Spark-X2.5-4B"
director_backend = "text"
director_temperature = 0.18
director_max_new_tokens = 3072
director_gpu_memory_gb = 9.0 # 12GB 显卡为渲染和临时张量预留约 3GB
director_cpu_memory_gb = 20.0
director_offload = true
director_contact_sheet_assets = 0 # Spark 是文本模型；改用多模态导演时可设为 24~32
director_prompt_tokens = 2600 # 导演提示词上限；Spark 的注意力开销随提示词长度平方增长，12GB 显存不要调高

[render]
width = 1920
height = 1080
fps = 30
crf = 19
intermediate_crf = 14 # 中间文件使用更高质量，减少多次编码造成的细节损失
preset = "medium"
encoder_tune = "film" # 也可用 grain/animation/none
min_shot_seconds = 1.8
max_shot_seconds = 5.5
subtitle_font = "auto"
subtitle_fonts_dir = "fonts" # 可放入自定义 ttf/otf；不存在也不影响系统字体
subtitle_size = 46
edit_style = "auto" # auto 按歌曲情绪自动选；也可固定为某个风格名；manual = 用下面的手工值
subtitle_effect = "auto" # 也可固定为某个特效名；可选值见 README「字幕特效」
subtitle_margin = 72
subtitle_layout = "free" # free = 分句自由排版并避开主体；band = 传统的底部居中一行
subtitle_fill = "solid" # knockout = 文字从画面里镂空，字中透出提亮虚化的同一帧
subtitle_outline = 1.1 # 描边宽度；0 为无描边（更融入画面，但需要画面本身够暗）
subtitle_highlight_color = "&H0000D7FF" # ASS 的金黄色（BGR）
visual_effects = true
image_composites = true # AI 按段落自动选择分屏、照片堆叠、双重曝光和节拍蒙太奇
image_composite_ratio = 0.24 # 多图镜头占比；副歌会适当提高
max_composite_images = 3 # 建议 2~3；4 仅适合短促高能蒙太奇
avoid_asset_repeats = true # 素材充足时每个镜头用不同素材；多图合成只消耗富余素材
blurred_image_background = true # 图片保持原比例，空余区域由同图模糊背景填满
image_background_blur = 26.0
image_foreground_scale = 0.92
vignette = true
film_grain = 1.6
look_strength = 0.72 # AI 导演色彩弧的应用强度
shot_match_strength = 0.3 # 不同来源素材的轻量曝光/饱和度匹配
professional_transitions = true
transition_min_seconds = 0.16
transition_max_seconds = 0.55
transition_density = 0.35 # 段落内部使用可见转场的比例；0 = 只在段落切换时转场，1 = 每个切点都转场

[render.subtitle_fonts]
energetic = "preset:energetic"
uplifting = "preset:modern"
melancholic = "preset:cinematic"
dreamy = "preset:dreamy"
romantic = "preset:lyrical"
dark = "preset:dark"
cinematic = "preset:cinematic"
"""

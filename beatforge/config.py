from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RenderConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crf: int = 19
    preset: str = "medium"
    min_shot_seconds: float = 1.8
    max_shot_seconds: float = 5.5
    subtitle_font: str = "auto"
    subtitle_fonts_dir: Path | None = None
    subtitle_fonts: dict[str, str] = Field(default_factory=lambda: {
        "energetic": "Arial Black",
        "uplifting": "Microsoft YaHei",
        "melancholic": "SimSun",
        "dreamy": "Microsoft YaHei Light",
        "romantic": "KaiTi",
        "dark": "SimHei",
        "cinematic": "Microsoft YaHei",
    })
    subtitle_size: int = 46
    subtitle_effect: Literal["auto", "karaoke", "cinematic", "bounce", "float", "glow", "typewriter"] = "auto"
    subtitle_margin: int = 72
    subtitle_highlight_color: str = "&H0000D7FF"
    visual_effects: bool = True
    vignette: bool = True
    film_grain: float = Field(default=1.6, ge=0, le=8)
    professional_transitions: bool = True
    transition_min_seconds: float = Field(default=.16, ge=.05, le=1.0)
    transition_max_seconds: float = Field(default=.55, ge=.1, le=1.5)


class AIConfig(BaseModel):
    enabled: bool = True
    device: Literal["auto", "cuda", "cpu"] = "auto"
    offline: bool = False
    asr_backend: Literal["qwen3", "faster-whisper"] = "qwen3"
    qwen_asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    clap_model: str = "laion/clap-htsat-fused"
    music_structure_backend: Literal["librosa", "beat-this", "allin1"] = "allin1"
    vision_backend: Literal["qwen3-vl-embedding", "siglip2"] = "qwen3-vl-embedding"
    vision_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    vision_reranker_model: str | None = "Qwen/Qwen3-VL-Reranker-2B"
    vision_rerank_top_k: int = Field(default=5, ge=0, le=20)
    frame_samples: int = Field(default=3, ge=1, le=12)


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
        if self.render.subtitle_fonts_dir and not self.render.subtitle_fonts_dir.is_absolute():
            self.render.subtitle_fonts_dir = (self.root / self.render.subtitle_fonts_dir).resolve()
        return self


def load_project(file: Path) -> ProjectConfig:
    project_file = file.resolve()
    with project_file.open("rb") as handle:
        data = tomllib.load(handle)
    data["root"] = project_file.parent
    data.setdefault("cache_dir", ".beatforge")
    return ProjectConfig.model_validate(data)


PROJECT_TEMPLATE = '''music = "music.mp3"
lyrics = "lyrics.lrc" # 可删除；缺失时由 Qwen3-ASR 自动转写
media_dir = "media"
output = "output.mp4"
cache_dir = ".beatforge"

[ai]
enabled = true
device = "auto"
offline = false
asr_backend = "qwen3"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B"
whisper_model = "small" # RTX 5070 可改为 large-v3-turbo
whisper_compute_type = "int8" # RTX 5070 可改为 int8_float16
clap_model = "laion/clap-htsat-fused"
music_structure_backend = "allin1" # 需要 music-ai extra；也可用 beat-this 或 librosa
vision_backend = "qwen3-vl-embedding"
vision_model = "Qwen/Qwen3-VL-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
vision_rerank_top_k = 5
frame_samples = 3

[render]
width = 1920
height = 1080
fps = 30
crf = 19
preset = "medium"
min_shot_seconds = 1.8
max_shot_seconds = 5.5
subtitle_font = "auto"
subtitle_fonts_dir = "fonts" # 可放入自定义 ttf/otf；不存在也不影响系统字体
subtitle_size = 46
subtitle_effect = "auto" # 也可固定为 karaoke/cinematic/bounce/float/glow/typewriter
subtitle_margin = 72
subtitle_highlight_color = "&H0000D7FF" # ASS 的金黄色（BGR）
visual_effects = true
vignette = true
film_grain = 1.6
professional_transitions = true
transition_min_seconds = 0.16
transition_max_seconds = 0.55

[render.subtitle_fonts]
energetic = "Arial Black"
uplifting = "Microsoft YaHei"
melancholic = "SimSun"
dreamy = "Microsoft YaHei Light"
romantic = "KaiTi"
dark = "SimHei"
cinematic = "Microsoft YaHei"
'''

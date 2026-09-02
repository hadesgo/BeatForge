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
    subtitle_font: str = "Microsoft YaHei"
    subtitle_size: int = 46


class AIConfig(BaseModel):
    enabled: bool = True
    device: Literal["auto", "cuda", "cpu"] = "auto"
    offline: bool = False
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    clap_model: str = "laion/clap-htsat-fused"
    vision_model: str = "google/siglip2-base-patch16-224"
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
        return self


def load_project(file: Path) -> ProjectConfig:
    project_file = file.resolve()
    with project_file.open("rb") as handle:
        data = tomllib.load(handle)
    data["root"] = project_file.parent
    data.setdefault("cache_dir", ".beatforge")
    return ProjectConfig.model_validate(data)


PROJECT_TEMPLATE = '''music = "music.mp3"
lyrics = "lyrics.lrc" # 可删除；缺失时由 Whisper 自动转写
media_dir = "media"
output = "output.mp4"
cache_dir = ".beatforge"

[ai]
enabled = true
device = "auto"
offline = false
whisper_model = "small" # RTX 5070 可改为 large-v3-turbo
whisper_compute_type = "int8" # RTX 5070 可改为 int8_float16
clap_model = "laion/clap-htsat-fused"
vision_model = "google/siglip2-base-patch16-224"
frame_samples = 3

[render]
width = 1920
height = 1080
fps = 30
crf = 19
preset = "medium"
min_shot_seconds = 1.8
max_shot_seconds = 5.5
subtitle_font = "Microsoft YaHei"
subtitle_size = 46
'''

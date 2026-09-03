from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from beatforge.config import AIConfig


WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


@dataclass(frozen=True, slots=True)
class ModelRequirement:
    component: str
    repo_id: str


@dataclass(frozen=True, slots=True)
class DownloadedModel:
    component: str
    repo_id: str
    local_path: str


class ModelDownloadError(RuntimeError):
    def __init__(self, completed: list[DownloadedModel], failures: list[tuple[ModelRequirement, Exception]]) -> None:
        self.completed = completed
        self.failures = failures
        details = "; ".join(f"{item.repo_id}: {error}" for item, error in failures)
        super().__init__(f"{len(failures)} 个模型下载失败：{details}")


def required_models(config: AIConfig) -> list[ModelRequirement]:
    """Return the deduplicated Hugging Face model set used by this project."""
    if not config.enabled:
        return []
    if config.asr_backend == "qwen3":
        items = [
            ModelRequirement("歌词识别", config.qwen_asr_model),
            ModelRequirement("歌词强制对齐", config.qwen_aligner_model),
        ]
    else:
        items = [ModelRequirement("歌词识别", WHISPER_REPOS.get(config.whisper_model, config.whisper_model))]
    items.extend([
        ModelRequirement("音乐情绪分析", config.clap_model),
        ModelRequirement("视觉语义检索", config.vision_model),
    ])
    if config.vision_reranker_model:
        items.append(ModelRequirement("视觉语义精排", config.vision_reranker_model))
    if config.director_enabled:
        items.append(ModelRequirement("AI 导演", config.director_model))
    unique: dict[str, ModelRequirement] = {}
    for item in items:
        unique.setdefault(item.repo_id, item)
    return list(unique.values())


def download_required_models(
    config: AIConfig,
    *,
    cache_dir: Path | None = None,
    max_workers: int = 4,
    progress: Callable[[str, ModelRequirement, str | None], None] | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> list[DownloadedModel]:
    """Download every configured HF model, continuing after individual failures."""
    if snapshot_download_fn is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("缺少 huggingface-hub，请先安装 BeatForge 的 ai extra") from exc
        snapshot_download_fn = snapshot_download
    completed: list[DownloadedModel] = []
    failures: list[tuple[ModelRequirement, Exception]] = []
    for item in required_models(config):
        if progress:
            progress("start", item, None)
        options: dict[str, object] = {"repo_id": item.repo_id, "max_workers": max_workers}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            options["cache_dir"] = str(cache_dir)
        try:
            local_path = str(snapshot_download_fn(**options))
            result = DownloadedModel(item.component, item.repo_id, local_path)
            completed.append(result)
            if progress:
                progress("complete", item, local_path)
        except Exception as exc:
            failures.append((item, exc))
            if progress:
                progress("failed", item, str(exc))
    if failures:
        raise ModelDownloadError(completed, failures)
    return completed


def write_download_manifest(models: list[DownloadedModel], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in models], ensure_ascii=False, indent=2), "utf-8")
    return target

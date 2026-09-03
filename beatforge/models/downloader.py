from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

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
    source: str = "huggingface"


class ModelDownloadError(RuntimeError):
    def __init__(self, completed: list[DownloadedModel], failures: list[tuple[ModelRequirement, Exception]]) -> None:
        self.completed = completed
        self.failures = failures
        details = "; ".join(f"{item.repo_id}: {error}" for item, error in failures)
        super().__init__(f"{len(failures)} 个模型下载失败：{details}")


def required_models(config: AIConfig) -> list[ModelRequirement]:
    """Return the deduplicated model set used by this project."""
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
    source: Literal["auto", "modelscope", "huggingface"] = "auto",
    fallback_to_huggingface: bool = True,
    progress: Callable[[str, ModelRequirement, str | None], None] | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
    modelscope_snapshot_download_fn: Callable[..., str] | None = None,
) -> list[DownloadedModel]:
    """Download configured models, preferring ModelScope for mainland China."""
    hf_download = snapshot_download_fn
    ms_download = modelscope_snapshot_download_fn
    needs_hf = source in {"auto", "huggingface"} or (source == "modelscope" and fallback_to_huggingface)
    if needs_hf and hf_download is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("缺少 huggingface-hub，请先安装 BeatForge 的 ai extra") from exc
        hf_download = snapshot_download
    if source in {"auto", "modelscope"} and ms_download is None:
        try:
            from modelscope import snapshot_download as modelscope_snapshot_download
        except ImportError as exc:
            raise RuntimeError("缺少 modelscope，请先安装 BeatForge 的 ai extra") from exc
        ms_download = modelscope_snapshot_download

    providers: list[tuple[str, Callable[..., str]]] = []
    if source in {"auto", "modelscope"}:
        assert ms_download is not None
        providers.append(("modelscope", ms_download))
    if source == "huggingface" or (source in {"auto", "modelscope"} and fallback_to_huggingface):
        assert hf_download is not None
        providers.append(("huggingface", hf_download))

    completed: list[DownloadedModel] = []
    failures: list[tuple[ModelRequirement, Exception]] = []
    for item in required_models(config):
        if progress:
            progress("start", item, None)
        provider_errors: list[str] = []
        for provider, download in providers:
            options: dict[str, object]
            if provider == "modelscope":
                options = {"model_id": item.repo_id}
                if cache_dir is not None:
                    target = cache_dir / "modelscope" / item.repo_id
                    target.mkdir(parents=True, exist_ok=True)
                    options["local_dir"] = str(target)
            else:
                options = {"repo_id": item.repo_id, "max_workers": max_workers}
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    options["cache_dir"] = str(cache_dir / "huggingface")
            try:
                local_path = str(download(**options))
                result = DownloadedModel(item.component, item.repo_id, local_path, provider)
                completed.append(result)
                if progress:
                    progress("complete", item, f"{provider}: {local_path}")
                break
            except Exception as exc:
                provider_errors.append(f"{provider}: {exc}")
        else:
            error = RuntimeError("；".join(provider_errors))
            failures.append((item, error))
            if progress:
                progress("failed", item, str(error))
    if failures:
        raise ModelDownloadError(completed, failures)
    return completed


def write_download_manifest(models: list[DownloadedModel], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(item) for item in models], ensure_ascii=False, indent=2), "utf-8")
    return target


def load_download_manifest(target: Path) -> dict[str, str]:
    """Return usable local model paths from a downloader manifest."""
    if not target.is_file():
        return {}
    try:
        entries = json.loads(target.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    resolved: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        repo_id, local_path = entry.get("repo_id"), entry.get("local_path")
        if isinstance(repo_id, str) and isinstance(local_path, str) and Path(local_path).exists():
            resolved[repo_id] = str(Path(local_path).resolve())
    return resolved

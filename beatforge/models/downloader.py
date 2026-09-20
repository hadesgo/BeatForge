from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from beatforge.config import AIConfig
from beatforge.models.separator import SeparationUnavailable

# ModelScope mirrors do not always use the same namespace as Hugging Face.
# Keep the configured repository ID canonical so the manifest can still map it
# to a local directory regardless of which provider supplied the snapshot.
MODELSCOPE_REPO_ALIASES = {
    "tencent/WeMM-Embedding-2B": "tencent-community/WeMM-Embedding-2B",
}


@dataclass(frozen=True, slots=True)
class ModelRequirement:
    """One thing to fetch, and how to fetch it.

    ``repo_id`` is a Hugging Face / ModelScope repository for the ``snapshot`` provider
    and a checkpoint filename for ``audio-separator``, which keeps its own catalogue and
    downloads single files rather than repository snapshots.
    """

    component: str
    repo_id: str
    provider: str = "snapshot"


@dataclass(frozen=True, slots=True)
class DownloadedModel:
    component: str
    repo_id: str
    local_path: str
    source: str = "huggingface"


class ModelDownloadError(RuntimeError):
    def __init__(
        self,
        completed: list[DownloadedModel],
        failures: list[tuple[ModelRequirement, Exception]],
    ) -> None:
        self.completed = completed
        self.failures = failures
        details = "; ".join(f"{item.repo_id}: {error}" for item, error in failures)
        super().__init__(f"{len(failures)} 个模型下载失败：{details}")


def required_models(config: AIConfig) -> list[ModelRequirement]:
    """Return the deduplicated model set used by this project."""
    if not config.enabled:
        return []
    items = [
        ModelRequirement("歌词识别", config.qwen_asr_model),
        ModelRequirement("歌词强制对齐", config.qwen_aligner_model),
    ]
    items.extend(
        [
            ModelRequirement("音乐情绪分析", config.clap_model),
            ModelRequirement("视觉语义检索", config.vision_model),
        ]
    )
    if config.vision_reranker_model:
        items.append(ModelRequirement("视觉语义精排", config.vision_reranker_model))
    if config.director_enabled:
        items.append(ModelRequirement(
            "AI 导演", config.director_model,
            # A snapshot would pull both packings - 13 GB to get one usable file - so the
            # GGUF provider fetches the single quantisation by name instead.
            provider="gguf",
        ))
    if config.separate_vocals and config.separation_model:
        # Not a repository snapshot: audio-separator keeps its own catalogue and fetches
        # the checkpoint plus its config YAML itself.
        items.append(ModelRequirement(
            "人声分离", config.separation_model, provider="audio-separator",
        ))
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
    separator_download_fn: Callable[[ModelRequirement, Path | None], DownloadedModel] | None = None,
) -> list[DownloadedModel]:
    """Download configured models, preferring ModelScope for mainland China."""
    hf_download = snapshot_download_fn
    ms_download = modelscope_snapshot_download_fn
    needs_hf = source in {"auto", "huggingface"} or (
        source == "modelscope" and fallback_to_huggingface
    )
    if needs_hf and hf_download is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "缺少 huggingface-hub，请先安装 BeatForge 的 ai extra"
            ) from exc
        hf_download = snapshot_download
    if source in {"auto", "modelscope"} and ms_download is None:
        try:
            from modelscope import snapshot_download as modelscope_snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "缺少 modelscope，请先安装 BeatForge 的 ai extra"
            ) from exc
        ms_download = modelscope_snapshot_download

    providers: list[tuple[str, Callable[..., str]]] = []
    if source in {"auto", "modelscope"}:
        assert ms_download is not None
        providers.append(("modelscope", ms_download))
    if source == "huggingface" or (
        source in {"auto", "modelscope"} and fallback_to_huggingface
    ):
        assert hf_download is not None
        providers.append(("huggingface", hf_download))

    completed: list[DownloadedModel] = []
    failures: list[tuple[ModelRequirement, Exception]] = []
    for item in required_models(config):
        if progress:
            progress("start", item, None)
        if item.provider == "gguf":
            try:
                result = _download_gguf(item, config, cache_dir, providers)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append((item, exc))
                if progress:
                    progress("failed", item, str(exc))
            else:
                completed.append(result)
                if progress:
                    progress("complete", item, f"gguf: {result.local_path}")
            continue
        if item.provider == "audio-separator":
            fetch = separator_download_fn or _download_separator_model
            try:
                result = fetch(item, cache_dir)
            except SeparationUnavailable as exc:
                # A missing optional extra is a configuration choice, not a failure.
                # Failing the whole command here would break the default experience for
                # anyone who has not installed the separation extra, while saying nothing
                # would leave them wondering later why separation never happens.
                if progress:
                    progress("skipped", item, str(exc))
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append((item, exc))
                if progress:
                    progress("failed", item, str(exc))
            else:
                completed.append(result)
                if progress:
                    progress("complete", item, f"audio-separator: {result.local_path}")
            continue
        provider_errors: list[str] = []
        for provider, download in providers:
            options: dict[str, object]
            if provider == "modelscope":
                provider_repo_id = MODELSCOPE_REPO_ALIASES.get(
                    item.repo_id, item.repo_id
                )
                options = {"model_id": provider_repo_id}
                if cache_dir is not None:
                    target = cache_dir / "modelscope" / provider_repo_id
                    target.mkdir(parents=True, exist_ok=True)
                    options["local_dir"] = str(target)
            else:
                options = {"repo_id": item.repo_id, "max_workers": max_workers}
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    options["cache_dir"] = str(cache_dir / "huggingface")
            try:
                local_path = str(download(**options))
                result = DownloadedModel(
                    item.component, item.repo_id, local_path, provider
                )
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


def _download_gguf(
    item: ModelRequirement,
    config: AIConfig,
    cache_dir: Path | None,
    providers: list[tuple[str, Callable[..., str]]],
) -> DownloadedModel:
    """Fetch one GGUF file out of a repository, first provider that answers wins.

    ``allow_patterns`` rather than a bare snapshot: the checkpoint repository holds both
    packings - PTQ1_0 and PQ2_0 - so a snapshot would spend 13 GB of bandwidth and disk
    on a choice the user already made. ModelScope goes first because that is where the
    checkpoint is mirrored for the project's target network; Hugging Face is the
    fallback, and both drop the file in the same directory so the runtime finds it there
    whichever one supplied it.
    """
    target_dir = (cache_dir or Path(".")) / "director"
    target_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for provider, download in providers:
        try:
            download(**_gguf_options(provider, item, config, target_dir))
        except Exception as exc:  # noqa: BLE001 - the next provider gets a turn
            errors.append(f"{provider}: {exc}")
            continue
        resolved = target_dir / config.director_gguf_file
        if not resolved.is_file():
            errors.append(f"{provider}: 下载后仍找不到 {resolved}")
            continue
        return DownloadedModel(item.component, item.repo_id, str(resolved), provider)
    raise RuntimeError("；".join(errors) or "没有可用的 GGUF 下载源")


def _gguf_options(
    provider: str, item: ModelRequirement, config: AIConfig, target_dir: Path,
) -> dict[str, object]:
    """Single-file download options for one provider.

    The two SDKs spell the repository argument differently, and ``allow_patterns`` has
    to be a list for Hugging Face: it iterates patterns, so a bare string is matched one
    character at a time and downloads nothing at all.
    """
    filename = config.director_gguf_file
    if provider == "modelscope":
        return {
            "model_id": MODELSCOPE_REPO_ALIASES.get(item.repo_id, item.repo_id),
            "allow_patterns": filename,
            "local_dir": str(target_dir),
        }
    return {
        "repo_id": item.repo_id,
        "allow_patterns": [filename],
        "local_dir": str(target_dir),
    }


def _download_separator_model(
    item: ModelRequirement, cache_dir: Path | None,
) -> DownloadedModel:
    """Fetch one audio-separator checkpoint, without loading it.

    ``download_model_files`` is the package's download-only entry point, so this does not
    need the model in memory or a GPU - but it does need the package, and it reads its
    catalogue from the network, so it cannot work offline.
    """
    from beatforge.models.separator import SEPARATOR_SUBDIR, SeparationUnavailable

    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise SeparationUnavailable(
            "未安装 audio-separator；跳过。需要时运行 uv sync --extra ai --extra separation"
        ) from exc
    if cache_dir is None:
        raise RuntimeError("下载人声分离模型需要指定 cache_dir")
    # ``cache_dir`` here is the models directory, so the checkpoint lands where the
    # runtime looks for it: <project cache>/models/separator.
    model_dir = cache_dir / SEPARATOR_SUBDIR[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(log_level=logging.ERROR, model_file_dir=str(model_dir))
    try:
        separator.download_model_files(item.repo_id)
    finally:
        del separator
    path = model_dir / item.repo_id
    if not path.is_file():
        raise RuntimeError(f"下载后仍找不到检查点：{path}")
    return DownloadedModel(item.component, item.repo_id, str(path), "audio-separator")


def write_download_manifest(models: list[DownloadedModel], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(item) for item in models], ensure_ascii=False, indent=2),
        "utf-8",
    )
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
        if (
            isinstance(repo_id, str)
            and isinstance(local_path, str)
            and Path(local_path).exists()
        ):
            resolved[repo_id] = str(Path(local_path).resolve())
    return resolved

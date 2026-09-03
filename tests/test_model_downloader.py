import json
from pathlib import Path

import pytest

from beatforge.config import AIConfig
from beatforge.models.downloader import (
    ModelDownloadError,
    download_required_models,
    load_download_manifest,
    required_models,
    write_download_manifest,
)


def test_default_manifest_contains_every_configured_huggingface_model_once() -> None:
    repos = [item.repo_id for item in required_models(AIConfig())]
    assert repos == [
        "Qwen/Qwen3-ASR-1.7B-hf",
        "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        "laion/clap-htsat-fused",
        "Qwen/Qwen3-VL-Embedding-8B",
        "Qwen/Qwen3-VL-Reranker-8B",
        "Qwen/Qwen3.5-9B",
    ]


def test_whisper_backend_resolves_short_model_name_and_deduplicates() -> None:
    config = AIConfig(
        asr_backend="faster-whisper", whisper_model="large-v3-turbo",
        director_model="laion/clap-htsat-fused",
    )
    repos = [item.repo_id for item in required_models(config)]
    assert repos[0] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert repos.count("laion/clap-htsat-fused") == 1


def test_downloader_uses_one_cache_and_writes_manifest(tmp_path: Path) -> None:
    calls = []

    def fake_download(**options):
        calls.append(options)
        return tmp_path / "snapshots" / options["repo_id"].replace("/", "--")

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config, cache_dir=tmp_path / "hf", max_workers=3,
        source="huggingface", snapshot_download_fn=fake_download,
    )
    manifest = write_download_manifest(models, tmp_path / "project" / "models.json")

    assert len(models) == 4
    assert all(call["cache_dir"] == str(tmp_path / "hf" / "huggingface") for call in calls)
    assert all(call["max_workers"] == 3 for call in calls)
    assert json.loads(manifest.read_text("utf-8"))[0]["component"] == "歌词识别"


def test_downloader_reports_failures_after_attempting_remaining_models(tmp_path: Path) -> None:
    attempted = []

    def fake_download(**options):
        attempted.append(options["repo_id"])
        if "ForcedAligner" in options["repo_id"]:
            raise OSError("network unavailable")
        return tmp_path / "cached"

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    with pytest.raises(ModelDownloadError) as captured:
        download_required_models(config, source="huggingface", snapshot_download_fn=fake_download)

    assert len(attempted) == 4
    assert len(captured.value.completed) == 3
    assert captured.value.failures[0][0].component == "歌词强制对齐"


def test_modelscope_is_preferred_and_manifest_resolves_local_paths(tmp_path: Path) -> None:
    calls = []

    def fake_modelscope_download(**options):
        calls.append(options)
        target = Path(options["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", "utf-8")
        return target

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config, cache_dir=tmp_path / "models", source="modelscope",
        fallback_to_huggingface=False,
        modelscope_snapshot_download_fn=fake_modelscope_download,
    )
    manifest = write_download_manifest(models, tmp_path / "models.json")
    resolved = load_download_manifest(manifest)

    assert len(calls) == 4
    assert all("model_id" in call and "repo_id" not in call for call in calls)
    assert all(model.source == "modelscope" for model in models)
    assert resolved[config.qwen_asr_model] == str(Path(models[0].local_path).resolve())


def test_auto_source_falls_back_to_huggingface(tmp_path: Path) -> None:
    def failed_modelscope(**options):
        raise OSError(f"missing: {options['model_id']}")

    def working_huggingface(**options):
        return tmp_path / options["repo_id"].replace("/", "--")

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config, source="auto", snapshot_download_fn=working_huggingface,
        modelscope_snapshot_download_fn=failed_modelscope,
    )
    assert len(models) == 4
    assert all(model.source == "huggingface" for model in models)

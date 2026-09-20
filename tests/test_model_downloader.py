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


def _stub_separator(tmp_path: Path):
    """Stand in for the real checkpoint fetch.

    Without this the downloader tests would reach the network and pull a 913 MB model
    into a temporary directory - slow, flaky, and nothing to do with what they test.
    """
    from beatforge.models.downloader import DownloadedModel

    def fetch(item, cache_dir):
        target = (cache_dir or tmp_path) / "separator" / item.repo_id
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ckpt")
        return DownloadedModel(item.component, item.repo_id, str(target), "audio-separator")

    return fetch


def test_default_manifest_contains_every_configured_huggingface_model_once() -> None:
    repos = [
        item.repo_id for item in required_models(AIConfig())
        if item.provider == "snapshot"
    ]
    assert repos == [
        "Qwen/Qwen3-ASR-1.7B-hf",
        "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        "laion/clap-htsat-fused",
        "tencent/WeMM-Embedding-2B",
        "Qwen/Qwen3-VL-Reranker-2B",
    ]


def test_model_manifest_deduplicates_shared_repository() -> None:
    config = AIConfig(director_model="laion/clap-htsat-fused")
    repos = [item.repo_id for item in required_models(config)]
    assert repos[0] == "Qwen/Qwen3-ASR-1.7B-hf"
    assert repos.count("laion/clap-htsat-fused") == 1


def test_downloader_uses_one_cache_and_writes_manifest(tmp_path: Path) -> None:
    calls = []

    def fake_download(**options):
        calls.append(options)
        return tmp_path / "snapshots" / options["repo_id"].replace("/", "--")

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config,
        cache_dir=tmp_path / "hf",
        max_workers=3,
        source="huggingface",
        snapshot_download_fn=fake_download,
        separator_download_fn=_stub_separator(tmp_path),
    )
    manifest = write_download_manifest(models, tmp_path / "project" / "models.json")

    assert len(models) == 5, "four snapshots plus the separation checkpoint"
    assert all(
        call["cache_dir"] == str(tmp_path / "hf" / "huggingface") for call in calls
    )
    assert all(call["max_workers"] == 3 for call in calls)
    assert json.loads(manifest.read_text("utf-8"))[0]["component"] == "歌词识别"


def test_downloader_reports_failures_after_attempting_remaining_models(
    tmp_path: Path,
) -> None:
    attempted = []

    def fake_download(**options):
        attempted.append(options["repo_id"])
        if "ForcedAligner" in options["repo_id"]:
            raise OSError("network unavailable")
        return tmp_path / "cached"

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    with pytest.raises(ModelDownloadError) as captured:
        download_required_models(
            config, source="huggingface", snapshot_download_fn=fake_download,
            separator_download_fn=_stub_separator(tmp_path),
        )

    assert len(attempted) == 4, "the separator does not go through snapshot_download"
    assert len(captured.value.completed) == 4
    assert captured.value.failures[0][0].component == "歌词强制对齐"


def test_modelscope_is_preferred_and_manifest_resolves_local_paths(
    tmp_path: Path,
) -> None:
    calls = []

    def fake_modelscope_download(**options):
        calls.append(options)
        target = Path(options["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", "utf-8")
        return target

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config,
        cache_dir=tmp_path / "models",
        source="modelscope",
        fallback_to_huggingface=False,
        modelscope_snapshot_download_fn=fake_modelscope_download,
        separator_download_fn=_stub_separator(tmp_path),
    )
    manifest = write_download_manifest(models, tmp_path / "models.json")
    resolved = load_download_manifest(manifest)

    assert len(calls) == 4
    assert all("model_id" in call and "repo_id" not in call for call in calls)
    assert calls[3]["model_id"] == "tencent-community/WeMM-Embedding-2B"
    assert sum(1 for model in models if model.source == "modelscope") == 4
    assert models[3].repo_id == "tencent/WeMM-Embedding-2B"
    assert resolved[config.qwen_asr_model] == str(Path(models[0].local_path).resolve())
    assert resolved[config.vision_model] == str(Path(models[3].local_path).resolve())


def test_auto_source_falls_back_to_huggingface(tmp_path: Path) -> None:
    def failed_modelscope(**options):
        raise OSError(f"missing: {options['model_id']}")

    def working_huggingface(**options):
        return tmp_path / options["repo_id"].replace("/", "--")

    config = AIConfig(vision_reranker_model=None, director_enabled=False)
    models = download_required_models(
        config,
        source="auto",
        snapshot_download_fn=working_huggingface,
        modelscope_snapshot_download_fn=failed_modelscope,
        separator_download_fn=_stub_separator(tmp_path),
    )
    assert len(models) == 5
    assert sum(1 for model in models if model.source == "huggingface") == 4


# ------------------------------------------------------ the separation checkpoint


def test_the_separation_model_joins_the_download_list() -> None:
    """It is a checkpoint filename, not a repository, so it needs its own provider."""
    items = {item.component: item for item in required_models(AIConfig())}

    separation = items["人声分离"]
    assert separation.provider == "audio-separator"
    assert separation.repo_id == AIConfig().separation_model


def test_turning_separation_off_removes_it_from_the_download_list() -> None:
    """Nobody should have to fetch a model their config says will not be used."""
    config = AIConfig(separate_vocals=False)

    assert "人声分离" not in {item.component for item in required_models(config)}


def test_the_downloader_and_the_runtime_agree_on_where_the_checkpoint_lives() -> None:
    """The whole point of the download step is that the runtime then finds the file.

    Two directories that disagree would mean the download silently not helping, and the
    model being fetched a second time on first use - with nothing anywhere to say so.
    """
    from beatforge.models.separator import SEPARATOR_SUBDIR, separator_model_dir

    project_cache = Path(".beatforge")
    models_dir = project_cache / "models"      # what the CLI hands the downloader

    assert models_dir / SEPARATOR_SUBDIR[-1] == separator_model_dir(project_cache)


def test_the_checkpoint_is_downloaded_without_loading_it(tmp_path: Path, monkeypatch) -> None:
    """``download_model_files`` is the package's download-only entry point.

    Loading the model to fetch it would need a GPU and several gigabytes of RAM to
    achieve a file copy.
    """
    import sys
    from types import SimpleNamespace

    from beatforge.models.downloader import _download_separator_model

    seen: dict[str, object] = {}

    class FakeSeparator:
        def __init__(self, **options):
            seen["options"] = options

        def download_model_files(self, filename):
            seen["filename"] = filename
            Path(seen["options"]["model_file_dir"], filename).write_bytes(b"ckpt")
            return (filename, "roformer", "vocals", "path", None)

    monkeypatch.setitem(sys.modules, "audio_separator", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "audio_separator.separator", SimpleNamespace(Separator=FakeSeparator),
    )
    item = next(x for x in required_models(AIConfig()) if x.provider == "audio-separator")
    models_dir = tmp_path / "models"

    result = _download_separator_model(item, models_dir)

    assert seen["filename"] == item.repo_id
    assert Path(result.local_path).is_file()
    assert result.source == "audio-separator"
    assert Path(result.local_path).parent == models_dir / "separator"


def test_a_missing_separation_extra_skips_instead_of_failing(tmp_path: Path, monkeypatch) -> None:
    """``separate_vocals`` defaults to on, so failing here would break the default run.

    Reporting a skip still tells the user which extra they are missing - and leaves a
    genuine download error as a failure, which is a different thing.
    """
    import sys

    from beatforge.models.downloader import download_required_models

    monkeypatch.setitem(sys.modules, "audio_separator", None)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", None)
    states: list[str] = []

    def snapshot_download(**options):
        # The director's GGUF is fetched by name rather than as a snapshot, so it carries
        # allow_patterns plus a local_dir; everything else is a plain repository snapshot.
        if "allow_patterns" in options:
            target = Path(options["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            for name in options["allow_patterns"]:
                (target / name).write_bytes(b"gguf")
            return str(target)
        target = Path(options.get("cache_dir", tmp_path)) / options["repo_id"].replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    models = download_required_models(
        AIConfig(), cache_dir=tmp_path / "models", source="huggingface",
        snapshot_download_fn=snapshot_download,
        progress=lambda state, item, detail: states.append(state),
    )

    assert "skipped" in states, "the missing extra was not reported"
    assert all(item.source != "audio-separator" for item in models)
    assert models, "the other models must still have been fetched"


def test_the_director_gguf_is_fetched_by_name_not_as_a_snapshot() -> None:
    """The checkpoint repository holds both packings; a snapshot costs 13 GB for one file.

    It also has to leave the snapshot provider, or the download loop would try to pull the
    whole repository as well.
    """
    director = next(
        item for item in required_models(AIConfig()) if item.component == "AI 导演"
    )

    assert director.provider == "gguf"
    assert director.repo_id == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert director.repo_id not in {
        item.repo_id for item in required_models(AIConfig()) if item.provider == "snapshot"
    }


def test_the_director_gguf_is_fetched_from_modelscope_by_default(tmp_path: Path) -> None:
    """The checkpoint is mirrored on ModelScope, which is the faster mirror for the
    project's target network, so that has to be the provider that answers first."""
    calls = []

    def fake_modelscope_download(**options):
        target = Path(options["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        if "allow_patterns" not in options:
            return str(target)  # a plain repository snapshot
        calls.append(options)
        # Only the single quantisation, via allow_patterns - never the whole repository.
        (target / options["allow_patterns"]).write_bytes(b"gguf")
        return str(target)

    def unexpected_huggingface(**options):
        raise AssertionError(f"should not have fallen back: {options}")

    config = AIConfig(vision_reranker_model=None, separate_vocals=False)
    models = download_required_models(
        config,
        cache_dir=tmp_path / "models",
        source="auto",
        snapshot_download_fn=unexpected_huggingface,
        modelscope_snapshot_download_fn=fake_modelscope_download,
    )

    director = next(model for model in models if model.component == "AI 导演")
    assert director.repo_id == "prism-ml/Ternary-Bonsai-2-27B-gguf"
    assert director.source == "modelscope"
    assert Path(director.local_path).name == config.director_gguf_file
    assert calls == [{
        "model_id": "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "allow_patterns": config.director_gguf_file,
        "local_dir": str(tmp_path / "models" / "director"),
    }]
    # The other snapshots still go through the snapshot provider.
    assert {model.component for model in models} == {
        "歌词识别", "歌词强制对齐", "音乐情绪分析", "视觉语义检索", "AI 导演",
    }
    # The download has to leave the file where the runtime then looks for it.
    from beatforge.models.llama_server import find_gguf

    assert find_gguf(None, tmp_path, config.director_model) == Path(director.local_path)


def test_the_director_gguf_falls_back_to_huggingface(tmp_path: Path) -> None:
    """Same fallback contract as the snapshots, with one provider-specific wrinkle:
    ``allow_patterns`` must be a list for Hugging Face, which iterates it as patterns
    and would match a bare string one character at a time."""
    calls = []

    def failed_modelscope(**options):
        raise OSError(f"missing: {options.get('model_id')}")

    def fake_huggingface(**options):
        calls.append(options)
        if "allow_patterns" not in options:
            return str(options["cache_dir"])  # a plain repository snapshot
        target = Path(options["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / options["allow_patterns"][0]).write_bytes(b"gguf")
        return str(target)

    config = AIConfig(vision_reranker_model=None, separate_vocals=False)
    models = download_required_models(
        config,
        cache_dir=tmp_path / "models",
        source="auto",
        snapshot_download_fn=fake_huggingface,
        modelscope_snapshot_download_fn=failed_modelscope,
    )

    director = next(model for model in models if model.component == "AI 导演")
    assert director.source == "huggingface"
    gguf_call = next(call for call in calls if "allow_patterns" in call)
    assert gguf_call["allow_patterns"] == [config.director_gguf_file]
    assert gguf_call["repo_id"] == config.director_model


def test_the_director_gguf_reports_a_provider_that_downloads_nothing(tmp_path: Path) -> None:
    """A provider that returns cleanly but leaves no file must not be treated as success.

    Otherwise the manifest would point at a path that is not there, and the failure
    would only surface much later as a missing checkpoint.
    """
    def empty_download(**options):
        target = Path(options["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    config = AIConfig(vision_reranker_model=None, separate_vocals=False)
    with pytest.raises(ModelDownloadError) as captured:
        download_required_models(
            config,
            cache_dir=tmp_path / "models",
            source="modelscope",
            fallback_to_huggingface=False,
            modelscope_snapshot_download_fn=empty_download,
        )

    assert "下载后仍找不到" in str(captured.value)
    assert captured.value.failures[0][0].component == "AI 导演"


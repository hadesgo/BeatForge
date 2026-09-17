import sys
from types import SimpleNamespace

import pytest

from beatforge.runtime import require_usable_ai_device


def test_cuda_build_with_unavailable_driver_fails_before_model_load(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="升级 NVIDIA 驱动"):
        require_usable_ai_device("auto", "cpu")


def test_explicit_cpu_allows_cuda_build_without_a_driver(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    require_usable_ai_device("cpu", "cpu")


def test_native_asr_version_check_survives_a_broken_environment() -> None:
    """``doctor`` exists to report a broken environment, so it must not break on one.

    An interrupted ``uv sync`` leaves a dist-info directory with no METADATA, and
    ``importlib.metadata.version`` answers that with ``None`` instead of raising. A
    pre-release version does not parse as a plain integer pair either.
    """
    from beatforge.cli import _supports_native_asr

    assert _supports_native_asr("5.16.1") is True
    assert _supports_native_asr("5.13.0") is True
    assert _supports_native_asr("5.12.9") is False
    assert _supports_native_asr(None) is False, "a half-written dist-info reports None"
    assert _supports_native_asr("garbage") is False


def test_release_gpu_survives_an_unusable_torch(monkeypatch) -> None:
    """It is best-effort cleanup and must never be the thing that fails a stage.

    A torch that imports but has no ``cuda`` attribute is exactly the state an
    interrupted install leaves behind, and it is the state cleanup is most likely to
    run in.
    """
    import sys
    from types import SimpleNamespace

    from beatforge.runtime import release_gpu

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    release_gpu()  # must not raise

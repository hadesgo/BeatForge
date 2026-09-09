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

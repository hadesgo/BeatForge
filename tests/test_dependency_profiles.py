from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_only_cpu_and_cuda130_pytorch_profiles_are_available() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    extras = project["project"]["optional-dependencies"]
    assert extras["ai-cuda"] == [
        "torch==2.14.0",
        "torchvision==0.29.0",
        "torchaudio==2.11.0",
        "torchcodec==0.16.0",
    ]
    assert "ai-cuda126" not in extras
    assert all(
        "bitsandbytes" not in dependency
        for dependencies in extras.values()
        for dependency in dependencies
    )

    indexes = {item["name"]: item["url"] for item in project["tool"]["uv"]["index"]}
    assert indexes["pytorch-cu130"] == "https://download.pytorch.org/whl/cu130"
    assert "pytorch-cu126" not in indexes

    for package in ("torch", "torchvision", "torchaudio", "torchcodec"):
        routes = project["tool"]["uv"]["sources"][package]
        assert {"index": "pytorch-cu130", "extra": "ai-cuda"} in routes
        assert all(route["extra"] != "ai-cuda126" for route in routes)


def test_pytorch_profiles_are_pairwise_exclusive() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    conflicts = {
        frozenset(item["extra"] for item in conflict)
        for conflict in project["tool"]["uv"]["conflicts"]
    }

    assert conflicts >= {
        frozenset(("ai-cpu", "ai-cuda")),
    }

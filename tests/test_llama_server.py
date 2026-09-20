"""The llama.cpp director engine, exercised without a checkpoint or a real server."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.models import llama_server
from beatforge.models.ai_director import (
    PROMPT_LADDER,
    _build_context,
    _treat_with_llamacpp,
)


def _analysis() -> AudioAnalysis:
    return AudioAnalysis(
        duration=8, bpm=120, beats=[0, 2, 4, 6, 8], sections=[0, 4, 8],
        energy_times=[0, 4, 8], energy_values=[.4, .7, .5], average_energy=.55,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["intro", "chorus"],
    )


def _lyrics() -> list[LyricLine]:
    return [LyricLine(0, 4, "独自醒来"), LyricLine(4, 8, "奔向天光")]


def _assets(count: int) -> list[MediaAsset]:
    return [
        MediaAsset(index, Path(f"{index}.jpg"), "image", float("inf"), 1920, 1080,
                   quality_score=.5 + index * .001)
        for index in range(count)
    ]


def _config(**overrides) -> AIConfig:
    base = {
        "director_gguf": Path("model.gguf"),
        "director_llama_server": Path("llama-server.exe"),
    }
    base.update(overrides)
    return AIConfig(**base)


# ------------------------------------------------------------ binary and model lookup


def test_a_missing_server_binary_names_the_fork(tmp_path: Path) -> None:
    """Stock llama.cpp either refuses a ternary checkpoint or answers with nonsense.

    Someone who installs the wrong build gets a confusing failure much later, so the
    error has to say which build is needed before they get there.
    """
    with pytest.raises(llama_server.LlamaServerError) as captured:
        llama_server.find_server_binary(None, tmp_path)

    assert "PrismML" in str(captured.value)
    assert "llama.cpp/releases" in str(captured.value)


def test_the_server_is_found_in_the_cache_directory(tmp_path: Path) -> None:
    """Unpacking a fork into the project cache is how it avoids shadowing a system build."""
    binary = tmp_path / "bin" / "llama-server.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"x")

    assert llama_server.find_server_binary(None, tmp_path) == binary


def test_an_explicit_binary_path_is_used_verbatim(tmp_path: Path) -> None:
    binary = tmp_path / "custom" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"x")

    assert llama_server.find_server_binary(binary, None) == binary


def test_a_missing_gguf_points_at_download_models(tmp_path: Path) -> None:
    with pytest.raises(llama_server.LlamaServerError) as captured:
        llama_server.find_gguf(None, tmp_path, "prism-ml/Ternary-Bonsai-2-27B-gguf")

    assert "download-models" in str(captured.value)


def test_the_gguf_is_found_where_download_models_puts_it(tmp_path: Path) -> None:
    model = tmp_path / "models" / "director" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"gguf")

    assert llama_server.find_gguf(None, tmp_path, "any/repo") == model


def test_a_free_port_is_found_when_the_preferred_one_is_taken() -> None:
    """A stale server from a previous run should not fail the whole director stage."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]

        assert llama_server._free_port(taken) != taken


# --------------------------------------------------------------- the chat request


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_the_request_constrains_the_output_with_a_schema(monkeypatch) -> None:
    """Constraining it server-side is what removes most of the repair rounds.

    Free-form JSON from a 2-bit checkpoint is exactly where a schema-less request
    fails, and every failure costs a full second generation.
    """
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", fake_urlopen)
    server = llama_server.LlamaServer(base_url="http://127.0.0.1:8917")
    schema = {"type": "object", "properties": {"concept": {"type": "string"}}}

    server.chat([{"role": "user", "content": "hi"}], json_schema=schema, temperature=0.5)

    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["response_format"] == {"type": "json_schema", "json_schema": schema}
    assert seen["body"]["temperature"] == 0.5
    assert seen["body"]["stream"] is False


def test_a_server_error_surfaces_the_body(monkeypatch) -> None:
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 500, "boom", {}, __import__("io").BytesIO(b"model not loaded"),
        )

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", fake_urlopen)
    server = llama_server.LlamaServer(base_url="http://127.0.0.1:8917")

    with pytest.raises(llama_server.LlamaServerError) as captured:
        server.chat([{"role": "user", "content": "hi"}])

    assert "model not loaded" in str(captured.value)


def test_an_unexpected_response_shape_is_reported_not_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_server.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse({"error": "nope"}),
    )
    server = llama_server.LlamaServer(base_url="http://127.0.0.1:8917")

    with pytest.raises(llama_server.LlamaServerError):
        server.chat([{"role": "user", "content": "hi"}])


# ----------------------------------------------------------------- the treatment


class _FakeServer:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.base_url = "http://127.0.0.1:8917"

    def chat(self, messages, *, json_schema=None, temperature=0.7, max_tokens=4096) -> str:
        self.requests.append({
            "messages": messages, "json_schema": json_schema,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        return self.replies.pop(0)


def _patch_server(monkeypatch, server: _FakeServer) -> None:
    from contextlib import contextmanager

    @contextmanager
    def fake_open(**kwargs):
        server.kwargs = kwargs
        yield server

    monkeypatch.setattr(llama_server, "open_server", fake_open)
    monkeypatch.setitem(sys.modules, "beatforge.models.llama_server", llama_server)


def test_the_llamacpp_engine_validates_and_returns_the_treatment(tmp_path, monkeypatch) -> None:
    payload = {
        "concept": "夜色中的独行", "narrative_arc": "从克制到释放", "visual_style": "冷调手持",
        "grade_profile": "cinematic", "sections": [],
    }
    server = _FakeServer([json.dumps(payload, ensure_ascii=False)])
    _patch_server(monkeypatch, server)

    result = _treat_with_llamacpp({}, _config(), tmp_path)

    assert result.concept == "夜色中的独行"
    assert server.requests[0]["json_schema"]["type"] == "object"
    assert server.kwargs["extra_args"] == _config().director_llama_args


def test_a_bad_first_answer_gets_exactly_one_repair_round(tmp_path, monkeypatch) -> None:
    """One correction, then give up - a second round would be a retry loop."""
    payload = {"concept": "修好了", "narrative_arc": "a", "visual_style": "b", "grade_profile": "dark"}
    server = _FakeServer(["这不是 JSON", json.dumps(payload, ensure_ascii=False)])
    _patch_server(monkeypatch, server)

    result = _treat_with_llamacpp({}, _config(), tmp_path)

    assert result.concept == "修好了"
    assert len(server.requests) == 2
    repair = server.requests[1]["messages"]
    assert repair[-1]["role"] == "user" and "校验" in repair[-1]["content"]
    assert repair[-2] == {"role": "assistant", "content": "这不是 JSON"}


def test_two_bad_answers_fail_rather_than_loop(tmp_path, monkeypatch) -> None:
    from pydantic import ValidationError

    server = _FakeServer(["nope", "still nope"])
    _patch_server(monkeypatch, server)

    with pytest.raises(ValidationError):
        _treat_with_llamacpp({}, _config(), tmp_path)

    assert len(server.requests) == 2, "a second repair round would be a retry loop"


def test_an_existing_server_url_skips_starting_one(monkeypatch) -> None:
    """Pointing at a server you already run avoids a second 7 GB copy in memory."""
    seen: dict = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        return 200, b'{"status":"ok"}'

    monkeypatch.setattr(llama_server, "_get", fake_get)
    monkeypatch.setattr(llama_server, "find_server_binary", lambda *a, **k: pytest.fail("started a server"))
    monkeypatch.setattr(llama_server, "find_gguf", lambda *a, **k: pytest.fail("looked for a model"))

    with llama_server.open_server(
        binary=None, gguf=None, cache_dir=None, repo_id="x", url="http://127.0.0.1:9999/",
    ) as server:
        assert server.base_url == "http://127.0.0.1:9999"
        assert not server.owned


# --------------------------------------------------------------- the prompt budget


def test_the_widest_ladder_rung_is_the_first_one() -> None:
    """The ladder can only take detail away, so it has to be ordered richest-first."""
    strides = [rung[0] for rung in PROMPT_LADDER]
    candidates = [rung[1] for rung in PROMPT_LADDER]
    assets = [rung[2] for rung in PROMPT_LADDER]

    assert strides == sorted(strides), "a bigger stride drops more lyrics"
    assert candidates == sorted(candidates, reverse=True)
    assert assets == sorted(assets, reverse=True)


def test_a_large_budget_keeps_every_candidate(tmp_path: Path) -> None:
    """The old ceiling of 24 assets was a memory workaround, not an editorial choice.

    With a budget that can hold the lot, the brief should carry every candidate rather
    than the first 24 in discovery order.
    """
    assets = _assets(60)
    context = _build_context(_analysis(), _lyrics(), assets, None)

    assert len(context["assets"]) == 60, "the brief was capped before the ladder ran"


def test_trimming_keeps_the_strongest_candidates(tmp_path: Path) -> None:
    """Trimming slices the list, so the list has to be in rank order.

    Built in discovery order, a raised cap would drop the best material first - the
    opposite of what the cap is for.
    """
    import numpy as np

    assets = _assets(40)
    similarities = np.zeros((len(_lyrics()), len(assets)))
    # The strongest material is last in discovery order, so a positional slice would
    # drop it first.
    similarities[:, 39] = 0.99
    similarities[:, 38] = 0.98

    context = _build_context(_analysis(), _lyrics(), assets, similarities)

    assert context["assets"][0]["id"] == assets[39].id
    assert context["assets"][1]["id"] == assets[38].id

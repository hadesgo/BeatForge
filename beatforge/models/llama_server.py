"""Serve a GGUF checkpoint with llama.cpp and talk to it over HTTP.

The director used to be an in-process transformers model. The ternary checkpoints it
moved to are GGUF-only and need a llama.cpp fork with the matching kernels - stock
llama.cpp either refuses them outright or, worse, loads them and produces garbage - so
the runtime has to be a separate process.

``llama-server`` rather than ``llama-cli`` for three reasons: it applies the model's own
chat template, it constrains the output with a JSON schema instead of asking the model
politely, and the checkpoint stays resident across calls. The process is started and
stopped by BeatForge, so there is nothing for the user to keep running.

Everything here is deliberately free of model knowledge, which is what lets the director
tests exercise the request shape and the retry logic without a 7 GB checkpoint on disk.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: How long to wait for the weights to load. A 7 GB checkpoint off a cold page cache is
#: slow, and llama.cpp gives no progress on the health endpoint while it reads.
STARTUP_TIMEOUT_SECONDS = 600
#: A single completion. The director emits a few thousand tokens at most.
REQUEST_TIMEOUT_SECONDS = 900
#: How often to poll the health endpoint while loading.
POLL_SECONDS = 0.5

#: Substrings that mean "this build cannot read that file", which is the failure mode
#: worth naming: a stock llama.cpp build loads a ternary checkpoint without complaint and
#: then answers with nonsense, or rejects it as an unknown quantisation type.
FORK_HINT = (
    "Ternary-Bonsai 需要 PrismML 的 llama.cpp 分支（自带三值混合注意力内核）。"
    "原版 llama.cpp 要么拒绝这种量化类型，要么加载后输出乱码。"
    "下载地址：https://github.com/PrismML-Eng/llama.cpp/releases"
)


class LlamaServerError(RuntimeError):
    """Raised when the server cannot be started, or answers with an error."""


def find_server_binary(configured: Path | None, cache_dir: Path | None = None) -> Path:
    """Locate ``llama-server``: explicit path, then PATH, then the cache directory.

    The cache lookup exists because unpacking a release archive into the project cache is
    the least invasive way to install a fork that must not shadow a system llama.cpp.
    """
    if configured is not None:
        path = Path(configured)
        if not path.is_file():
            raise LlamaServerError(f"director_llama_server 指向的文件不存在：{path}")
        return path
    found = shutil.which("llama-server")
    if found:
        return Path(found)
    if cache_dir is not None:
        for candidate in (
            cache_dir / "bin" / "llama-server.exe",
            cache_dir / "bin" / "llama-server",
            cache_dir / "bin" / "llama-server" / "llama-server.exe",
            cache_dir / "bin" / "llama-server" / "llama-server",
        ):
            if candidate.is_file():
                return candidate
    raise LlamaServerError(
        "找不到 llama-server。把 PrismML 分支的发布包解压到 "
        "<cache>/bin/，或在 project.toml 里设置 director_llama_server。\n" + FORK_HINT
    )


def find_gguf(configured: Path | None, cache_dir: Path | None, repo_id: str) -> Path:
    """Locate the checkpoint: explicit path, then whatever ``download-models`` cached."""
    if configured is not None:
        path = Path(configured)
        if not path.is_file():
            raise LlamaServerError(f"director_gguf 指向的文件不存在：{path}")
        return path
    if cache_dir is not None:
        for root in (
            cache_dir / "models" / "director",
            cache_dir / "models" / "huggingface" / f"models--{repo_id.replace('/', '--')}",
        ):
            if not root.is_dir():
                continue
            matches = sorted(root.rglob("*.gguf"))
            if matches:
                return matches[0]
    raise LlamaServerError(
        f"找不到 GGUF 检查点。先运行 beatforge download-models 拉取 {repo_id}，"
        "或在 project.toml 里设置 director_gguf。"
    )


def _free_port(preferred: int) -> int:
    """``preferred`` if it is free, otherwise the next one that is.

    A stale server from a previous run should not make the whole director stage fail, and
    binding is the only reliable way to find out whether a port is actually usable.
    """
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        return candidate
    raise LlamaServerError(f"从 {preferred} 起连续 20 个端口都被占用")


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        raise LlamaServerError(f"无法连接 {url}：{exc}") from exc


@dataclass
class LlamaServer:
    """A running llama-server, either one we started or one the user already had."""

    base_url: str
    process: subprocess.Popen | None = None

    @property
    def owned(self) -> bool:
        """Whether we are responsible for shutting it down."""
        return self.process is not None

    def chat(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """One completion, optionally constrained to a JSON schema.

        The schema matters more than it looks: the director's output is parsed straight
        into a pydantic model, and free-form JSON from a 2-bit model is where the retries
        used to come from. Constraining it server-side turns most of those into a single
        clean answer.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            raise LlamaServerError(f"llama-server 返回 {exc.code}：{detail}") from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise LlamaServerError(f"llama-server 请求失败：{exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServerError(f"llama-server 响应结构异常：{str(body)[:400]}") from exc

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.process = None


def _wait_until_ready(server: LlamaServer, process: subprocess.Popen) -> None:
    """Poll ``/health`` until the weights are loaded, or the process dies."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if process.stderr is not None:
                tail = process.stderr.read().decode("utf-8", "ignore")[-800:]
            raise LlamaServerError(
                f"llama-server 启动后立即退出（代码 {process.returncode}）。\n{tail}\n{FORK_HINT}"
            )
        try:
            status, body = _get(f"{server.base_url}/health", timeout=2.0)
        except LlamaServerError:
            time.sleep(POLL_SECONDS)
            continue
        if status == 200:
            try:
                state = json.loads(body.decode("utf-8")).get("status")
            except (json.JSONDecodeError, AttributeError):
                state = "ok"
            if state in ("ok", "no slot available"):
                return
        time.sleep(POLL_SECONDS)
    raise LlamaServerError(f"llama-server 在 {STARTUP_TIMEOUT_SECONDS}s 内没有就绪")


@contextmanager
def open_server(
    *,
    binary: Path | None,
    gguf: Path | None,
    cache_dir: Path | None,
    repo_id: str,
    extra_args: list[str] | None = None,
    port: int = 8917,
    url: str | None = None,
    log: Path | None = None,
) -> Iterator[LlamaServer]:
    """Yield a usable server, starting one only if the caller did not bring their own.

    Passing ``url`` skips process management entirely - that is the escape hatch for
    anyone who keeps a server running, and the reason the tests can drive this without a
    checkpoint.
    """
    if url:
        base = url.rstrip("/")
        server = LlamaServer(base_url=base)
        status, _ = _get(f"{base}/health", timeout=5.0)
        if status != 200:
            raise LlamaServerError(f"{base}/health 返回 {status}，该地址上似乎没有 llama-server")
        yield server
        return

    executable = find_server_binary(binary, cache_dir)
    model = find_gguf(gguf, cache_dir, repo_id)
    chosen = _free_port(port)
    command = [
        str(executable),
        "-m", str(model),
        "--host", "127.0.0.1",
        "--port", str(chosen),
        *(extra_args or []),
    ]
    sink = open(log, "wb") if log is not None else subprocess.DEVNULL
    try:
        process = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.PIPE if log is None else sink,
        )
    except OSError as exc:
        raise LlamaServerError(f"无法启动 {executable}：{exc}") from exc
    server = LlamaServer(base_url=f"http://127.0.0.1:{chosen}", process=process)
    try:
        _wait_until_ready(server, process)
        yield server
    finally:
        server.stop()
        if log is None and sink is not subprocess.DEVNULL:
            sink.close()

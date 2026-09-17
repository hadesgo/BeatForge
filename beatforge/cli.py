from __future__ import annotations

import json
import importlib.util
import importlib.metadata
import platform
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from beatforge.config import PROJECT_TEMPLATE, load_project
from beatforge.pipeline import run_project
from beatforge.runtime import resolve_device

app = typer.Typer(no_args_is_help=True, help="使用本地 AI 根据音乐、歌词和用户素材生成 MV")
console = Console()


@app.command()
def init(directory: Path = typer.Argument(..., help="项目目录")) -> None:
    """创建一个新的 MV 项目。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "media").mkdir(exist_ok=True)
    (directory / "fonts").mkdir(exist_ok=True)
    (directory / "project.toml").write_text(PROJECT_TEMPLATE, "utf-8")
    (directory / "lyrics.lrc").write_text("[00:00.00]在这里填写歌词\n[00:05.00]或删除本文件让 ASR 自动转写\n", "utf-8")
    console.print(f"[green]项目已创建[/green] {directory.resolve() / 'project.toml'}")


@app.command("run")
def run_command(
    project: Path = typer.Argument(Path("project.toml"), help="项目配置"),
    plan_only: bool = typer.Option(False, "--plan-only", help="只生成剪辑方案"),
    no_ai: bool = typer.Option(False, "--no-ai", help="不用模型，便于测试渲染链路"),
) -> None:
    """分析、编排并渲染一个项目。"""
    result = run_project(load_project(project), plan_only=plan_only, no_ai=no_ai)
    console.print(f"[bold green]完成[/bold green] {result}")


def _supports_native_asr(version: str | None) -> bool:
    """Qwen3-ASR needs ``transformers>=5.13``.

    ``importlib.metadata.version`` returns ``None`` - rather than raising - when a
    dist-info directory exists but its METADATA was never written, which is exactly what
    an interrupted ``uv sync`` leaves behind. A pre-release like ``5.16.1rc1`` also does
    not parse as a plain integer pair. Neither is worth crashing a diagnostic command
    over: both simply mean "not ready", which is the answer the command exists to give.
    """
    try:
        return tuple(int(part) for part in str(version).split(".")[:2]) >= (5, 13)
    except (TypeError, ValueError):
        return False


@app.command()
def doctor() -> None:
    """检查 FFmpeg、PyTorch、CUDA 和 AI 依赖。"""
    table = Table("项目", "状态", "信息")
    table.add_row("Python", "OK", platform.python_version())
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        table.add_row(binary, "OK" if found else "缺失", found or "请加入 PATH")
    try:
        import torch
        cuda = torch.cuda.is_available()
        info = torch.cuda.get_device_name(0) if cuda else "CPU 模式"
        enough_vram = False
        if cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 2**30
            enough_vram = vram >= 11.5
            info += f" · {vram:.1f} GB · CUDA {torch.version.cuda}"
        table.add_row("PyTorch", "OK", torch.__version__)
        table.add_row("GPU", "OK" if cuda else "未启用", info)
        if cuda:
            table.add_row("12GB 显存", "OK" if enough_vram else "不足", f"检测到 {vram:.1f} GB")
    except ImportError:
        table.add_row("AI 依赖", "缺失", "uv sync --extra ai --extra ai-cpu")
    except (AttributeError, RuntimeError, OSError) as exc:
        # torch is importable but unusable: an interrupted install leaves a namespace
        # package behind, and a driver can refuse to initialise. That is precisely the
        # state this command exists to report, so it must not be the state that breaks it.
        table.add_row("PyTorch", "损坏", f"{type(exc).__name__}: {exc} · 重新 uv sync")
    try:
        import torchaudio
        table.add_row("TorchAudio", "OK", torchaudio.__version__)
    except (ImportError, OSError, RuntimeError) as exc:
        table.add_row(
            "TorchAudio", "不兼容",
            f"{exc} · 请用同一个 uv profile 重新同步 torch/vision/audio",
        )
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    table.add_row(
        "Qwen3-ASR Native", "OK" if _supports_native_asr(transformers_version) else "未就绪",
        f"Transformers {transformers_version or '未安装'}",
    )
    separation = importlib.util.find_spec("audio_separator")
    table.add_row(
        "人声分离", "OK" if separation else "未安装",
        "separation extra（MelBand-RoFormer，转录前分离人声）",
    )
    sentence_transformers = importlib.util.find_spec("sentence_transformers")
    table.add_row("WeMM / 视觉精排", "OK" if sentence_transformers else "未安装", "sentence-transformers>=5.7")
    table.add_row(
        "Music Structure", "OK" if importlib.util.find_spec("allin1_infer") else "未安装",
        "music-ai extra（All-In-One）",
    )
    console.print(table)


@app.command("download-models")
def download_models(
    project: Path = typer.Argument(Path("project.toml"), help="用于确定模型名称的项目配置"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="可选的统一模型缓存目录"),
    workers: int = typer.Option(4, "--workers", min=1, max=16, help="单个模型的并行下载数"),
    source: str = typer.Option("auto", "--source", help="auto（魔搭优先）、modelscope 或 huggingface"),
    no_fallback: bool = typer.Option(False, "--no-fallback", help="魔搭失败时不回退 Hugging Face"),
) -> None:
    """统一下载项目启用的全部模型，默认优先使用 ModelScope。"""
    try:
        from beatforge.models.downloader import (
            ModelDownloadError,
            download_required_models,
            write_download_manifest,
        )
        loaded = load_project(project)
        if source not in {"auto", "modelscope", "huggingface"}:
            raise typer.BadParameter("--source 必须是 auto、modelscope 或 huggingface")

        def report(state, item, detail):
            if state == "start":
                console.print(f"[cyan]下载[/cyan] {item.component} · {item.repo_id}")
            elif state == "complete":
                console.print(f"[green]完成[/green] {item.repo_id}")
            else:
                console.print(f"[red]失败[/red] {item.repo_id} · {detail}")

        models = download_required_models(
            loaded.ai, cache_dir=cache_dir or loaded.cache_dir / "models",
            max_workers=workers, source=source,
            fallback_to_huggingface=not no_fallback, progress=report,
        )
        manifest = write_download_manifest(models, loaded.cache_dir / "models.json")
    except ModelDownloadError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[bold green]全部模型已缓存[/bold green] · 清单 {manifest}")
    console.print("可在 project.toml 中设置 offline = true")


if __name__ == "__main__":
    app()

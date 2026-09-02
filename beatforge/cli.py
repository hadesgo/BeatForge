from __future__ import annotations

import json
import importlib.util
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
        if cuda:
            info += f" · {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB · CUDA {torch.version.cuda}"
        table.add_row("PyTorch", "OK", torch.__version__)
        table.add_row("GPU", "OK" if cuda else "未启用", info)
    except ImportError:
        table.add_row("AI 依赖", "缺失", "uv sync --extra ai --extra ai-cpu")
    try:
        import ctranslate2
        types = ", ".join(sorted(ctranslate2.get_supported_compute_types(resolve_device("auto"))))
        table.add_row("CTranslate2", "OK", types)
    except (ImportError, RuntimeError) as exc:
        table.add_row("CTranslate2", "未就绪", str(exc))
    table.add_row("Qwen3-ASR", "OK" if importlib.util.find_spec("qwen_asr") else "未安装", "qwen extra")
    qwen_vl = importlib.util.find_spec("qwen3_vl_embedding")
    if qwen_vl is None:
        try:
            qwen_vl = importlib.util.find_spec("src.models.qwen3_vl_embedding")
        except ModuleNotFoundError:
            qwen_vl = None
    table.add_row("Qwen3-VL Embedding", "OK" if qwen_vl else "未安装", "qwen extra")
    table.add_row(
        "Music Structure", "OK" if importlib.util.find_spec("allin1_infer") else "未安装",
        "music-ai extra（All-In-One）",
    )
    console.print(table)


@app.command("download-models")
def download_models(
    project: Path = typer.Argument(Path("project.toml"), help="用于确定模型名称的项目配置"),
) -> None:
    """预下载默认模型，之后可完全离线运行。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise typer.BadParameter("请先安装 AI 环境；当前电脑使用 uv sync --extra ai --extra ai-cpu") from exc
    config = load_project(project).ai
    whisper_repos = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large-v3": "Systran/faster-whisper-large-v3",
        "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    }
    asr_models = (
        [config.qwen_asr_model, config.qwen_aligner_model]
        if config.asr_backend == "qwen3"
        else [whisper_repos.get(config.whisper_model, config.whisper_model)]
    )
    models = [*asr_models, config.clap_model, config.vision_model]
    if config.vision_reranker_model:
        models.append(config.vision_reranker_model)
    for model in models:
        console.print(f"下载 {model}")
        snapshot_download(model)
    console.print("[green]模型已全部缓存，可在 project.toml 中设置 offline = true[/green]")


if __name__ == "__main__":
    app()

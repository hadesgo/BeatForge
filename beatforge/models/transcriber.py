from __future__ import annotations

from pathlib import Path

from beatforge.lyrics import LyricLine


def transcribe(
    audio: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    offline: bool,
) -> list[LyricLine]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("缺少 AI 依赖；CPU 电脑请运行 uv sync --extra ai --extra ai-cpu") from exc

    if device == "cpu" and "float16" in compute_type:
        compute_type = "int8"
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        local_files_only=offline,
    )
    segments, _ = model.transcribe(
        str(audio),
        language="zh",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    lines: list[LyricLine] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            lines.append(LyricLine(float(segment.start), float(segment.end), text))
    return lines

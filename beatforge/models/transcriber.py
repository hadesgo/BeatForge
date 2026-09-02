from __future__ import annotations

from pathlib import Path
from typing import Any

from beatforge.lyrics import LyricLine, LyricToken


def transcribe(
    audio: Path, *, backend: str, qwen_model: str, qwen_aligner: str,
    whisper_model: str, device: str, compute_type: str, offline: bool,
) -> list[LyricLine]:
    if backend == "qwen3":
        return _transcribe_qwen(audio, qwen_model, qwen_aligner, device, offline)
    return _transcribe_whisper(audio, whisper_model, device, compute_type, offline)


def _transcribe_qwen(audio: Path, model_name: str, aligner_name: str, device: str, offline: bool) -> list[LyricLine]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError("Qwen3-ASR 未安装，请安装 uv 的 qwen extra") from exc
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"
    common = {"dtype": dtype, "device_map": device_map, "local_files_only": offline}
    model = Qwen3ASRModel.from_pretrained(
        model_name, forced_aligner=aligner_name, forced_aligner_kwargs=common,
        max_inference_batch_size=1, max_new_tokens=2048, **common,
    )
    result = model.transcribe(
        audio=str(audio), language=None, return_time_stamps=True,
        context="这是一首歌曲。请忠实识别演唱歌词，不要补写重复句。",
    )[0]
    items = list(result.time_stamps.items) if result.time_stamps else []
    return group_aligned_tokens(items)


def group_aligned_tokens(items: list[Any], max_characters: int = 18, gap_seconds: float = .75) -> list[LyricLine]:
    """Group Qwen forced-aligner character/word spans into readable subtitle lines."""
    lines: list[LyricLine] = []
    current: list[LyricToken] = []
    punctuation = set("。！？!?；;，,")
    for item in items:
        token = LyricToken(str(item.text), float(item.start_time), float(item.end_time))
        previous = current[-1] if current else None
        visible_length = sum(len(part.text.strip()) for part in current)
        should_break = bool(previous and (
            token.start - previous.end > gap_seconds
            or visible_length >= max_characters
            or previous.text[-1:] in punctuation
        ))
        if should_break:
            lines.append(_line_from_tokens(current))
            current = []
        current.append(token)
    if current:
        lines.append(_line_from_tokens(current))
    return lines


def _line_from_tokens(tokens: list[LyricToken]) -> LyricLine:
    text = "".join(token.text for token in tokens).strip()
    return LyricLine(tokens[0].start, tokens[-1].end, text, tokens)


def _transcribe_whisper(audio: Path, model_name: str, device: str, compute_type: str, offline: bool) -> list[LyricLine]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("缺少 Faster Whisper 依赖") from exc
    if device == "cpu" and "float16" in compute_type:
        compute_type = "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=offline)
    segments, _ = model.transcribe(
        str(audio), language="zh", beam_size=5, vad_filter=True,
        word_timestamps=True, condition_on_previous_text=False,
    )
    return [
        LyricLine(float(segment.start), float(segment.end), segment.text.strip())
        for segment in segments if segment.text.strip()
    ]

from __future__ import annotations

from pathlib import Path
from typing import Any

from beatforge.lyrics import LyricLine, LyricToken
from beatforge.runtime import release_gpu


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
        from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("原生 Qwen3-ASR 需要 transformers>=5.13 与 PyTorch") from exc
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else {"": "cpu"}
    common = {"dtype": dtype, "device_map": device_map, "local_files_only": offline}

    processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, **common).eval()
    inputs = processor.apply_transcription_request(
        audio=str(audio),
        prompt="这是一首歌曲。请忠实识别演唱歌词，不要补写重复句。",
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    transcript = str(parsed.get("transcription", "")).strip()
    language = parsed.get("language") or "Chinese"
    del output_ids, generated_ids, inputs, model, processor
    release_gpu()
    if not transcript:
        return []

    aligner_processor = AutoProcessor.from_pretrained(aligner_name, local_files_only=offline)
    aligner = AutoModelForTokenClassification.from_pretrained(aligner_name, **common).eval()
    aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
        audio=str(audio), transcript=transcript, language=language,
    )
    aligner_inputs = aligner_inputs.to(aligner.device, aligner.dtype)
    try:
        with torch.inference_mode():
            outputs = aligner(**aligner_inputs)
        items = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner.config.timestamp_token_id,
        )[0]
        return group_aligned_tokens(list(items))
    finally:
        del aligner_inputs, aligner, aligner_processor
        release_gpu()


def group_aligned_tokens(items: list[Any], max_characters: int = 18, gap_seconds: float = .75) -> list[LyricLine]:
    """Group Qwen forced-aligner character/word spans into readable subtitle lines."""
    lines: list[LyricLine] = []
    current: list[LyricToken] = []
    punctuation = set("。！？!?；;，,")
    for item in items:
        value = item.get if isinstance(item, dict) else lambda key: getattr(item, key)
        token = LyricToken(str(value("text")), float(value("start_time")), float(value("end_time")))
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

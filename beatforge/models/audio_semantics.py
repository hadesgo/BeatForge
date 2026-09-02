from __future__ import annotations

from pathlib import Path

import numpy as np

MOOD_LABELS = {
    "energetic": "energetic, intense, fast music",
    "uplifting": "uplifting, bright, hopeful music",
    "melancholic": "melancholic, sad, emotional music",
    "dreamy": "dreamy, atmospheric, soft music",
    "romantic": "romantic, warm, intimate music",
    "dark": "dark, tense, mysterious music",
    "cinematic": "cinematic, dramatic, epic music",
}


def classify_music(
    audio_file: Path,
    *,
    model_name: str,
    device: str,
    offline: bool,
) -> dict[str, float]:
    try:
        import librosa
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("缺少 AI 依赖；CPU 电脑请运行 uv sync --extra ai --extra ai-cpu") from exc

    samples, _ = librosa.load(audio_file, sr=48_000, mono=True)
    if len(samples) > 48_000 * 30:
        centers = np.linspace(5, len(samples) / 48_000 - 5, 3)
        clips = [samples[max(0, int((c - 5) * 48_000)):int((c + 5) * 48_000)] for c in centers]
    else:
        clips = [samples]

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, local_files_only=offline,
    ).to(device).eval()
    labels = list(MOOD_LABELS)
    text = processor(text=list(MOOD_LABELS.values()), return_tensors="pt", padding=True)
    text = {key: value.to(device) for key, value in text.items()}
    audio = processor(audios=clips, sampling_rate=48_000, return_tensors="pt", padding=True)
    audio = {key: value.to(device) for key, value in audio.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text)
        audio_features = model.get_audio_features(**audio)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
        scores = (audio_features @ text_features.T).mean(0).softmax(0).float().cpu().numpy()
    return {label: round(float(score), 5) for label, score in zip(labels, scores)}

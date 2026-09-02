from __future__ import annotations

from pathlib import Path
from typing import Any


def analyze_beats(audio: Path, backend: str, device: str) -> dict[str, Any] | None:
    if backend == "librosa":
        return None
    if backend == "allin1":
        try:
            import allin1_infer
        except ImportError as exc:
            raise RuntimeError("All-In-One-Infer 未安装，请增加 uv 的 music-ai extra") from exc
        result = allin1_infer.analyze(str(audio), device=device)
        return {
            "beats": [round(float(value), 3) for value in result.beats],
            "downbeats": [round(float(value), 3) for value in result.downbeats],
            "sections": [
                {"start": float(segment.start), "end": float(segment.end), "label": str(segment.label)}
                for segment in result.segments if str(segment.label) not in {"start", "end"}
            ],
        }
    try:
        from beat_this.inference import File2Beats
    except ImportError as exc:
        raise RuntimeError("Beat This! 未安装，请增加 uv 的 music-ai extra") from exc
    tracker = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = tracker(str(audio))
    return {
        "beats": [round(float(value), 3) for value in beats],
        "downbeats": [round(float(value), 3) for value in downbeats],
    }

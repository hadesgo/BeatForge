from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(slots=True)
class AudioAnalysis:
    duration: float
    bpm: float
    beats: list[float]
    sections: list[float]
    energy_times: list[float]
    energy_values: list[float]
    average_energy: float
    brightness: float
    mood: str
    mood_scores: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)

    def energy_at(self, time: float) -> float:
        if not self.energy_times:
            return self.average_energy
        index = int(np.searchsorted(self.energy_times, time, side="right") - 1)
        return self.energy_values[max(0, min(index, len(self.energy_values) - 1))]


def analyze_music(file: Path, mood_scores: dict[str, float] | None = None) -> AudioAnalysis:
    samples, sample_rate = librosa.load(file, sr=22_050, mono=True)
    total = librosa.get_duration(y=samples, sr=sample_rate)
    harmonic, percussive = librosa.effects.hpss(samples)
    tempo, beat_frames = librosa.beat.beat_track(y=percussive, sr=sample_rate, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=512)[0]
    lo, hi = np.quantile(rms, [0.1, 0.95]) if len(rms) else (0.0, 1.0)
    normalized = np.clip((rms - lo) / max(hi - lo, 1e-8), 0, 1)
    energy_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
    brightness = float(np.clip(np.mean(centroid) / 5000, 0, 1))

    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate)
    section_count = max(2, min(12, round(total / 20)))
    if chroma.shape[1] >= section_count:
        boundaries = librosa.segment.agglomerative(chroma, section_count)
        sections = librosa.frames_to_time(boundaries, sr=sample_rate).tolist()
    else:
        sections = [0.0]
    sections = sorted(set([0.0, *sections, float(total)]))

    scores = mood_scores or _heuristic_mood(float(np.mean(normalized)), brightness, float(np.asarray(tempo).item()))
    mood = max(scores, key=scores.get)
    return AudioAnalysis(
        duration=round(float(total), 3),
        bpm=round(float(np.asarray(tempo).item()), 2),
        beats=[round(float(x), 3) for x in beats],
        sections=[round(float(x), 3) for x in sections],
        energy_times=[round(float(x), 3) for x in energy_times],
        energy_values=[round(float(x), 4) for x in normalized],
        average_energy=round(float(np.mean(normalized)), 4),
        brightness=round(brightness, 4),
        mood=mood,
        mood_scores=scores,
    )


def _heuristic_mood(energy: float, brightness: float, bpm: float) -> dict[str, float]:
    scores = {
        "energetic": energy * 0.6 + min(bpm / 160, 1) * 0.4,
        "uplifting": brightness * 0.6 + min(bpm / 140, 1) * 0.4,
        "melancholic": (1 - energy) * 0.6 + (1 - brightness) * 0.4,
        "dreamy": (1 - energy) * 0.7 + 0.2,
        "cinematic": 0.45,
    }
    total = sum(scores.values())
    return {key: round(value / total, 5) for key, value in scores.items()}

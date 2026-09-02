from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset


@dataclass(slots=True)
class Shot:
    index: int
    start: float
    end: float
    duration: float
    media_id: int
    file: str
    kind: str
    source_start: float
    lyric: str
    energy: float
    motion: str
    transition: str
    semantic_score: float
    melody: float = 0.0
    section: str = "unknown"
    edit_intent: str = "continuity"

    def as_dict(self) -> dict:
        return asdict(self)


def create_plan(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    *,
    min_shot: float,
    max_shot: float,
) -> list[Shot]:
    boundaries = _boundaries(analysis, lyrics, min_shot, max_shot)
    lyric_rows = {id(line): i for i, line in enumerate(lyrics)}
    usage: dict[int, int] = {}
    recent: list[int] = []
    previous: MediaAsset | None = None
    chorus_motifs: list[int] = []
    shots: list[Shot] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        midpoint = (start + end) / 2
        line = next((line for line in lyrics if line.start <= midpoint < line.end), None)
        energy = analysis.energy_at(midpoint)
        section = _section_at(analysis, midpoint)
        ranked: list[tuple[float, MediaAsset, float]] = []
        for asset in assets:
            semantic = float(similarities[lyric_rows[id(line)], asset.id]) if line and similarities is not None else _tag_score(line, asset)
            mood = 0.12 if asset.mood == analysis.mood else 0.0
            movement = energy * 0.10 if asset.kind == "video" else (1 - energy) * 0.06
            repeat = usage.get(asset.id, 0) * 0.09 + (0.22 if asset.id in recent[-2:] else 0)
            quality = asset.quality_score * .16
            continuity = _color_similarity(previous, asset) * (.08 if section != "chorus" else .03)
            shot_variety = -.05 if previous and previous.shot_size != "unknown" and previous.shot_size == asset.shot_size else 0
            section_fit = .10 if section == "chorus" and asset.kind == "video" else .06 if section in {"intro", "outro"} and asset.kind == "image" else 0
            motif = .12 if section == "chorus" and asset.id in chorus_motifs else 0
            score = semantic + mood + movement + quality + continuity + shot_variety + section_fit + motif - repeat
            ranked.append((score, asset, semantic))
        _, selected, semantic = max(ranked, key=lambda item: item[0])
        usage[selected.id] = usage.get(selected.id, 0) + 1
        recent.append(selected.id)
        previous = selected
        if section == "chorus" and selected.id not in chorus_motifs and len(chorus_motifs) < 2:
            chorus_motifs.append(selected.id)
        shot_duration = end - start
        available = max(0.0, selected.duration - shot_duration - 0.1) if math.isfinite(selected.duration) else 0.0
        source_start = ((index * 0.61803398875) % 1) * available
        shots.append(Shot(
            index=index, start=round(start, 3), end=round(end, 3), duration=round(shot_duration, 3),
            media_id=selected.id, file=str(selected.file), kind=selected.kind,
            source_start=round(source_start, 3), lyric=line.text if line else "",
            energy=round(energy, 4),
            motion="dynamic" if energy > 0.68 else "gentle" if energy < 0.3 else "steady",
            transition="flash" if energy > 0.75 else "dip" if index % 4 == 0 else "fade",
            semantic_score=round(semantic, 4),
            melody=round(analysis.melody_at(midpoint), 4),
            section=section,
            edit_intent="impact" if section == "chorus" and energy > .65 else "breathe" if section in {"intro", "outro"} else "continuity",
        ))
    return shots


def _boundaries(analysis: AudioAnalysis, lyrics: list[LyricLine], minimum: float, maximum: float) -> list[float]:
    anchors = sorted(set([0.0, analysis.duration, *analysis.sections, *(line.start for line in lyrics)]))
    output = [0.0]
    for target in anchors[1:]:
        cursor = output[-1]
        while target - cursor > maximum:
            section = _section_at(analysis, cursor)
            section_scale = .78 if section == "chorus" else 1.18 if section in {"intro", "outro", "bridge"} else 1.0
            ideal = cursor + np.clip((3.8 - analysis.energy_at(cursor) * 1.5) * section_scale, minimum, maximum)
            grid = analysis.downbeats or analysis.beats
            candidates = [beat for beat in grid if minimum <= beat - cursor <= maximum and beat < target - minimum / 2]
            if not candidates and grid is analysis.downbeats:
                candidates = [beat for beat in analysis.beats if minimum <= beat - cursor <= maximum and beat < target - minimum / 2]
            cut = min(candidates, key=lambda beat: abs(beat - ideal)) if candidates else float(ideal)
            if cut <= cursor + 0.1:
                break
            output.append(round(cut, 3))
            cursor = cut
        if target - output[-1] >= minimum or target == analysis.duration:
            output.append(round(target, 3))
    if output[-1] != analysis.duration:
        output.append(analysis.duration)
    return sorted(set(output))


def _tag_score(line: LyricLine | None, asset: MediaAsset) -> float:
    if not line:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{1,4}", line.text.lower()))
    haystack = " ".join([asset.description, *asset.tags]).lower()
    return sum(0.15 for token in tokens if token in haystack)


def _section_at(analysis: AudioAnalysis, time: float) -> str:
    for index, (start, end) in enumerate(zip(analysis.sections, analysis.sections[1:])):
        if start <= time < end:
            return analysis.section_labels[index] if index < len(analysis.section_labels) else "unknown"
    return analysis.section_labels[-1] if analysis.section_labels else "unknown"


def _color_similarity(previous: MediaAsset | None, current: MediaAsset) -> float:
    if previous is None:
        return 0.0
    distance = np.linalg.norm(np.asarray(previous.dominant_color) - np.asarray(current.dominant_color))
    return float(max(0, 1 - distance / 441.7))

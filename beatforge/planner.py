from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset

if TYPE_CHECKING:
    from beatforge.models.ai_director import DirectorTreatment, SectionDirection


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
    transition_tone: str = "neutral"
    camera_motion: str = "unknown"
    section_index: int = -1

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
    treatment: DirectorTreatment | None = None,
    source_starts: np.ndarray | None = None,
) -> list[Shot]:
    boundaries = _boundaries(analysis, lyrics, min_shot, max_shot, treatment)
    lyric_rows = {id(line): i for i, line in enumerate(lyrics)}
    usage: dict[int, int] = {}
    recent: list[int] = []
    previous: MediaAsset | None = None
    chorus_motifs: list[int] = []
    video_cursors: dict[int, float] = {}
    shots: list[Shot] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        midpoint = (start + end) / 2
        line = next((line for line in lyrics if line.start <= midpoint < line.end), None)
        energy = analysis.energy_at(midpoint)
        section, section_index = _section_info(analysis, midpoint)
        direction = treatment.section(section_index) if treatment else None
        shot_duration = end - start
        ranked: list[tuple[float, MediaAsset, float, int]] = []
        for asset_column, asset in enumerate(assets):
            semantic = float(similarities[lyric_rows[id(line)], asset_column]) if line and similarities is not None else _tag_score(line, asset)
            mood = 0.12 if asset.mood == analysis.mood else 0.0
            movement = _motion_fit(asset, energy)
            repeat = usage.get(asset.id, 0) * 0.09 + (0.22 if asset.id in recent[-2:] else 0)
            quality = asset.quality_score * .16
            continuity = _color_similarity(previous, asset) * (.08 if section != "chorus" else .03)
            shot_variety = -.09 if previous and previous.shot_size != "unknown" and previous.shot_size == asset.shot_size else 0
            section_fit = .10 if section == "chorus" and asset.kind == "video" else .06 if section in {"intro", "outro"} and asset.kind == "image" else 0
            motif = .12 if section == "chorus" and asset.id in chorus_motifs else 0
            director_score = _director_asset_score(asset, direction, treatment)
            duration_penalty = .24 if asset.kind == "video" and asset.duration < shot_duration + .25 else 0.0
            framing_penalty = _framing_penalty(asset)
            score = semantic + mood + movement + quality + continuity + shot_variety + section_fit + motif + director_score - repeat - duration_penalty - framing_penalty
            ranked.append((score, asset, semantic, asset_column))
        _, selected, semantic, selected_column = max(ranked, key=lambda item: item[0])
        continues_previous = previous is not None and selected.id == previous.id
        usage[selected.id] = usage.get(selected.id, 0) + 1
        recent.append(selected.id)
        if section == "chorus" and selected.id not in chorus_motifs and len(chorus_motifs) < 2:
            chorus_motifs.append(selected.id)
        available = max(0.0, selected.duration - shot_duration - 0.1) if math.isfinite(selected.duration) else 0.0
        if selected.kind == "video" and continues_previous and video_cursors.get(selected.id, 0) <= available:
            source_start = video_cursors[selected.id]
        elif selected.kind == "video" and line and source_starts is not None:
            center = float(source_starts[lyric_rows[id(line)], selected_column])
            source_start = float(np.clip(center - shot_duration / 2, 0, available))
        else:
            source_start = ((index * 0.61803398875) % 1) * available
        if selected.kind == "video":
            video_cursors[selected.id] = source_start + shot_duration
        previous = selected
        shots.append(Shot(
            index=index, start=round(start, 3), end=round(end, 3), duration=round(shot_duration, 3),
            media_id=selected.id, file=str(selected.file), kind=selected.kind,
            source_start=round(source_start, 3), lyric=line.text if line else "",
            energy=round(energy, 4),
            motion="dynamic" if energy > 0.68 else "gentle" if energy < 0.3 else "steady",
            transition="cut",
            semantic_score=round(semantic, 4),
            melody=round(analysis.melody_at(midpoint), 4),
            section=section,
            edit_intent=direction.edit_intent if direction else "impact" if section == "chorus" and energy > .65 else "breathe" if section in {"intro", "outro"} else "continuity",
            transition_tone=direction.transition_tone if direction else "neutral",
            camera_motion=selected.camera_motion,
            section_index=section_index,
        ))
    _assign_transitions(shots)
    return shots


def _boundaries(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    minimum: float,
    maximum: float,
    treatment: DirectorTreatment | None = None,
) -> list[float]:
    # Lyrics drive semantic shot choice, but must not force a cut on every line.
    # Structural boundaries and musical beat grids are the editing clock.
    anchors = sorted(set([0.0, analysis.duration, *analysis.sections]))
    output = [0.0]
    for target in anchors[1:]:
        cursor = output[-1]
        while target - cursor > maximum:
            section, section_index = _section_info(analysis, cursor)
            section_scale = .78 if section == "chorus" else 1.18 if section in {"intro", "outro", "bridge"} else 1.0
            direction = treatment.section(section_index) if treatment else None
            if direction:
                section_scale *= 1.25 - direction.cut_intensity * .65
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
    return _section_info(analysis, time)[0]


def _section_info(analysis: AudioAnalysis, time: float) -> tuple[str, int]:
    for index, (start, end) in enumerate(zip(analysis.sections, analysis.sections[1:])):
        if start <= time < end:
            return (analysis.section_labels[index] if index < len(analysis.section_labels) else "unknown", index)
    index = max(0, len(analysis.sections) - 2)
    return (analysis.section_labels[-1] if analysis.section_labels else "unknown", index)


def _director_asset_score(
    asset: MediaAsset,
    direction: SectionDirection | None,
    treatment: DirectorTreatment | None,
) -> float:
    score = .10 if treatment and asset.id in treatment.motif_asset_ids else 0.0
    if not direction:
        return score
    if asset.id in direction.preferred_asset_ids:
        score += .22 - direction.preferred_asset_ids.index(asset.id) * .025
    if direction.preferred_media == asset.kind:
        score += .08
    if asset.shot_size in direction.preferred_shot_sizes:
        score += .06
    return score


def _color_similarity(previous: MediaAsset | None, current: MediaAsset) -> float:
    if previous is None:
        return 0.0
    distance = np.linalg.norm(np.asarray(previous.dominant_color) - np.asarray(current.dominant_color))
    return float(max(0, 1 - distance / 441.7))


def _motion_fit(asset: MediaAsset, energy: float) -> float:
    if asset.kind == "image":
        return (1 - energy) * .06
    motion = asset.camera_motion.lower()
    active = any(word in motion for word in ("fast", "handheld", "whip", "tracking", "dynamic", "快速", "手持", "跟拍"))
    calm = any(word in motion for word in ("static", "locked", "slow", "tripod", "固定", "缓慢"))
    if active:
        return energy * .13 - (1 - energy) * .04
    if calm:
        return (1 - energy) * .10
    return .04 + energy * .04


def _framing_penalty(asset: MediaAsset, target_aspect: float = 16 / 9) -> float:
    if asset.width <= 0 or asset.height <= 0:
        return 0.0
    source_aspect = asset.width / asset.height
    retained = min(source_aspect / target_aspect, target_aspect / source_aspect)
    return max(0.0, 1 - retained) * .12


def _assign_transitions(shots: list[Shot]) -> None:
    """Use editorial cuts by default and reserve visible transitions for structural changes."""
    for shot, following in zip(shots, shots[1:]):
        section_change = shot.section_index != following.section_index
        if section_change:
            shot.transition = "flash" if following.energy > .78 else "dip"
        elif shot.edit_intent == "breathe" or following.edit_intent == "breathe":
            shot.transition = "dissolve"
        else:
            shot.transition = "cut"
    if shots:
        shots[-1].transition = "none"

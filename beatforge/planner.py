from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset

if TYPE_CHECKING:
    from beatforge.models.ai_director import DirectorTreatment, SectionDirection


@dataclass(slots=True)
class ShotLayer:
    media_id: int
    file: str
    kind: str = "image"
    role: str = "secondary"
    focus_point: list[float] = field(default_factory=lambda: [.5, .5])
    source_width: int = 0
    source_height: int = 0
    source_color: list[int] = field(default_factory=lambda: [128, 128, 128])
    enter_offset: float = 0.0


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
    source_color: list[int] = field(default_factory=lambda: [128, 128, 128])
    focus_point: list[float] = field(default_factory=lambda: [.5, .5])
    source_width: int = 0
    source_height: int = 0
    image_effect: str = "cinematic_depth"
    layers: list[ShotLayer] = field(default_factory=list)

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
    target_width: int = 1920,
    target_height: int = 1080,
    image_composites: bool = True,
    image_composite_ratio: float = .24,
    max_composite_images: int = 3,
    avoid_asset_repeats: bool = True,
    transition_density: float = .35,
) -> list[Shot]:
    """Lay out shots on the musical grid and pick media for each of them.

    Reuse policy: an asset that is already on screen is only chosen again when the
    remaining untouched assets can no longer cover the remaining shots. While the
    supply lasts, every shot gets something the audience has not seen yet.
    """
    boundaries = _boundaries(analysis, lyrics, min_shot, max_shot, treatment)
    lyric_rows = {id(line): i for i, line in enumerate(lyrics)}
    usage: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    previous: MediaAsset | None = None
    chorus_motifs: list[int] = []
    video_cursors: dict[int, float] = {}
    lyric_visual_history: dict[str, set[int]] = {}
    active_line_id: int | None = None
    active_lyric_key = ""
    active_line_visual_assets: set[int] = set()
    shots: list[Shot] = []
    shot_count = len(boundaries) - 1
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        midpoint = (start + end) / 2
        line = next((line for line in lyrics if line.start <= midpoint < line.end), None)
        line_id = id(line) if line is not None else None
        if line_id != active_line_id:
            if active_lyric_key and active_line_visual_assets:
                lyric_visual_history.setdefault(active_lyric_key, set()).update(active_line_visual_assets)
            active_line_id = line_id
            active_lyric_key = _lyric_key(line.text) if line is not None else ""
            active_line_visual_assets = set()
        previously_visible_for_lyric = lyric_visual_history.get(active_lyric_key, set())
        remaining_shots = shot_count - index
        minimum_usage, least_used = _least_used_assets(assets, usage)
        # Assets left over after covering every remaining shot with a distinct one.
        # Only this surplus may be spent on composite layers.
        spare_assets = len(least_used) - remaining_shots
        energy = analysis.energy_at(midpoint)
        section, section_index = _section_info(analysis, midpoint)
        direction = treatment.section(section_index) if treatment else None
        shot_duration = end - start
        ranked: list[tuple[float, MediaAsset, float, int]] = []
        for asset_column, asset in enumerate(assets):
            semantic = float(similarities[lyric_rows[id(line)], asset_column]) if line and similarities is not None else _tag_score(line, asset)
            mood = 0.12 if asset.mood == analysis.mood else 0.0
            movement = _motion_fit(asset, energy)
            repeat = _reuse_penalty(usage.get(asset.id, 0), last_seen.get(asset.id), index)
            quality = asset.quality_score * .16
            continuity = _color_similarity(previous, asset) * (.08 if section != "chorus" else .03)
            shot_variety = -.09 if previous and previous.shot_size != "unknown" and previous.shot_size == asset.shot_size else 0
            section_fit = .10 if section == "chorus" and asset.kind == "video" else .06 if section in {"intro", "outro"} and asset.kind == "image" else 0
            motif = .12 if section == "chorus" and asset.id in chorus_motifs else 0
            director_score = _director_asset_score(asset, direction, treatment)
            duration_penalty = .24 if asset.kind == "video" and asset.duration < shot_duration + .25 else 0.0
            framing_penalty = _framing_penalty(asset, target_width / max(target_height, 1))
            upscale_penalty = _upscale_penalty(asset, target_width, target_height)
            score = semantic + mood + movement + quality + continuity + shot_variety + section_fit + motif + director_score - repeat - duration_penalty - framing_penalty - upscale_penalty
            ranked.append((score, asset, semantic, asset_column))
        # A repeated lyric line should not show the same picture twice.
        candidates = [item for item in ranked if item[1].id not in previously_visible_for_lyric] or ranked
        if avoid_asset_repeats:
            # Prefer the least-used assets so the audience keeps seeing new material;
            # an asset is only shown again once everything else has caught up.
            tier = [item for item in candidates if usage.get(item[1].id, 0) == minimum_usage]
            candidates = tier or candidates
        _, selected, semantic, selected_column = max(candidates, key=lambda item: item[0])
        continues_previous = previous is not None and selected.id == previous.id
        usage[selected.id] = usage.get(selected.id, 0) + 1
        last_seen[selected.id] = index
        if line is not None:
            active_line_visual_assets.add(selected.id)
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
        image_effect = "source_video"
        layers: list[ShotLayer] = []
        if selected.kind == "image":
            # Composites may draw on the whole pool, but they always favour the
            # least-used images so they cannot drain the shots still to come.
            layer_pool = candidates if (spare_assets >= 0 or not avoid_asset_repeats) else ranked
            image_candidates = [
                item for item in sorted(
                    layer_pool, key=lambda item: (usage.get(item[1].id, 0), -item[0]),
                )
                if item[1].kind == "image" and item[1].id != selected.id
                and item[1].id not in previously_visible_for_lyric
            ]
            if avoid_asset_repeats and spare_assets >= 0:
                # Composite layers are a luxury: they may only spend assets that are
                # surplus to the distinct ones still needed to cover the remaining shots.
                spare_pool = [
                    item for item in image_candidates
                    if usage.get(item[1].id, 0) == minimum_usage
                ]
                image_candidates = (spare_pool or image_candidates)[:spare_assets]
            image_effect, layer_count = _choose_image_effect(
                index, section, energy, analysis.melody_at(midpoint), len(image_candidates),
                enabled=image_composites, ratio=image_composite_ratio,
                max_images=max_composite_images,
                edit_intent=direction.edit_intent if direction else "continuity",
            )
            entry_offsets = _layer_entry_offsets(analysis, start, end, layer_count)
            for layer_index, (_score, layer_asset, _layer_semantic, _column) in enumerate(image_candidates[:layer_count]):
                layers.append(ShotLayer(
                    media_id=layer_asset.id, file=str(layer_asset.file),
                    role="secondary", focus_point=layer_asset.focus_point.copy(),
                    source_width=layer_asset.width, source_height=layer_asset.height,
                    source_color=layer_asset.dominant_color.copy(),
                    enter_offset=entry_offsets[layer_index],
                ))
                usage[layer_asset.id] = usage.get(layer_asset.id, 0) + 1
                last_seen[layer_asset.id] = index
                active_line_visual_assets.add(layer_asset.id)
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
            source_color=selected.dominant_color.copy(),
            focus_point=selected.focus_point.copy(),
            source_width=selected.width,
            source_height=selected.height,
            image_effect=image_effect,
            layers=layers,
        ))
    _assign_transitions(shots, mood=analysis.mood, density=transition_density)
    return shots


def _least_used_assets(
    assets: list[MediaAsset], usage: dict[int, int],
) -> tuple[int, list[MediaAsset]]:
    """Return the lowest on-screen count and every asset still sitting at it."""
    if not assets:
        return 0, []
    counts = [usage.get(asset.id, 0) for asset in assets]
    minimum = min(counts)
    return minimum, [asset for asset, count in zip(assets, counts) if count == minimum]


def _reuse_penalty(visible: int, last_index: int | None, index: int) -> float:
    """Price a repeat by how often and how recently the asset was already shown."""
    if visible <= 0:
        return 0.0
    penalty = visible * .09
    if last_index is None:
        return penalty
    gap = index - last_index
    if gap <= 1:
        return penalty + .34
    if gap == 2:
        return penalty + .18
    return penalty


def _choose_image_effect(
    index: int, section: str, energy: float, melody: float, available: int,
    *, enabled: bool, ratio: float, max_images: int,
    edit_intent: str = "continuity",
) -> tuple[str, int]:
    """Choose a restrained, section-consistent still-image treatment."""
    if not enabled or available <= 0 or max_images < 2:
        return _single_image_effect(index, section, energy, melody, edit_intent), 0

    # A stable gate keeps composites special instead of turning the MV into a slide template.
    gate = ((index * 37 + 17) % 100) / 100
    intent_scale = 1.4 if edit_intent == "impact" else .45 if edit_intent == "breathe" else 1.0
    section_ratio = min(1.0, ratio * intent_scale * (1.85 if section == "chorus" else 1.25 if section in {"bridge", "solo"} else 1.0))
    if gate >= section_ratio:
        return _single_image_effect(index, section, energy, melody, edit_intent), 0

    if section == "chorus":
        effect = ("beat_montage", "photo_stack", "split_screen")[index % 3]
    elif section in {"bridge", "solo"}:
        effect = "double_exposure" if index % 2 else "photo_stack"
    else:
        effect = ("split_screen", "photo_stack", "double_exposure")[index % 3]
    wanted_total = 4 if effect == "beat_montage" else 3 if effect == "photo_stack" else 2
    total = min(max_images, wanted_total, available + 1)
    if total < 2:
        return _single_image_effect(index, section, energy, melody, edit_intent), 0
    return effect, total - 1


def _single_image_effect(
    index: int, section: str, energy: float, melody: float, edit_intent: str,
) -> str:
    """Pick one camera move, or occasionally a framing treatment instead.

    The moves are grouped by what the music is doing: a release at the end of a
    phrase, an arrival on an impact, an unhurried drift through a verse. Selection
    is a fixed rotation rather than a random draw, so a given plan always renders
    the same way.
    """
    framing = _framing_effect(index, section, energy, edit_intent)
    if framing:
        return framing
    if edit_intent == "breathe" or section in {"intro", "outro"}:
        return ("breathe", "cinematic_depth", "pull_back", "dolly_out")[index % 4]
    if edit_intent == "impact" or (section == "chorus" and energy > .76):
        return ("punch_in", "dolly_in", "arc", "pan_reveal")[index % 4]
    if energy > .62:
        return ("pan_reveal", "drift", "arc", "dolly_in")[index % 4]
    if melody > .62:
        return ("tilt_up", "focus_pull", "dolly_out", "tilt_down")[index % 4]
    return ("cinematic_depth", "drift", "tilt_down", "arc")[index % 4]


def _framing_effect(index: int, section: str, energy: float, edit_intent: str) -> str:
    """Return a framing treatment for roughly one shot in twelve, else ``""``.

    Letterboxing, an iris and parallax are the still-image equivalent of a
    composite: each is worth seeing once and tedious if every shot gets one. The
    gate keeps them occasional without making them random.
    """
    if ((index * 53 + 7) % 100) / 100 >= .085:
        return ""
    if section == "intro":
        return "iris"
    if edit_intent == "breathe" or section in {"outro", "bridge", "solo"}:
        return "parallax"
    return "film_bars" if energy > .7 else "parallax"


def _layer_entry_offsets(
    analysis: AudioAnalysis, start: float, end: float, count: int,
) -> list[float]:
    """Place layer reveals on real beats, falling back to even musical phrasing."""
    if count <= 0:
        return []
    duration = end - start
    grid = analysis.downbeats or analysis.beats
    candidates = [beat - start for beat in grid if start + .12 < beat < end - .12]
    offsets: list[float] = []
    for index in range(count):
        target = duration * (index + 1) / (count + 1)
        unused = [beat for beat in candidates if all(abs(beat - used) > .08 for used in offsets)]
        offsets.append(min(unused, key=lambda beat: abs(beat - target)) if unused else target)
    return [round(value, 3) for value in sorted(offsets)]


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


def _lyric_key(text: str) -> str:
    """Normalize cosmetic lyric differences when tracking repeated lines."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.casefold(), flags=re.UNICODE)


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


def _upscale_penalty(asset: MediaAsset, target_width: int, target_height: int) -> float:
    if asset.width <= 0 or asset.height <= 0:
        return .04
    scale = max(target_width / asset.width, target_height / asset.height)
    return float(np.clip((scale - 1.15) * .055, 0, .18))


# Families subtle enough to repeat back to back without reading as a template.
# Anything else gets rotated away from its predecessor.
_SUBTLE_TRANSITIONS = {"cut", "none", "dissolve", "blur"}

_TRANSITION_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "wipe": ("reveal", "slide", "diag"),
    "slide": ("wipe", "reveal", "squeeze"),
    "reveal": ("slide", "wipe", "diag"),
    "diag": ("wipe", "reveal", "slide"),
    "slice": ("pixel", "squeeze", "wipe"),
    "pixel": ("slice", "squeeze", "wipe"),
    "squeeze": ("slice", "pixel", "slide"),
    "zoom": ("radial", "pixel", "slice"),
    "radial": ("zoom", "circle", "slice"),
    "flash": ("zoom", "radial", "slice"),
    "circle": ("radial", "zoom", "dissolve"),
    "dip": ("dissolve", "blur", "circle"),
}


def _assign_transitions(shots: list[Shot], *, mood: str = "", density: float = .35) -> None:
    """Pick a transition family per cut, then keep it from repeating itself.

    Hard cuts stay the default. A visible transition is punctuation, and
    punctuating every cut turns the edit into a slideshow, so only structural
    changes - a new section, a change of intent - and a bounded share of the cuts
    inside a section earn one.
    """
    for index, (shot, following) in enumerate(zip(shots, shots[1:])):
        shot.transition = _transition_family(shot, following, index, mood, density)
    if shots:
        shots[-1].transition = "none"
    _break_transition_repeats(shots)


def _transition_family(shot: Shot, following: Shot, index: int, mood: str, density: float) -> str:
    """Name the transition family the edit is asking for at this cut."""
    tone = following.transition_tone if following.transition_tone != "neutral" else shot.transition_tone
    dreamy = mood in {"dreamy", "romantic"} or tone == "soft"
    restless = mood in {"energetic", "dark"} or tone == "bright"

    if shot.section_index != following.section_index:
        # Structural punctuation: the strongest move the music can justify.
        if following.energy > .78:
            return "flash"
        if following.section == "chorus":
            return "radial" if restless else "zoom"
        if following.section in {"intro", "outro"} or shot.section == "chorus":
            return "dip"
        if dreamy:
            return "circle"
        if restless:
            return "slice"
        return "dissolve"

    if "impact" in {shot.edit_intent, following.edit_intent}:
        return "zoom" if index % 2 else "flash"
    if "breathe" in {shot.edit_intent, following.edit_intent}:
        return "blur" if dreamy else "dissolve"

    # Inside a section, spend a bounded share of the cut points on visible moves.
    if ((index * 29 + 11) % 100) / 100 >= density:
        return "cut"
    energy = max(shot.energy, following.energy)
    if energy > .70:
        return ("wipe", "slide", "diag", "pixel", "squeeze")[index % 5]
    if energy < .32:
        return "blur" if dreamy else "dissolve"
    return ("wipe", "reveal", "dissolve")[index % 3]


def _break_transition_repeats(shots: list[Shot]) -> None:
    """Never play the same visible transition twice in a row.

    Two identical wipes back to back stop reading as punctuation and start reading
    as a template, which is the one thing an edit like this cannot afford.
    """
    previous = ""
    for shot in shots:
        family = shot.transition
        if family == previous and family not in _SUBTLE_TRANSITIONS:
            options = _TRANSITION_ALTERNATIVES.get(family, ())
            family = next((name for name in options if name != previous), "dissolve")
            shot.transition = family
        previous = family

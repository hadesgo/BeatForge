from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from beatforge.audio import AudioAnalysis, section_at
from beatforge.editing import EditStyle
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
    min_shot: float | None = None,
    max_shot: float | None = None,
    treatment: DirectorTreatment | None = None,
    source_starts: np.ndarray | None = None,
    target_width: int = 1920,
    target_height: int = 1080,
    image_composites: bool = True,
    image_composite_ratio: float | None = None,
    max_composite_images: int = 3,
    avoid_asset_repeats: bool = True,
    transition_density: float | None = None,
    style: EditStyle | None = None,
) -> list[Shot]:
    """Lay out shots on the musical grid and pick media for each of them.

    Reuse policy: an asset that is already on screen is only chosen again when the
    remaining untouched assets can no longer cover the remaining shots. While the
    supply lasts, every shot gets something the audience has not seen yet.

    A style owns the settings it has an opinion about - shot length, transition
    density, composite ratio - because a style and a hand-set value are two answers to
    the same question. Everything it does not name is taken from the arguments.
    """
    if style is not None:
        min_shot, max_shot = style.shot_min, style.shot_max
        transition_density = style.transition_density
        image_composite_ratio = style.composite_ratio
    if min_shot is None or max_shot is None:
        raise ValueError("需要给出 min_shot/max_shot，或指定一个 edit_style")
    transition_density = .35 if transition_density is None else transition_density
    image_composite_ratio = .24 if image_composite_ratio is None else image_composite_ratio
    boundaries = _boundaries(analysis, lyrics, min_shot, max_shot, treatment, style)
    # How hard this edit insists on a shot-size change across a cut.
    contrast = style.shot_size_contrast if style else .09
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
    # Camera-move rotations advance on this, not on the shot index. Composites claim a
    # share of the shots, and a rotation keyed off the global index would then lose
    # whichever slots those shots occupied - with enough composites in one section, a
    # whole group of moves can never be reached at all.
    single_image_cursor = 0
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
        section, section_index = section_at(analysis, midpoint)
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
            shot_variety = -contrast if previous and previous.shot_size != "unknown" and previous.shot_size == asset.shot_size else 0
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
                single_image_cursor, index, section, energy, analysis.melody_at(midpoint),
                len(image_candidates),
                enabled=image_composites, ratio=image_composite_ratio,
                max_images=max_composite_images,
                edit_intent=direction.edit_intent if direction else "continuity",
            )
            if layer_count == 0:
                single_image_cursor += 1
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
    _assign_transitions(
        shots, mood=analysis.mood, density=transition_density,
        flavour=style.transition_flavour if style else "balanced",
    )
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


# Multi-image layouts: name -> (pictures it is designed for, fewest it can be built with).
#
# Every one of these gives each picture its own region of the frame, or shows one
# picture at a time. None of them draws one picture over another's pixels - two
# pictures sharing an area have no way to both stay legible, and the frame reads as
# mud rather than as a design. That entire family (a screen-blended double exposure,
# a stack of cards laid over another picture) was removed for this reason.
IMAGE_COMPOSITES: dict[str, tuple[int, int]] = {
    "split_screen": (2, 2),      # 等宽双栏
    "hero_split": (2, 2),        # 2:1 主副双栏
    "diagonal_split": (2, 2),    # 斜线硬边分割
    "triptych": (3, 3),          # 三联竖排
    "hero_grid": (3, 3),         # 左大 + 右双堆叠
    "beat_montage": (4, 2),      # 镜头内按节拍依次切换，一次只显示一张
}

# What to fall back to when a layout asks for more pictures than the shot can afford.
_COMPOSITE_SMALLER = {
    "triptych": "split_screen",
    "hero_grid": "hero_split",
}


def _fit_composite(effect: str, capacity: int) -> str | None:
    """Walk down to the largest layout this many pictures can fill, or give up.

    The capacity comes from ``max_composite_images`` and from how many surplus
    assets are actually left, so a three-panel layout has to be able to become a
    two-panel one instead of rendering a hole where its third picture should be.
    """
    while effect is not None:
        if IMAGE_COMPOSITES[effect][1] <= capacity:
            return effect
        effect = _COMPOSITE_SMALLER.get(effect)
    return None


def _composite_rotation(section: str) -> tuple[str, ...]:
    """Which layouts this part of the song rotates through.

    The song picks the *kind* of division: a chorus splits into three or switches
    between pictures, a quiet passage divides the frame once and lets it sit.
    """
    if section == "chorus":
        return ("beat_montage", "hero_grid", "triptych")
    if section in {"bridge", "solo", "intro", "outro"}:
        return ("hero_split", "diagonal_split", "split_screen")
    return ("split_screen", "hero_split", "triptych", "diagonal_split", "hero_grid")


def _choose_image_effect(
    cursor: int, index: int, section: str, energy: float, melody: float, available: int,
    *, enabled: bool, ratio: float, max_images: int,
    edit_intent: str = "continuity",
) -> tuple[str, int]:
    """Choose a restrained, section-consistent still-image treatment.

    Three questions, three counters, and they must not be answered by the same one:

    - ``index`` is the shot's position in the film. It drives the gate and the layout
      rotation. The gate has to move on *every* shot, because a gate keyed on anything
      else is self-reinforcing: a shot that becomes a composite freezes the very
      counter that decided it, so the gate returns the same answer forever. That is
      exactly what happened while both the gate and the move rotation shared the
      single-image counter - ``image_composite_ratio`` had no effect at all, and every
      image shot in the film came back a composite whatever the config said.
    - ``cursor`` counts the single-image shots chosen so far, so the camera-move
      rotation stays dense. Keying it on ``index`` instead would lose every slot the
      composites claimed.
    """
    if not enabled or available <= 0 or max_images < 2:
        return _single_image_effect(cursor, section, energy, melody, edit_intent), 0

    # A stable gate keeps composites special instead of turning the MV into a slide template.
    gate = ((index * 37 + 17) % 100) / 100
    intent_scale = 1.4 if edit_intent == "impact" else .45 if edit_intent == "breathe" else 1.0
    section_ratio = min(1.0, ratio * intent_scale * (1.85 if section == "chorus" else 1.25 if section in {"bridge", "solo"} else 1.0))
    if gate >= section_ratio:
        return _single_image_effect(cursor, section, energy, melody, edit_intent), 0

    rotation = _composite_rotation(section)
    capacity = min(max_images, available + 1)
    effect = _fit_composite(rotation[index % len(rotation)], capacity)
    if effect is None:
        return _single_image_effect(cursor, section, energy, melody, edit_intent), 0
    return effect, min(capacity, IMAGE_COMPOSITES[effect][0]) - 1


def _single_image_effect(
    cursor: int, section: str, energy: float, melody: float, edit_intent: str,
) -> str:
    """Pick one camera move, or occasionally a framing treatment instead.

    The moves are grouped by what the music is doing: a release at the end of a
    phrase, an arrival on an impact, an unhurried drift through a verse. Selection
    is a fixed rotation rather than a random draw, so a given plan always renders
    the same way.

    The rotations are ordered so that neighbouring shots in the same group differ in
    *character*, not just in direction: a push, then a turn, then a travel.
    """
    framing = _framing_effect(cursor, section, energy, edit_intent)
    if framing:
        return framing
    if edit_intent == "breathe" or section in {"intro", "outro"}:
        return ("breathe", "cinematic_depth", "pull_back", "handheld", "dolly_out")[cursor % 5]
    if edit_intent == "impact" or (section == "chorus" and energy > .76):
        # Impact cuts get the loud moves: a slam, a 3D turn, a whip, a spiral.
        return ("punch_in", "tilt3d_back", "whip_pan", "spiral_in", "dolly_in")[cursor % 5]
    if energy > .62:
        return ("pan_reveal", "tilt3d_right", "drift", "pulse_in", "arc", "dolly_in")[cursor % 6]
    if melody > .62:
        # Melodic passages turn and roll rather than push.
        return ("tilt_up", "spiral_in", "focus_pull", "roll_drift", "tilt3d_left", "tilt_down")[cursor % 6]
    return ("cinematic_depth", "drift", "tilt3d_front", "tilt_down", "arc", "handheld")[cursor % 6]


def _framing_effect(cursor: int, section: str, energy: float, edit_intent: str) -> str:
    """Return a framing treatment for roughly one shot in twelve, else ``""``.

    Letterboxing, an iris and parallax are the still-image equivalent of a
    composite: each is worth seeing once and tedious if every shot gets one. The
    gate keeps them occasional without making them random.
    """
    if ((cursor * 53 + 7) % 100) / 100 >= .085:
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
    style: EditStyle | None = None,
) -> list[float]:
    """Lay the cut points on the musical grid.

    Lyrics drive semantic shot choice, but must not force a cut on every line - unless
    the style says they should. Which grid a cut may land on is the most audible
    decision an editor makes: every beat reads as the cut playing percussion, bar lines
    read as phrase punctuation, and lyric starts read as the words carrying the edit.
    """
    tempo = style.tempo if style else 3.8
    gain = style.energy_gain if style else 1.5
    speedup = style.section_speedup if style else 0.0
    alignment = style.cut_alignment if style else "downbeat"

    anchors = sorted(set([0.0, analysis.duration, *analysis.sections]))
    output = [0.0]
    for target in anchors[1:]:
        cursor = output[-1]
        while target - cursor > maximum:
            section, section_index = section_at(analysis, cursor)
            section_scale = .78 if section == "chorus" else 1.18 if section in {"intro", "outro", "bridge"} else 1.0
            direction = treatment.section(section_index) if treatment else None
            if direction:
                section_scale *= 1.25 - direction.cut_intensity * .65
            # Tighten toward the end of a section. An editor accelerates into the drop;
            # the reverse - shots getting longer as the section builds - reads as an
            # edit that has run out of ideas right where it should be peaking.
            section_scale *= 1 - speedup * _section_progress(analysis, section_index, cursor)
            ideal = cursor + np.clip((tempo - analysis.energy_at(cursor) * gain) * section_scale, minimum, maximum)
            candidates = _cut_candidates(
                analysis, lyrics, cursor, minimum, maximum, target, alignment,
            )
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


def _cut_candidates(
    analysis: AudioAnalysis, lyrics: list[LyricLine], cursor: float,
    minimum: float, maximum: float, target: float, alignment: str,
) -> list[float]:
    """Every position this style is willing to put a cut, inside the allowed window.

    Each alignment carries a finer fallback. A coarse grid can leave a window with no
    candidate at all - a four-bar phrase grid especially - and falling through to the
    next grid down keeps the cut musical instead of dropping it on an arbitrary frame.
    """
    primary, fallback = _alignment_grids(analysis, lyrics, alignment, maximum)
    return (
        _within_window(primary, cursor, minimum, maximum, target)
        or _within_window(fallback, cursor, minimum, maximum, target)
    )


def _alignment_grids(
    analysis: AudioAnalysis, lyrics: list[LyricLine], alignment: str, maximum: float,
) -> tuple[list[float], list[float]]:
    beats = list(analysis.beats or analysis.downbeats)
    downbeats = list(analysis.downbeats or analysis.beats)
    if alignment == "lyric":
        return sorted({line.start for line in lyrics}), downbeats
    if alignment == "phrase":
        return _phrase_grid(downbeats, maximum), downbeats
    if alignment == "beat":
        return beats, downbeats
    return downbeats, beats


def _phrase_grid(downbeats: list[float], maximum: float) -> list[float]:
    """Cut on musical phrases: the smallest whole number of bars that fits the window.

    A fixed four-bar phrase is the textbook answer and the wrong one here. At 120 bpm a
    bar is two seconds, so a four-bar phrase is eight, and a style whose shots top out
    at six would never find a single candidate - it would silently fall back to cutting
    on bar lines and the alignment would be decoration. Sizing the phrase to the style's
    own window is what makes the promise keepable across tempos.
    """
    if len(downbeats) < 2:
        return downbeats
    bar = float(np.median(np.diff(downbeats)))
    stride = max(1, min(4, int(maximum / bar))) if bar > 0 else 1
    return downbeats[::stride]


def _within_window(
    grid: list[float], cursor: float, minimum: float, maximum: float, target: float,
) -> list[float]:
    return [
        beat for beat in grid
        if minimum <= beat - cursor <= maximum and beat < target - minimum / 2
    ]


def _section_progress(analysis: AudioAnalysis, section_index: int, time: float) -> float:
    """How far into its section a moment sits, 0..1."""
    starts = analysis.sections or [0.0, analysis.duration]
    index = max(0, min(section_index, len(starts) - 1))
    begin = starts[index]
    finish = starts[index + 1] if index + 1 < len(starts) else analysis.duration
    return float(np.clip((time - begin) / max(finish - begin, .01), 0, 1))



def _tag_score(line: LyricLine | None, asset: MediaAsset) -> float:
    if not line:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{1,4}", line.text.lower()))
    haystack = " ".join([asset.description, *asset.tags]).lower()
    return sum(0.15 for token in tokens if token in haystack)


def _lyric_key(text: str) -> str:
    """Normalize cosmetic lyric differences when tracking repeated lines."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.casefold(), flags=re.UNICODE)




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
    "mask": ("circle", "open", "reveal"),
    "open": ("close", "mask", "circle"),
    "close": ("open", "mask", "circle"),
    "smooth": ("slide", "wipe", "reveal"),
    "corner": ("diag", "wipe", "slice"),
    "wind": ("slice", "pixel", "squeeze"),
    "soft": ("blur", "dissolve", "circle"),
    # The effect transitions are the loudest thing in the vocabulary, so their
    # replacements stay loud - swapping a glitch for a dissolve would deflate the cut.
    "glitch": ("flash", "film_burn", "zoom"),
    "light_leak": ("dip", "flash", "soft"),
    "film_burn": ("flash", "glitch", "zoom"),
}


# Families an edit that does not want to shout will not use. A cut still has to go
# somewhere, so each is swapped for something in the same place in the sentence but
# spoken quietly - and rotated, so the substitution does not collapse to one name.
_QUIET_SWAPS: dict[str, tuple[str, ...]] = {
    "flash": ("dip", "blur", "dissolve"),
    "glitch": ("dip", "blur", "dissolve"),
    "film_burn": ("dip", "blur"),
    "light_leak": ("dip", "soft"),
    "zoom": ("blur", "dissolve", "circle"),
    "pixel": ("blur", "soft"),
    "squeeze": ("soft", "smooth"),
    "slice": ("soft", "smooth", "blur"),
    "radial": ("circle", "soft"),
    "wind": ("soft", "smooth"),
    "corner": ("smooth", "soft"),
    "mask": ("circle", "soft"),
    "open": ("circle", "soft"),
    "close": ("circle", "soft"),
}

# Inside a section, the pool a visible cut draws from. Same musical moment, three
# different volumes: a quiet edit dissolves, a loud one is part of the percussion.
_INSIDE_POOLS: dict[str, tuple[str, ...]] = {
    "subtle": ("blur", "soft", "smooth", "mask", "reveal", "circle", "wipe", "slide"),
    "balanced": ("wipe", "slide", "diag", "pixel", "squeeze", "corner", "wind", "glitch"),
    "impact": ("glitch", "flash", "pixel", "squeeze", "slice", "zoom", "wind", "corner"),
}


def _assign_transitions(
    shots: list[Shot], *, mood: str = "", density: float = .35,
    flavour: str = "balanced",
) -> None:
    """Pick a transition family per cut, then keep it from repeating itself.

    Hard cuts stay the default. A visible transition is punctuation, and
    punctuating every cut turns the edit into a slideshow, so only structural
    changes - a new section, a change of intent - and a bounded share of the cuts
    inside a section earn one.
    """
    visible = 0
    for index, (shot, following) in enumerate(zip(shots, shots[1:])):
        family = _transition_family(
            shot, following, index=index, visible=visible, mood=mood, density=density,
            flavour=flavour,
        )
        shot.transition = family
        if family not in _SUBTLE_TRANSITIONS and family not in {"cut", "none", ""}:
            visible += 1
    if shots:
        shots[-1].transition = "none"
    _break_transition_repeats(shots)


def _transition_family(
    shot: Shot, following: Shot, *, index: int, visible: int, mood: str, density: float,
    flavour: str = "balanced",
) -> str:
    """Name the transition family the edit is asking for at this cut.

    Two counters, because they answer different questions. ``index`` is the cut's
    position in the timeline, which is what the density gate wants - a bounded share
    of the cuts, spread across the whole edit. ``visible`` counts the visible
    transitions handed out so far, and it is what the rotations index on.

    Rotating on the shot index starves whole families. A branch is narrow - ``open``
    only fires on a section change in a dreamy song - so it might fire three times at
    cuts 7, 19 and 31, every one of them ``% 3 == 1``, and two of its three names can
    never be reached. Rotating on the visible count makes consecutive firings take
    consecutive slots, so a branch that fires as often as its rotation is long covers
    all of it.
    """
    tone = following.transition_tone if following.transition_tone != "neutral" else shot.transition_tone
    dreamy = mood in {"dreamy", "romantic"} or tone == "soft"
    restless = mood in {"energetic", "dark"} or tone == "bright"

    if shot.section_index != following.section_index:
        # Structural punctuation: the strongest move the music can justify. A seam is
        # always punctuated, which is why it sits above the density gate - the gate is
        # about how many *ordinary* cuts get to be visible, not about the seams.
        return _quieten(_structural_family(shot, following, visible, dreamy, restless), flavour, visible)

    # Inside a section, spend a bounded share of the cut points on visible moves. This
    # has to come before the intent branches: a breathe or an impact cut is still a cut
    # inside a section, and a low density has to be able to silence it. Letting those
    # branches answer first made ``transition_density`` almost meaningless - a
    # documentary edit asking for 0.10 still got a visible transition on most cuts.
    if ((index * 29 + 11) % 100) / 100 >= density:
        return "cut"

    if "impact" in {shot.edit_intent, following.edit_intent}:
        family = ("zoom", "film_burn", "flash", "glitch")[visible % 4]
    elif "breathe" in {shot.edit_intent, following.edit_intent}:
        family = ("blur", "soft")[visible % 2] if dreamy else "dissolve"
    else:
        energy = max(shot.energy, following.energy)
        if energy > .70:
            pool = _INSIDE_POOLS.get(flavour, _INSIDE_POOLS["balanced"])
            family = pool[visible % len(pool)]
        elif energy < .32:
            family = ("blur", "soft", "dissolve")[visible % 3] if dreamy else ("dissolve", "soft")[visible % 2]
        else:
            family = ("wipe", "reveal", "dissolve", "smooth", "mask")[visible % 5]
    return _quieten(family, flavour, visible)


def _structural_family(
    shot: Shot, following: Shot, visible: int, dreamy: bool, restless: bool,
) -> str:
    """The family a seam between two sections earns."""
    if following.energy > .78:
        # A cut this loud wants a flash or a glitch, not a blend.
        return ("flash", "glitch", "light_leak")[visible % 3]
    if following.section == "chorus":
        return ("radial", "wind", "mask")[visible % 3] if restless else ("zoom", "mask", "smooth")[visible % 3]
    if following.section in {"intro", "outro"} or shot.section == "chorus":
        return "dip"
    if dreamy:
        return ("circle", "soft", "open")[visible % 3]
    if restless:
        return ("slice", "corner", "close")[visible % 3]
    return "dissolve"


def _quieten(family: str, flavour: str, visible: int) -> str:
    """Tone a family down when the style has asked for restraint.

    An edit that never raises its voice still has to punctuate; it just does it with a
    dip or a blur instead of a flash. The swap is rotated so the quiet alternative does
    not become its own tic.
    """
    if flavour != "subtle":
        return family
    alternatives = _QUIET_SWAPS.get(family)
    return alternatives[visible % len(alternatives)] if alternatives else family


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

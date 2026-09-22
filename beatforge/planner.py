from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from beatforge.audio import AudioAnalysis, section_at
from beatforge.editing import EditStyle
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset

if TYPE_CHECKING:
    from beatforge.audit import ConfigAudit
    from beatforge.models.ai_director import DirectorTreatment, SectionDirection

#: Share of the film a director's motifs should occupy, in total. Kept as a module
#: constant so ``scripts/reuse_probe.py`` can be pointed at it: the acceptance band is
#: 8%–30%, and this is the single number that moves it.
_MOTIF_SHARE = 0.10
#: Hard ceiling on motif shots, whatever the share works out to. A song where
#: ``shot_count * share / len(motifs)`` would push the motif share past this has too
#: many motifs enabled, not too few repeats - so the schedule drops motifs instead.
_MOTIF_CAP_SHARE = 0.30
#: Every motif is expected to come back at least this often, when the song is long
#: enough to hold it. The number is a floor to aim for, not a guarantee: at eight shots,
#: three appearances for even one motif is more than the cap allows.
_MOTIF_MIN_REPEATS = 3


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
    section_index: int = -1
    source_color: list[int] = field(default_factory=lambda: [128, 128, 128])
    focus_point: list[float] = field(default_factory=lambda: [.5, .5])
    source_width: int = 0
    source_height: int = 0
    image_effect: str = "cinematic_depth"
    layers: list[ShotLayer] = field(default_factory=list)
    #: This shot is a reserved appearance of one of the director's motifs.
    is_motif: bool = False
    #: Brightness/exposure of the source, carried off the asset for the grade and the
    #: vignette gate. The renderer reads the shot, never the media file again.
    luma: float = 0.5
    edge_luma: float = 0.5
    noise_score: float = 0.0
    #: The subject's ``(x, y)`` extents in the source, off the asset. The full-bleed
    #: crop consults it before throwing picture away; ``None`` (videos) means no gate.
    subject_span: list[float] | None = None

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
    motifs: list[int] | None = None,
    video_quota: float = 0.0,
    quality_floor: float = 0.0,
    audit: ConfigAudit | None = None,
) -> list[Shot]:
    """Lay out shots on the musical grid and pick media for each of them.

    Reuse policy (R-03): a **non-motif** asset that is already on screen is only chosen
    again once every non-motif asset has been seen - while the untouched supply lasts,
    each shot gets something new. The director's **motifs** are the deliberate exception:
    they are scheduled into reserved slots ahead of time and come back on purpose, so the
    film has a handful of images the audience recognises instead of 255 strangers.

    A style still sets the *habit* - which grid a cut lands on, how loud a transition may
    be - but it no longer owns the shot-length window: that is decided in
    ``audit.apply_style`` (explicit config first), and this function only falls back to
    the style's window when no explicit one was passed. ``min_shot``/``max_shot`` given
    here are used as-is when present.
    """
    if min_shot is None or max_shot is None:
        if style is None:
            raise ValueError("需要给出 min_shot/max_shot，或指定一个 edit_style")
        min_shot = style.shot_min if min_shot is None else min_shot
        max_shot = style.shot_max if max_shot is None else max_shot
    if transition_density is None:
        transition_density = style.transition_density if style else .35
    if image_composite_ratio is None:
        image_composite_ratio = style.composite_ratio if style else .24
    boundaries = _boundaries(analysis, lyrics, min_shot, max_shot, treatment, style)
    # How hard this edit insists on a shot-size change across a cut.
    contrast = style.shot_size_contrast if style else .09
    lyric_rows = {id(line): i for i, line in enumerate(lyrics)}
    usage: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    previous: MediaAsset | None = None
    video_cursors: dict[int, float] = {}
    lyric_visual_history: dict[str, set[int]] = {}
    active_line_id: int | None = None
    active_lyric_key = ""
    active_line_visual_assets: set[int] = set()
    shots: list[Shot] = []
    shot_count = len(boundaries) - 1
    columns = {asset.id: index for index, asset in enumerate(assets)}
    by_id = {asset.id: asset for asset in assets}
    active_motifs = [motif for motif in dict.fromkeys(motifs if motifs is not None
                    else (treatment.motif_asset_ids if treatment else [])) if motif in by_id]
    motif_ids = set(active_motifs)
    # Motifs are reserved *before* the main loop, from the section each shot lands in, so
    # the schedule never depends on which asset the scorer would have picked - and the
    # plan stays reproducible.
    shot_sections = [
        section_at(analysis, (start + end) / 2)[0] for start, end in pairwise(boundaries)
    ]
    reserved = _motif_schedule(shot_sections, active_motifs)
    video_target = round(shot_count * max(0.0, video_quota))
    used_videos = 0
    # Camera-move rotations advance on this, not on the shot index. Composites and motif
    # slots claim a share of the shots, and a rotation keyed off the global index would
    # then lose whichever slots those shots occupied - with enough composites in one
    # section, a whole group of moves can never be reached at all.
    single_image_cursor = 0
    for index, (start, end) in enumerate(pairwise(boundaries)):
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
        reserved_remaining = sum(1 for slot in reserved if slot >= index)
        non_motif_assets = [asset for asset in assets if asset.id not in motif_ids]
        non_motif_min = (
            min(usage.get(asset.id, 0) for asset in non_motif_assets)
            if non_motif_assets else 0
        )
        non_motif_least = [
            asset for asset in non_motif_assets if usage.get(asset.id, 0) == non_motif_min
        ]
        # Assets left over after covering every remaining shot with a distinct one.
        # Only this surplus may be spent on composite layers, and motifs are not part of
        # the supply - they are already spoken for by ``reserved``.
        spare_assets = len(non_motif_least) - (remaining_shots - reserved_remaining)
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
            director_score = _director_asset_score(asset, direction, treatment)
            duration_penalty = .24 if asset.kind == "video" and asset.duration < shot_duration + .25 else 0.0
            framing_penalty = _framing_penalty(asset, target_width / max(target_height, 1))
            upscale_penalty = _upscale_penalty(asset, target_width, target_height)
            score = semantic + mood + movement + quality + continuity + shot_variety + section_fit + director_score - repeat - duration_penalty - framing_penalty - upscale_penalty
            ranked.append((score, asset, semantic, asset_column))
        reserved_motif = reserved.get(index)
        if reserved_motif is not None:
            # A motif slot: the asset was chosen before scoring, and the least-used tier
            # logic deliberately does not apply - that is the whole point of reserving it.
            selected = by_id[reserved_motif]
            semantic = next(
                (item[2] for item in ranked if item[1].id == reserved_motif), 0.0
            )
            selected_column = columns[reserved_motif]
            is_motif = True
        else:
            is_motif = False
            # The non-motif supply, unless there is none - a pool where every asset is a
            # motif (a three-photo project, say) has nothing else to draw on, so the
            # ranking stands rather than leaving the slot with no candidate at all.
            pool = [item for item in ranked if item[1].id not in motif_ids] or ranked
            if quality_floor > 0:
                gated = [item for item in pool if item[1].quality_score >= quality_floor]
                if gated:
                    pool = gated
                elif audit is not None:
                    # The gate only gives way when nothing above the floor is left, and
                    # that surrender is declared rather than silent (R-02/R-09).
                    low = [item for item in pool if item[1].quality_score < quality_floor]
                    audit.record(
                        "quality_floor", requested=quality_floor, effective=0.0,
                        overridden_by="quality-floor-degraded",
                        reason=(
                            f"素材池中已无 ≥ Q{quality_floor * 100:.0f} 分位的主镜头可用，"
                            f"{len(low)} 个低质素材放行"
                        ),
                    )
            # A repeated lyric line should not show the same picture twice.
            candidates = [item for item in pool if item[1].id not in previously_visible_for_lyric] or pool
            if avoid_asset_repeats:
                # Prefer the least-used non-motif assets so the audience keeps seeing new
                # material; a non-motif is only shown again once every other non-motif has
                # caught up.
                tier = [item for item in candidates if usage.get(item[1].id, 0) == non_motif_min]
                candidates = tier or candidates
            if video_target > 0 and used_videos < video_target and (
                used_videos * max(shot_count, 1) < video_target * (index + 1)
            ):
                # R-08: a soft quota. Videos starve because the pool is mostly stills and
                # the scorer runs shot-by-shot; while the edit is *behind* its video
                # target for this point in the timeline, prefer the videos that are
                # already in the zero-reuse tier. Videos the tier does not offer are left
                # alone rather than forced - that keeps the non-motif zero-reuse rule
                # intact, and the pace is measured against the running position rather
                # than a fixed count so the clips spread across the whole film.
                video_tier = [item for item in candidates if item[1].kind == "video"]
                if video_tier:
                    candidates = video_tier
            _, selected, semantic, selected_column = max(candidates, key=lambda item: item[0])
        continues_previous = previous is not None and selected.id == previous.id
        usage[selected.id] = usage.get(selected.id, 0) + 1
        last_seen[selected.id] = index
        if selected.kind == "video":
            used_videos += 1
        if line is not None:
            active_line_visual_assets.add(selected.id)
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
            # least-used images so they cannot drain the shots still to come - and never
            # a motif, which is reserved for its own slots.
            layer_pool = [item for item in ranked if item[1].id not in motif_ids]
            if not (spare_assets >= 0 or not avoid_asset_repeats):
                layer_pool = ranked
            image_candidates = [
                item for item in sorted(
                    layer_pool, key=lambda item: (usage.get(item[1].id, 0), -item[0]),
                )
                if item[1].kind == "image" and item[1].id != selected.id
                and item[1].id not in motif_ids
                and item[1].id not in previously_visible_for_lyric
            ]
            if avoid_asset_repeats and spare_assets >= 0:
                # Composite layers are a luxury: they may only spend assets that are
                # surplus to the distinct ones still needed to cover the remaining shots.
                spare_pool = [
                    item for item in image_candidates
                    if usage.get(item[1].id, 0) == non_motif_min
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
            section_index=section_index,
            source_color=selected.dominant_color.copy(),
            focus_point=selected.focus_point.copy(),
            source_width=selected.width,
            source_height=selected.height,
            image_effect=image_effect,
            layers=layers,
            is_motif=is_motif,
            luma=selected.luma,
            edge_luma=selected.edge_luma,
            noise_score=selected.noise_score,
            subject_span=selected.subject_span,
        ))
    _assign_transitions(
        shots, mood=analysis.mood, density=transition_density,
        flavour=style.transition_flavour if style else "balanced",
    )
    return shots


def _motif_schedule(
    shot_sections: list[str],
    motifs: list[int],
    *,
    min_repeats: int = _MOTIF_MIN_REPEATS,
    share: float = _MOTIF_SHARE,
    cap_share: float = _MOTIF_CAP_SHARE,
) -> dict[int, int]:
    """Reserve the shots where a director's motifs come back. Pure, deterministic.

    Motifs have to *repeat* to be motifs - a visual theme that appears once is a shot.
    The reservations are laid out before any asset is scored so the schedule does not
    depend on the scorer, and they are spread across the whole film with a per-motif
    phase so the theme recurs throughout rather than clumping at the front.

    Two clamps keep the schedule honest on short songs. The total is capped at
    ``cap_share`` of the shots, and when the floor of ``min_repeats`` cannot fit inside
    that cap the schedule **drops motifs** rather than padding counts - five themes on a
    forty-shot song asking for three appearances each is 37%, not a motif. On a very
    short song even one motif cannot reach the floor, and the schedule degrades to fewer
    appearances, which is the best that is arithmetically available.
    """
    shot_count = len(shot_sections)
    active = list(dict.fromkeys(motifs))
    if shot_count <= 0 or not active:
        return {}
    budget = int(shot_count * cap_share)
    if budget <= 0:
        return {}
    max_motifs = max(1, budget // max(1, min_repeats))
    if len(active) > max_motifs:
        active = active[:max_motifs]
    per = max(min_repeats, round(shot_count * share / len(active)))
    per = max(1, min(per, budget // len(active)))
    if per <= 0:
        return {}
    schedule: dict[int, int] = {}
    span = per * len(active)
    for position, motif in enumerate(active):
        for step in range(per):
            fraction = (step + 0.5) / per
            fraction = (fraction + position / (span + 1)) % 1.0
            index = min(shot_count - 1, int(fraction * shot_count))
            index = _nearest_free_slot(index, schedule, shot_count)
            if index is not None:
                schedule[index] = motif
    return schedule


def _nearest_free_slot(index: int, taken: dict[int, int], shot_count: int) -> int | None:
    """The nearest shot index not already reserved, searching forward then back."""
    if index not in taken:
        return index
    for step in range(1, shot_count):
        for candidate in (index + step, index - step):
            if 0 <= candidate < shot_count and candidate not in taken:
                return candidate
    return None


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

    A letterbox and an iris are the still-image equivalent of a composite: each is
    worth seeing once and tedious if every shot gets one. The gate keeps them
    occasional without making them random.

    ``parallax`` used to be the third name here. It was removed with the single-image
    rework (R-04): its whole mechanism was splitting one still into a blurred backdrop
    and a scaled foreground - the same-image double it was meant to avoid - and a
    genuine depth cue is not recoverable from a flat photo. Two names instead of three
    also lowers how often a frame gets any frame at all, which is the point.
    """
    if ((cursor * 53 + 7) % 100) / 100 >= .085:
        return ""
    if section == "intro":
        return "iris"
    return "film_bars" if energy > .7 else ""


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

    When a director's treatment is present, the *intensity arc decides the target shot
    length* and the style only supplies the window and the grid (R-01): a section the
    director called intense cuts short, a section it called calm holds. With no
    treatment - a ``--no-ai`` run, or a rule-director fallback - the original
    tempo/energy formula answers instead, so every AI-less render is unchanged.
    """
    tempo = style.tempo if style else 3.8
    gain = style.energy_gain if style else 1.5
    speedup = style.section_speedup if style else 0.0
    alignment = style.cut_alignment if style else "downbeat"

    anchors = sorted({0.0, analysis.duration, *analysis.sections})
    # A director rarely uses the whole 0..1 intensity scale. The my-mv treatment spans
    # 0.20-0.65; fed in raw, that spends under half the window, and the film comes out
    # uniformly slow (measured: per-section means 3.36-3.60s, p90/p10 = 1.76, Pearson r
    # = -0.44) instead of contrasted. Rescaling the arc *within this song* onto the full
    # window keeps the director's relative intent - calm stays calmer than its chorus -
    # while restoring the contrast R-01 exists to produce. Songs whose arc is genuinely
    # flat are left alone: there is no contrast to recover.
    arc_lo = arc_hi = None
    if treatment is not None and treatment.sections:
        levels = [section.cut_intensity for section in treatment.sections]
        arc_lo, arc_hi = min(levels), max(levels)
    output = [0.0]
    for target in anchors[1:]:
        cursor = output[-1]
        while target - cursor > maximum:
            section, section_index = section_at(analysis, cursor)
            direction = treatment.section(section_index) if treatment else None
            if direction is not None:
                intensity = direction.cut_intensity
                if arc_hi is not None and arc_lo is not None and arc_hi > arc_lo:
                    intensity = (intensity - arc_lo) / (arc_hi - arc_lo)
                progress = _section_progress(analysis, section_index, cursor)
                ideal = cursor + _section_target_length(
                    intensity, minimum, maximum, speedup, progress,
                )
            else:
                section_scale = .78 if section == "chorus" else 1.18 if section in {"intro", "outro", "bridge"} else 1.0
                # Tighten toward the end of a section. An editor accelerates into the
                # drop; the reverse - shots getting longer as the section builds - reads
                # as an edit that has run out of ideas right where it should be peaking.
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


def _section_target_length(
    intensity: float, minimum: float, maximum: float, speedup: float, progress: float,
) -> float:
    """The shot length a director's per-section intensity asks for (R-01).

    Intensity runs 0..1 and maps straight onto the window: 1 wants the shortest shot the
    style allows, 0 the longest. That inversion is the point - the arc is the target, and
    the window is only the range it may pick inside. ``speedup`` then tightens the target
    toward the end of the section, the same way an editor accelerates into the drop.
    """
    span = maximum - minimum
    target = maximum - float(intensity) * span
    target *= 1 - speedup * float(progress)
    return float(np.clip(target, minimum, maximum))


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
    """How well a source suits a moment, by energy alone.

    Stills get a gentle bias toward the quiet end: a still cut into a loud section works,
    but it should not be the first choice when moving footage is on the table. Video used
    to read ``camera_motion`` here, but the field was never populated - every asset read
    "unknown", so the branch scored everything the same while pretending to know. It is
    gone (R-11) and neither kind carries fabricated movement metadata any more.
    """
    if asset.kind == "image":
        return (1 - energy) * .06
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

#: The transitions that shout. Under R-07 each of these needs the *music* to authorise
#: it - a director section marked ``transition_tone`` bright or dark, or an ``impact``
#: intent on one side of the cut. Falling back to an unnamed energy gate is what put 58
#: flashes and glitches into a "warm documentary" cut.
_IMPACT_FAMILIES = frozenset({"flash", "glitch", "film_burn", "light_leak", "zoom"})

# Inside a section, the pool a visible cut draws from. Same musical moment, three
# different volumes: a quiet edit dissolves, a loud one is part of the percussion.
# The rotations are kept short (four to six names) so one song cannot walk the whole
# library - R-07 asks for at most eight families in any single film. Every family still
# appears in some pool or some branch, so the coverage sweep can still reach it.
_INSIDE_POOLS: dict[str, tuple[str, ...]] = {
    "subtle": ("blur", "soft", "mask", "wipe"),
    "balanced": ("wipe", "slide", "diag", "pixel", "corner"),
    "impact": ("glitch", "flash", "pixel", "zoom", "squeeze"),
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
    for index, (shot, following) in enumerate(pairwise(shots)):
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


def _impact_authorised(shot: Shot, following: Shot) -> bool:
    """Is the music allowed a loud transition at this cut?

    The old code asked only how loud the *following* shot was, which is a property of the
    mix, not of the film: a quiet, warm documentary still has loud moments. R-07 moves the
    permission to the director - a section the director called bright or dark may shout,
    as may a cut either side of which carries an ``impact`` intent. Everything else is
    quiet by default even where the meter is high.
    """
    tone = following.transition_tone if following.transition_tone != "neutral" else shot.transition_tone
    if "impact" in {shot.edit_intent, following.edit_intent}:
        return True
    return tone in {"bright", "dark"}


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
    authorised = _impact_authorised(shot, following)

    if shot.section_index != following.section_index:
        # Structural punctuation: the strongest move the music can justify. A seam is
        # always punctuated, which is why it sits above the density gate - the gate is
        # about how many *ordinary* cuts get to be visible, not about the seams.
        return _quieten(
            _structural_family(shot, following, visible, dreamy, restless, authorised),
            flavour, visible,
        )

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
            if not authorised:
                # A loud meter is not a licence to shout (R-07): fall back to the
                # balanced pool, whose names read as energy rather than as impact.
                pool = _INSIDE_POOLS["balanced"]
            family = pool[visible % len(pool)]
        elif energy < .32:
            family = ("blur", "soft", "dissolve")[visible % 3] if dreamy else ("dissolve", "soft")[visible % 2]
        else:
            family = ("wipe", "reveal", "dissolve", "smooth", "mask")[visible % 5]
    return _quieten(family, flavour, visible)


def _structural_family(
    shot: Shot, following: Shot, visible: int, dreamy: bool, restless: bool,
    authorised: bool,
) -> str:
    """The family a seam between two sections earns."""
    if following.energy > .78 and authorised:
        # A cut this loud wants a flash or a glitch, not a blend - but only when the
        # director's tone or intent has actually asked the edit to shout (R-07).
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

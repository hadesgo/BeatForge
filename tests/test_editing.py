"""The editing styles, and the craft they are supposed to encode."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from beatforge.audio import AudioAnalysis
from beatforge.editing import (
    CUT_ALIGNMENTS,
    EDIT_STYLES,
    TRANSITION_FLAVOURS,
    EditStyle,
    resolve_style,
)
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import _boundaries, create_plan
from beatforge.renderer import _EFFECT_TRANSITIONS


def _song(duration: float = 120.0, *, mood: str = "cinematic", density: float = 60.0) -> AudioAnalysis:
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    step = duration / len(labels)
    return AudioAnalysis(
        duration=duration, bpm=120,
        beats=[x / 2 for x in range(int(duration * 2) + 1)],
        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
        sections=[index * step for index in range(len(labels) + 1)],
        energy_times=[0, duration * .3, duration * .7, duration],
        energy_values=[.2, .9, .5, .25],
        average_energy=.55, brightness=.5, mood=mood, mood_scores={mood: 1},
        section_labels=labels,
        melody_times=[0, duration / 2, duration], melody_values=[.2, .8, .3],
        melodic_motion=.5, rhythmic_density=density,
    )


def _media(count: int, *, blocked: bool = True) -> list[MediaAsset]:
    sizes = ["wide", "medium", "closeup", "detail"]
    return [
        MediaAsset(
            index, Path(f"p{index}.jpg"), "image", float("inf"), 1920, 1080,
            shot_size=sizes[(index // 7) % 4] if blocked else sizes[index % 4],
        )
        for index in range(count)
    ]


def _lines(duration: float) -> list[LyricLine]:
    return [LyricLine(t, t + 3, f"第{int(t)}句") for t in np.arange(0, duration - 3, 3)]


# --------------------------------------------------------------------- the table


@pytest.mark.parametrize("name", sorted(EDIT_STYLES))
def test_every_style_is_internally_consistent(name: str) -> None:
    """A style is a promise about pacing; the numbers have to keep it."""
    style = EDIT_STYLES[name]
    assert style.shot_min < style.shot_max, name
    assert style.shot_min <= style.tempo <= style.shot_max, f"{name}: tempo outside its own range"
    assert style.cut_alignment in CUT_ALIGNMENTS, name
    assert style.transition_flavour in TRANSITION_FLAVOURS, name
    assert 0 <= style.transition_density <= 1, name
    assert 0 <= style.composite_ratio <= 1, name
    assert style.subtitle_layout in {"band", "free"}, name
    assert style.label and style.summary, name


def test_the_styles_actually_span_a_range_of_tempos() -> None:
    """Eight names that all cut at the same speed are one style with eight labels."""
    tempos = [style.tempo for style in EDIT_STYLES.values()]
    assert max(tempos) / min(tempos) > 3, tempos
    assert len({style.cut_alignment for style in EDIT_STYLES.values()}) >= 3
    assert len({style.transition_flavour for style in EDIT_STYLES.values()}) == 3


def test_resolve_style_maps_mood_and_honours_manual() -> None:
    assert resolve_style("manual", "cinematic") is None
    assert resolve_style("beat", "melancholic") is EDIT_STYLES["beat"], "a named style wins over mood"
    assert resolve_style("auto", "melancholic") is EDIT_STYLES["cinematic"]
    assert resolve_style("auto", "dreamy") is EDIT_STYLES["dream"]


def test_auto_prefers_a_fast_style_for_a_relentless_song() -> None:
    """Mood describes the colour, not the tempo.

    A song that is loud and busy the whole way through is not a long-take song, even if
    its mood reads as cinematic - the style has to follow what the cut can do.
    """
    assert resolve_style(
        "auto", "cinematic", average_energy=.8, rhythmic_density=95,
    ) is EDIT_STYLES["beat"]
    assert resolve_style(
        "auto", "cinematic", average_energy=.3, rhythmic_density=40,
    ) is EDIT_STYLES["cinematic"]


# ------------------------------------------------------------------ the craft


def test_cut_alignment_changes_where_cuts_are_allowed_to_land() -> None:
    """The grid a cut may land on is the most audible decision in the file."""
    analysis = _song(60)
    lines = _lines(60)
    # A window wide enough that every grid has candidates. A style whose shots top out
    # below one bar cannot cut on phrases at all, and would fall back silently.
    grids = {}
    for alignment in ("beat", "downbeat", "phrase"):
        style = EditStyle(
            label=alignment, summary="", shot_min=2.0, shot_max=6.0, tempo=3.5,
            energy_gain=1.0, cut_alignment=alignment, section_speedup=.0,
            transition_density=.3, transition_flavour="balanced",
            shot_size_contrast=.1, camera_intensity=1.0, composite_ratio=.2,
            subtitle_layout="free",
        )
        grids[alignment] = _boundaries(analysis, lines, 2.0, 6.0, None, style)

    assert grids["beat"] != grids["downbeat"], "cutting on every beat must differ from bar lines"
    assert grids["phrase"] != grids["downbeat"], "phrase cuts must differ from bar lines"
    assert grids["beat"] != grids["phrase"]
    beats = set(analysis.beats)
    downbeats = set(analysis.downbeats)
    # A beat-grid cut is allowed anywhere on the beat; a bar-grid cut only on bar lines.
    assert all(any(abs(cut - beat) < 1e-6 for beat in beats) for cut in grids["beat"][1:-1])
    assert all(any(abs(cut - bar) < 1e-6 for bar in downbeats) for cut in grids["downbeat"][1:-1])


def test_cutting_on_lyric_lines_puts_the_cuts_on_the_words() -> None:
    analysis = _song(60)
    lines = _lines(60)
    style = EditStyle(
        label="lyric", summary="", shot_min=1.5, shot_max=4.0, tempo=3.0, energy_gain=.8,
        cut_alignment="lyric", section_speedup=.0, transition_density=.3,
        transition_flavour="subtle", shot_size_contrast=.1, camera_intensity=1.0,
        composite_ratio=.2, subtitle_layout="free",
    )
    boundaries = _boundaries(analysis, lines, 1.5, 4.0, None, style)
    starts = {round(line.start, 3) for line in lines}

    assert len(boundaries) > 5
    inside = [cut for cut in boundaries[1:-1] if cut not in starts]
    # Interior cuts land on lyric starts or on a section seam, nowhere else.
    assert len(inside) <= len(analysis.sections) + 1, inside


def test_a_style_with_no_speedup_keeps_its_shot_length_flat() -> None:
    """Tightening toward the drop is a choice, not a law."""
    analysis = _song(120)
    lines = _lines(120)

    def lengths(speedup: float) -> list[float]:
        style = EditStyle(
            label="s", summary="", shot_min=1.5, shot_max=4.0, tempo=3.0, energy_gain=.8,
            cut_alignment="downbeat", section_speedup=speedup, transition_density=.3,
            transition_flavour="balanced", shot_size_contrast=.1, camera_intensity=1.0,
            composite_ratio=.2, subtitle_layout="free",
        )
        boundaries = _boundaries(analysis, lines, 1.5, 4.0, None, style)
        return [round(b - a, 3) for a, b in pairwise(boundaries)]

    flat, tightening = lengths(0.0), lengths(.45)
    assert tightening != flat
    # Both cover the same song, so the tightening one has to fit more cuts in.
    assert len(tightening) >= len(flat)


def test_shot_size_contrast_keeps_the_picture_changing() -> None:
    """Two mediums next to each other is a list; a wide next to a close-up is a sentence.

    This is what stops fast cutting from turning into mush: at four shots a second the
    audience needs the shot size to carry the difference between one cut and the next.
    """
    analysis = _song(60)
    assets = _media(40)
    sizes = {asset.id: asset.shot_size for asset in assets}

    def clashes(contrast: float) -> float:
        style = EditStyle(
            label="s", summary="", shot_min=1.5, shot_max=3.0, tempo=2.2, energy_gain=1.0,
            cut_alignment="downbeat", section_speedup=.1, transition_density=.3,
            transition_flavour="balanced", shot_size_contrast=contrast,
            camera_intensity=1.0, composite_ratio=.2, subtitle_layout="free",
        )
        shots = create_plan(analysis, _lines(60), assets, None, min_shot=1.5, max_shot=3.0,
                            style=style)
        return sum(
            1 for index in range(1, len(shots))
            if sizes[shots[index].media_id] == sizes[shots[index - 1].media_id]
        ) / (len(shots) - 1)

    assert clashes(.22) < clashes(0.0) / 3, "the contrast rule barely did anything"


def test_the_style_owns_the_pacing_it_was_given() -> None:
    """A style and a hand-set shot length are two answers to the same question."""
    analysis = _song(90)
    fast = create_plan(
        analysis, _lines(90), _media(80), None, min_shot=4.0, max_shot=6.0,
        style=EDIT_STYLES["beat"],
    )
    slow = create_plan(
        analysis, _lines(90), _media(80), None, min_shot=4.0, max_shot=6.0,
        style=EDIT_STYLES["cinematic"],
    )
    assert len(fast) > len(slow) * 2, "the style did not override the explicit shot length"


def test_a_quiet_style_never_reaches_for_a_loud_transition() -> None:
    analysis = _song(150, mood="dreamy")
    shots = create_plan(
        analysis, _lines(150), _media(80), None, style=EDIT_STYLES["dream"],
    )
    used = {shot.transition for shot in shots}
    assert not used & set(_EFFECT_TRANSITIONS), f"a quiet edit used {used & set(_EFFECT_TRANSITIONS)}"
    assert "flash" not in used and "zoom" not in used, used


def test_an_impact_style_is_allowed_to_shout() -> None:
    analysis = _song(150, mood="energetic", density=95)
    shots = create_plan(
        analysis, _lines(150), _media(120), None, style=EDIT_STYLES["impact"],
    )
    used = {shot.transition for shot in shots}
    assert used & (set(_EFFECT_TRANSITIONS) | {"flash", "zoom", "pixel", "slice"})


def test_transition_density_now_controls_how_many_cuts_are_visible() -> None:
    """It has to silence intent-driven cuts too, not just the ones inside a section.

    The impact and breathe branches used to answer before the gate, so a documentary
    edit asking for 0.10 still got a visible transition on most of its cuts.
    """
    analysis = _song(150)

    def visible(style_name: str) -> float:
        shots = create_plan(analysis, _lines(150), _media(90), None, style=EDIT_STYLES[style_name])
        names = [shot.transition for shot in shots[:-1]]
        return sum(1 for name in names if name not in {"cut", "none"}) / len(names)

    quiet = visible("documentary")
    loud = visible("impact")
    assert quiet < .3, f"documentary edit was still punctuating {quiet:.0%} of its cuts"
    assert loud > .5, f"impact edit only punctuated {loud:.0%} of its cuts"
    assert loud > quiet * 2


def test_styles_do_not_collapse_onto_one_another() -> None:
    """Two names that produce the same edit are one style with two labels."""
    analysis = _song(120)
    lines = _lines(120)
    assets = _media(90)
    fingerprints = {}
    for name, style in EDIT_STYLES.items():
        shots = create_plan(analysis, lines, assets, None, style=style)
        names = [shot.transition for shot in shots[:-1]]
        fingerprints[name] = (
            len(shots),
            round(float(np.median([shot.duration for shot in shots])), 2),
            # Two styles can share a tempo and still be different edits - what separates
            # a documentary from a dream piece is mostly how loud the cuts are.
            round(sum(1 for name in names if name not in {"cut", "none"}) / len(names), 2),
        )
    assert len(set(fingerprints.values())) == len(fingerprints), fingerprints

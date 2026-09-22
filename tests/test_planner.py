from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.audit import ConfigAudit
from beatforge.editing import EDIT_STYLES
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.models.ai_director import DirectorTreatment, SectionDirection
from beatforge.planner import (
    IMAGE_COMPOSITES,
    Shot,
    _break_transition_repeats,
    _choose_image_effect,
    _least_used_assets,
    _motif_schedule,
    _reuse_penalty,
    _section_target_length,
    _transition_family,
    _upscale_penalty,
    create_plan,
)


def _grid(duration: float, labels: list[str] | None = None) -> AudioAnalysis:
    return AudioAnalysis(
        duration=duration, bpm=120, beats=[x / 2 for x in range(int(duration * 2) + 1)],
        sections=[0, duration / 2, duration],
        energy_times=[0, duration], energy_values=[.5, .6], average_energy=.55,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
        section_labels=labels or ["verse", "chorus"],
    )


def _lines(duration: float) -> list[LyricLine]:
    return [LyricLine(i * 4, (i + 1) * 4, f"第{i}句") for i in range(int(duration / 4))]


def _images(count: int) -> list[MediaAsset]:
    return [
        MediaAsset(i, Path(f"p{i}.jpg"), "image", float("inf"), 1920, 1080)
        for i in range(count)
    ]


def _videos(count: int, start: int = 0, duration: float = 30.0) -> list[MediaAsset]:
    return [
        MediaAsset(start + i, Path(f"v{i}.mp4"), "video", duration, 1920, 1080)
        for i in range(count)
    ]


def _visible_ids(shots) -> list[int]:
    return [shot.media_id for shot in shots] + [
        layer.media_id for shot in shots for layer in shot.layers
    ]


# --- R-01 / R-03 fixtures -------------------------------------------------------------

def _section_arc(arc: str, count: int) -> list[float]:
    """The ``cut_intensity`` shape a director speaks, as one value per section."""
    if arc == "flat_low":
        return [.2] * count
    if arc == "flat_high":
        return [.85] * count
    if arc == "rise":
        return [round(float(value), 3) for value in np.linspace(.15, .9, count)]
    if arc == "fall":
        return [round(float(value), 3) for value in np.linspace(.9, .15, count)]
    if arc == "alternate":
        return [.85 if index % 2 else .2 for index in range(count)]
    raise ValueError(f"unknown arc {arc!r}")


def _treatment(
    section_count: int, *, arc: str = "flat_low", tone: str = "neutral",
    intent: str = "continuity", motifs: list[int] | None = None,
) -> DirectorTreatment:
    """A director treatment whose ``cut_intensity`` follows ``arc`` (R-01/R-07).

    The coverage sweeps build songs through this so the *phase* of every rotation moves
    with the arc, and so a loud seam can actually be authorised to shout - a treatment
    whose ``transition_tone`` stays neutral would silently starve the impact families.
    """
    values = _section_arc(arc, section_count)
    return DirectorTreatment(
        concept="测试导演方案",
        narrative_arc="从安静走向释放的测试叙事",
        visual_style="克制统一的测试视觉风格",
        grade_profile="cinematic",
        motif_asset_ids=list(motifs or []),
        sections=[
            SectionDirection(
                section_index=index,
                narrative_role=f"第 {index} 段的叙事作用",
                cut_intensity=value,
                transition_tone=tone,
                edit_intent=intent,
            )
            for index, value in enumerate(values)
        ],
    )


def _reach_song(
    duration: float, labels: list[str], *, mood: str = "cinematic",
    peak: float = .5, melody: float = .5,
) -> AudioAnalysis:
    """A song that visits every section type, for the coverage sweeps."""
    return AudioAnalysis(
        duration=duration, bpm=120,
        beats=[x / 2 for x in range(int(duration * 2) + 1)],
        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
        sections=[index * duration / len(labels) for index in range(len(labels) + 1)],
        energy_times=[0, duration * .25, duration * .45, duration * .75, duration],
        energy_values=[.15, .95, .6, peak, .2],
        average_energy=.5, brightness=.5, mood=mood, mood_scores={mood: 1},
        section_labels=labels,
        melody_times=[0, duration / 2, duration], melody_values=[.2, melody, .3],
        melodic_motion=.6, rhythmic_density=80,
    )


def test_plan_is_continuous() -> None:
    analysis = AudioAnalysis(
        duration=12, bpm=120, beats=[x / 2 for x in range(24)], sections=[0, 6, 12],
        energy_times=[0, 6], energy_values=[.2, .8], average_energy=.5,
        brightness=.5, mood="uplifting", mood_scores={"uplifting": 1},
    )
    lyrics = [LyricLine(0, 6, "阳光天空"), LyricLine(6, 12, "城市跳舞")]
    assets = [
        MediaAsset(0, Path("sun.jpg"), "image", float("inf"), 100, 100, ["阳光"], "天空", "uplifting"),
        MediaAsset(1, Path("city.mp4"), "video", 20, 100, 100, ["城市"], "跳舞", "energetic"),
    ]
    shots = create_plan(analysis, lyrics, assets, None, min_shot=1.5, max_shot=4)
    assert shots[0].start == 0
    assert shots[-1].end == 12
    assert all(a.end == b.start for a, b in pairwise(shots))


def test_ai_similarity_controls_selection() -> None:
    analysis = AudioAnalysis(
        duration=4, bpm=100, beats=[0, 2, 4], sections=[0, 4],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
    )
    lyrics = [LyricLine(0, 4, "歌词")]
    assets = [
        MediaAsset(0, Path("a.jpg"), "image", float("inf"), 100, 100),
        MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100, dominant_color=[20, 40, 80]),
    ]
    shots = create_plan(analysis, lyrics, assets, np.array([[.1, .9]]), min_shot=1.5, max_shot=5)
    assert shots[0].media_id == 1
    assert shots[0].semantic_score == .9
    assert shots[0].source_color == [20, 40, 80]


def _shot(
    index: int, section: str, energy: float, intent: str = "continuity",
    *, tone: str = "neutral", section_index: int = 0,
) -> Shot:
    return Shot(
        index, index * 2, index * 2 + 2, 2, index, "a.jpg", "image", 0, "",
        energy, "dynamic", "cut", .5, section=section, edit_intent=intent,
        transition_tone=tone, section_index=section_index,
    )


def test_plan_punctuates_section_changes_and_keeps_the_rest_mostly_cut() -> None:
    """A visible transition is punctuation; giving every cut one makes a slideshow."""
    analysis = AudioAnalysis(
        duration=8, bpm=120, beats=[0, 2, 4, 6, 8], sections=[0, 4, 8],
        energy_times=[0, 4], energy_values=[.5, .7], average_energy=.6,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse", "chorus"],
    )
    lyrics = [LyricLine(0, 2, "一"), LyricLine(2, 4, "二"), LyricLine(4, 6, "三"), LyricLine(6, 8, "四")]
    assets = [
        MediaAsset(0, Path("a.mp4"), "video", 20, 1920, 1080),
        MediaAsset(1, Path("b.mp4"), "video", 20, 1920, 1080),
    ]

    shots = create_plan(analysis, lyrics, assets, None, min_shot=1.5, max_shot=4)

    assert shots[-1].transition == "none"
    boundary = next(
        index for index, (shot, following) in enumerate(pairwise(shots))
        if shot.section_index != following.section_index
    )
    assert shots[boundary].transition != "cut", "a section change is always punctuated"
    inside = [shot.transition for index, shot in enumerate(shots[:-1]) if index != boundary]
    visible = [name for name in inside if name != "cut"]
    assert len(visible) <= len(inside) * .6, f"too many visible transitions: {inside}"


def test_transition_family_reads_the_music_not_the_calendar() -> None:
    """Inside a section, energy decides which visible move the cut earns."""
    def family(energy: float) -> str:
        return _transition_family(
            _shot(0, "verse", energy), _shot(1, "verse", energy),
            index=0, visible=0, mood="cinematic", density=1.0,
        )

    assert family(.9) in {"wipe", "slide", "diag", "pixel", "squeeze", "corner", "wind", "glitch"}
    assert family(.1) == "dissolve"
    assert family(.5) in {"wipe", "reveal", "dissolve", "smooth", "mask"}


def test_transition_density_zero_suppresses_only_inside_section_moves() -> None:
    """``transition_density = 0`` must not silence a section change."""
    inside = _transition_family(
        _shot(0, "verse", .9), _shot(1, "verse", .9),
        index=0, visible=0, mood="cinematic", density=0.0,
    )
    structural = _transition_family(
        _shot(0, "verse", .5, section_index=0), _shot(1, "chorus", .5, section_index=1),
        index=0, visible=0, mood="cinematic", density=0.0,
    )
    assert inside == "cut"
    assert structural != "cut"


def test_a_section_change_outranks_an_impact_inside_one() -> None:
    """The strongest move the music can justify belongs at the structural seam.

    R-07 moved the permission to shout to the director: a loud seam earns a flash only
    when the section's ``transition_tone`` - or an ``impact`` intent on one side - has
    authorised it. Loudness on its own is not a licence any more, which is what used to
    put a burst of flashes into a warm documentary.
    """
    into_chorus = _transition_family(
        _shot(0, "verse", .5, section_index=0),
        _shot(1, "chorus", .9, section_index=1, tone="bright"),
        index=0, visible=0, mood="cinematic", density=.35,
    )
    into_outro = _transition_family(
        _shot(0, "chorus", .5, section_index=1), _shot(1, "outro", .3, section_index=2),
        index=0, visible=0, mood="cinematic", density=.35,
    )
    unauthorised = _transition_family(
        _shot(0, "verse", .5, section_index=0), _shot(1, "chorus", .9, section_index=1),
        index=0, visible=0, mood="cinematic", density=.35,
    )

    assert into_chorus == "flash"
    assert into_outro == "dip"
    assert unauthorised not in {"flash", "glitch", "film_burn", "light_leak"}, (
        "an unauthorised seam still shouted"
    )


def test_transition_rotations_advance_on_visible_transitions_not_shot_index() -> None:
    """A narrow branch must not be able to lock onto one name for the whole song.

    ``open`` only fires on a section change in a dreamy song, and a song has only a
    handful of those. Keyed off the shot index they can all land on the same residue
    and two of the three names become unreachable; keyed off how many visible
    transitions have already been handed out, consecutive firings take consecutive
    slots.
    """
    def family(visible: int) -> str:
        return _transition_family(
            _shot(0, "verse", .5, section_index=0), _shot(1, "bridge", .5, section_index=1),
            index=7, visible=visible, mood="dreamy", density=1.0,
        )

    # Same cut position, different visible count: the rotation has to move.
    assert len({family(count) for count in range(6)}) == 3


def test_identical_visible_transitions_are_rotated_apart() -> None:
    """Two identical wipes in a row stop reading as punctuation and read as a template."""
    shots = [_shot(index, "verse", .5) for index in range(4)]
    for shot in shots:
        shot.transition = "wipe"
    _break_transition_repeats(shots)

    families = [shot.transition for shot in shots]
    assert families[0] == "wipe"
    for previous, current in pairwise(families):
        assert current != previous, families


def test_subtle_transitions_are_allowed_to_repeat() -> None:
    """A dissolve is invisible enough that repeating it reads as continuity, not a tic."""
    shots = [_shot(index, "verse", .5) for index in range(3)]
    for shot in shots:
        shot.transition = "dissolve"
    _break_transition_repeats(shots)
    assert [shot.transition for shot in shots] == ["dissolve"] * 3


def test_plan_penalizes_video_that_would_need_visible_loop() -> None:
    analysis = AudioAnalysis(
        duration=4, bpm=100, beats=[0, 4], sections=[0, 4],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
    )
    assets = [
        MediaAsset(0, Path("short.mp4"), "video", 1, 1920, 1080, quality_score=.8),
        MediaAsset(1, Path("long.mp4"), "video", 12, 1920, 1080, quality_score=.7),
    ]

    shots = create_plan(analysis, [], assets, None, min_shot=1.5, max_shot=5)

    assert shots[0].media_id == 1


def test_lyrics_choose_content_without_forcing_a_cut_per_line() -> None:
    analysis = AudioAnalysis(
        duration=8, bpm=120, beats=[x / 2 for x in range(17)], sections=[0, 8],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
    )
    lyrics = [LyricLine(i, i + 1, str(i)) for i in range(8)]
    assets = [MediaAsset(0, Path("a.jpg"), "image", float("inf"), 1920, 1080)]

    shots = create_plan(analysis, lyrics, assets, None, min_shot=1.5, max_shot=4)

    assert len(shots) < len(lyrics)


def test_video_starts_near_the_frame_that_matches_the_lyric() -> None:
    analysis = AudioAnalysis(
        duration=4, bpm=120, beats=[0, 2, 4], sections=[0, 4],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
    )
    lyrics = [LyricLine(0, 4, "海边日落")]
    assets = [MediaAsset(0, Path("story.mp4"), "video", 20, 1920, 1080)]

    shots = create_plan(
        analysis, lyrics, assets, np.array([[.9]]), min_shot=1.5, max_shot=5,
        source_starts=np.array([[10.0]]),
    )

    assert shots[0].source_start == 8.0


def test_low_resolution_asset_receives_upscale_penalty() -> None:
    low = MediaAsset(0, Path("low.jpg"), "image", float("inf"), 320, 180)
    high = MediaAsset(1, Path("high.jpg"), "image", float("inf"), 3840, 2160)
    assert _upscale_penalty(low, 1920, 1080) > _upscale_penalty(high, 1920, 1080)


def test_repeated_lyrics_prefer_different_assets() -> None:
    analysis = AudioAnalysis(
        duration=8, bpm=120, beats=[0, 2, 4, 6, 8], sections=[0, 4, 8],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse", "verse"],
    )
    lyrics = [LyricLine(0, 4, "再次飞翔！"), LyricLine(4, 8, " 再次飞翔 ")]
    assets = [
        MediaAsset(0, Path("best.jpg"), "image", float("inf"), 1920, 1080),
        MediaAsset(1, Path("alternate.jpg"), "image", float("inf"), 1920, 1080),
    ]
    similarities = np.array([[1.0, .1], [1.0, .1]])

    shots = create_plan(
        analysis, lyrics, assets, similarities, min_shot=1.5, max_shot=5,
    )

    assert [shot.media_id for shot in shots] == [0, 1]


def test_repeated_lyrics_reuse_only_when_no_alternative_exists() -> None:
    analysis = AudioAnalysis(
        duration=8, bpm=120, beats=[0, 4, 8], sections=[0, 4, 8],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse", "verse"],
    )
    lyrics = [LyricLine(0, 4, "同一句"), LyricLine(4, 8, "同一句")]
    assets = [MediaAsset(0, Path("only.jpg"), "image", float("inf"), 1920, 1080)]

    shots = create_plan(analysis, lyrics, assets, None, min_shot=1.5, max_shot=5)

    assert [shot.media_id for shot in shots] == [0, 0]


def test_assets_are_never_shown_twice_when_supply_is_sufficient() -> None:
    """R-03: while untouched non-motif material remains, no non-motif comes back.

    With no motifs enabled this is the original zero-reuse guarantee, strengthened: the
    plan is not allowed to reach for a repeated picture while a fresh one is on the
    table. The motif contract has its own test below.
    """
    analysis = _grid(100)
    lyrics = _lines(100)
    assets = _images(50)
    similarities = np.random.default_rng(3).uniform(.2, .9, size=(len(lyrics), len(assets)))

    shots = create_plan(
        analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5,
        image_composite_ratio=1,
    )

    visible = _visible_ids(shots)
    assert len(visible) == len(set(visible))
    # Composites are still allowed to spend the surplus material.
    assert any(shot.layers for shot in shots)


def test_motifs_recur_while_non_motifs_stay_fresh() -> None:
    """R-03: the director's motifs come back on purpose; everything else stays new.

    A motif that appears once is a shot, not a theme. The schedule reserves each motif
    its appearances ahead of scoring, and a non-motif is still never shown twice while
    fresh non-motif material is left. The total motif share has to sit inside the
    8%–30% band, which is what stops a short song from padding motifs to hit ``≥3``.
    """
    analysis = _grid(120)
    lyrics = _lines(120)
    assets = _images(60)
    motifs = [0, 1, 2]
    similarities = np.random.default_rng(5).uniform(.2, .9, size=(len(lyrics), len(assets)))

    shots = create_plan(
        analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5,
        motifs=motifs, image_composite_ratio=0,
    )

    counts = Counter(shot.media_id for shot in shots)
    assert {shot.media_id for shot in shots if shot.is_motif} == set(motifs), (
        "an enabled motif never came back"
    )
    for motif in motifs:
        assert counts[motif] >= 3, f"motif {motif} only appeared {counts[motif]} times"

    non_motif = [shot.media_id for shot in shots if not shot.is_motif]
    assert len(non_motif) == len(set(non_motif)), (
        "a non-motif returned while fresh material was still available"
    )
    motif_share = sum(counts[motif] for motif in motifs) / len(shots)
    assert .08 <= motif_share <= .30, f"motif share {motif_share:.0%} left the 8%–30% band"


def test_motif_schedule_drops_motifs_rather_than_padding_a_short_song() -> None:
    """The double clamp: cap the total at 30%, then drop motifs that will not fit.

    Five themes on a forty-shot song asking for three appearances each would be 37% -
    not a motif, a slideshow. The schedule has to *reduce the number of motifs*, which
    is the whole point of the user's edge-case decision, not pad each one to ``≥3``.
    """
    sections = ["verse", "chorus"] * 20  # forty shots
    schedule = _motif_schedule(sections, [1, 2, 3, 4, 5])

    kept = set(schedule.values())
    assert len(kept) < 5, f"too many motifs survived on a short song: {sorted(kept)}"
    assert len(schedule) <= int(len(sections) * .30), "the 30% cap was exceeded"
    assert all(list(schedule.values()).count(motif) >= 3 for motif in kept)


def test_motif_schedule_is_empty_for_a_film_too_short_to_hold_one() -> None:
    """Two shots cannot hold a recurring theme, and the schedule says so rather than lie."""
    assert _motif_schedule(["verse", "chorus"], [1, 2, 3]) == {}


def test_section_target_length_inverts_intensity_into_the_window() -> None:
    """R-01: intensity 1 asks for the shortest shot, intensity 0 for the longest."""
    minimum, maximum = 1.5, 5.0
    assert _section_target_length(1.0, minimum, maximum, 0.0, 0.0) == minimum
    assert _section_target_length(0.0, minimum, maximum, 0.0, 0.0) == maximum
    # ``speedup`` tightens toward the end of the section but never leaves the window.
    late = _section_target_length(0.0, minimum, maximum, .5, 1.0)
    assert minimum <= late < maximum


def test_shot_lengths_follow_the_director_intensity_arc() -> None:
    """R-01: the arc decides the shot length; the window only bounds it.

    A ``flat_low`` arc (calm) has to cut longer shots than a ``flat_high`` one (intense),
    and neither may leave the effective window - which is the explicit config's, not the
    style's, per the priority decision.
    """
    analysis = _reach_song(160, ["intro", "verse", "chorus", "bridge", "outro"])
    calm = create_plan(
        analysis, _lines(160), _images(90), None, min_shot=1.8, max_shot=5.5,
        treatment=_treatment(5, arc="flat_low"),
    )
    intense = create_plan(
        analysis, _lines(160), _images(90), None, min_shot=1.8, max_shot=5.5,
        treatment=_treatment(5, arc="flat_high"),
    )

    calm_mean = sum(shot.duration for shot in calm) / len(calm)
    intense_mean = sum(shot.duration for shot in intense) / len(intense)
    assert calm_mean > intense_mean + .5, (
        f"the arc barely moved the cut: calm {calm_mean:.2f}s vs intense {intense_mean:.2f}s"
    )
    assert max(shot.duration for shot in intense) <= 5.5 + 1e-6
    assert min(shot.duration for shot in intense) >= 1.8 - 1e-6


def test_scarce_assets_are_spread_evenly_instead_of_repeating_hero_shots() -> None:
    analysis = _grid(100)
    lyrics = _lines(100)
    assets = _images(8)
    similarities = np.full((len(lyrics), len(assets)), .2)
    similarities[:, 0] = 1.0

    shots = create_plan(analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5)

    visible = Counter(_visible_ids(shots))
    assert len(visible) == len(assets)
    assert max(visible.values()) - min(visible.values()) <= 1


def test_asset_repeat_policy_can_be_turned_off() -> None:
    analysis = _grid(20)
    lyrics = _lines(20)
    assets = _images(10)
    similarities = np.full((len(lyrics), len(assets)), .2)
    similarities[:, 0] = 1.0

    strict = create_plan(analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5)
    loose = create_plan(
        analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5,
        avoid_asset_repeats=False,
    )

    assert len({shot.media_id for shot in strict}) == len(strict)
    assert len({shot.media_id for shot in loose}) < len(loose)


def test_video_quota_lifts_real_motion_above_the_still_floor() -> None:
    """R-08: a soft quota so real footage does not drown in the still library.

    The clips here are shorter than the shots, so the scorer's loop penalty pushes them
    below the stills and an unquoted edit leaves them on the shelf. The quota keeps the
    video share at or above the 15% target, and the non-motif zero-reuse rule stays
    intact - the quota only ever prefers videos the tier already offers.
    """
    analysis = _grid(120)
    lyrics = _lines(120)
    # A deep still library is the starvation case: the clips are short enough to score
    # below the stills, and there are far more stills than shots, so an unquoted edit
    # leaves every clip on the shelf.
    assets = _images(120) + _videos(10, start=120, duration=2.0)

    off = create_plan(analysis, lyrics, assets, None, min_shot=1.8, max_shot=5.5)
    on = create_plan(
        analysis, lyrics, assets, None, min_shot=1.8, max_shot=5.5, video_quota=.15,
    )

    on_share = sum(1 for shot in on if shot.kind == "video") / len(on)
    off_share = sum(1 for shot in off if shot.kind == "video") / len(off)
    assert on_share >= .15, f"the video quota left real motion at {on_share:.0%}"
    assert on_share > off_share, "the quota did not change anything"


def test_quality_floor_keeps_the_worst_footage_off_screen() -> None:
    """R-09: an unusable source (a photo of a screen) is gated by quality.

    With plenty of good material above the P5 floor the gate holds and the bad asset is
    never chosen as a primary shot.
    """
    analysis = _grid(60)
    lyrics = _lines(60)
    assets = [
        *_images(20),
        MediaAsset(20, Path("screen-photo.jpg"), "image", float("inf"), 1920, 1080,
                   quality_score=.02),
    ]

    shots = create_plan(
        analysis, lyrics, assets, None, min_shot=1.8, max_shot=5.5, quality_floor=.45,
    )

    assert all(shot.media_id != 20 for shot in shots), "the P5 gate let a bad source through"


def test_quality_floor_gives_way_and_records_it_when_nothing_is_good_enough() -> None:
    """R-09 + R-02: the gate may release, but never silently.

    When the pool holds nothing above the floor the gate has to release - there is a film
    to make - and the surrender is written into the ``ConfigAudit`` rather than swallowed.
    """
    analysis = _grid(40)
    lyrics = _lines(40)
    assets = [
        MediaAsset(0, Path("bad-a.jpg"), "image", float("inf"), 1920, 1080, quality_score=.10),
        MediaAsset(1, Path("bad-b.jpg"), "image", float("inf"), 1920, 1080, quality_score=.12),
    ]
    audit = ConfigAudit()

    shots = create_plan(
        analysis, lyrics, assets, None, min_shot=1.8, max_shot=5.5,
        quality_floor=.90, audit=audit,
    )

    assert shots, "the gate must release rather than render nothing"
    assert any(item["key"] == "quality_floor" for item in audit.as_list()), (
        "the quality gate degraded without declaring it"
    )


def test_a_pool_where_every_asset_is_a_motif_still_plans() -> None:
    """R-03 edge case: with no non-motif supply the ranking stands, no slot is left empty.

    A three-photo project can have all three named as motifs, which empties the non-motif
    pool the main loop draws from. The fallback must be "rank everything", not ``max()``
    over an empty list.
    """
    shots = create_plan(
        _grid(60), _lines(60), _images(3), None, min_shot=1.8, max_shot=5.5,
        motifs=[0, 1, 2],
    )

    assert shots
    assert all(shot.media_id in {0, 1, 2} for shot in shots)


def test_least_used_tier_and_reuse_penalty_follow_visible_history() -> None:
    assets = _images(3)
    minimum, tier = _least_used_assets(assets, {0: 2, 1: 2, 2: 1})
    assert minimum == 1
    assert [asset.id for asset in tier] == [2]

    assert _reuse_penalty(0, None, 5) == 0.0
    assert _reuse_penalty(1, 4, 5) > _reuse_penalty(1, 3, 5) > _reuse_penalty(1, 0, 5)
    assert _reuse_penalty(2, 0, 5) > _reuse_penalty(1, 0, 5)


def test_plan_builds_semantically_ranked_multi_image_layers() -> None:
    analysis = AudioAnalysis(
        duration=4, bpm=120, beats=[0, 2, 4], sections=[0, 4],
        energy_times=[0], energy_values=[.7], average_energy=.7,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse"],
    )
    lyrics = [LyricLine(0, 4, "海边")]
    assets = [
        MediaAsset(0, Path("primary.jpg"), "image", float("inf"), 1200, 1800),
        MediaAsset(1, Path("second.jpg"), "image", float("inf"), 1920, 1080),
        MediaAsset(2, Path("third.jpg"), "image", float("inf"), 1080, 1080),
    ]

    shots = create_plan(
        analysis, lyrics, assets, np.array([[1.0, .8, .6]]),
        min_shot=1.5, max_shot=5, image_composite_ratio=1,
    )

    assert shots[0].image_effect == "split_screen"
    assert [layer.media_id for layer in shots[0].layers] == [1]
    assert shots[0].layers[0].enter_offset == 2


def test_a_layout_asking_for_more_pictures_than_it_can_afford_steps_down() -> None:
    """``max_composite_images`` and the surplus rule both cap what a shot may spend.

    A three-panel layout has to become a two-panel one rather than render a hole where
    its third picture should be, and the plan has to record the layout it actually got -
    the renderer reads the name, not the number of layers.
    """
    # The verse rotation is (split_screen, hero_split, triptych, diagonal_split,
    # hero_grid), so shot 2 lands on the three-panel layouts and shot 4 on the grid.
    narrow = _choose_image_effect(
        0, 2, "verse", .8, .5, available=1, enabled=True, ratio=1, max_images=3,
    )
    assert narrow == ("split_screen", 1)

    roomy = _choose_image_effect(
        0, 2, "verse", .8, .5, available=2, enabled=True, ratio=1, max_images=3,
    )
    assert roomy == ("triptych", 2)

    capped = _choose_image_effect(
        0, 4, "verse", .8, .5, available=6, enabled=True, ratio=1, max_images=2,
    )
    assert capped == ("hero_split", 1)


def test_the_composite_gate_follows_the_configured_share() -> None:
    """``image_composite_ratio`` is the share of image shots that may be composites.

    The gate used to run off the single-image counter, which only advances on the shots
    the gate *refused*. It was therefore self-reinforcing: one composite froze the
    counter, the gate returned the same answer forever, and every image shot in the film
    came back a composite no matter what the config said - 24%, 50% and 100% all gave
    the same film. Counting the same song at two ratios is what catches it.
    """
    analysis = _varied_song()
    shares = {}
    for ratio in (.12, .5):
        shots = create_plan(
            analysis, _lines(120), _images(300), None,
            min_shot=1.5, max_shot=4, image_composite_ratio=ratio,
        )
        stills = [shot for shot in shots if shot.kind == "image"]
        composites = [shot for shot in stills if shot.image_effect in IMAGE_COMPOSITES]
        shares[ratio] = len(composites) / max(1, len(stills))

    # Generous bounds: the gate is deterministic, but the section multipliers and the
    # surplus rule move the real share around the configured one.
    assert .02 <= shares[.12] <= .30, f"{shares[.12]:.0%} composites at ratio .12"
    assert shares[.5] > shares[.12] + .15, (
        f"the ratio knob barely moved the share: {shares[.12]:.0%} vs {shares[.5]:.0%}"
    )


def test_every_multi_image_layout_is_reachable_from_some_song() -> None:
    """A layout no song can select is dead weight, and nothing would say so.

    The rotation only ever offers one name at a time and the cursor advances on the
    shots that are *not* composites, so reachability depends on the song's sections and
    on how many pictures each shot can afford. Sweep both.
    """
    labels = ["intro", "verse", "chorus", "bridge", "solo", "outro"]
    used: set[str] = set()
    for peak in (.55, .95):
        for capacity in (2, 3):
            analysis = AudioAnalysis(
                duration=160, bpm=120,
                beats=[x / 2 for x in range(321)],
                downbeats=[float(x) for x in range(0, 161, 2)],
                sections=[index * 160 / len(labels) for index in range(len(labels) + 1)],
                energy_times=[0, 40, 72, 120, 160],
                energy_values=[.2, .9, .55, peak, .25],
                average_energy=.5, brightness=.5, mood="uplifting",
                mood_scores={"uplifting": 1}, section_labels=labels,
                melody_times=[0, 80, 160], melody_values=[.3, .8, .4],
                melodic_motion=.5, rhythmic_density=70,
            )
            shots = create_plan(
                analysis, _lines(160), _images(90), None,
                min_shot=1.5, max_shot=4, image_composite_ratio=.5,
                max_composite_images=capacity,
            )
            used.update(shot.image_effect for shot in shots)

    missing = sorted(set(IMAGE_COMPOSITES) - used)
    assert not missing, f"no song can select these layouts: {missing}"


def _varied_song(duration: float = 120.0) -> AudioAnalysis:
    """A song that visits every section type, so every effect branch can fire."""
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    step = duration / len(labels)
    return AudioAnalysis(
        duration=duration, bpm=120,
        beats=[x / 2 for x in range(int(duration * 2) + 1)],
        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
        sections=[i * step for i in range(len(labels) + 1)],
        energy_times=[0, duration * .4, duration], energy_values=[.3, .85, .35],
        average_energy=.5, brightness=.5, mood="cinematic",
        mood_scores={"cinematic": 1}, section_labels=labels,
    )


def test_planned_effects_are_all_names_the_renderer_can_build() -> None:
    """A name the renderer does not recognise degrades silently to a default move.

    That failure is invisible in the plan and only shows up as a whole video of
    identical push-ins, so pin the vocabulary from both ends. ``parallax`` is gone
    (R-04): its whole mechanism was the same-image double the single-image rework
    forbids, so it is not a name the planner may still emit.
    """
    from beatforge.renderer import _CAMERA_MOVES

    known = set(_CAMERA_MOVES) | set(IMAGE_COMPOSITES) | {
        "film_bars", "iris", "source_video",
    }
    assert "parallax" not in known
    shots = create_plan(
        _varied_song(), _lines(120), _images(90), None,
        min_shot=1.5, max_shot=4, image_composite_ratio=.35,
    )

    used = {shot.image_effect for shot in shots}
    assert used <= known, used - known
    assert len(used) >= 8, f"the plan barely used the vocabulary: {sorted(used)}"


def test_planned_transition_families_are_all_known_to_the_renderer() -> None:
    from beatforge.renderer import _EFFECT_TRANSITIONS, _TRANSITION_LIBRARY

    known = set(_TRANSITION_LIBRARY) | set(_EFFECT_TRANSITIONS) | {"cut", "none"}
    shots = create_plan(
        _varied_song(), _lines(120), _images(90), None,
        min_shot=1.5, max_shot=4, transition_density=1,
    )

    used = {shot.transition for shot in shots}
    assert used <= known, used - known
    assert len(used) >= 8, f"the plan barely used the vocabulary: {sorted(used)}"


def test_every_camera_move_is_reachable_from_some_song() -> None:
    """A move the planner can never pick is dead weight, and nothing would say so.

    Reachability is song-dependent. A shot lands in a branch based on its section,
    energy, melody and intent, and a rotation like ``index % 5`` only ever offers one
    of its five names at a time - so a single song can leave a whole group untouched
    while every other test still passes.

    Which names a narrow branch can reach depends on the *phase* of the rotation where
    the branch starts, and that phase comes from how many single-image shots preceded
    it. Sweep the song length, the mood *and* the director's ``cut_intensity`` arc (R-01
    rewrites the cut grid, so the same song with a different arc enters every branch at
    a different counter). ``flat_low`` and ``flat_high`` differ in shot count as well as
    phase, and the three lengths alone are not enough once a treatment is driving the
    boundaries.
    """
    from beatforge.renderer import _CAMERA_MOVES

    labels = ["intro", "verse", "chorus", "bridge", "solo", "outro"]
    used: set[str] = set()
    for duration in (160.0, 152.0, 168.0):
        for mood in ("energetic", "uplifting", "melancholic", "dreamy", "romantic", "dark", "cinematic"):
            for peak in (.5, .95):
                for melody in (.3, .9):
                    analysis = _reach_song(duration, labels, mood=mood, peak=peak, melody=melody)
                    shots = create_plan(
                        analysis, _lines(duration), _images(80), None,
                        min_shot=1.5, max_shot=4, image_composite_ratio=.35,
                    )
                    used.update(shot.image_effect for shot in shots)

    # The director arc then moves the same song's boundaries, and with them the phase of
    # every rotation - the R-01 dimension the original sweep did not have.
    for arc in ("flat_low", "flat_high", "rise", "fall", "alternate"):
        for intent in ("continuity", "breathe", "impact"):
            analysis = _reach_song(160.0, labels, mood="cinematic", peak=.8, melody=.6)
            shots = create_plan(
                analysis, _lines(160), _images(80), None,
                min_shot=1.5, max_shot=4, image_composite_ratio=.35,
                treatment=_treatment(len(labels), arc=arc, intent=intent),
            )
            used.update(shot.image_effect for shot in shots)

    missing = sorted(set(_CAMERA_MOVES) - used)
    assert not missing, f"no song can select these moves: {missing}"


def test_every_transition_family_is_reachable_from_some_song() -> None:
    """Same starvation risk as the camera moves, but with three extra causes.

    A transition branch is narrow - ``open`` only fires on a section change in a
    dreamy song, ``radial`` only on a section change into the chorus in a restless
    one, ``close`` only on a *quiet* seam in a restless song - so rotating on the cut
    index lets its few firings land on the same residue and starve the rest of the
    group. Rotating on the visible count fixes that, but reachability still depends on
    the *song*, the *style flavour* and, since R-07, the *director's authorisation*:

    - ``flash``/``glitch``/``film_burn``/``light_leak`` need a ``transition_tone`` of
      bright or dark (or an ``impact`` intent) before they may appear at all;
    - ``squeeze`` lives only in the style's impact pool, so a balanced sweep alone can
      never reach it;
    - ``close`` needs a restless song *and* a flavour that will not quieten it away.

    Sweep all three. A NEUTRAL treatment - the old fixture - would quietly starve the
    loud families and nothing would have caught it.
    """
    from beatforge.renderer import _EFFECT_TRANSITIONS, _TRANSITION_LIBRARY

    layouts = {
        "six": ["intro", "verse", "chorus", "bridge", "solo", "outro"],
        "vcvc": ["verse", "chorus", "verse", "chorus", "outro"],
        # Ten alternating sections so the restless "quiet seam" branch fires often
        # enough to walk its whole rotation and expose ``close``.
        "solo": ["verse", "solo"] * 5,
    }
    shapes = {
        "fall": ([0, 45, 90, 160], [.9, .25, .85, .3]),
        "rise": ([0, 30, 55, 100, 160], [.15, .95, .5, .9, .2]),
        "early": ([0, 25, 50, 80, 160], [.95, .3, .95, .35, .2]),
        # A flat mid level: never loud enough for the flash branch, so the quiet and
        # mid-energy branches get their turn.
        "flat": ([0, 160], [.6, .6]),
        # A flat high level: always above the energy gate, for the inside-section pool.
        "high": ([0, 160], [.82, .82]),
    }
    used: set[str] = set()

    def run(layout: str, shape: str, mood: str, *, style=None, treatment=None) -> None:
        labels = layouts[layout]
        times, values = shapes[shape]
        analysis = AudioAnalysis(
            duration=160, bpm=120,
            beats=[x / 2 for x in range(321)],
            downbeats=[float(x) for x in range(0, 161, 2)],
            sections=[index * 160 / len(labels) for index in range(len(labels) + 1)],
            energy_times=times, energy_values=values,
            average_energy=.5, brightness=.5, mood=mood, mood_scores={mood: 1},
            section_labels=labels,
            melody_times=[0, 80, 160], melody_values=[.2, .9, .3],
            melodic_motion=.6, rhythmic_density=80,
        )
        shots = create_plan(
            analysis, _lines(160), _images(80), None,
            min_shot=1.5, max_shot=4, transition_density=1,
            style=style, treatment=treatment,
        )
        used.update(shot.transition for shot in shots)

    # (a) the three style flavours over a spread of songs. The flavour decides which
    # inside-section pool is used, and whether impact families are quietened away.
    for style_name in ("beat", "montage", "cinematic"):
        style = EDIT_STYLES[style_name]
        for layout in ("six", "vcvc", "solo"):
            for shape in ("fall", "rise", "early", "flat"):
                for mood in ("energetic", "dreamy"):
                    run(layout, shape, mood, style=style)

    # (b) director arcs, with the tone R-07 needs to authorise the impact families.
    # Without a bright/dark tone a loud seam falls back to a quiet family.
    for arc in ("flat_low", "flat_high", "rise", "fall", "alternate"):
        for tone in ("dark", "bright", "soft"):
            for intent in ("continuity", "impact"):
                treatment = _treatment(6, arc=arc, tone=tone, intent=intent)
                run("six", "rise", "energetic", treatment=treatment)

    # (c) style *and* treatment together: an impact-flavoured edit that the director has
    # authorised, on a song that never drops below the inside-section energy gate. This
    # is the only path to ``squeeze``, which lives solely in the impact pool.
    for tone in ("dark", "bright"):
        for layout in ("six", "solo"):
            run(layout, "high", "energetic",
                style=EDIT_STYLES["beat"], treatment=_treatment(6, tone=tone))

    known = set(_TRANSITION_LIBRARY) | set(_EFFECT_TRANSITIONS)
    missing = sorted(known - used)
    assert not missing, f"no song can select these transitions: {missing}"

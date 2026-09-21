from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import (
    IMAGE_COMPOSITES,
    Shot,
    _break_transition_repeats,
    _choose_image_effect,
    _least_used_assets,
    _reuse_penalty,
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


def _visible_ids(shots) -> list[int]:
    return [shot.media_id for shot in shots] + [
        layer.media_id for shot in shots for layer in shot.layers
    ]


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
    """The strongest move the music can justify belongs at the structural seam."""
    into_chorus = _transition_family(
        _shot(0, "verse", .5, section_index=0), _shot(1, "chorus", .9, section_index=1),
        index=0, visible=0, mood="cinematic", density=.35,
    )
    into_outro = _transition_family(
        _shot(0, "chorus", .5, section_index=1), _shot(1, "outro", .3, section_index=2),
        index=0, visible=0, mood="cinematic", density=.35,
    )
    assert into_chorus == "flash"
    assert into_outro == "dip"


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
    identical push-ins, so pin the vocabulary from both ends.
    """
    from beatforge.renderer import _CAMERA_MOVES

    known = set(_CAMERA_MOVES) | set(IMAGE_COMPOSITES) | {
        "film_bars", "iris", "parallax", "source_video",
    }
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
    it. Sweep the song length as well as its mood: an impact branch that is entered at
    counter 11 only ever offers four of its five moves, however many moods it is tried
    against, because every song of the same length enters it at the same place. Two
    lengths is enough to shift the phase.
    """
    from beatforge.renderer import _CAMERA_MOVES

    labels = ["intro", "verse", "chorus", "bridge", "solo", "outro"]
    used: set[str] = set()
    for duration in (160.0, 152.0):
        for mood in ("energetic", "uplifting", "melancholic", "dreamy", "romantic", "dark", "cinematic"):
            for peak in (.5, .95):
                for melody in (.3, .9):
                    analysis = AudioAnalysis(
                        duration=duration, bpm=120,
                        beats=[x / 2 for x in range(int(duration * 2) + 1)],
                        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
                        sections=[index * duration / len(labels) for index in range(len(labels) + 1)],
                        energy_times=[0, duration * .25, duration * .45, duration * .75, duration],
                        energy_values=[.15, .95, .6, peak, .2],
                        average_energy=.5, brightness=.5, mood=mood,
                        mood_scores={mood: 1}, section_labels=labels,
                        melody_times=[0, duration / 2, duration], melody_values=[.2, melody, .3],
                        melodic_motion=.6, rhythmic_density=80,
                    )
                    shots = create_plan(
                        analysis, _lines(duration), _images(80), None,
                        min_shot=1.5, max_shot=4, image_composite_ratio=.35,
                    )
                    used.update(shot.image_effect for shot in shots)

    missing = sorted(set(_CAMERA_MOVES) - used)
    assert not missing, f"no song can select these moves: {missing}"


def test_every_transition_family_is_reachable_from_some_song() -> None:
    """Same starvation risk as the camera moves, but with a different cause.

    A transition branch is narrow - ``open`` only fires on a section change in a
    dreamy song, ``radial`` only on a section change into the chorus in a restless
    one - so rotating on the cut index lets its few firings land on the same residue
    and starve the rest of the group. Rotating on the visible count fixes that, but
    reachability still depends on the *song*: a layout whose section boundaries all
    sit above the ``.78`` energy gate never reaches the chorus branch at all. These
    eight layouts are the smallest set that covers the whole library, found by
    sweeping layouts against energy shapes; a single fixed song covers about half.
    """
    from beatforge.renderer import _EFFECT_TRANSITIONS, _TRANSITION_LIBRARY

    layouts = {
        "six": ["intro", "verse", "chorus", "bridge", "solo", "outro"],
        "vcvc": ["verse", "chorus", "verse", "chorus", "outro"],
    }
    shapes = {
        "fall": ([0, 45, 90, 160], [.9, .25, .85, .3]),
        "rise": ([0, 30, 55, 100, 160], [.15, .95, .5, .9, .2]),
        "early": ([0, 25, 50, 80, 160], [.95, .3, .95, .35, .2]),
    }
    sweep = [
        ("six", "fall", "energetic"), ("six", "fall", "dreamy"),
        ("six", "fall", "cinematic"), ("six", "rise", "energetic"),
        ("six", "rise", "dreamy"), ("six", "early", "energetic"),
        ("six", "early", "dreamy"), ("vcvc", "early", "energetic"),
    ]

    used: set[str] = set()
    for layout, shape, mood in sweep:
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
        )
        used.update(shot.transition for shot in shots)

    known = set(_TRANSITION_LIBRARY) | set(_EFFECT_TRANSITIONS)
    missing = sorted(known - used)
    assert not missing, f"no song can select these transitions: {missing}"

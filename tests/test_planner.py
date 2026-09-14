from collections import Counter
from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import _least_used_assets, _reuse_penalty, _upscale_penalty, create_plan


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
    assert all(a.end == b.start for a, b in zip(shots, shots[1:]))


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


def test_plan_prefers_hard_cuts_and_reserves_transition_for_section_change() -> None:
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

    assert shots[0].transition in {"dip", "flash"}
    assert shots[-1].transition == "none"


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

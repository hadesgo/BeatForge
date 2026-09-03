from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import _upscale_penalty, create_plan


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

"""Probe how often the planner repeats an asset, before/after the reuse rule."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import create_plan


def build(duration: float, asset_count: int, *, video: bool = False, composite: bool = True):
    step = 0.5
    beats = [i * step for i in range(int(duration / step) + 1)]
    sections = [0.0, duration / 3, duration * 2 / 3, duration]
    analysis = AudioAnalysis(
        duration=duration, bpm=120, beats=beats, sections=sections,
        energy_times=[0.0, duration / 2, duration], energy_values=[.35, .75, .4],
        average_energy=.5, brightness=.5, mood="cinematic",
        mood_scores={"cinematic": 1},
        downbeats=[i * 2.0 for i in range(int(duration / 2) + 1)],
        section_labels=["intro", "verse", "chorus"],
    )
    lyrics = [
        LyricLine(i * 4.0, min((i + 1) * 4.0, duration), f"第{i}句歌词")
        for i in range(int(duration / 4.0))
    ]
    assets = [
        MediaAsset(
            i, Path(f"asset-{i}.{'mp4' if video else 'jpg'}"),
            "video" if video else "image",
            30.0 if video else float("inf"),
            1920, 1080,
            quality_score=round(.9 - (i % 7) * .04, 4),
            dominant_color=[(i * 37) % 256, (i * 71) % 256, (i * 113) % 256],
        )
        for i in range(asset_count)
    ]
    # Strong semantic preference for a handful of assets, like a real MV where
    # a few hero shots match the lyrics much better than the rest.
    rng = np.random.default_rng(7)
    base = rng.uniform(.25, .45, size=(len(lyrics), asset_count))
    for column in range(min(3, asset_count)):
        base[:, column] = rng.uniform(.72, .95, size=len(lyrics))
    return analysis, lyrics, assets, base


def report(title: str, duration: float, count: int, *, video: bool = False) -> None:
    analysis, lyrics, assets, similarities = build(duration, count, video=video)
    shots = create_plan(
        analysis, lyrics, assets, similarities, min_shot=1.8, max_shot=5.5,
    )
    primary = collections.Counter(shot.media_id for shot in shots)
    layers = collections.Counter(layer.media_id for shot in shots for layer in shot.layers)
    visible = collections.Counter(primary)
    visible.update(layers)
    repeated = {key: value for key, value in visible.items() if value > 1}
    reused_slots = sum(value - 1 for value in visible.values())
    unused = [asset.id for asset in assets if asset.id not in visible]
    print(f"{title}")
    print(f"  assets={count}  shots={len(shots)}  layer_slots={sum(layers.values())}")
    print(f"  max_uses={max(visible.values()) if visible else 0}  "
          f"assets_used={len(visible)}  assets_never_used={len(unused)}")
    print(f"  repeated_slots={reused_slots}  repeated_assets={len(repeated)}")
    print(f"  usage={dict(sorted(visible.items()))}")


if __name__ == "__main__":
    print("=" * 72)
    report("A. 20 images / 20 shots (enough assets)", 100.0, 20)
    report("B. 40 images / 20 shots (plenty of assets)", 100.0, 40)
    report("C. 30 images / 60 shots (exact)", 300.0, 30)
    report("D. 3 images / 20 shots (scarce)", 100.0, 3)
    report("E. 12 videos / 20 shots (enough videos)", 100.0, 12, video=True)
    print("=" * 72)

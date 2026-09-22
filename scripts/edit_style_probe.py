"""Compare what each editing style actually does to the same song.

A style is only worth having if it changes the edit, and the numbers that show that are
the ones an editor would quote in a notes session: how long a shot gets to live, how
often the cut is visible at all, whether the shot size ever changes, and how loud the
transitions are. Run this after touching a style, or after adding one.

Run: uv run python scripts/edit_style_probe.py
     uv run python scripts/edit_style_probe.py --duration 180 --mood energetic
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.editing import EDIT_STYLES
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.planner import create_plan
from beatforge.renderer import _EFFECT_TRANSITIONS, _TRANSITION_LIBRARY

# Transitions that read as a blend rather than as punctuation.
_QUIET = {"dissolve", "blur", "soft", "smooth", "circle", "dip", "fade"}


def song(mood: str, duration: float) -> AudioAnalysis:
    labels = ["intro", "verse", "chorus", "bridge", "outro"]
    step = duration / len(labels)
    return AudioAnalysis(
        duration=duration, bpm=124,
        beats=[x / 2 for x in range(int(duration * 2) + 1)],
        downbeats=[float(x) for x in range(0, int(duration) + 1, 2)],
        sections=[index * step for index in range(len(labels) + 1)],
        energy_times=[0, duration * .25, duration * .5, duration * .8, duration],
        energy_values=[.18, .55, .95, .45, .2],
        average_energy=.55, brightness=.5, mood=mood, mood_scores={mood: 1},
        section_labels=labels,
        melody_times=[0, duration * .5, duration], melody_values=[.2, .85, .3],
        melodic_motion=.6, rhythmic_density=78,
    )


def media(count: int) -> list[MediaAsset]:
    """Assets whose shot sizes come in blocks, the way a real shoot does.

    Cyclic sizes would flatter every style: consecutive picks would alternate on their
    own and the shot-size rule would never be tested. Blocks make a size clash the
    likely outcome, which is exactly when an edit has to spend a decision on avoiding
    it - and that decision is what ``shot_size_contrast`` is.
    """
    sizes = ["wide", "medium", "closeup", "detail"]
    return [
        MediaAsset(
            index, Path(f"p{index}.jpg"), "image", float("inf"), 1920, 1080,
            shot_size=sizes[(index // 7) % len(sizes)],
        )
        for index in range(count)
    ]


def report(shots, duration: float, assets: list[MediaAsset]) -> dict[str, object]:
    lengths = np.array([shot.duration for shot in shots])
    transitions = Counter(shot.transition for shot in shots[:-1])
    visible = sum(count for name, count in transitions.items() if name not in {"cut", "none"})
    loud = sum(
        count for name, count in transitions.items()
        if name in _EFFECT_TRANSITIONS or name in {"flash", "zoom", "pixel", "squeeze", "slice", "radial", "wind"}
    )
    quiet = sum(count for name, count in transitions.items() if name in _QUIET)
    sizes = {asset.id: asset.shot_size for asset in assets}
    clashes = sum(
        1 for index in range(1, len(shots))
        if sizes.get(shots[index].media_id) == sizes.get(shots[index - 1].media_id)
    )
    return {
        "shots": len(shots),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "shortest": float(lengths.min()),
        "longest": float(lengths.max()),
        # R-01 acceptance readings: how lopsided the length distribution is (p90/p10) and
        # how much of the film is under a second (the "everything is a flash" tell).
        "p10": float(np.percentile(lengths, 10)),
        "p90": float(np.percentile(lengths, 90)),
        "short": float((lengths < .8).mean()),
        "visible": visible / max(1, len(shots) - 1),
        "loud": loud / max(1, visible) if visible else 0.0,
        "quiet": quiet / max(1, visible) if visible else 0.0,
        # R-07: how many distinct transition kinds the film spends, and how many of them
        # are the loud-impact vocabulary.
        "families": len(transitions),
        "impact_kinds": len({name for name in transitions if name in _EFFECT_TRANSITIONS}),
        "moves": len({shot.image_effect for shot in shots}),
        "size_clash": clashes / max(1, len(shots) - 1),
        "known": set(transitions) <= (set(_TRANSITION_LIBRARY) | set(_EFFECT_TRANSITIONS) | {"cut", "none"}),
        "duration": duration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--mood", default="cinematic")
    parser.add_argument("--assets", type=int, default=90)
    args = parser.parse_args()

    analysis = song(args.mood, args.duration)
    assets = media(args.assets)
    lyrics = [
        LyricLine(t, t + 3.5, f"第{int(t)}秒这一句歌词")
        for t in np.arange(0, args.duration - 3.5, 3.5)
    ]

    rows = {}
    for name, style in EDIT_STYLES.items():
        shots = create_plan(
            analysis, lyrics, assets, None,
            min_shot=style.shot_min, max_shot=style.shot_max,
            image_composite_ratio=style.composite_ratio,
            transition_density=style.transition_density,
            style=style,
        )
        rows[name] = report(shots, args.duration, assets)

    header = f"{'风格':<12}{'镜头':>5}{'均长':>7}{'中位':>7}{'最短':>7}{'最长':>7}" \
             f"{'p90/p10':>9}{'短镜':>6}{'可见转场':>9}{'其中冲击':>9}{'含蓄':>7}" \
             f"{'族数':>6}{'冲击族':>7}{'运镜':>5}{'同景别':>8}"
    print(f"同一首歌（{args.mood} · {args.duration:.0f}s · {len(assets)} 个素材）下各风格的编排结果")
    print()
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        label = EDIT_STYLES[name].label
        spread = row["p90"] / max(row["p10"], .01)
        print(
            f"{label:<12}{row['shots']:>5}{row['mean']:>6.2f}s{row['median']:>6.2f}s"
            f"{row['shortest']:>6.2f}s{row['longest']:>6.2f}s"
            f"{spread:>9.2f}{row['short']:>6.0%}"
            f"{row['visible']:>8.0%}{row['loud']:>9.0%}{row['quiet']:>7.0%}"
            f"{row['families']:>6}{row['impact_kinds']:>7}{row['moves']:>5}{row['size_clash']:>8.0%}"
        )

    unknown = [name for name, row in rows.items() if not row["known"]]
    print()
    print("出现未登记的转场族：", unknown or "无")
    busiest = max(rows, key=lambda name: rows[name]["families"])
    print(f"族数最多 {EDIT_STYLES[busiest].label}（{rows[busiest]['families']} 个族） · "
          f"全风格冲击族上限 {max(row['impact_kinds'] for row in rows.values())} 个")
    slowest = max(rows, key=lambda name: rows[name]["median"])
    fastest = min(rows, key=lambda name: rows[name]["median"])
    print(f"最慢 {EDIT_STYLES[slowest].label}（中位 {rows[slowest]['median']:.2f}s） · "
          f"最快 {EDIT_STYLES[fastest].label}（中位 {rows[fastest]['median']:.2f}s） · "
          f"相差 {rows[slowest]['median'] / max(rows[fastest]['median'], .01):.1f} 倍")
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())

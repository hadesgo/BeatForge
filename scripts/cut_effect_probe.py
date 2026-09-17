"""Render every effect transition on a real cut, and report what it does to the frame.

Effect transitions are burned into the two shots' own frames rather than blended by
``xfade``, so nothing in the xfade name list proves they work. This renders a two-shot
sequence per effect and measures the luminance around the cut: a flash has to spike,
a light leak has to warm the picture, and neither may leak into the rest of the shot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import ArtDirection
from beatforge.planner import Shot
from beatforge.renderer import _EFFECT_TRANSITIONS, _render_shot, _transition_spec
from beatforge.runtime import command, duration

ART = ArtDirection(
    concept="", narrative_arc="", visual_style="", color_arc=[], motifs=[],
    mood="cinematic", font="sans", highlight_color="&H0000D7FF",
    base_subtitle_effect="", line_effects=[], grade_filter="",
    camera_intensity=1.0, transition_tone="neutral", grain=0.0, vignette=False,
)
ANALYSIS = AudioAnalysis(
    duration=8, bpm=100, beats=[0, 2, 4, 6, 8], sections=[0, 8], energy_times=[0],
    energy_values=[.5], average_energy=.5, brightness=.5,
    mood="cinematic", mood_scores={"cinematic": 1},
)


def shot(index: int, start: float, family: str, media: Path) -> Shot:
    return Shot(
        index, start, start + 2, 2, index, str(media), "image", 0, "", .7,
        "steady", family, .8, melody=.5, section_index=1,
    )


def frame_stats(clip: Path, frame: int, tmp: Path) -> tuple[float, float, float]:
    """Mean luminance, spatial standard deviation and warmth for one frame.

    Mean alone is the wrong instrument: sensor noise and a channel split barely move
    the average, so a glitch would look like it did nothing. The deviation is what
    catches those, and warmth (red minus blue) catches a light leak.
    """
    target = tmp / f"g{frame:04d}.png"
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(clip),
        "-vf", rf"select=eq(n\,{frame})", "-vsync", "0", "-frames:v", "1", str(target),
    ])
    pixels = np.asarray(Image.open(target).convert("RGB"), dtype=float)
    grey = pixels.mean(axis=2)
    return float(grey.mean()), float(grey.std()), float(pixels[:, :, 0].mean() - pixels[:, :, 2].mean())


def main() -> int:
    tmp = Path(".probe/transitions")
    if tmp.exists():
        for item in tmp.iterdir():
            item.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = RenderConfig(width=320, height=180, fps=15, crf=30, preset="ultrafast",
                       film_grain=0, vignette=False)
    grey = tmp / "grey.jpg"
    Image.new("RGB", (320, 180), (120, 120, 120)).save(grey)

    print(f"{'effect':<12} {'rest':>14} {'at cut':>14} {'peak d':>8} {'peak sd':>8} {'warmth':>7}")
    failures = []
    for name in sorted(_EFFECT_TRANSITIONS):
        outgoing = shot(0, 0, name, grey)
        incoming = shot(1, 2, "cut", grey)
        transition = _transition_spec(outgoing, incoming, ART, cfg)
        clip = tmp / f"{name}.mp4"
        _render_shot(outgoing, clip, cfg, ART, outgoing.duration, 2,
                     outgoing=transition)
        frames = round(duration(clip) * cfg.fps)
        rest = np.mean([frame_stats(clip, f, tmp) for f in range(2, max(3, frames // 3))], axis=0)
        window = [frame_stats(clip, f, tmp) for f in range(max(0, frames - 6), frames)]
        peak_delta = max(abs(stats[0] - rest[0]) for stats in window)
        peak_sd = max(stats[1] for stats in window)
        warmth = max(stats[2] for stats in window)
        ok = peak_delta > 8 or peak_sd > rest[1] + 4 or warmth > rest[2] + 8
        print(f"{name:<12} {rest[0]:>6.1f}/{rest[1]:<7.1f} "
              f"{window[-1][0]:>6.1f}/{window[-1][1]:<7.1f} "
              f"{peak_delta:>8.1f} {peak_sd:>8.1f} {warmth:>7.1f}  {'' if ok else '<-- no effect'}")
        if not ok:
            failures.append(name)
    print()
    print("effects that did nothing:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

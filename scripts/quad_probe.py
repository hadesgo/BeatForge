"""Measure the crop quad of every camera move, to size rolls and keystones.

Rolling or skewing the quad swings its corners away from the crop box, so a move can
leave the picture even though its zoom looks generous. This walks every move at every
frame and reports the worst excursion, plus how far the corners travel.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.planner import Shot
from beatforge.renderer import _CAMERA_MOVES, _image_filter_graph

CFG = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
ANALYSIS = AudioAnalysis(
    duration=8, bpm=100, beats=[0, 2, 4, 6, 8], sections=[0, 8], energy_times=[0],
    energy_values=[.5], average_energy=.5, brightness=.5,
    mood="cinematic", mood_scores={"cinematic": 1},
)


def quad_for(effect: str, seconds: float = 3.0) -> tuple[str, int]:
    shot = Shot(
        0, 0, seconds, seconds, 0, "frame.jpg", "image", 0, "", .6, "dynamic", "none", .8,
        melody=.5, edit_intent="continuity", image_effect=effect,
    )
    filters, _ = _image_filter_graph(shot, CFG, create_art_direction(ANALYSIS, [], CFG), seconds, 1)
    return next(item for item in filters if "perspective" in item), round(seconds * CFG.fps)


def evaluate(filter_string: str, frame: int) -> dict[str, float]:
    points = dict(re.findall(r"([xy][0-3])='([^']*)'", filter_string))
    namespace = {
        "W": CFG.width, "H": CFG.height, "on": frame,
        "clip": lambda v, lo, hi: max(lo, min(hi, v)),
        "pow": pow, "cos": math.cos, "sin": math.sin, "sqrt": math.sqrt, "PI": math.pi,
    }
    return {
        name: float(eval(expr, {"__builtins__": {"max": max, "min": min, "abs": abs}}, namespace))
        for name, expr in points.items()
    }


def main() -> int:
    print(f"{'effect':<18} {'overshoot':>10} {'x-span':>8} {'y-span':>8} {'size drift':>11}")
    worst = []
    for effect in sorted(_CAMERA_MOVES):
        filter_string, frames = quad_for(effect)
        overshoot = 0.0
        spans, sizes = [], []
        for frame in range(1, frames + 1):
            quad = evaluate(filter_string, frame)
            overshoot = max(
                overshoot,
                -min(quad["x0"], quad["x1"], quad["x2"], quad["x3"]),
                max(quad["x0"], quad["x1"], quad["x2"], quad["x3"]) - CFG.width,
                -min(quad["y0"], quad["y1"], quad["y2"], quad["y3"]),
                max(quad["y0"], quad["y1"], quad["y2"], quad["y3"]) - CFG.height,
            )
            spans.append((max(quad["x0"], quad["x1"], quad["x2"], quad["x3"])
                          - min(quad["x0"], quad["x1"], quad["x2"], quad["x3"]),
                          max(quad["y0"], quad["y1"], quad["y2"], quad["y3"])
                          - min(quad["y0"], quad["y1"], quad["y2"], quad["y3"])))
            sizes.append(math.dist((quad["x0"], quad["y0"]), (quad["x1"], quad["y1"])))
        x_span = max(span[0] for span in spans)
        y_span = max(span[1] for span in spans)
        drift = max(sizes) - min(sizes)
        flag = "  <-- OUT" if overshoot > 1e-6 else ""
        print(f"{effect:<18} {overshoot:>10.2f} {x_span:>8.1f} {y_span:>8.1f} {drift:>11.1f}{flag}")
        if overshoot > 1e-6:
            worst.append(effect)
    print()
    print("outside the frame:", worst or "none")
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())

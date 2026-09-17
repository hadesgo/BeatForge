"""Measure the crop quad of every camera move, to size rolls and keystones.

Rolling or skewing the quad swings its corners away from the crop box, so a move can
leave the picture even though its zoom looks generous. This walks every move at every
frame and reports the worst excursion, plus how far the corners travel.

The sweep over ``amount`` matters as much as the sweep over frames. ``amount`` is the
per-shot intensity the renderer derives from the art direction, the shot's melody and
its motion, and it scales the zoom, the roll and the keystone alike - so the zoom lift
that keeps the quad inside the frame has to be sized from the scaled strengths. A move
whose generated quad applies ``amount`` to one field but not another looks fine at the
single ``amount`` a fixed test shot happens to produce and still kills the render on a
real one (a ``tilt3d_front`` shot did exactly that at frame 218 of 253).

Run: uv run python scripts/quad_probe.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import RenderConfig
from beatforge.renderer import _CAMERA_MOVES, _camera_quad, _perspective_filter

CFG = RenderConfig(width=640, height=360, fps=30, film_grain=0, vignette=False)
ASPECT = CFG.height / CFG.width
FRAMES = round(3.0 * CFG.fps)
# ``amount`` is `max(.35, art.camera_intensity) * (1 + melody*.18) * motion`; across the
# profiles, the melodies and the three motion classes it spans roughly this range.
AMOUNTS = (0.245, 0.35, 0.5, 0.7, 0.8714, 1.0, 1.35, 2.0, 2.5, 3.2)


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


def excursion(effect: str, amount: float) -> float:
    """Worst distance any corner sits outside the frame, in pixels.

    Negative means every corner stayed inside: the value is then the *clearance* the
    move has left, so a move that is one pixel from the edge reports ``-1``.
    """
    move = _CAMERA_MOVES[effect]
    filter_string = _perspective_filter(
        _camera_quad(move, amount, 1, FRAMES, CFG.fps, ASPECT)
    )
    worst = -math.inf
    # One frame past the end: the still is fed from `-loop 1`, so filters further down
    # the chain do pull frames beyond the shot.
    for frame in range(1, FRAMES + 2):
        quad = evaluate(filter_string, frame)
        worst = max(
            worst,
            -min(quad["x0"], quad["x1"], quad["x2"], quad["x3"]),
            max(quad["x0"], quad["x1"], quad["x2"], quad["x3"]) - CFG.width,
            -min(quad["y0"], quad["y1"], quad["y2"], quad["y3"]),
            max(quad["y0"], quad["y1"], quad["y2"], quad["y3"]) - CFG.height,
        )
    return worst


def main() -> int:
    header = f"{'effect':<18}{'clearance':>11}{'@amount':>9}{'x-span':>8}{'y-span':>8}{'drift':>8}"
    print(f"每帧逐角点求值 · {FRAMES} 帧 · 参考 amount=1.0 · 扫描 {len(AMOUNTS)} 个 amount")
    print("clearance 为负表示越界像素数，为正表示离画面边缘最近的余量")
    print(header)
    print("-" * len(header))
    outside = []
    for effect in sorted(_CAMERA_MOVES):
        # The worst case across amounts is the one with the least room to spare, and
        # excursion() returns a negative value while the quad stays inside.
        worst, tightest_amount = -math.inf, AMOUNTS[0]
        for amount in AMOUNTS:
            value = excursion(effect, amount)
            if value > worst:
                worst, tightest_amount = value, amount
        filter_string = _perspective_filter(
            _camera_quad(_CAMERA_MOVES[effect], 1.0, 1, FRAMES, CFG.fps, ASPECT)
        )
        spans, sizes = [], []
        for frame in range(1, FRAMES + 1):
            quad = evaluate(filter_string, frame)
            spans.append((max(quad["x0"], quad["x1"], quad["x2"], quad["x3"])
                          - min(quad["x0"], quad["x1"], quad["x2"], quad["x3"]),
                          max(quad["y0"], quad["y1"], quad["y2"], quad["y3"])
                          - min(quad["y0"], quad["y1"], quad["y2"], quad["y3"])))
            sizes.append(math.dist((quad["x0"], quad["y0"]), (quad["x1"], quad["y1"])))
        x_span = max(span[0] for span in spans)
        y_span = max(span[1] for span in spans)
        drift = max(sizes) - min(sizes)
        flag = "  <-- OUT" if worst > 1e-6 else ""
        print(f"{effect:<18}{-worst:>11.2f}{tightest_amount:>9.4g}{x_span:>8.1f}"
              f"{y_span:>8.1f}{drift:>8.1f}{flag}")
        if worst > 1e-6:
            outside.append(effect)
    print()
    print("outside the frame:", outside or "none")
    return 1 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())

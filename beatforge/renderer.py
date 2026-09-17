from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from beatforge.config import RenderConfig
from beatforge.director import ArtDirection
from beatforge.lyrics import LyricLine, write_ass
from beatforge.planner import Shot
from beatforge.runtime import command, duration


def render(
    shots: list[Shot], lyrics: list[LyricLine], music: Path, output: Path,
    cache: Path, config: RenderConfig, art: ArtDirection,
) -> None:
    clips = cache / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    transitions = [
        _transition_spec(shot, shots[index + 1], art, config)
        for index, shot in enumerate(shots[:-1])
    ] if config.professional_transitions else []
    section_count = max((shot.section_index for shot in shots), default=0) + 1
    for index, shot in enumerate(shots):
        print(f"\r渲染镜头 {index + 1}/{len(shots)}", end="", flush=True)
        handle = transitions[index][1] if index < len(transitions) else 0.0
        _render_shot(
            shot, clips / f"{index:05}.mp4", config, art,
            shot.duration + handle, section_count,
        )
    print()
    picture = cache / "picture.mp4"
    if transitions:
        _compose_transitions(shots, clips, picture, transitions, config)
    else:
        concat_file = cache / "clips.txt"
        concat_file.write_text("\n".join(f"file '{(clips / f'{i:05}.mp4').as_posix()}'" for i in range(len(shots))), "utf-8")
        command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(picture)])
    subtitle = cache / "lyrics.ass"
    write_ass(
        lyrics, subtitle, width=config.width, height=config.height,
        font=art.font, size=config.subtitle_size,
        margin=config.subtitle_margin, effect=art.base_subtitle_effect,
        highlight_color=art.highlight_color, line_effects=art.line_effects,
    )
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(music)]
    if lyrics:
        escaped = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        subtitle_filter = f"ass='{escaped}'"
        if config.subtitle_fonts_dir and config.subtitle_fonts_dir.exists():
            fonts = config.subtitle_fonts_dir.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
            subtitle_filter += f":fontsdir='{fonts}'"
        args += ["-vf", subtitle_filter]
    args += ["-map", "0:v:0", "-map", "1:a:0", *_video_encode_args(config, intermediate=False),
             "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(output)]
    command(args)


def _render_shot(
    shot: Shot, output: Path, cfg: RenderConfig, art: ArtDirection,
    render_duration: float, section_count: int,
) -> None:
    frames = max(1, round(render_duration * cfg.fps))
    if shot.kind == "image":
        _render_image_shot(shot, output, cfg, art, render_duration, section_count)
        return

    # Preserve source cinematography: synthetic oscillation on moving footage looks seasick.
    overscan = 1.025
    scaled_width, scaled_height = round(cfg.width * overscan / 2) * 2, round(cfg.height * overscan / 2) * 2
    focus_x, focus_y = _safe_focus(shot.focus_point)
    visual = (f"scale={scaled_width}:{scaled_height}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={cfg.width}:{cfg.height}:"
              f"x='clip(iw*{focus_x}-ow/2,0,iw-ow)':y='clip(ih*{focus_y}-oh/2,0,ih-oh)',setsar=1")
    grade = art.grade_filter
    effects = [
        _shot_match_filter(shot, cfg.shot_match_strength),
        grade,
        _section_color_filter(shot, art, section_count, cfg.look_strength),
    ]
    if cfg.visual_effects:
        upscale = (
            max(cfg.width / shot.source_width, cfg.height / shot.source_height)
            if shot.source_width > 0 and shot.source_height > 0 else 1.0
        )
        if shot.motion == "dynamic" and upscale <= 1.35:
            effects.append("unsharp=5:5:0.55:5:5:0")
        elif shot.motion == "gentle":
            effects.append("gblur=sigma=0.18")
        if art.vignette:
            effects.append("vignette=PI/5")
        if art.grain > 0:
            effects.append(f"noise=alls={art.grain}:allf=t+u")
    args = ["ffmpeg", "-y", "-v", "error"]
    args += ["-stream_loop", "-1", "-ss", str(shot.source_start)]
    args += ["-i", shot.file, "-t", str(render_duration), "-an", "-vf", ",".join(
        [visual, *(item for item in effects if item)]
    ), "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output)]
    command(args)


def _render_image_shot(
    shot: Shot, output: Path, cfg: RenderConfig, art: ArtDirection,
    render_duration: float, section_count: int,
) -> None:
    files = [shot.file, *(layer.file for layer in shot.layers if layer.kind == "image")]
    args = ["ffmpeg", "-y", "-v", "error"]
    for file in files:
        args += ["-loop", "1", "-framerate", str(cfg.fps), "-i", file]
    filters, current = _image_filter_graph(shot, cfg, art, render_duration, len(files))
    finishing = [
        _shot_match_filter(shot, cfg.shot_match_strength), art.grade_filter,
        _section_color_filter(shot, art, section_count, cfg.look_strength),
    ]
    if cfg.visual_effects:
        if shot.motion == "dynamic":
            finishing.append("unsharp=5:5:0.42:5:5:0")
        if art.vignette:
            finishing.append("vignette=PI/5")
        if art.grain > 0:
            finishing.append(f"noise=alls={art.grain}:allf=t+u")
    finish = ",".join(item for item in finishing if item)
    # Normalize to the exact target canvas: composite effects (e.g. split_screen's
    # hstack) can emit odd widths, and xfade/concat reject mismatched input sizes.
    normalize = (
        f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    filters.append(
        f"{current}{finish + ',' if finish else ''}{normalize},fps={cfg.fps},"
        f"trim=duration={render_duration:.4f},setpts=PTS-STARTPTS,format=yuv420p[vout]"
    )
    args += [
        "-t", str(render_duration), "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-an", "-r", str(cfg.fps),
        *_video_encode_args(cfg, intermediate=True), str(output),
    ]
    command(args)


@dataclass(frozen=True, slots=True)
class _CameraMove:
    """One still-image camera move, expressed so it reads the same at any size.

    ``zoom_from``/``zoom_to`` are *added* magnification: ``0.06`` means the crop is
    ``W / 1.06`` wide, so the frame never crops outside itself. ``x_from``..``y_to``
    are offsets in units of the panning range ``W - W/zoom`` — ``0`` is centred and
    ``.09`` spends nine percent of the travel available at that moment. Keeping every
    offset inside about ±.15 stops the crop from reaching its ``clip()`` bound, which
    would park the move against the edge and stall it mid-shot.
    """

    zoom_from: float
    zoom_to: float
    x_from: float = 0.0
    x_to: float = 0.0
    y_from: float = 0.0
    y_to: float = 0.0
    ease: str = "linear"
    breathe: bool = False


# Movement vocabulary for stills. These are deliberately distinct in *character*,
# not just in magnitude: a drift holds its framing and slides, a dolly commits to
# the push, a punch arrives fast and settles, a breathe swells and returns.
_CAMERA_MOVES: dict[str, _CameraMove] = {
    # The workhorses. Small, unhurried pushes for verses, intros and outros.
    "cinematic_depth": _CameraMove(0.0, 0.055, y_to=-0.035),
    "focus_pull": _CameraMove(0.0, 0.045, y_to=-0.030),
    # Reveals: the framing travels, so the shot uncovers something.
    "pan_reveal": _CameraMove(0.0, 0.075, x_from=-0.090, x_to=0.090, y_to=-0.035),
    "tilt_up": _CameraMove(0.0, 0.050, y_from=0.075, y_to=-0.075),
    "tilt_down": _CameraMove(0.0, 0.050, y_from=-0.075, y_to=0.075),
    "arc": _CameraMove(0.0, 0.060, x_from=-0.070, x_to=0.070, y_from=0.060, y_to=-0.060),
    # A drift barely zooms at all, which is what makes it read as a slide rather
    # than a push. ``direction`` flips it per shot, so there is no left/right pair.
    "drift": _CameraMove(0.035, 0.035, x_from=0.105, x_to=-0.105),
    # Committed moves, for cuts that ask for one.
    "dolly_in": _CameraMove(0.0, 0.095),
    "dolly_out": _CameraMove(0.095, 0.0),
    "punch_in": _CameraMove(0.0, 0.130, ease="out"),
    "pull_back": _CameraMove(0.120, 0.0, ease="out"),
    # Long, low-energy shots: the frame swells and settles instead of leaving.
    "breathe": _CameraMove(0.0, 0.050, breathe=True),
}

# Parallax drives two planes of the same image at different rates; the gap between
# the rates is the depth cue. The backdrop is magnified less and travels less.
_PARALLAX_BACK = _CameraMove(0.0, 0.040, x_from=-0.050, x_to=0.050, y_to=-0.020)
_PARALLAX_FRONT = _CameraMove(0.0, 0.055, x_from=-0.115, x_to=0.115, y_to=-0.045)

_EASES = {
    "linear": lambda p: p,
    # Decelerating: covers most of the distance early, then settles. This is what
    # makes an impact move feel like it lands rather than coasts.
    "out": lambda p: f"(1-pow(1-{p},3))",
    "in": lambda p: f"pow({p},3)",
    "inout": lambda p: f"(0.5-0.5*cos(PI*{p}))",
}

# Every move is lifted this far above 1.0 magnification. A crop that exactly fills
# the frame is a degenerate perspective mapping: its travel range ``W - W/zoom`` is
# zero, so the crop origin expression is ``clip(0, 0, 0)`` at best and any
# floating-point noise in the zoom curve makes the upper bound negative — which
# ``clip`` does *not* rescue (it computes ``min(max(x, min), max)``, so a negative
# ``max`` comes straight back out) and the filter then rejects the frame. Paying a
# 0.4% crop — under a pixel at 1080p — removes the degenerate case for every move,
# including zoom curves that end exactly where they started.
_MIN_ZOOM = 0.004


def _camera_expressions(
    move: _CameraMove, amount: float, direction: int, frames: int,
) -> tuple[str, str, str]:
    """Return ffmpeg expressions for ``(zoom, x_fraction, y_fraction)``.

    ``perspective`` exposes ``on`` — the output frame index — and nothing else: no
    ``t``, and no ``n``. ``on`` is **1-based**; ``vf_perspective.c`` fills it from
    ``outl->frame_count_in + 1``, so the last frame of an N-frame shot reports
    ``on == N``. Progress is therefore ``(on-1)/(frames-1)``. Writing
    ``on/(frames-1)`` instead overshoots the tail by a whole frame, and for a move
    that ends at 1.0 magnification that is enough to push the zoom below 1.0 — at
    which point ``W - W/zoom`` goes negative and the filter fails the frame.

    The clamp matters as much as the offset. The still is fed from ``-loop 1``, so
    it is infinitely long, and filters further down the chain (``fps``, ``trim``)
    pull frames past the end of the shot to settle their own timestamps. Without the
    clamp those extra frames keep walking the curve — past its end, so a move that
    was supposed to arrive and hold instead reverses, and once the zoom drops under
    1.0 the crop collapses and the frame is rejected. Holding the final value costs
    nothing: those frames are trimmed away.
    """
    progress = f"clip((on-1)/{max(1, frames - 1)},0,1)"
    ease = _EASES[move.ease](progress)
    low = 1 + _MIN_ZOOM + move.zoom_from * amount
    high = 1 + _MIN_ZOOM + move.zoom_to * amount
    if move.breathe:
        # 0.5-0.5*cos(2*pi*p) runs 0 -> 1 -> 0, so the frame swells and returns.
        zoom = f"({low:.6f}+{high - low:.6f}*(0.5-0.5*cos(2*PI*{progress})))"
    elif abs(high - low) < 1e-9:
        zoom = f"{low:.6f}"
    else:
        zoom = f"({low:.6f}+{high - low:.6f}*{ease})"
    x = f"{direction * move.x_from:.5f}+{direction * (move.x_to - move.x_from):.5f}*{ease}"
    y = f"{move.y_from:.5f}+{move.y_to - move.y_from:.5f}*{ease}"
    return zoom, x, y


def _perspective_filter(zoom: str, x_fraction: str, y_fraction: str) -> str:
    """Crop ``W/zoom`` wide and slide it inside the frame, per frame.

    ``perspective`` is used instead of ``zoompan`` because zoompan truncates the
    crop origin to whole pixels of its input frame. The camera moves here are only
    a fraction of a pixel per frame, so zoompan holds the image still for several
    frames and then snaps it by a whole pixel - a visible stutter on any straight
    edge. Feeding it a supersampled frame only shrinks the jump; perspective
    evaluates the crop rectangle per frame with true sub-pixel interpolation, so
    the move stays smooth and nothing is resampled twice.

    The crop is ``W/zoom`` wide and may travel ``W - W/zoom`` before leaving the
    frame; the two are easy to confuse and swapping them turns the move into a
    hard punch-in.
    """
    crop_width = f"(W/({zoom}))"
    crop_height = f"(H/({zoom}))"
    travel = f"(W-W/({zoom}))"
    rise = f"(H-H/({zoom}))"
    origin_x = f"clip({travel}/2+{travel}*({x_fraction}),0,{travel})"
    origin_y = f"clip({rise}/2+{rise}*({y_fraction}),0,{rise})"
    return (
        f"perspective="
        f"x0='{origin_x}':y0='{origin_y}':"
        f"x1='{origin_x}+{crop_width}':y1='{origin_y}':"
        f"x2='{origin_x}':y2='{origin_y}+{crop_height}':"
        f"x3='{origin_x}+{crop_width}':y3='{origin_y}+{crop_height}':"
        f"interpolation=cubic:sense=source:eval=frame"
    )


def _plane_duration(duration: float, cfg: RenderConfig) -> float:
    """How long to generate a synthetic plane that is composited into a shot.

    Planes built from ``color`` meet the picture through ``overlay=shortest=1``, so
    whichever side runs out first ends the shot — and a plane generated for exactly
    the shot's length does run out first, because it also passes through filters that
    hold it a frame behind the picture. The shot then renders one frame short, which
    is a hole at the end of every shot that uses the effect. A frame of surplus costs
    nothing: the finishing ``trim`` cuts the shot to length afterwards.
    """
    return duration + 1 / cfg.fps


def _image_filter_graph(
    shot: Shot, cfg: RenderConfig, art: ArtDirection,
    duration: float, input_count: int,
) -> tuple[list[str], str]:
    effect = shot.image_effect if cfg.visual_effects else "cinematic_depth"
    if input_count < 2 and effect in {"split_screen", "photo_stack", "double_exposure", "beat_montage"}:
        effect = "cinematic_depth"
    filters: list[str] = []

    if effect == "split_screen":
        gap = max(2, round(cfg.width * .004))
        left_width = (cfg.width - gap) // 2
        right_width = cfg.width - gap - left_width
        _adapt_image(filters, 0, "left", left_width, cfg.height, cfg)
        _adapt_image(filters, 1, "right", right_width, cfg.height, cfg)
        filters.append(f"[left][right]hstack=inputs=2[panels]")
        filters.append(
            f"color=c=white@0.22:s={gap}x{cfg.height}:r={cfg.fps}:d={_plane_duration(duration, cfg):.4f}[divider];"
            f"[panels][divider]overlay=x={left_width}:y=0:shortest=1[composite]"
        )
        return filters, "[composite]"

    if effect == "photo_stack":
        _adapt_image(filters, 0, "base", cfg.width, cfg.height, cfg)
        current = "[base]"
        card_width, card_height = round(cfg.width * .56), round(cfg.height * .64)
        for layer_index in range(1, min(input_count, 3)):
            angle = -.026 if layer_index % 2 else .022
            filters.append(
                f"[{layer_index}:v]scale={card_width}:{card_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad=iw+14:ih+14:7:7:color=white,format=rgba,"
                f"rotate={angle}:ow=rotw(iw):oh=roth(ih):c=none[card{layer_index}]"
            )
            planned_enter = shot.layers[layer_index - 1].enter_offset
            enter = planned_enter if planned_enter > 0 else min(duration * .58, duration * (.18 + .22 * (layer_index - 1)))
            x = round(cfg.width * (.08 if layer_index % 2 else .40))
            y = round(cfg.height * (.09 if layer_index % 2 else .15))
            out = f"[stack{layer_index}]"
            filters.append(
                f"{current}[card{layer_index}]overlay="
                f"x='{x}+(1-min(max((t-{enter:.3f})/.35,0),1))*{(-90 if layer_index % 2 else 90)}':"
                f"y={y}:enable='gte(t,{enter:.3f})':shortest=1{out}"
            )
            current = out
        return filters, current

    if effect == "double_exposure":
        _adapt_image(filters, 0, "exposure0", cfg.width, cfg.height, cfg)
        _adapt_image(filters, 1, "exposure1", cfg.width, cfg.height, cfg)
        filters.append("[exposure0][exposure1]blend=all_mode=screen:all_opacity=0.34[composite]")
        return filters, "[composite]"

    if effect == "beat_montage":
        count = min(input_count, 4)
        for index in range(count):
            _adapt_image(filters, index, f"montage{index}", cfg.width, cfg.height, cfg)
        current = "[montage0]"
        planned_starts = [0.0, *(layer.enter_offset for layer in shot.layers[:count - 1])]
        if any(value <= 0 for value in planned_starts[1:]):
            planned_starts = [duration * index / count for index in range(count)]
        for index in range(1, count):
            start = planned_starts[index]
            end = duration if index == count - 1 else planned_starts[index + 1]
            out = f"[sequence{index}]"
            filters.append(
                f"{current}[montage{index}]overlay=0:0:"
                f"enable='between(t,{start:.4f},{end:.4f})':shortest=1{out}"
            )
            current = out
        return filters, current

    # --- single-image camera moves, and the treatments layered on top of them ---
    frames = max(1, round(duration * cfg.fps))
    direction = -1 if (shot.media_id + max(0, shot.section_index)) % 2 else 1
    amount = (max(.35, art.camera_intensity) * (1 + shot.melody * .18)
              * {"dynamic": 1.35, "gentle": .7}.get(shot.motion, 1.0))

    # Parallax needs the backdrop and the foreground as separate streams, so it has
    # to claim the input before anything composites the two together. With no
    # blurred backdrop there is no second plane to offset, so it falls through to a
    # plain camera move instead.
    if effect == "parallax" and cfg.blurred_image_background:
        _adapt_image_layers(filters, 0, "px", cfg.width, cfg.height, cfg)
        for plane, plane_move in (("bg", _PARALLAX_BACK), ("fg", _PARALLAX_FRONT)):
            plane_zoom, plane_x, plane_y = _camera_expressions(plane_move, amount, direction, frames)
            filters.append(f"[px{plane}]{_perspective_filter(plane_zoom, plane_x, plane_y)}[px{plane}m]")
        filters.append("[pxbgm][pxfgm]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[composite]")
        return filters, "[composite]"

    _adapt_image(filters, 0, "adapted", cfg.width, cfg.height, cfg)
    move = _CAMERA_MOVES.get(effect, _CAMERA_MOVES["cinematic_depth"])
    # A breathe shot, or the outro, releases rather than drives: reverse a push so
    # the frame opens up instead of closing in. Moves that already end where they
    # started - a drift, a breathe - are left alone.
    if (shot.edit_intent == "breathe" or shot.section == "outro") and move.zoom_to > move.zoom_from:
        move = replace(move, zoom_from=move.zoom_to, zoom_to=move.zoom_from, ease="out")
    zoom, x_fraction, y_fraction = _camera_expressions(move, amount, direction, frames)
    filters.append(f"[adapted]{_perspective_filter(zoom, x_fraction, y_fraction)}[moved]")

    if effect == "film_bars":
        # Overlay the bars rather than cropping to them: the audience reads 2.35:1
        # while the image underneath keeps its whole frame.
        bar = max(2, round(cfg.height * .055 / 2) * 2)
        filters.append(
            f"[moved]drawbox=x=0:y=0:w=iw:h={bar}:t=fill:color=black,"
            f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:t=fill:color=black[composite]"
        )
        return filters, "[composite]"

    if effect == "iris":
        # A circular mask that opens from a point until it clears the frame corners,
        # so the picture is revealed rather than left with a permanent vignette.
        #
        # ``geq`` is evaluated per pixel per frame and costs roughly 0.2s a frame at
        # 1080p — more than every other effect here combined. The mask is a smooth
        # disc, so it is built at an eighth of the canvas and scaled up: the same
        # shape for a fraction of the work, and the bilinear upsample gives the edge
        # a softness that a hard one-pixel circle would not have.
        #
        # ``geq`` also speaks a different expression dialect from ``perspective``: it
        # has no ``on`` at all, and numbers frames with the 0-based ``N``. The clamp
        # is for the same reason as the camera moves — frames pulled past the end of
        # the shot must hold rather than run backwards.
        mask_width = max(64, round(cfg.width / 8 / 2) * 2)
        mask_height = max(36, round(cfg.height / 8 / 2) * 2)
        reach = math.hypot(mask_width, mask_height) / 2
        progress = f"clip(N/{max(1, frames - 1)},0,1)"
        mask_duration = _plane_duration(duration, cfg)
        filters.append(
            f"color=c=black:s={mask_width}x{mask_height}:r={cfg.fps}:d={mask_duration:.4f},"
            f"format=gray,geq=lum='if(lte((X-W/2)^2+(Y-H/2)^2,"
            f"pow({reach:.2f}*(0.18+0.82*{progress}),2)),255,0)',"
            f"scale={cfg.width}:{cfg.height}:flags=bilinear[irismask];"
            f"[moved]format=rgba[irisfg];"
            f"[irisfg][irismask]alphamerge[iriscut];"
            f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}:d={mask_duration:.4f}[irisbg];"
            f"[irisbg][iriscut]overlay=0:0:shortest=1[composite]"
        )
        return filters, "[composite]"

    return filters, "[moved]"


def _adapt_image_layers(
    filters: list[str], input_index: int, label: str,
    width: int, height: int, cfg: RenderConfig,
) -> None:
    """Emit the blurred backdrop and the sharp foreground as two separate labels.

    ``_adapt_image`` overlays them immediately, which is what most effects want.
    Parallax has to move each plane at its own rate *before* they meet, so it asks
    for the labels and does the overlay itself.
    """
    foreground_width = max(2, round(width * cfg.image_foreground_scale / 2) * 2)
    foreground_height = max(2, round(height * cfg.image_foreground_scale / 2) * 2)
    blur = cfg.image_background_blur if cfg.blurred_image_background else 0
    background_effect = f"gblur=sigma={blur:.2f},eq=brightness=-.055:saturation=.88" if blur > 0 else "null"
    filters.append(
        f"[{input_index}:v]split=2[{label}bgsrc][{label}fgsrc];"
        f"[{label}bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},{background_effect},setsar=1[{label}bg];"
        f"[{label}fgsrc]scale={foreground_width}:{foreground_height}:"
        f"force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[{label}fg]"
    )


def _adapt_image(
    filters: list[str], input_index: int, label: str,
    width: int, height: int, cfg: RenderConfig,
) -> None:
    """Fit without distortion and fill any letterbox area with a blurred copy."""
    _adapt_image_layers(filters, input_index, label, width, height, cfg)
    filters.append(
        f"[{label}bg][{label}fg]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[{label}]"
    )


def _safe_focus(value: list[float]) -> tuple[float, float]:
    if len(value) != 2:
        return .5, .5
    try:
        return (
            round(float(np.clip(float(value[0]), .08, .92)), 4),
            round(float(np.clip(float(value[1]), .08, .92)), 4),
        )
    except (TypeError, ValueError):
        return .5, .5


def _shot_match_filter(shot: Shot, strength: float) -> str:
    """Apply restrained normalization so mixed cameras do not visibly jump."""
    if strength <= 0 or len(shot.source_color) != 3 or shot.source_color == [128, 128, 128]:
        return ""
    red, green, blue = (float(value) / 255 for value in shot.source_color)
    luminance = red * .2126 + green * .7152 + blue * .0722
    chroma = max(red, green, blue) - min(red, green, blue)
    brightness = float(np.clip((.48 - luminance) * .12 * strength, -.035, .035))
    saturation = float(np.clip(1 + (.32 - chroma) * .16 * strength, .94, 1.06))
    return f"eq=brightness={brightness:.4f}:saturation={saturation:.4f}"


def _section_color_filter(shot: Shot, art: ArtDirection, section_count: int, strength: float) -> str:
    """Turn the director's color arc into a subtle, section-consistent tint."""
    if strength <= 0 or not art.color_arc:
        return ""
    progress = max(0, shot.section_index) / max(1, section_count - 1)
    palette_index = min(len(art.color_arc) - 1, round(progress * (len(art.color_arc) - 1)))
    look = art.color_arc[palette_index].casefold()
    warm = ("warm", "amber", "gold", "orange", "sunset", "romantic", "暖", "琥珀", "金", "夕阳")
    cool = ("cool", "cold", "blue", "teal", "cyan", "melancholic", "cinematic", "冷", "蓝", "青")
    purple = ("purple", "violet", "magenta", "dreamy", "紫", "梦幻")
    green = ("green", "emerald", "forest", "绿", "森林")
    mono = ("mono", "desatur", "black and white", "dark", "黑白", "低饱和")
    amount = .032 * strength
    if any(key in look for key in warm):
        return f"colorbalance=rs={amount:.4f}:gs={amount * .25:.4f}:bs={-amount * .75:.4f}"
    if any(key in look for key in cool):
        return f"colorbalance=rs={-amount * .55:.4f}:gs={amount * .18:.4f}:bs={amount:.4f}"
    if any(key in look for key in purple):
        return f"colorbalance=rs={amount * .65:.4f}:gs={-amount * .35:.4f}:bs={amount:.4f}"
    if any(key in look for key in green):
        return f"colorbalance=rs={-amount * .35:.4f}:gs={amount * .65:.4f}:bs={-amount * .15:.4f}"
    if any(key in look for key in mono):
        return f"eq=saturation={1 - .18 * strength:.4f}"
    return ""


def _video_encode_args(cfg: RenderConfig, *, intermediate: bool) -> list[str]:
    args = [
        "-c:v", "libx264", "-preset", cfg.preset,
        "-crf", str(cfg.intermediate_crf if intermediate else cfg.crf),
        "-pix_fmt", "yuv420p",
    ]
    if cfg.encoder_tune != "none":
        args.extend(["-tune", cfg.encoder_tune])
    return args


# Every name here is a real ``xfade`` transition; ``ffmpeg -h filter=xfade`` lists the
# full set. BeatForge used to reach for six of them, which is why an edit could run
# for three minutes and never once change its vocabulary. These are the ones that
# read as editorial decisions rather than as presets.
_TRANSITION_LIBRARY: dict[str, tuple[str, ...]] = {
    "dissolve": ("dissolve", "fade", "fadegrays"),
    # ``dip`` and ``flash`` are defined by their colour rather than by a shape, so
    # ``_transition_spec`` picks those two from the tone instead of rotating.
    "dip": ("fadeblack", "fadewhite"),
    "flash": ("fadewhite", "fadefast"),
    "wipe": ("wipeleft", "wiperight", "wipeup", "wipedown"),
    "slide": ("slideleft", "slideright", "slideup", "slidedown"),
    "circle": ("circleopen", "circleclose"),
    "radial": ("radial",),
    "zoom": ("zoomin",),
    "blur": ("hblur",),
    "slice": ("hlslice", "hrslice", "vuslice", "vdslice"),
    "diag": ("diagtl", "diagtr", "diagbl", "diagbr"),
    "pixel": ("pixelize",),
    "squeeze": ("squeezeh", "squeezev"),
    "reveal": ("revealleft", "revealright", "revealup", "revealdown",
               "coverleft", "coverright", "coverup", "coverdown"),
}

# Families with a left/right reading. Index 0 travels left, index 1 travels right.
# The shot's own drift direction chooses, so a wipe continues the motion the camera
# was already making instead of fighting it.
_TRANSITION_DIRECTION: dict[str, tuple[str, str]] = {
    "wipe": ("wipeleft", "wiperight"),
    "slide": ("slideleft", "slideright"),
    "diag": ("diagtl", "diagbr"),
    "reveal": ("revealleft", "revealright"),
    "slice": ("hlslice", "hrslice"),
}

# How long each family wants to take. An impact transition has to be short or it
# stops reading as impact; the soft ones need room to breathe.
_TRANSITION_PACE: dict[str, float] = {
    "flash": .16, "zoom": .20, "pixel": .22, "squeeze": .24, "slice": .26,
    "radial": .26, "wipe": .28, "slide": .28, "diag": .30, "reveal": .32,
    "dip": .36, "circle": .42, "blur": .44, "dissolve": .44,
}


def _transition_spec(shot: Shot, following: Shot, art: ArtDirection, cfg: RenderConfig) -> tuple[str, float]:
    """Resolve the planner's transition family into a concrete ``xfade`` name.

    The planner picks *what* the edit needs - a wipe, a dip, a burst - and this
    decides how to realise it, using the tone of the incoming shot and the outgoing
    shot's drift direction.
    """
    family = shot.transition
    if family in {"cut", "none", ""}:
        return "cut", 0.0
    tone = following.transition_tone if following.transition_tone != "neutral" else art.transition_tone
    duration = _TRANSITION_PACE.get(family, .28)

    if family == "dip":
        # A dip is defined by its colour: to black for weight, to white for release.
        name = "fadewhite" if tone == "bright" else "fadeblack"
    elif family == "flash":
        name = "fadewhite" if tone in {"bright", "soft"} else "fadefast"
    elif family == "dissolve":
        name = "fadegrays" if tone == "dark" else "dissolve"
    elif family in _TRANSITION_DIRECTION:
        direction = -1 if (shot.media_id + max(0, shot.section_index)) % 2 else 1
        name = _TRANSITION_DIRECTION[family][1 if direction > 0 else 0]
    else:
        # Unknown family - a hand-edited plan.json, or a director field this build
        # does not know yet. Fall back to a dissolve rather than a hard cut, so the
        # shot still reads as the edit intended it to.
        options = _TRANSITION_LIBRARY.get(family) or _TRANSITION_LIBRARY["dissolve"]
        name = options[shot.index % len(options)]

    duration = max(cfg.transition_min_seconds, min(cfg.transition_max_seconds, duration, shot.duration / 3, following.duration / 3))
    return name, round(duration, 3)


def _compose_transitions(
    shots: list[Shot], clips: Path, output: Path,
    transitions: list[tuple[str, float]], cfg: RenderConfig,
) -> None:
    args = ["ffmpeg", "-y", "-v", "error"]
    for index in range(len(shots)):
        args += ["-i", str(clips / f"{index:05}.mp4")]
    filters = [f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[v{index}]" for index in range(len(shots))]
    # Anchor every fade to the actual on-disk clip durations: frame rounding makes
    # rendered clips slightly shorter than shot.duration + handle, so a timeline
    # built from ideal durations drifts past the real streams and xfade rejects
    # the offset as invalid.
    actual = [duration(clips / f"{index:05}.mp4") for index in range(len(shots))]
    current = "[v0]"
    timeline = actual[0]
    for index, (name, transition_duration) in enumerate(transitions):
        label = f"[x{index + 1}]"
        if name == "cut" or transition_duration <= 0:
            filters.append(f"{current}[v{index + 1}]concat=n=2:v=1:a=0{label}")
            timeline += actual[index + 1]
        else:
            offset = max(0.0, timeline - transition_duration - .001)
            filters.append(
                f"{current}[v{index + 1}]xfade=transition={name}:duration={transition_duration}:offset={round(offset, 3)}{label}"
            )
            timeline = offset + actual[index + 1]
        current = label
    end_fade = min(.5, shots[-1].duration / 3)
    filters.append(f"{current}fade=t=in:st=0:d=0.25,fade=t=out:st={max(0, timeline - end_fade):.3f}:d={end_fade:.3f}[vout]")
    args += [
        "-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
        "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output),
    ]
    command(args)



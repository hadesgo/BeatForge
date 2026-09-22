from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from beatforge.config import RenderConfig
from beatforge.director import ArtDirection
from beatforge.fonts import stage_fonts
from beatforge.legibility import adaptive_outline, estimate_region_luma, needs_local_dim
from beatforge.lyrics import LyricLine, Placement, plan_placements, write_ass
from beatforge.planner import IMAGE_COMPOSITES, Shot
from beatforge.runtime import command, duration

# How the knockout fill treats the frame it shows through the letters. The letters are
# lifted and the picture around them is pulled down, so the contrast between the two is
# guaranteed rather than borrowed from whatever happened to be behind the text - which
# matters because the free layout deliberately puts the type where the frame is empty.
_KNOCKOUT_LIFT = 0.16
_KNOCKOUT_DIM = -0.10
_KNOCKOUT_SATURATION = 1.25

#: How far the glyph-shaped patch of picture is pulled down under a bright-backdrop line
#: (R-05's local dim). Smaller than the knockout dim because it is confined to the
#: letters' own halo rather than the whole frame - enough to seat the type, no more.
_SUBTITLE_DIM = -0.14

#: Half the padding drawn around a line's placements when its backdrop is measured, as
#: fractions of the frame. The measured box is a little larger than the type itself, so a
#: bright edge just outside the letters still counts towards the reading.
_LINE_REGION_X = .16
_LINE_REGION_Y = .07


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
        outgoing = transitions[index] if index < len(transitions) else None
        incoming = transitions[index - 1] if index > 0 else None
        # The extra footage has to cover the *incoming* transition, not the outgoing one.
        # Each fade is one `xfade` whose first input is everything joined so far, so the
        # blend shows this shot from its own frame 0: the shot's frames therefore have to
        # fill the incoming overlap as well as its own duration, and the stream it feeds
        # is exhausted at `offset + seconds`. Rendering the outgoing overlap instead put
        # the slack on the wrong end - the accumulated stream ran longer than the plan and
        # every later cut drifted with it (0.58s by the end of a 253-shot edit), while the
        # clip was still short of what its own incoming fade needed.
        _render_shot(
            shot, clips / f"{index:05}.mp4", config, art,
            shot.duration + (incoming.handle if incoming else 0.0), section_count,
            incoming=incoming, outgoing=outgoing,
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
    # One directory for libass to read, holding the project's fonts and the shipped
    # ones - see ``fonts.stage_fonts``. Collecting it here rather than passing the
    # project's own directory means the bundled families are actually reachable.
    fonts_dir = stage_fonts(config.subtitle_fonts_dir)
    placements = _subtitle_placements(lyrics, shots, config)
    # Read the backdrop under each line, so a line over a bright beach gets a heavier
    # outline than one over a night scene (R-05). The reading is purely CPU.
    line_lumas = _subtitle_lumas(lyrics, shots, placements, config, cache)
    line_outlines = [
        adaptive_outline(
            luma, config.subtitle_outline, max_outline=config.subtitle_max_outline,
        )
        for luma in line_lumas
    ] if lyrics else None
    dim_windows = _dim_windows(lyrics, line_lumas)
    write_ass(
        lyrics, subtitle, width=config.width, height=config.height,
        font=art.font, size=config.subtitle_size, weight=art.font_weight,
        margin=config.subtitle_margin, effect=art.base_subtitle_effect,
        highlight_color=art.highlight_color, line_effects=art.line_effects,
        placements=placements, outline=config.subtitle_outline,
        line_outlines=line_outlines,
        mask_only=config.subtitle_fill == "knockout",
    )
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(music)]
    if lyrics and config.subtitle_fill == "knockout":
        args += ["-filter_complex", _knockout_graph(subtitle, duration(picture), config, fonts_dir),
                 "-map", "[vout]"]
    elif lyrics and dim_windows:
        # A line sits over a bright enough backdrop that an outline alone will not hold
        # it: dim the glyph-shaped patch of picture under just those lines (R-05). No
        # backdrop is added anywhere else - the default frame has no scrim.
        args += ["-filter_complex",
                 _legibility_graph(subtitle, duration(picture), config, fonts_dir, dim_windows),
                 "-map", "[vout]"]
    elif lyrics:
        args += ["-vf", _subtitle_filter(subtitle, config, fonts_dir), "-map", "0:v:0"]
    else:
        args += ["-map", "0:v:0"]
    args += ["-map", "1:a:0", *_video_encode_args(config, intermediate=False),
             "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", str(output)]
    command(args)


def _subtitle_filter(
    subtitle: Path, cfg: RenderConfig, fonts_dir: Path | None = None,
) -> str:
    """The ``ass`` filter that draws the lyric script over the picture.

    ``fonts_dir`` is one directory holding every font this run may use - the project's
    own and the shipped ones, collected by ``fonts.stage_fonts``. libass accepts exactly
    one, and it needs it: without a ``fontsdir`` it sees only the fonts the machine has
    installed, so every bundled family silently falls back to whatever is there.
    """
    escaped = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
    filters = f"ass='{escaped}'"
    if fonts_dir and fonts_dir.is_dir():
        fonts = fonts_dir.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        filters += f":fontsdir='{fonts}'"
    return filters


def _knockout_graph(subtitle: Path, seconds: float, cfg: RenderConfig, fonts_dir: Path | None = None) -> str:
    """Cut the lyric out of the picture so a treated version of the frame shows through.

    This is the literal reading of putting the type *into* the image: there is no
    caption layer at all, only a hole in the frame shaped like the words. The fill is
    the same frame, lifted and blurred, which is what makes the letters read as a
    window rather than as white text.

    The mask is the lyric script rendered white on black; ``alphamerge`` takes its luma
    as the alpha, so whatever the script animates - a fade, a scale, a per-fragment
    delay - the hole follows it exactly.

    The picture around the letters is pulled down at the same time as the letters are
    lifted. Relying on the lift alone would leave the type invisible wherever it landed
    on a smooth, dark part of the frame - which is exactly where the free layout puts it.
    """
    blur = max(2.0, cfg.width / 320)
    return (
        f"[0:v]split=2[base][treat];"
        f"[treat]gblur=sigma={blur:.2f},"
        f"eq=brightness={_KNOCKOUT_LIFT:.3f}:saturation={_KNOCKOUT_SATURATION:.2f}[lit];"
        f"[base]eq=brightness={_KNOCKOUT_DIM:.3f}[dim];"
        f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}:d={seconds:.3f},"
        f"{_subtitle_filter(subtitle, cfg, fonts_dir)},format=gray[mask];"
        f"[lit]format=rgba[fill];"
        f"[fill][mask]alphamerge[hole];"
        f"[dim][hole]overlay=0:0:format=auto[vout]"
    )


def _legibility_graph(
    subtitle: Path, seconds: float, cfg: RenderConfig, fonts_dir: Path | None,
    windows: list[tuple[float, float]],
) -> str:
    """Draw the lyric over a glyph-shaped patch of slightly dimmed picture.

    This is the optional half of R-05: the outline already thickens on a bright backdrop,
    but where the picture is *very* bright the letters can still wash out. Rather than lay
    a translucent scrim across the whole frame - which would dim the picture the audience
    came to see - the frame is darkened only inside the shape of the letters, and the
    whole patch is gated to the on-screen window of the lines that asked for it. A line is
    only visible during its own window, so gating the patch by time confines the dim to
    exactly the lines that needed it.

    When no line needs it the caller simply draws the plain ``ass`` filter, so the default
    frame carries no scrim at all.
    """
    enable = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in windows)
    return (
        f"[0:v]split=2[base][treat];"
        f"[treat]eq=brightness={_SUBTITLE_DIM:.3f},format=rgba[halosink];"
        f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}:d={seconds:.3f},"
        f"{_subtitle_filter(subtitle, cfg, fonts_dir)},format=gray[mask];"
        f"[halosink][mask]alphamerge[halo];"
        f"[base][halo]overlay=0:0:format=auto:enable='{enable}',"
        f"{_subtitle_filter(subtitle, cfg, fonts_dir)},format=yuv420p[vout]"
    )


def _subtitle_placements(
    lyrics: list[LyricLine], shots: list[Shot], cfg: RenderConfig,
) -> list[list[Placement]] | None:
    """Free-layout positions for each lyric, or ``None`` for the classic bottom band.

    The point of the free layout is that the type shares the frame with the subject
    instead of sitting under it, so each line needs to know where its subject is -
    which is the shot it lands in, and the focus point the media pass already
    estimated for that shot.
    """
    if cfg.subtitle_layout != "free" or not lyrics:
        return None
    return plan_placements(
        lyrics, _lyric_focus_points(lyrics, shots),
        width=cfg.width, height=cfg.height,
        margin=cfg.subtitle_margin, size=cfg.subtitle_size,
    )


def _lyric_shot_indices(lyrics: list[LyricLine], shots: list[Shot]) -> list[int]:
    """Which shot each lyric line lands in.

    Shots and lyrics are both in time order, so one walk covers the whole list rather
    than searching the shot list again for every line. The line's midpoint decides the
    shot it belongs to, which is what a viewer reads as "the picture the words are over".
    """
    indices: list[int] = []
    cursor = 0
    for line in lyrics:
        midpoint = (line.start + line.end) / 2
        while cursor < len(shots) - 1 and shots[cursor].end <= midpoint:
            cursor += 1
        indices.append(cursor)
    return indices


def _lyric_focus_points(
    lyrics: list[LyricLine], shots: list[Shot],
) -> list[tuple[float, float] | None]:
    """The subject position of the shot each lyric lands in."""
    points: list[tuple[float, float] | None] = []
    for index in _lyric_shot_indices(lyrics, shots):
        focus = shots[index].focus_point if shots else None
        points.append(
            (float(focus[0]), float(focus[1]))
            if focus and len(focus) == 2 else None
        )
    return points


def _subtitle_lumas(
    lyrics: list[LyricLine], shots: list[Shot], placements: list[list[Placement]] | None,
    cfg: RenderConfig, cache: Path,
) -> list[float]:
    """The backdrop luminance under each lyric line, for the outline and dim decisions.

    The reading is taken from the *source* media of the shot the line lands in, over the
    region the line occupies - the band it sits in, or the box its placements cover. A
    video contributes a single cached frame. Anything unreadable returns a neutral
    mid-grey, so a lyric always gets an outline and never blocks a render.
    """
    if not lyrics:
        return []
    legibility_cache = cache / "legibility"
    lumas: list[float] = []
    for index in range(len(lyrics)):
        if not shots:
            lumas.append(.5)
            continue
        shot = shots[_lyric_shot_indices(lyrics, shots)[index]]
        focus = shot.focus_point if shot.focus_point and len(shot.focus_point) == 2 else None
        lumas.append(estimate_region_luma(
            shot.file, shot.kind, focus, _line_region(index, placements, cfg),
            legibility_cache, at=shot.source_start,
        ))
    return lumas


def _line_region(
    index: int, placements: list[list[Placement]] | None, cfg: RenderConfig,
) -> tuple[float, float, float, float]:
    """The normalised box a line occupies: its placements, or the bottom band.

    With the free layout the box is drawn around the fragments the line was placed at; in
    the band layout it is the strip at the foot of the frame the centred line sits in.
    """
    if placements and index < len(placements) and placements[index]:
        row = placements[index]
        xs = [fragment.x / max(cfg.width, 1) for fragment in row]
        ys = [fragment.y / max(cfg.height, 1) for fragment in row]
        return (
            max(0.0, min(xs) - _LINE_REGION_X), max(0.0, min(ys) - _LINE_REGION_Y),
            min(1.0, max(xs) + _LINE_REGION_X), min(1.0, max(ys) + _LINE_REGION_Y),
        )
    margin = cfg.subtitle_margin / max(cfg.height, 1)
    size = cfg.subtitle_size / max(cfg.height, 1)
    return (0.0, max(0.0, 1.0 - margin - size), 1.0, 1.0)


def _dim_windows(
    lyrics: list[LyricLine], lumas: list[float],
) -> list[tuple[float, float]]:
    """On-screen windows of the lines whose backdrop is bright enough to need a dim."""
    return [
        (line.start, line.end)
        for line, luma in zip(lyrics, lumas)
        if needs_local_dim(luma)
    ]


def _render_shot(
    shot: Shot, output: Path, cfg: RenderConfig, art: ArtDirection,
    render_duration: float, section_count: int,
    *, incoming: _Transition | None = None, outgoing: _Transition | None = None,
) -> None:
    if shot.kind == "image":
        _render_image_shot(shot, output, cfg, art, render_duration, section_count,
                           incoming=incoming, outgoing=outgoing)
        return

    # Preserve source cinematography: synthetic oscillation on moving footage looks seasick.
    overscan = 1.025
    scaled_width, scaled_height = round(cfg.width * overscan / 2) * 2, round(cfg.height * overscan / 2) * 2
    focus_x, focus_y = _safe_focus(shot.focus_point)
    visual = (f"scale={scaled_width}:{scaled_height}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={cfg.width}:{cfg.height}:"
              f"x='clip(iw*{focus_x}-ow/2,0,iw-ow)':y='clip(ih*{focus_y}-oh/2,0,ih-oh)',setsar=1")
    # One merged grade, not three stacked ones (R-06): the shot match, the director's
    # grade and the section tint all land on the same couple of operators.
    effects = _grade_filter(shot, art, section_count, cfg)
    if cfg.visual_effects:
        upscale = (
            max(cfg.width / shot.source_width, cfg.height / shot.source_height)
            if shot.source_width > 0 and shot.source_height > 0 else 1.0
        )
        if shot.motion == "dynamic" and upscale <= 1.35:
            effects.append("unsharp=5:5:0.55:5:5:0")
        elif shot.motion == "gentle":
            effects.append("gblur=sigma=0.18")
        vignette = _vignette(shot, art)
        if vignette:
            effects.append(vignette)
        grain = _grain(shot, art)
        if grain:
            effects.append(grain)
    effects += _cut_effects(incoming, outgoing, render_duration, cfg)
    args = ["ffmpeg", "-y", "-v", "error"]
    args += ["-stream_loop", "-1", "-ss", str(shot.source_start)]
    args += ["-i", shot.file, "-t", str(render_duration), "-an", "-vf", ",".join(
        [visual, *(item for item in effects if item)]
    ), "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output)]
    command(args)


def _cut_effects(
    incoming: _Transition | None, outgoing: _Transition | None,
    duration: float, cfg: RenderConfig,
) -> list[str]:
    """Effect-transition filters for both ends of one shot.

    A shot sits between two cuts, so it can carry a head effect, a tail effect or
    both; ``_transition_effect_filters`` returns nothing for cuts and xfades, which
    are handled by the timeline instead.
    """
    filters: list[str] = []
    if incoming is not None:
        filters += _transition_effect_filters(incoming, "in", duration, cfg)
    if outgoing is not None:
        filters += _transition_effect_filters(outgoing, "out", duration, cfg)
    return filters


def _render_image_shot(
    shot: Shot, output: Path, cfg: RenderConfig, art: ArtDirection,
    render_duration: float, section_count: int,
    *, incoming: _Transition | None = None, outgoing: _Transition | None = None,
) -> None:
    files = [shot.file, *(layer.file for layer in shot.layers if layer.kind == "image")]
    args = ["ffmpeg", "-y", "-v", "error"]
    for file in files:
        args += ["-loop", "1", "-framerate", str(cfg.fps), "-i", file]
    filters, current = _image_filter_graph(shot, cfg, art, render_duration, len(files))
    # The three color stages are merged into at most two operators (R-06), so a shot
    # that used to carry six color ops now carries two.
    finishing = _grade_filter(shot, art, section_count, cfg)
    if cfg.visual_effects:
        if shot.motion == "dynamic":
            finishing.append("unsharp=5:5:0.42:5:5:0")
        vignette = _vignette(shot, art)
        if vignette:
            finishing.append(vignette)
        grain = _grain(shot, art)
        if grain:
            finishing.append(grain)
    finishing += _cut_effects(incoming, outgoing, render_duration, cfg)
    finish = ",".join(item for item in finishing if item)
    # Normalize to the exact target canvas: a filter chain can emit an odd width, and
    # xfade/concat reject mismatched input sizes. The multi-image layouts no longer
    # depend on this - their panels add up to the canvas and their seams are drawn over
    # the join, precisely so that this step has nothing to scale or pad.
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

    The remaining fields are what let one table cover moves that are more than "push
    in a bit". ``curve`` changes how progress maps onto the interpolation - a ramp
    arrives, a breathe returns, a pulse repeats; ``roll`` turns the frame about its
    centre; ``keystone``/``keystone_v`` skew the quad as if the picture were turning
    in space; ``shake`` adds handheld jitter and ``drag`` smears the frame along its
    own motion so a fast throw reads as fast rather than as a jump cut.
    """

    zoom_from: float
    zoom_to: float
    x_from: float = 0.0
    x_to: float = 0.0
    y_from: float = 0.0
    y_to: float = 0.0
    ease: str = "linear"
    curve: str = "ramp"
    pulses: int = 1
    roll: float = 0.0
    keystone: float = 0.0
    keystone_v: float = 0.0
    shake: float = 0.0
    drag: int = 0


# Movement vocabulary for stills. These are deliberately distinct in *character*,
# not just in magnitude: a drift holds its framing and slides, a dolly commits to
# the push, a punch arrives fast and settles, a breathe swells and returns, a
# handheld holds still and trembles.
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
    "breathe": _CameraMove(0.0, 0.050, curve="breathe"),
    # --- the wider vocabulary: moves that are not just "push in a bit" ---------
    # A throw. A pan can only travel ``W - W/zoom``, so a whip needs a real crop to
    # have anywhere to go: at a 5% zoom the whole move covers about eight pixels and
    # reads as a slow drift however fast it is nominally going. The deep crop is what
    # buys the travel, and ``drag`` blurs along it so the speed is visible rather than
    # implied by a jump.
    "whip_pan": _CameraMove(0.060, 0.420, x_from=-0.140, x_to=0.140, ease="inout", drag=5),
    # A push that also turns: the two together read as travelling around a subject
    # rather than straight at it.
    "spiral_in": _CameraMove(0.0, 0.115, ease="inout", roll=2.8),
    # A long slow dutch-angle drift. Barely any push, so the roll carries it.
    "roll_drift": _CameraMove(0.050, 0.075, x_from=-0.075, x_to=0.075, roll=-3.4),
    # A hold that breathes like a hand rather than a tripod, for intimate or
    # documentary material.
    "handheld": _CameraMove(0.0, 0.030, shake=0.055),
    # Turning in space. The top and bottom edges travel at different widths, which
    # is what makes a flat still read as a plane with a front and a back.
    "tilt3d_back": _CameraMove(0.0, 0.055, keystone=0.14),
    "tilt3d_front": _CameraMove(0.0, 0.055, keystone=-0.14),
    "tilt3d_left": _CameraMove(0.0, 0.055, keystone_v=-0.13),
    "tilt3d_right": _CameraMove(0.0, 0.055, keystone_v=0.13),
    # Two swells instead of one long push, so the move lands on the beat twice.
    "pulse_in": _CameraMove(0.0, 0.105, curve="pulse", pulses=2),
}

# Parallax was removed with the single-image rework (R-04). Its two planes were the
# same image split into a blurred backdrop and a scaled foreground - the dual-scale
# double the rework forbids - and a real depth cue is not recoverable from a flat photo.
# ``_framing_effect`` no longer emits it and the planner's vocabulary no longer carries it.

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

# Handheld jitter is built from sines rather than noise because ``perspective``
# exposes no random source. Two incommensurate rates per axis, plus a slow envelope
# so the tremor waxes and wanes, keeps it from reading as a metronome.
_SHAKE_HZ = (4.3, 7.1)
_SHAKE_ENVELOPE_HZ = 0.6


def _curve_parameter(move: _CameraMove, progress: str) -> str:
    """How progress drives the move's interpolation, as an FFmpeg expression.

    ``ramp`` is the usual one-way trip. ``breathe`` runs 0 -> 1 -> 0, so the frame
    swells and returns to where it started. ``pulse`` is the same shape repeated, so
    a single shot can land on the beat more than once.
    """
    if move.curve == "breathe":
        return f"(0.5-0.5*cos(2*PI*{progress}))"
    if move.curve == "pulse":
        return f"(0.5-0.5*cos(2*PI*{move.pulses}*{progress}))"
    return _EASES[move.ease](progress)


def _shake_expressions(amplitude: float, progress: str, frames: int, fps: int) -> tuple[str, str]:
    """Handheld tremor for the crop origin, in travel units.

    Driven by ``on`` divided by ``fps`` rather than by progress, so the tremor keeps
    the same frequency in seconds whatever the shot length or frame rate - a shake
    that slowed down on long shots would read as a wobble, not as a hand.

    ``on`` is clamped at both ends rather than left to run. Like every other curve
    here it has to hold once the shot is over: frames pulled past the end by
    ``fps``/``trim`` would otherwise keep trembling, and a shot that will not sit
    still at its end cannot be cut against anything.
    """
    seconds = f"((max(1,min(on,{frames}))-1)/{fps})"
    envelope = f"(0.62+0.38*sin(2*PI*{_SHAKE_ENVELOPE_HZ}*{seconds}+0.9))"
    slow, fast = _SHAKE_HZ
    x = (
        f"({amplitude:.5f}*{envelope}*"
        f"(sin(2*PI*{slow}*{seconds})+0.55*sin(2*PI*{fast}*{seconds}+1.3)))"
    )
    y = (
        f"({amplitude * .72:.5f}*{envelope}*"
        f"(sin(2*PI*{fast}*{seconds}+2.1)+0.5*sin(2*PI*{slow}*{seconds})))"
    )
    return x, y


def _quad_spread(move: _CameraMove, amount: float, unit: float, aspect: float) -> float:
    """How much bigger than the crop box the quad gets at curve position ``unit``.

    A keystone widens one edge and a roll swings the corners outward, so the box that
    has to fit inside the frame is the quad's *bounding* box, not ``W/zoom``. Returns
    the factor the zoom must reach to keep the widest case inside the picture, for the
    horizontal and vertical axes (the larger of the two).
    """
    keystone = 1 + abs(move.keystone * amount * unit)
    keystone_v = 1 + abs(move.keystone_v * amount * unit)
    roll = math.radians(move.roll * amount * unit)
    cos_r, sin_r = abs(math.cos(roll)), abs(math.sin(roll))
    # The extra 0.4% is the same margin ``_MIN_ZOOM`` buys an unskewed move: it keeps
    # a corner from landing exactly on the frame edge, where float noise decides
    # whether the filter sees it as inside or out.
    return 1.004 * max(
        keystone * cos_r + aspect * keystone_v * sin_r,
        keystone_v * cos_r + keystone * sin_r / aspect,
    )


def _required_zoom(move: _CameraMove, amount: float, low: float, high: float, aspect: float) -> float:
    """How far the zoom curve must be lifted so the quad never leaves the frame.

    Sampling the curve is enough because every shape here is monotonic in ``unit``
    between the points sampled, and the curves that are not - a breathe, a pulse -
    still sweep the whole 0..1 range. Lifting the whole curve rather than clamping it
    keeps the move's intended zoom *delta* intact; only its baseline moves.
    """
    deficit = 0.0
    for step in range(9):
        unit = step / 8
        available = low + (high - low) * unit
        deficit = max(deficit, _quad_spread(move, amount, unit, aspect) - available)
    return deficit


def _camera_quad(
    move: _CameraMove, amount: float, direction: int, frames: int, fps: int,
    aspect: float = 9 / 16,
) -> tuple[str, ...]:
    """Return the eight corner expressions of the crop quad, per frame.

    ``aspect`` is the frame's height over its width, which is what converts a
    horizontal keystone into a vertical footprint and back.
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

    Every corner comes from one centre and one half-size, then optionally skewed and
    rolled. Deriving them that way is what lets a roll or a 3D turn reuse the same
    single ``perspective`` pass instead of stacking another resample on the crop.
    """
    progress = f"clip((on-1)/{max(1, frames - 1)},0,1)"
    unit = _curve_parameter(move, progress)

    low = 1 + _MIN_ZOOM + move.zoom_from * amount
    high = 1 + _MIN_ZOOM + move.zoom_to * amount
    lift = _required_zoom(move, amount, low, high, aspect)
    low, high = low + lift, high + lift
    zoom = f"{low:.6f}" if abs(high - low) < 1e-9 else f"({low:.6f}+{high - low:.6f}*{unit})"

    x = f"{direction * move.x_from:.5f}+{direction * (move.x_to - move.x_from):.5f}*{unit}"
    y = f"{move.y_from:.5f}+{move.y_to - move.y_from:.5f}*{unit}"
    if move.shake:
        shake_x, shake_y = _shake_expressions(move.shake * amount, progress, frames, fps)
        x, y = f"({x}+{shake_x})", f"({y}+{shake_y})"

    crop_width = f"(W/({zoom}))"
    crop_height = f"(H/({zoom}))"
    half_width = f"({crop_width}/2)"
    half_height = f"({crop_height}/2)"

    # A keystone makes one edge wider (or taller) than its opposite, which is what a
    # flat plane looks like when it turns in space. ``keystone`` splits the top and
    # bottom widths, ``keystone_v`` the left and right heights.
    #
    # ``amount`` scales every one of these fields - zoom, roll, shake and keystone
    # alike - and ``_quad_spread`` sizes the zoom lift from the *scaled* strength.
    # Leaving it out here made the quad up to 14% wider than the estimate, so a shot
    # with ``amount < 1`` was lifted too little and its corners left the frame
    # (ffmpeg rejects such a frame with EINVAL halfway through the render).
    top_bottom = f"({move.keystone * amount:.5f}*{unit})" if move.keystone else None
    left_right = f"({move.keystone_v * amount:.5f}*{unit})" if move.keystone_v else None
    wide_half = f"({half_width}*(1+abs({top_bottom})))" if top_bottom else half_width
    tall_half = f"({half_height}*(1+abs({left_right})))" if left_right else half_height

    if move.roll:
        angle = f"({move.roll * amount * math.pi / 180:.6f}*{unit})"
        cosine, sine = f"cos({angle})", f"sin({angle})"
        extent_x = f"({wide_half}*abs({cosine})+{tall_half}*abs({sine}))"
        extent_y = f"({wide_half}*abs({sine})+{tall_half}*abs({cosine}))"
    else:
        cosine, sine = "1", "0"
        extent_x, extent_y = wide_half, tall_half

    footprint_x = f"(2*{extent_x})"
    footprint_y = f"(2*{extent_y})"
    travel_x = f"(W-{footprint_x})"
    travel_y = f"(H-{footprint_y})"
    centre_x = f"(clip({travel_x}/2+{travel_x}*({x}),0,{travel_x})+{footprint_x}/2)"
    centre_y = f"(clip({travel_y}/2+{travel_y}*({y}),0,{travel_y})+{footprint_y}/2)"

    corners = []
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        offset_x = f"({sx}*{half_width}*(1-({sy})*{top_bottom}))" if top_bottom else f"({sx}*{half_width})"
        offset_y = f"({sy}*{half_height}*(1+({sx})*{left_right}))" if left_right else f"({sy}*{half_height})"
        if move.roll:
            corners.append(f"{centre_x}+{offset_x}*{cosine}-{offset_y}*{sine}")
            corners.append(f"{centre_y}+{offset_x}*{sine}+{offset_y}*{cosine}")
        else:
            corners.append(f"{centre_x}+{offset_x}")
            corners.append(f"{centre_y}+{offset_y}")
    return tuple(corners)


def _perspective_filter(quad: tuple[str, ...]) -> str:
    """Crop a quad out of the frame and stretch it back to full size, per frame.

    ``perspective`` is used instead of ``zoompan`` because zoompan truncates the
    crop origin to whole pixels of its input frame. The camera moves here are only
    a fraction of a pixel per frame, so zoompan holds the image still for several
    frames and then snaps it by a whole pixel - a visible stutter on any straight
    edge. Feeding it a supersampled frame only shrinks the jump; perspective
    evaluates the crop quad per frame with true sub-pixel interpolation, so the move
    stays smooth and nothing is resampled twice.

    The crop is ``W/zoom`` wide and may travel ``W - W/zoom`` before leaving the
    frame; the two are easy to confuse and swapping them turns the move into a
    hard punch-in. ``sense=source`` means the coordinates describe where in the
    *source* each output corner comes from.
    """
    x0, y0, x1, y1, x2, y2, x3, y3 = quad
    return (
        f"perspective="
        f"x0='{x0}':y0='{y0}':"
        f"x1='{x1}':y1='{y1}':"
        f"x2='{x2}':y2='{y2}':"
        f"x3='{x3}':y3='{y3}':"
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
    filters: list[str] = []

    if effect in IMAGE_COMPOSITES:
        composite = _composite_graph(shot, cfg, duration, input_count, filters)
        if composite is not None:
            return composite
        # Not enough pictures to fill the layout this plan asked for. A camera move
        # is a poor stand-in for a division of the frame, but it is a shot.
        effect = "cinematic_depth"

    # --- single-image camera moves, and the treatments layered on top of them ---
    frames = max(1, round(duration * cfg.fps))
    direction = -1 if (shot.media_id + max(0, shot.section_index)) % 2 else 1
    amount = (max(.35, art.camera_intensity) * (1 + shot.melody * .18)
              * {"dynamic": 1.35, "gentle": .7}.get(shot.motion, 1.0))

    # R-04: a single image is cropped full-bleed to the canvas, exactly like the video
    # branch - subject-aware, driven by ``focus_point``. There is no blurred same-source
    # backdrop and no scaled foreground any more; that pair was one picture at two
    # scales, which is the double the rework exists to kill. The scale also converts
    # the still's full-range decode to the limited range the delivery expects.
    _fill_frame(filters, 0, "adapted", cfg.width, cfg.height,
                focus=shot.focus_point, full_range=True)
    move = _CAMERA_MOVES.get(effect, _CAMERA_MOVES["cinematic_depth"])
    # A breathe shot, or the outro, releases rather than drives: reverse a push so
    # the frame opens up instead of closing in. Moves that already end where they
    # started - a drift, a breathe - are left alone.
    if (shot.edit_intent == "breathe" or shot.section == "outro") and move.zoom_to > move.zoom_from:
        move = replace(move, zoom_from=move.zoom_to, zoom_to=move.zoom_from, ease="out")
    quad = _camera_quad(move, amount, direction, frames, cfg.fps, cfg.height / cfg.width)
    filters.append(f"[adapted]{_perspective_filter(quad)}[moved]")
    current = "[moved]"
    if effect == "focus_pull":
        # R-04 redefines focus_pull as a same-scale rack focus rather than a
        # blurred-backdrop double: one full-bleed frame split into a sharp copy and a
        # blurred copy, cross-faded over the shot so the picture comes *into* focus.
        # The two copies are the same size, so this is not the dual-scale double the
        # single-image rework forbids - it is the sharpness axis, not the framing axis.
        current = _focus_pull(filters, current, cfg, duration)
    if move.drag:
        # A whip pan travels several pixels in a single frame, which on its own reads
        # as a jump cut rather than as speed. Blending the neighbouring frames along
        # the path supplies the smear a real shutter would have left - the effect
        # filters below then work on the dragged stream instead of the clean one.
        # The kernel is a triangle, so the centre frame dominates and the tails fade.
        weights = " ".join(str(min(i + 1, move.drag - i)) for i in range(move.drag))
        filters.append(f"[moved]tmix=frames={move.drag}:weights='{weights}'[dragged]")
        current = "[dragged]"

    if effect == "film_bars":
        # Overlay the bars rather than cropping to them: the audience reads 2.35:1
        # while the image underneath keeps its whole frame.
        bar = max(2, round(cfg.height * .055 / 2) * 2)
        filters.append(
            f"{current}drawbox=x=0:y=0:w=iw:h={bar}:t=fill:color=black,"
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
            f"{current}format=rgba[irisfg];"
            f"[irisfg][irismask]alphamerge[iriscut];"
            f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}:d={mask_duration:.4f}[irisbg];"
            f"[irisbg][iriscut]overlay=0:0:shortest=1[composite]"
        )
        return filters, "[composite]"

    return filters, current


def _even(value: int) -> int:
    """Round down to an even length. Codecs and chroma subsampling want even edges."""
    return value - value % 2


def _seam_thickness(size: int) -> int:
    """Hairline thickness for a seam across an axis this many pixels long."""
    return max(2, round(size * .004))


def _seam(
    filters: list[str], cfg: RenderConfig, duration: float, current: str, *,
    x: int, y: int, width: int, height: int, label: str,
) -> str:
    """Lay a translucent hairline across the join between two panels.

    It is drawn *over* the two panels rather than carved out of them, which is what
    lets the panels add up to the canvas exactly. Leaving the seam as a gap makes the
    mosaic a few pixels narrower than the frame, and the finishing normalisation then
    scales the whole shot up and pads it with black bars - a dark edge down both sides
    of every split screen, for a seam that was never meant to be visible as a gap.
    """
    filters.append(
        f"color=c=white@0.22:s={width}x{height}:r={cfg.fps}:d={_plane_duration(duration, cfg):.4f}[{label}]"
    )
    filters.append(f"{current}[{label}]overlay=x={x}:y={y}:shortest=1[{label}joined]")
    return f"[{label}joined]"


def _composite_graph(
    shot: Shot, cfg: RenderConfig, duration: float, input_count: int, filters: list[str],
) -> tuple[list[str], str] | None:
    """Build a multi-image layout, or ``None`` when there are not enough pictures.

    Every layout here hands each picture its own region of the frame, or shows one
    picture at a time. None of them paints one picture over another's pixels: two
    pictures sharing an area have no way to both stay legible, and the frame reads as
    mud rather than as a design. That whole family - a screen-blended double exposure,
    a pile of cards laid over another picture - was removed for this reason, and the
    layouts below are what replaced it.

    The panels of a division are full-bleed inside their own region. Letting one keep
    the letterbox treatment while its neighbour filled its half gave the two sides
    visibly different framing and exposure, which read as two pictures that happen to
    be adjacent rather than one frame divided.
    """
    effect = shot.image_effect
    if input_count < IMAGE_COMPOSITES.get(effect, (2, 2))[1]:
        return None

    # --- divisions of the frame: each picture gets an exclusive region ---
    if effect in {"split_screen", "hero_split", "triptych"}:
        # The widths do *not* reserve anything for the seam; it is drawn on top of the
        # join below. Reserving it would make three panels of a triptych come out 634,
        # 634 and 652 on a 1920 canvas - an imbalance the eye reads as a mistake long
        # before it can say why.
        if effect == "triptych":
            column = _even(cfg.width // 3)
            widths = [column, column, cfg.width - 2 * column]
        elif effect == "hero_split":
            # Two to one rather than even halves: the wide side is the shot, the narrow
            # side is a second look at the same moment. Equal halves simply read as a
            # screen split down the middle.
            hero = _even(round(cfg.width * 2 / 3))
            widths = [hero, cfg.width - hero]
        else:
            left = _even(cfg.width // 2)
            widths = [left, cfg.width - left]
        for index, width in enumerate(widths):
            _fill_frame(filters, index, f"panel{index}", width, cfg.height)
        inputs = "".join(f"[panel{index}]" for index in range(len(widths)))
        filters.append(f"{inputs}hstack=inputs={len(widths)}[panels]")
        seam = _seam_thickness(cfg.width)
        current = "[panels]"
        for index in range(len(widths) - 1):
            current = _seam(
                filters, cfg, duration, current,
                x=sum(widths[:index + 1]) - seam // 2, y=0,
                width=seam, height=cfg.height, label=f"seam{index}",
            )
        return filters, current

    # --- a hero with a column of smaller pictures beside it ---
    if effect == "hero_grid":
        hero = _even(round(cfg.width * 2 / 3))
        side = cfg.width - hero
        upper = _even(cfg.height // 2)
        lower = cfg.height - upper
        _fill_frame(filters, 0, "hero", hero, cfg.height)
        _fill_frame(filters, 1, "cell0", side, upper)
        _fill_frame(filters, 2, "cell1", side, lower)
        filters.append("[cell0][cell1]vstack=inputs=2[side];[hero][side]hstack=inputs=2[panels]")
        current = _seam(
            filters, cfg, duration, "[panels]",
            x=hero - _seam_thickness(cfg.width) // 2, y=0,
            width=_seam_thickness(cfg.width), height=cfg.height, label="column",
        )
        row_seam = _seam_thickness(cfg.height)
        return filters, _seam(
            filters, cfg, duration, current,
            x=hero, y=upper - row_seam // 2,
            width=side, height=row_seam, label="row",
        )

    # --- a diagonal edge instead of a straight one ---
    if effect == "diagonal_split":
        # Four ways to cut the frame in half, rotated by shot so one song does not repeat
        # the same slope. ``X/W`` and ``Y/H`` normalise the mask so one expression covers
        # both landscape and portrait canvases.
        slope = "X/W+Y/H" if shot.index % 2 else "X/W-Y/H+1"
        keep = "gt" if (shot.index // 2) % 2 else "lt"
        # Half resolution, not the eighth the iris mask uses: at an eighth a straight
        # edge lands on the eight-pixel grid and upscales into a visible staircase,
        # while at half the bilinear edge is a couple of pixels of softness - a seam
        # rather than a jitter. The mask is one rectangle either way, so it stays cheap.
        mask_width = max(64, _even(round(cfg.width / 2)))
        mask_height = max(36, _even(round(cfg.height / 2)))
        filters.append(
            f"color=c=black:s={mask_width}x{mask_height}:r={cfg.fps}:d={_plane_duration(duration, cfg):.4f},"
            f"format=gray,geq=lum='if({keep}({slope},1),255,0)',"
            f"scale={cfg.width}:{cfg.height}:flags=bilinear[dmask]"
        )
        # The second picture is cut with the mask into a hard-edged half and laid over
        # the first. Its pixels either win or lose; none of them are mixed.
        filters.append(
            f"[1:v]scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={cfg.width}:{cfg.height},setsar=1,format=rgba[dcut];"
            f"[dcut][dmask]alphamerge[dupper]"
        )
        _fill_frame(filters, 0, "dlower", cfg.width, cfg.height)
        filters.append("[dlower][dupper]overlay=0:0:shortest=1[composite]")
        return filters, "[composite]"

    # --- one picture at a time, switching inside the shot ---
    if effect == "beat_montage":
        count = min(input_count, 4)
        # Deliberately the one composite that keeps the letterbox treatment: it shows
        # one image at a time rather than several at once, so there is no second field
        # to compete with, and keeping each frame inset makes the montage read as a
        # sequence of photographs rather than as a hard cut between full frames. The
        # backdrop is now a flat tile of the shot's dominant colour (``_matte_frame``)
        # rather than a blurred copy of the same image - the same-source double is
        # exactly what R-04 removes, and this was its last remaining user.
        for index in range(count):
            _matte_frame(filters, index, f"montage{index}", cfg.width, cfg.height,
                         shot.source_color, cfg, duration)
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

    return None


def _fill_frame(
    filters: list[str], input_index: int, label: str, width: int, height: int,
    *, focus: list[float] | None = None, full_range: bool = False,
) -> None:
    """Scale to cover the region and crop the overflow, leaving no letterbox at all.

    This is the single-image path (R-04): the picture is cropped full-bleed to the
    canvas and the overflow is thrown away. When ``focus`` is given the crop is driven
    off the subject instead of the centre, exactly as the video branch does it, so a
    lone portrait still keeps its subject. There is deliberately no ``decrease`` variant
    - the removed ``_adapt_image`` pair was the same picture again at a second scale,
    and nothing draws a picture inside a picture any more.

    The format is pinned because these labels get stacked: a JPEG decodes as full-range
    ``yuvj420p`` and a PNG as RGB, and a stack of mismatched panels only works if ffmpeg
    happens to insert the right conversion. ``full_range`` additionally converts the
    still's full-range decode to the limited range the delivery expects (R-06); the
    composite panels stay full-range through the graph and are converted once at the end.
    """
    if focus is not None:
        focus_x, focus_y = _safe_focus(focus)
        crop = (f"crop={width}:{height}:x='clip(iw*{focus_x}-ow/2,0,iw-ow)':"
                f"y='clip(ih*{focus_y}-oh/2,0,ih-oh)'")
    else:
        crop = f"crop={width}:{height}"
    range_arg = ":in_range=full:out_range=limited" if full_range else ""
    filters.append(
        f"[{input_index}:v]scale={width}:{height}:"
        f"force_original_aspect_ratio=increase:flags=lanczos{range_arg},"
        f"{crop},setsar=1,format=yuv420p[{label}]"
    )


def _matte_frame(
    filters: list[str], input_index: int, label: str, width: int, height: int,
    colour: list[int], cfg: RenderConfig, duration: float,
) -> None:
    """Lay an inset full-bleed picture over a flat tile of its own dominant colour.

    The one composite that keeps the letterbox treatment (R-04): it shows one picture at
    a time, so the inset carries its own reading - a run of photographs. What it must not
    keep is the blurred same-source backdrop; a still, flat tile of the shot's dominant
    colour fills the letterbox instead, which is the same intent without the double.
    """
    inset_width = _even(round(width * .92)) or 2
    inset_height = _even(round(height * .92)) or 2
    _fill_frame(filters, input_index, f"{label}pic", inset_width, inset_height)
    base = list(colour)[:3]
    padded = base + [128] * (3 - len(base))
    red, green, blue = (max(0, min(255, int(value))) for value in padded)
    tile = f"0x{red:02X}{green:02X}{blue:02X}"
    filters.append(
        f"color=c={tile}:s={width}x{height}:r={cfg.fps}:d={_plane_duration(duration, cfg):.4f}[{label}bg];"
        f"[{label}bg][{label}pic]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[{label}]"
    )


def _focus_pull(filters: list[str], label: str, cfg: RenderConfig, duration: float) -> str:
    """Rack focus on one full-bleed frame, without a second-scale copy.

    ``focus_pull`` used to mean "a sharp foreground over a softened backdrop", which was
    the same-image double R-04 removes. Focus is the *sharpness* axis, not the framing
    axis, so it moves from space to time: the frame is split into a sharp copy and a
    blurred copy - same size, so not a dual-scale double - and the two are cross-faded
    over the shot. The picture comes into focus. The blend is driven by ``T`` (seconds),
    clamped so the frames pulled past the shot's end hold on the sharp copy.

    ``blend`` only needs ``all_expr`` to drive a custom cross-fade; there is no ``custom``
    *mode* in this ffmpeg (``all_mode=custom`` is rejected as an invalid value), and a mode
    would be ignored anyway once an expression is present. ``A`` is the sharp input, ``B``
    the blurred one, so the weight starts fully on ``B`` and lands fully on ``A``.
    """
    blur = max(1.6, cfg.width / 420)
    span = max(0.01, duration)
    filters.append(
        f"{label}split=2[focussharp][focussoft];"
        f"[focussoft]gblur=sigma={blur:.2f}[focusblur];"
        f"[focussharp][focusblur]blend="
        f"all_expr='A*clip(T/{span:.4f},0,1)+B*(1-clip(T/{span:.4f},0,1))'[focused]"
    )
    return "[focused]"


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


#: The two operators every color stage has to funnel through. Merging the shot match,
#: the director's grade and the section tint onto these two is R-06 - the three stages
#: used to stack up to six operators on one shot.
_EQ_KNOB = re.compile(r"\b(brightness|contrast|saturation)=(-?\d*\.?\d+)")
_CB_KNOB = re.compile(r"\b(rs|gs|bs)=(-?\d*\.?\d+)")

#: Ceiling on the *multiplied* saturation gain. Each stage is "subtle" on its own, but
#: three of them multiplied can push a warm, low-chroma source far past a natural grade.
_MAX_SATURATION = 1.10

#: A vignette only helps a picture whose edges are bright enough to be pulled down
#: without going muddy. Below this the corners are already dark and it only crushes them.
_VIGNETTE_EDGE_LUMA = 0.19

#: A source carrying this much high-frequency energy is already as grainy as the grade
#: wants; adding more only feeds the encoder noise it then spends bits preserving.
_GRAIN_NOISE_CEILING = 0.6


def _grade_filter(
    shot: Shot, art: ArtDirection, section_count: int, cfg: RenderConfig,
) -> list[str]:
    """The shot's whole color grade, as at most two operators (R-06).

    The shot match, the director's ``grade_filter`` and the section tint used to run one
    after another - up to six ``eq``/``colorbalance`` operators on a single shot, which
    makes the grade impossible to read back. They are folded here into at most one ``eq``
    (brightness summed, contrast and saturation multiplied) and at most one
    ``colorbalance`` (the vectors summed). Neutral operators are dropped and the merged
    saturation gain is clamped, so a pile of individually-"subtle" stages cannot add up
    to a look nobody chose.
    """
    brightness, contrast, saturation = 0.0, 1.0, 1.0
    red, green, blue = 0.0, 0.0, 0.0
    stages = (
        _shot_match_filter(shot, cfg.shot_match_strength),
        art.grade_filter,
        _section_color_filter(shot, art, section_count, cfg.look_strength),
    )
    for stage in stages:
        if not stage:
            continue
        for name, value in _EQ_KNOB.findall(stage):
            amount = float(value)
            if name == "brightness":
                brightness += amount
            elif name == "contrast":
                contrast *= amount
            else:
                saturation *= amount
        for name, value in _CB_KNOB.findall(stage):
            amount = float(value)
            if name == "rs":
                red += amount
            elif name == "gs":
                green += amount
            else:
                blue += amount
    saturation = min(saturation, _MAX_SATURATION)
    operators: list[str] = []
    if abs(brightness) > 1e-6 or abs(contrast - 1) > 1e-6 or abs(saturation - 1) > 1e-6:
        operators.append(
            f"eq=brightness={brightness:.4f}:contrast={contrast:.4f}:saturation={saturation:.4f}"
        )
    if max(abs(red), abs(green), abs(blue)) > 1e-6:
        operators.append(f"colorbalance=rs={red:.4f}:gs={green:.4f}:bs={blue:.4f}")
    return operators


def _vignette(shot: Shot, art: ArtDirection) -> str:
    """The vignette filter, or ``""`` when the source would only be crushed by it.

    R-06 makes the vignette conditional. It is a tool for a bright, flat-edged frame;
    applied to one already dark at the edges it simply eats the corners. The gate reads
    ``edge_luma`` - the brightness of the *border*, measured at discovery - so a dark
    subject against a bright wall still earns its vignette.
    """
    if not art.vignette or shot.edge_luma < _VIGNETTE_EDGE_LUMA:
        return ""
    return "vignette=PI/5"


def _grain(shot: Shot, art: ArtDirection) -> str:
    """The grain filter, or ``""`` when the source already carries noise (R-06).

    Grain used to be applied unconditionally, so a double-compressed source got a second
    layer of noise and the encoder then paid to preserve it - which is the R-12 volume
    problem in reverse. The gate reads ``noise_score`` so a clean source keeps its film
    texture.
    """
    if art.grain <= 0 or shot.noise_score >= _GRAIN_NOISE_CEILING:
        return ""
    return f"noise=alls={art.grain}:allf=t+u"


def _video_encode_args(cfg: RenderConfig, *, intermediate: bool) -> list[str]:
    args = [
        "-c:v", "libx264", "-preset", cfg.preset,
        "-crf", str(cfg.intermediate_crf if intermediate else cfg.crf),
        "-pix_fmt", "yuv420p",
    ]
    if cfg.encoder_tune != "none":
        args.extend(["-tune", cfg.encoder_tune])
    # R-06: deliver limited range explicitly. The still links convert to limited at the
    # source, and this tag is what makes the player read the frames that way instead of
    # inferring the range from the picture.
    args.extend(["-color_range", "tv"])
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
    # --- the rest of what xfade offers, grouped the way an editor would think -----
    # A shape opens over the cut, the way a mask on an adjustment layer would.
    "mask": ("circlecrop", "rectcrop"),
    # The frame comes apart, or comes back together.
    "open": ("vertopen", "horzopen"),
    "close": ("vertclose", "horzclose"),
    # Like ``slide``, but the incoming shot eases in instead of shoving.
    "smooth": ("smoothleft", "smoothright", "smoothup", "smoothdown"),
    # Diagonal corners of a wipe, for cuts that want to move off-axis.
    "corner": ("wipetl", "wipetr", "wipebl", "wipebr"),
    # A fast rolling slice - reads as a shutter rather than as a wipe.
    "wind": ("hlwind", "hrwind", "vuwind", "vdwind"),
    # Weightless: long, low-contrast, and best saved for the end of a section.
    "soft": ("distance", "fadeslow"),
}

# Families that are not xfade at all. These are applied inside each shot's own frames
# - there is no overlap - so they read as a flash or a glitch *on* the cut rather than
# as a blend across it. That is the difference between a dissolve and an impact.
_EFFECT_TRANSITIONS: dict[str, float] = {
    "glitch": .30,
    "light_leak": .34,
    "film_burn": .30,
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
    "smooth": ("smoothleft", "smoothright"),
    "corner": ("wipetl", "wipebr"),
    "wind": ("hlwind", "hrwind"),
}

# How long each family wants to take. An impact transition has to be short or it
# stops reading as impact; the soft ones need room to breathe.
_TRANSITION_PACE: dict[str, float] = {
    "flash": .16, "zoom": .20, "pixel": .22, "wind": .22, "squeeze": .24,
    "slice": .26, "radial": .26, "wipe": .28, "slide": .28, "corner": .28,
    "smooth": .30, "diag": .30, "reveal": .32, "mask": .34, "dip": .36,
    "open": .38, "close": .38, "circle": .42, "blur": .44, "dissolve": .44,
    "soft": .50,
}


@dataclass(frozen=True, slots=True)
class _Transition:
    """One cut in the timeline, and how it should be realised.

    ``xfade`` needs the *incoming* shot to render extra footage: the fade's first input
    is everything joined so far and its second is the incoming clip, which the blend
    shows from its own frame 0 - so the incoming clip has to fill the overlap as well as
    its own duration. A ``cut`` and an ``effect`` transition both need none: the effect
    lives inside the frames each shot already has.
    """

    kind: str
    name: str
    seconds: float

    @property
    def handle(self) -> float:
        return self.seconds if self.kind == "xfade" else 0.0

    @property
    def visible(self) -> bool:
        return self.kind != "cut"


_CUT = _Transition("cut", "cut", 0.0)


def _transition_spec(
    shot: Shot, following: Shot, art: ArtDirection, cfg: RenderConfig,
) -> _Transition:
    """Resolve the planner's transition family into something the renderer can build.

    The planner picks *what* the edit needs - a wipe, a dip, a burst - and this
    decides how to realise it, using the tone of the incoming shot and the outgoing
    shot's drift direction.
    """
    family = shot.transition
    if family in {"cut", "none", ""}:
        return _CUT
    if family in _EFFECT_TRANSITIONS:
        seconds = _clamp_transition(_EFFECT_TRANSITIONS[family], cfg, shot, following)
        return _Transition("effect", family, seconds)
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

    return _Transition("xfade", name, _clamp_transition(duration, cfg, shot, following))


def _clamp_transition(duration: float, cfg: RenderConfig, shot: Shot, following: Shot) -> float:
    """Keep a transition inside the configured bounds and inside both of its shots."""
    return round(max(
        cfg.transition_min_seconds,
        min(cfg.transition_max_seconds, duration, shot.duration / 3, following.duration / 3),
    ), 3)


def _transition_effect_filters(
    transition: _Transition, side: str, duration: float, cfg: RenderConfig,
) -> list[str]:
    """Filters that turn a hard cut into one of the effect transitions.

    ``side`` is ``"out"`` for the outgoing shot's tail or ``"in"`` for the incoming
    shot's head. Everything here is timeline-gated, so the shot is untouched outside
    the window around the cut - the point is a flash *on* the beat, not a look for
    the whole shot.

    The dip is timed to *arrive* at its colour on the boundary frame. ``fade`` reaches
    full colour at ``st + d``, so the outgoing side starts one frame interval early:
    stopping at ``duration`` would leave the last frame only part of the way there,
    and a flash that peaks at 30% is just a slight brightening.
    """
    if transition.kind != "effect":
        return []
    seconds = min(transition.seconds, duration / 2)
    interval = 1 / cfg.fps
    texture = (
        f"enable='gte(t,{max(0.0, duration - seconds):.4f})'" if side == "out"
        else f"enable='lte(t,{seconds:.4f})'"
    )

    if transition.name == "glitch":
        # Channel separation and sensor noise, then a hard flash into the cut. The
        # split is what makes it read as a broken signal rather than as a blur.
        shift = max(2, round(cfg.width * .0035))
        flash = min(.12, max(2 * interval, seconds / 3))
        return [
            f"rgbashift=rh={shift}:bh={-shift}:edge=wrap:{texture}",
            f"noise=alls=26:allf=t:{texture}",
            _dip(side, duration, flash, interval, "white"),
        ]
    if transition.name == "light_leak":
        # A warm bloom washing over the cut, the way light gets in at the edge of a
        # lens. No texture and no hard flash: the colour is the whole effect, and it
        # wants the full window to arrive.
        return [_dip(side, duration, seconds, interval, "0xFFB060")]
    if transition.name == "film_burn":
        # A hot flash plus grain: the frame looks briefly overexposed and dirty, the
        # way a splice does when it catches.
        flash = min(.15, max(3 * interval, seconds / 2))
        return [
            f"noise=alls=30:allf=t:{texture}",
            _dip(side, duration, flash, interval, "0xFFE8C0"),
        ]
    return []


def _dip(side: str, duration: float, seconds: float, interval: float, colour: str) -> str:
    """A fade to ``colour`` that reaches it exactly on the frame at the cut."""
    if side == "out":
        start = max(0.0, duration - seconds - interval)
        return f"fade=t=out:st={start:.4f}:d={seconds:.4f}:color={colour}"
    return f"fade=t=in:st=0:d={seconds:.4f}:color={colour}"



def _compose_transitions(
    shots: list[Shot], clips: Path, output: Path,
    transitions: list[_Transition], cfg: RenderConfig,
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
    # `shots[i].start` is the edit's own timeline, so every transition is anchored to it
    # rather than to the running total of measured clip lengths. A clip is a whole number
    # of frames and `trim=duration` rounds *up*, so each one came out up to a frame longer
    # than the shot it stands for; summed over 253 shots that pushed the picture 0.65s
    # past the plan, and `-shortest` quietly cut the difference off the end of the film
    # (the last shot lost a third of its length). Anchoring keeps each shot's rounding
    # error to itself instead of letting it accumulate.
    for index, transition in enumerate(transitions):
        label = f"[x{index + 1}]"
        start = shots[index + 1].start
        if transition.kind == "xfade" and transition.seconds > 0:
            # xfade discards whatever the accumulated stream holds past
            # `offset + seconds`, so moving the offset back to the plan's position needs
            # no trim. The upper bound is what the real stream can offer - an offset past
            # it is exactly what xfade rejects as invalid.
            latest = timeline - transition.seconds - .001
            offset = max(0.0, min(start - transition.seconds, latest))
            filters.append(
                f"{current}[v{index + 1}]xfade=transition={transition.name}"
                f":duration={transition.seconds}:offset={round(offset, 3)}{label}"
            )
            timeline = offset + actual[index + 1]
        else:
            # A cut, and every effect transition: the flash, glitch or leak was already
            # burned into the two shots' own frames, so the timeline just butts them
            # together. A concat cannot shift a stream, so an accumulated stream that
            # frame rounding has pushed past the plan's position is trimmed back to it
            # first.
            if timeline > start + 1e-6:
                filters.append(f"{current}trim=end={start:.3f},setpts=PTS-STARTPTS[c{index}]")
                current = f"[c{index}]"
                timeline = start
            filters.append(f"{current}[v{index + 1}]concat=n=2:v=1:a=0{label}")
            timeline += actual[index + 1]
        current = label
    end_fade = min(.5, shots[-1].duration / 3)
    filters.append(f"{current}fade=t=in:st=0:d=0.25,fade=t=out:st={max(0, timeline - end_fade):.3f}:d={end_fade:.3f}[vout]")
    # Windows caps a single command line at 32,767 characters, and a beat-cut edit runs
    # long: 253 shots put ~27k characters of filter graph and ~9k of `-i` paths on one
    # line, which fails with `FileNotFoundError: [WinError 206]` before ffmpeg is even
    # started. The graph therefore goes into a file and is pulled in with the generic
    # "read this option's value from a file" form, ``-/opt file``. The dedicated
    # ``-filter_complex_script`` that used to do this is gone in ffmpeg 9, and the input
    # list stays inline because it is a third of the size and a few hundred shots fit.
    script = output.with_suffix(".filter")
    script.write_text(";\n".join(filters), encoding="utf-8")
    args += [
        "-/filter_complex", str(script), "-map", "[vout]", "-an",
        "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output),
    ]
    command(args)



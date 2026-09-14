from __future__ import annotations

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
            f"color=c=white@0.22:s={gap}x{cfg.height}:r={cfg.fps}:d={duration:.4f}[divider];"
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

    _adapt_image(filters, 0, "adapted", cfg.width, cfg.height, cfg)
    frames = max(1, round(duration * cfg.fps))
    intensity = max(.35, art.camera_intensity) * (1 + shot.melody * .18)
    amount = ({"pan_reveal": .075, "focus_pull": .045}.get(effect, .055)
              * intensity * ({"dynamic": 1.35, "gentle": .7}.get(shot.motion, 1.0)))
    if shot.edit_intent == "breathe" or shot.section == "outro":
        zoom = f"max(1+{amount:.5f}-on/{frames}*{amount:.5f},1)"
    else:
        zoom = f"min(1+on/{frames}*{amount:.5f},{1 + amount:.5f})"
    direction = -1 if (shot.media_id + max(0, shot.section_index)) % 2 else 1
    pan = .18 if effect == "pan_reveal" else .06
    progress = f"on/{max(1, frames - 1)}"
    # ``perspective`` is used instead of ``zoompan`` because zoompan truncates the
    # crop origin to whole pixels of its input frame. The camera move here is only
    # a fraction of a pixel per frame, so zoompan holds the image still for several
    # frames and then snaps it by a whole pixel - a visible stutter on any straight
    # edge. Feeding it a supersampled frame only shrinks the jump; perspective
    # evaluates the crop rectangle per frame with true sub-pixel interpolation, so
    # the move stays smooth and nothing is resampled twice.
    #
    # The crop is ``W/zoom`` wide and may travel ``W - W/zoom`` before leaving the
    # frame; the two are easy to confuse and swapping them turns the move into a
    # hard punch-in.
    crop_width = f"(W/({zoom}))"
    crop_height = f"(H/({zoom}))"
    travel = f"(W-W/({zoom}))"
    rise = f"(H-H/({zoom}))"
    origin_x = f"clip((W-W/({zoom}))/2+{direction}*{travel}*{pan}*({progress}-.5),0,W-W/({zoom}))"
    origin_y = f"clip((H-H/({zoom}))/2-{rise}*.035*{progress},0,H-H/({zoom}))"
    filters.append(
        f"[adapted]perspective="
        f"x0='{origin_x}':y0='{origin_y}':"
        f"x1='{origin_x}+{crop_width}':y1='{origin_y}':"
        f"x2='{origin_x}':y2='{origin_y}+{crop_height}':"
        f"x3='{origin_x}+{crop_width}':y3='{origin_y}+{crop_height}':"
        f"interpolation=cubic:sense=source:eval=frame[composite]"
    )
    return filters, "[composite]"


def _adapt_image(
    filters: list[str], input_index: int, label: str,
    width: int, height: int, cfg: RenderConfig,
) -> None:
    """Fit without distortion and fill any letterbox area with a blurred copy."""
    foreground_width = max(2, round(width * cfg.image_foreground_scale / 2) * 2)
    foreground_height = max(2, round(height * cfg.image_foreground_scale / 2) * 2)
    blur = cfg.image_background_blur if cfg.blurred_image_background else 0
    background_effect = f"gblur=sigma={blur:.2f},eq=brightness=-.055:saturation=.88" if blur > 0 else "null"
    filters.append(
        f"[{input_index}:v]split=2[{label}bgsrc][{label}fgsrc];"
        f"[{label}bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},{background_effect},setsar=1[{label}bg];"
        f"[{label}fgsrc]scale={foreground_width}:{foreground_height}:"
        f"force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[{label}fg];"
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


def _transition_spec(shot: Shot, following: Shot, art: ArtDirection, cfg: RenderConfig) -> tuple[str, float]:
    tone = following.transition_tone if following.transition_tone != "neutral" else art.transition_tone
    if shot.transition == "cut":
        return "cut", 0.0
    if shot.transition == "flash":
        name, duration = ("fadewhite", .14) if tone == "bright" else ("smoothleft", .18)
    elif shot.transition == "dip":
        name, duration = ("fadeblack", .3) if tone in {"dark", "neutral"} else ("dissolve", .32)
    elif shot.transition == "dissolve":
        name, duration = "dissolve", .42
    elif following.section == "outro":
        name, duration = "fadeblack", .5
    elif tone == "soft" or art.mood == "dreamy":
        name, duration = "dissolve", .48
    elif tone == "dark" or art.mood in {"melancholic", "cinematic", "dark"}:
        name, duration = "fadeblack", .34
    else:
        name, duration = "fade", .24
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



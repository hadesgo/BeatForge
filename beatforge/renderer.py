from __future__ import annotations

from pathlib import Path

import numpy as np

from beatforge.config import RenderConfig
from beatforge.director import ArtDirection
from beatforge.lyrics import LyricLine, write_ass
from beatforge.planner import Shot
from beatforge.runtime import command


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
        melody_boost = 1 + shot.melody * .22
        zoom_amount = {"dynamic": .14, "gentle": .045}.get(shot.motion, .08) * art.camera_intensity * melody_boost
        progress = f"on/{max(1, frames - 1)}"
        direction = 1 if (shot.media_id + max(0, shot.section_index)) % 2 == 0 else -1
        pan_x = (
            f"clip((iw-iw/zoom)/2+{direction}*(iw-iw/zoom)*0.10*({progress}-.5),"
            "0,iw-iw/zoom)"
        )
        vertical_drift = .04 if shot.edit_intent == "breathe" else .015
        pan_y = (
            f"clip((ih-ih/zoom)/2-(ih-ih/zoom)*{vertical_drift}*{progress},"
            "0,ih-ih/zoom)"
        )
        if shot.edit_intent == "breathe" or shot.section == "outro":
            zoom = f"max(1+{zoom_amount}-on/{frames}*{zoom_amount},1)"
        else:
            zoom = f"min(1+on/{frames}*{zoom_amount},{1 + zoom_amount})"
        focus_x, focus_y = _safe_focus(shot.focus_point)
        visual = (f"scale={cfg.width * 2}:{cfg.height * 2}:force_original_aspect_ratio=increase:flags=lanczos,"
                  f"crop={cfg.width * 2}:{cfg.height * 2}:"
                  f"x='clip(iw*{focus_x}-ow/2,0,iw-ow)':y='clip(ih*{focus_y}-oh/2,0,ih-oh)',"
                  f"zoompan=z='{zoom}':"
                  f"x='{pan_x}':y='{pan_y}':d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps},setsar=1")
    else:
        # Preserve the source cinematography. Adding a synthetic sinusoidal pan to moving
        # footage creates the characteristic automated, seasick look.
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
    if shot.kind == "image":
        args += ["-loop", "1", "-framerate", str(cfg.fps)]
    else:
        args += ["-stream_loop", "-1", "-ss", str(shot.source_start)]
    args += ["-i", shot.file, "-t", str(render_duration), "-an", "-vf", ",".join(
        [visual, *(item for item in effects if item)]
    ), "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output)]
    command(args)


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
    current = "[v0]"
    timeline = shots[0].duration
    for index, (name, duration) in enumerate(transitions):
        label = f"[x{index + 1}]"
        if name == "cut" or duration <= 0:
            filters.append(f"{current}[v{index + 1}]concat=n=2:v=1:a=0{label}")
        else:
            filters.append(
                f"{current}[v{index + 1}]xfade=transition={name}:duration={duration}:offset={round(timeline, 3)}{label}"
            )
        current = label
        timeline += shots[index + 1].duration
    end_fade = min(.5, shots[-1].duration / 3)
    filters.append(f"{current}fade=t=in:st=0:d=0.25,fade=t=out:st={max(0, timeline - end_fade):.3f}:d={end_fade:.3f}[vout]")
    args += [
        "-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
        "-r", str(cfg.fps), *_video_encode_args(cfg, intermediate=True), str(output),
    ]
    command(args)

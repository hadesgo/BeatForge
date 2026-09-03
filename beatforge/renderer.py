from __future__ import annotations

from pathlib import Path

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
    for index, shot in enumerate(shots):
        print(f"\r渲染镜头 {index + 1}/{len(shots)}", end="", flush=True)
        handle = transitions[index][1] if index < len(transitions) else 0.0
        _render_shot(shot, clips / f"{index:05}.mp4", config, art, shot.duration + handle)
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
    args += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", config.preset,
             "-crf", str(config.crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
             "-shortest", "-movflags", "+faststart", str(output)]
    command(args)


def _render_shot(shot: Shot, output: Path, cfg: RenderConfig, art: ArtDirection, render_duration: float) -> None:
    frames = max(1, round(render_duration * cfg.fps))
    if shot.kind == "image":
        melody_boost = 1 + shot.melody * .22
        zoom_amount = {"dynamic": .14, "gentle": .045}.get(shot.motion, .08) * art.camera_intensity * melody_boost
        progress = f"on/{max(1, frames - 1)}"
        pan_x = (
            f"(iw-iw/zoom)*{progress}"
            if shot.index % 2 == 0 else f"(iw-iw/zoom)*(1-{progress})"
        )
        pan_y = "ih/2-ih/zoom/2"
        visual = (f"scale={cfg.width * 2}:{cfg.height * 2}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width * 2}:{cfg.height * 2},"
                  f"zoompan=z='min(1+on/{frames}*{zoom_amount},{1 + zoom_amount})':"
                  f"x='{pan_x}':y='{pan_y}':d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps}")
    else:
        # Preserve the source cinematography. Adding a synthetic sinusoidal pan to moving
        # footage creates the characteristic automated, seasick look.
        overscan = 1.025
        scaled_width, scaled_height = round(cfg.width * overscan / 2) * 2, round(cfg.height * overscan / 2) * 2
        visual = (f"scale={scaled_width}:{scaled_height}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width}:{cfg.height}:x='(iw-ow)/2':y='(ih-oh)/2'")
    grade = art.grade_filter
    effects: list[str] = []
    if cfg.visual_effects:
        if shot.motion == "dynamic":
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
    args += ["-i", shot.file, "-t", str(render_duration), "-an", "-vf", ",".join([visual, grade, *effects]),
             "-r", str(cfg.fps), "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
             "-pix_fmt", "yuv420p", str(output)]
    command(args)


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
        "-r", str(cfg.fps), "-c:v", "libx264", "-preset", cfg.preset,
        "-crf", str(cfg.crf), "-pix_fmt", "yuv420p", str(output),
    ]
    command(args)

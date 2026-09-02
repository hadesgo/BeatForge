from __future__ import annotations

from pathlib import Path

from beatforge.config import RenderConfig
from beatforge.lyrics import LyricLine, write_ass
from beatforge.planner import Shot
from beatforge.runtime import command


def render(shots: list[Shot], lyrics: list[LyricLine], music: Path, output: Path, cache: Path, config: RenderConfig) -> None:
    clips = cache / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, shot in enumerate(shots):
        print(f"\r渲染镜头 {index + 1}/{len(shots)}", end="", flush=True)
        _render_shot(shot, clips / f"{index:05}.mp4", config, index == len(shots) - 1)
    print()
    concat_file = cache / "clips.txt"
    concat_file.write_text("\n".join(f"file '{(clips / f'{i:05}.mp4').as_posix()}'" for i in range(len(shots))), "utf-8")
    picture = cache / "picture.mp4"
    command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(picture)])
    subtitle = cache / "lyrics.ass"
    write_ass(
        lyrics, subtitle, width=config.width, height=config.height,
        font=config.subtitle_font, size=config.subtitle_size,
        margin=config.subtitle_margin, effect=config.subtitle_effect,
        highlight_color=config.subtitle_highlight_color,
    )
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(music)]
    if lyrics:
        escaped = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        args += ["-vf", f"ass='{escaped}'"]
    args += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", config.preset,
             "-crf", str(config.crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
             "-shortest", "-movflags", "+faststart", str(output)]
    command(args)


def _render_shot(shot: Shot, output: Path, cfg: RenderConfig, is_last: bool) -> None:
    frames = max(1, round(shot.duration * cfg.fps))
    if shot.kind == "image":
        zoom_amount = {"dynamic": .14, "gentle": .045}.get(shot.motion, .08)
        pan_x = "iw/2-iw/zoom/2" if shot.index % 2 == 0 else "iw/2-iw/zoom/2+sin(on/28)*iw*0.012"
        pan_y = "ih/2-ih/zoom/2-cos(on/34)*ih*0.010" if shot.index % 3 == 0 else "ih/2-ih/zoom/2"
        visual = (f"scale={cfg.width * 2}:{cfg.height * 2}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width * 2}:{cfg.height * 2},"
                  f"zoompan=z='min(1+on/{frames}*{zoom_amount},{1 + zoom_amount})':"
                  f"x='{pan_x}':y='{pan_y}':d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps}")
    else:
        overscan = 1.08 if shot.motion == "dynamic" else 1.04
        scaled_width, scaled_height = round(cfg.width * overscan / 2) * 2, round(cfg.height * overscan / 2) * 2
        drift = max(2, round((scaled_width - cfg.width) * .38))
        visual = (f"scale={scaled_width}:{scaled_height}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width}:{cfg.height}:x='(iw-ow)/2+sin(n/32)*{drift}':"
                  f"y='(ih-oh)/2+cos(n/41)*{max(2, drift // 2)}'")
    grade = "eq=contrast=1.06:saturation=1.08"
    fade = .07 if shot.transition == "flash" else .2 if shot.transition == "dip" else .12
    fade_color = "white" if shot.transition == "flash" else "black"
    transitions = [f"fade=t=in:st=0:d={fade}:color={fade_color}"]
    out_fade = min(.5 if is_last else fade, shot.duration / 3)
    transitions.append(f"fade=t=out:st={max(0, shot.duration - out_fade)}:d={out_fade}:color={fade_color}")
    effects: list[str] = []
    if cfg.visual_effects:
        if shot.motion == "dynamic":
            effects.append("unsharp=5:5:0.55:5:5:0")
        elif shot.motion == "gentle":
            effects.append("gblur=sigma=0.18")
        if cfg.vignette:
            effects.append("vignette=PI/5")
        if cfg.film_grain > 0:
            effects.append(f"noise=alls={cfg.film_grain}:allf=t+u")
    args = ["ffmpeg", "-y", "-v", "error"]
    if shot.kind == "image":
        args += ["-loop", "1", "-framerate", str(cfg.fps)]
    else:
        args += ["-stream_loop", "-1", "-ss", str(shot.source_start)]
    args += ["-i", shot.file, "-t", str(shot.duration), "-an", "-vf", ",".join([visual, grade, *effects, *transitions]),
             "-r", str(cfg.fps), "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
             "-pix_fmt", "yuv420p", str(output)]
    command(args)

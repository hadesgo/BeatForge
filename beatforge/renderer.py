from __future__ import annotations

from pathlib import Path

from beatforge.config import RenderConfig
from beatforge.lyrics import LyricLine, write_srt
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
    srt = cache / "lyrics.srt"
    write_srt(lyrics, srt)
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(music)]
    if lyrics:
        escaped = srt.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
        style = f"FontName={config.subtitle_font},FontSize={config.subtitle_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,Outline=2.2,Alignment=2,MarginV=72"
        args += ["-vf", f"subtitles='{escaped}':charenc=UTF-8:force_style='{style}'"]
    args += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", config.preset,
             "-crf", str(config.crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
             "-shortest", "-movflags", "+faststart", str(output)]
    command(args)


def _render_shot(shot: Shot, output: Path, cfg: RenderConfig, is_last: bool) -> None:
    frames = max(1, round(shot.duration * cfg.fps))
    if shot.kind == "image":
        zoom_amount = {"dynamic": .14, "gentle": .045}.get(shot.motion, .08)
        visual = (f"scale={cfg.width * 2}:{cfg.height * 2}:force_original_aspect_ratio=increase,"
                  f"crop={cfg.width * 2}:{cfg.height * 2},"
                  f"zoompan=z='min(1+on/{frames}*{zoom_amount},{1 + zoom_amount})':"
                  f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps}")
    else:
        visual = f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,crop={cfg.width}:{cfg.height}"
    grade = "eq=contrast=1.06:saturation=1.08"
    fade = .07 if shot.transition == "flash" else .2 if shot.transition == "dip" else .12
    transitions = [f"fade=t=in:st=0:d={fade}"]
    out_fade = min(.5 if is_last else fade, shot.duration / 3)
    transitions.append(f"fade=t=out:st={max(0, shot.duration - out_fade)}:d={out_fade}")
    args = ["ffmpeg", "-y", "-v", "error"]
    if shot.kind == "image":
        args += ["-loop", "1", "-framerate", str(cfg.fps)]
    else:
        args += ["-stream_loop", "-1", "-ss", str(shot.source_start)]
    args += ["-i", shot.file, "-t", str(shot.duration), "-an", "-vf", ",".join([visual, grade, *transitions]),
             "-r", str(cfg.fps), "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
             "-pix_fmt", "yuv420p", str(output)]
    command(args)

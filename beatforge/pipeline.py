from __future__ import annotations

import json
from pathlib import Path

from beatforge.audio import analyze_music
from beatforge.config import ProjectConfig
from beatforge.director import create_art_direction
from beatforge.editing import resolve_style
from beatforge.lyrics import read_lrc, write_srt
from beatforge.media import discover_media
from beatforge.planner import create_plan
from beatforge.renderer import render
from beatforge.runtime import (
    duration,
    optimize_torch_runtime,
    release_gpu,
    require_binaries,
    require_usable_ai_device,
    resolve_device,
)


def speech_source(project: ProjectConfig, device: str) -> tuple[Path, str]:
    """The audio the recogniser should listen to, and a label for the log.

    An ASR model asked to transcribe a full mix is hearing a voice through a drum kit,
    and its failure mode is not silence - it is a confident transcript with the wrong
    words where the arrangement is dense. Separating first hands it the stem it is good
    at, and the same stem makes forced alignment easier for the same reason.

    Separation is a heavy optional dependency, so a missing extra falls back to the mix
    instead of failing the run. It says so loudly, because a transcript taken from the
    mix is a different and worse transcript and nothing downstream can tell.
    """
    if not project.ai.separate_vocals:
        return project.music, "原始混音（未做人声分离）"
    from beatforge.models.separator import SeparationUnavailable, separate_vocals

    try:
        stem = separate_vocals(
            project.music, project.cache_dir,
            model=project.ai.separation_model,
            device=device, offline=project.ai.offline,
        )
    except SeparationUnavailable as exc:
        return project.music, f"原始混音（人声分离不可用：{exc}）"
    return stem, f"人声轨 {stem.name}"


def run_project(project: ProjectConfig, *, plan_only: bool = False, no_ai: bool = False) -> Path:
    require_binaries()
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    if not project.music.exists():
        raise FileNotFoundError(f"音乐文件不存在: {project.music}")
    use_ai = project.ai.enabled and not no_ai
    device = resolve_device(project.ai.device)
    if use_ai:
        require_usable_ai_device(project.ai.device, device)
    optimize_torch_runtime(device)
    from beatforge.models.downloader import load_download_manifest
    downloaded = load_download_manifest(project.cache_dir / "models.json")

    def model_path(repo_id: str) -> str:
        return downloaded.get(repo_id, repo_id)
    total_duration = duration(project.music)

    print(f"1/5 歌词时间轴 · {'本地 AI / ' + device if use_ai else 'LRC'}")
    if project.lyrics and project.lyrics.exists():
        lyrics = read_lrc(project.lyrics, total_duration)
    elif use_ai:
        from beatforge.models.transcriber import transcribe
        source, source_label = speech_source(project, device)
        print(f"    {source_label}")
        lyrics = transcribe(
            source,
            qwen_model=model_path(project.ai.qwen_asr_model),
            qwen_aligner=model_path(project.ai.qwen_aligner_model),
            device=device, offline=project.ai.offline,
        )
        write_srt(lyrics, project.cache_dir / "asr.srt")
        release_gpu()
    else:
        lyrics = []
    print(f"    {len(lyrics)} 句")

    print("2/5 音乐结构与氛围分析")
    mood_scores = None
    structure = None
    if use_ai:
        from beatforge.models.audio_semantics import classify_music
        mood_scores = classify_music(
            project.music, model_name=model_path(project.ai.clap_model),
            device=device, offline=project.ai.offline,
        )
        release_gpu()
        if project.ai.music_structure_backend != "librosa":
            from beatforge.models.music_structure import analyze_beats
            structure = analyze_beats(project.music, project.ai.music_structure_backend, device)
            release_gpu()
    analysis = analyze_music(project.music, mood_scores, structure)
    print(f"    {analysis.bpm:.1f} BPM · {analysis.mood} · {len(analysis.sections) - 1} 个章节")

    print("3/5 素材视觉语义索引")
    assets = discover_media(project.media_dir)
    similarities = None
    source_starts = None
    if use_ai and lyrics:
        from beatforge.models.vision_index import VisionIndex
        index = VisionIndex(
            model_path(project.ai.vision_model), device, project.ai.offline, project.cache_dir,
            backend=project.ai.vision_backend,
            reranker_model=model_path(project.ai.vision_reranker_model) if project.ai.vision_reranker_model else None,
            rerank_top_k=project.ai.vision_rerank_top_k,
            batch_size=project.ai.vision_batch_size,
            input_pixels=project.ai.vision_input_pixels,
        )
        similarities = index.similarities([line.text for line in lyrics], assets, project.ai.frame_samples)
        source_starts = getattr(index, "best_source_starts", None)
        del index
        release_gpu()
    print(f"    {len(assets)} 个素材 · {project.ai.vision_backend if similarities is not None else '文件标签'}")

    print("4/5 AI 导演与镜头编排")
    treatment = None
    if use_ai and project.ai.director_enabled:
        try:
            from beatforge.models.ai_director import direct_mv
            director_config = project.ai.model_copy(update={
                "director_model": model_path(project.ai.director_model),
            })
            treatment = direct_mv(
                analysis, lyrics, assets, similarities, director_config, project.cache_dir,
                source_starts,
            )
            print(f"    导演概念：{treatment.concept}")
        except Exception as exc:
            print(f"    本地导演不可用，使用规则导演：{type(exc).__name__}: {exc}")
        finally:
            release_gpu()
    style = resolve_style(
        project.render.edit_style, analysis.mood,
        average_energy=analysis.average_energy,
        rhythmic_density=analysis.rhythmic_density,
    )
    if style is not None:
        print(f"    剪辑风格：{style.label} · {style.summary}")
    # A style owns the settings it has an opinion about; the render config keeps the
    # ones it does not, and everything the style does not name is passed through as-is.
    render_config = project.render if style is None else project.render.model_copy(update={
        "subtitle_layout": style.subtitle_layout,
    })
    shots = create_plan(
        analysis, lyrics, assets, similarities,
        min_shot=project.render.min_shot_seconds,
        max_shot=project.render.max_shot_seconds,
        treatment=treatment,
        source_starts=source_starts,
        target_width=project.render.width,
        target_height=project.render.height,
        image_composites=project.render.image_composites,
        image_composite_ratio=project.render.image_composite_ratio,
        max_composite_images=project.render.max_composite_images,
        avoid_asset_repeats=project.render.avoid_asset_repeats,
        transition_density=project.render.transition_density,
        style=style,
    )
    art = create_art_direction(analysis, lyrics, project.render, treatment, style)
    plan_file = project.cache_dir / "plan.json"
    plan = {
        "version": 3,
        "models": {
            "asr": project.ai.qwen_asr_model if use_ai else None,
            "aligner": project.ai.qwen_aligner_model if use_ai else None,
            # Recorded so a plan can be traced back to the audio the words came from: a
            # transcript off the mix and one off the vocal stem are different edits.
            "separation": (
                project.ai.separation_model
                if use_ai and project.ai.separate_vocals and not (project.lyrics and project.lyrics.exists())
                else None
            ),
            "audio": project.ai.clap_model if use_ai else None,
            "vision": project.ai.vision_model if similarities is not None else None,
            "vision_reranker": project.ai.vision_reranker_model if similarities is not None else None,
            "director": project.ai.director_model if treatment is not None else None,
        },
        "render": project.render.model_dump(mode="json"),
        "art_direction": art.as_dict(),
        "director_treatment": treatment.model_dump(mode="json") if treatment else None,
        "analysis": analysis.as_dict(),
        "lyrics": [line.as_dict() for line in lyrics],
        "media": [asset.as_dict() for asset in assets],
        "shots": [shot.as_dict() for shot in shots],
    }
    plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), "utf-8")
    print(f"    {len(shots)} 个镜头 · {plan_file}")
    if plan_only:
        return plan_file

    print("5/5 FFmpeg 成片渲染")
    render(shots, lyrics, project.music, project.output, project.cache_dir, render_config, art)
    return project.output

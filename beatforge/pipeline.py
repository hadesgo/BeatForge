from __future__ import annotations

import json
from pathlib import Path

from beatforge.audio import analyze_music
from beatforge.config import ProjectConfig
from beatforge.director import create_art_direction
from beatforge.lyrics import read_lrc, write_srt
from beatforge.media import discover_media
from beatforge.planner import create_plan
from beatforge.renderer import render
from beatforge.runtime import duration, optimize_torch_runtime, release_gpu, require_binaries, resolve_device


def run_project(project: ProjectConfig, *, plan_only: bool = False, no_ai: bool = False) -> Path:
    require_binaries()
    project.cache_dir.mkdir(parents=True, exist_ok=True)
    if not project.music.exists():
        raise FileNotFoundError(f"音乐文件不存在: {project.music}")
    device = resolve_device(project.ai.device)
    optimize_torch_runtime(device)
    use_ai = project.ai.enabled and not no_ai
    total_duration = duration(project.music)

    print(f"1/5 歌词时间轴 · {'本地 AI / ' + device if use_ai else 'LRC'}")
    if project.lyrics and project.lyrics.exists():
        lyrics = read_lrc(project.lyrics, total_duration)
    elif use_ai:
        from beatforge.models.transcriber import transcribe
        lyrics = transcribe(
            project.music, backend=project.ai.asr_backend,
            qwen_model=project.ai.qwen_asr_model,
            qwen_aligner=project.ai.qwen_aligner_model,
            whisper_model=project.ai.whisper_model, device=device,
            compute_type=project.ai.whisper_compute_type, offline=project.ai.offline,
        )
        write_srt(lyrics, project.cache_dir / "whisper.srt")
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
            project.music, model_name=project.ai.clap_model,
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
            project.ai.vision_model, device, project.ai.offline, project.cache_dir,
            backend=project.ai.vision_backend,
            reranker_model=project.ai.vision_reranker_model,
            rerank_top_k=project.ai.vision_rerank_top_k,
            quantization=project.ai.vision_quantization,
            batch_size=project.ai.vision_batch_size,
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
            treatment = direct_mv(
                analysis, lyrics, assets, similarities, project.ai, device, project.cache_dir,
                source_starts,
            )
            print(f"    导演概念：{treatment.concept}")
        except Exception as exc:
            print(f"    本地导演不可用，使用规则导演：{type(exc).__name__}: {exc}")
        finally:
            release_gpu()
    shots = create_plan(
        analysis, lyrics, assets, similarities,
        min_shot=project.render.min_shot_seconds,
        max_shot=project.render.max_shot_seconds,
        treatment=treatment,
        source_starts=source_starts,
    )
    art = create_art_direction(analysis, lyrics, project.render, treatment)
    plan_file = project.cache_dir / "plan.json"
    plan = {
        "version": 2,
        "models": {
            "asr": project.ai.qwen_asr_model if use_ai and project.ai.asr_backend == "qwen3" else project.ai.whisper_model if use_ai else None,
            "aligner": project.ai.qwen_aligner_model if use_ai and project.ai.asr_backend == "qwen3" else None,
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
    render(shots, lyrics, project.music, project.output, project.cache_dir, project.render, art)
    return project.output

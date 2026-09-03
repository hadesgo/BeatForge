from __future__ import annotations

import json
import gc
import math
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, ValidationError

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.runtime import command


SubtitleEffect = Literal["karaoke", "cinematic", "bounce", "float", "glow", "typewriter"]


class SectionDirection(BaseModel):
    section_index: int = Field(ge=0)
    narrative_role: str = Field(min_length=1, max_length=160)
    lyric_relation: Literal["literal", "metaphorical", "emotional", "contrast", "abstract"] = "emotional"
    cut_intensity: float = Field(default=.5, ge=0, le=1)
    preferred_media: Literal["any", "image", "video"] = "any"
    preferred_shot_sizes: list[Literal["wide", "medium", "closeup", "detail", "unknown"]] = Field(
        default_factory=list, max_length=3,
    )
    preferred_asset_ids: list[int] = Field(default_factory=list, max_length=5)
    subtitle_effect: SubtitleEffect = "cinematic"
    transition_tone: Literal["bright", "dark", "soft", "neutral"] = "neutral"
    edit_intent: Literal["continuity", "impact", "breathe"] = "continuity"


class DirectorTreatment(BaseModel):
    concept: str = Field(min_length=1, max_length=300)
    narrative_arc: str = Field(min_length=1, max_length=500)
    visual_style: str = Field(min_length=1, max_length=240)
    color_arc: list[str] = Field(default_factory=list, max_length=8)
    motif_asset_ids: list[int] = Field(default_factory=list, max_length=5)
    grade_profile: Literal["energetic", "uplifting", "melancholic", "dreamy", "romantic", "dark", "cinematic"]
    transition_tone: Literal["bright", "dark", "soft", "neutral"] = "neutral"
    sections: list[SectionDirection] = Field(default_factory=list)

    def section(self, index: int) -> SectionDirection | None:
        return next((item for item in self.sections if item.section_index == index), None)


SYSTEM_PROMPT = """你是一位经验丰富的音乐录影带导演和剪辑指导。根据已经完成的音乐分析、逐句歌词、素材元数据和视觉检索候选，制定一份可执行的导演方案。
要求：保持全片统一的视觉母题和色彩发展；主歌重视叙事连续性，副歌建立可识别的视觉记忆点，桥段创造反差，结尾留有呼吸；歌词与画面可以直译、隐喻、情绪呼应或有意对照；不要虚构不存在的素材 ID；不要输出时间码或 FFmpeg 命令。只返回符合 JSON Schema 的数据。"""


def direct_mv(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    config: AIConfig,
    device: str,
    cache_dir: Path,
    source_starts: np.ndarray | None = None,
) -> DirectorTreatment:
    context = _build_context(analysis, lyrics, assets, similarities)
    visual_reference = _build_contact_sheet(
        assets, similarities, cache_dir, config.director_contact_sheet_assets, source_starts,
    )
    treatment = _generate_treatment(context, config, device, cache_dir, visual_reference)
    return _sanitize(treatment, len(analysis.sections) - 1, {asset.id for asset in assets})


def _generate_treatment(
    context: dict,
    config: AIConfig,
    device: str,
    cache_dir: Path,
    visual_reference: Path | None = None,
) -> DirectorTreatment:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("AI 导演需要 ai 与 ai-cpu/ai-cuda extra") from exc

    offload_dir = cache_dir / "director-offload"
    load_options: dict = {
        "device_map": "auto" if device == "cuda" else {"": "cpu"},
        "dtype": "auto",
        "local_files_only": config.offline,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        gpu_limit = min(config.director_gpu_memory_gb, max(1.0, total_gb - 1.5))
        load_options["max_memory"] = {0: f"{gpu_limit:.1f}GiB", "cpu": f"{config.director_cpu_memory_gb:.1f}GiB"}
        if config.director_offload:
            offload_dir.mkdir(parents=True, exist_ok=True)
            load_options.update({"offload_folder": str(offload_dir), "offload_state_dict": True})

    processor = None
    model = None
    try:
        processor = AutoProcessor.from_pretrained(config.director_model, local_files_only=config.offline)
        model = AutoModelForMultimodalLM.from_pretrained(config.director_model, **load_options)
        model.eval()
        schema = json.dumps(DirectorTreatment.model_json_schema(), ensure_ascii=False)
        project_text = (
            f"JSON Schema:\n{schema}\n\n项目数据:\n{json.dumps(context, ensure_ascii=False)}"
        )
        user_content: str | list[dict] = project_text
        if visual_reference is not None:
            user_content = [
                {"type": "image", "image": str(visual_reference)},
                {"type": "text", "text": (
                    "上图是候选素材联系表，画面左上角编号对应项目数据里的素材 ID。"
                    "请同时判断构图、主体、景别、色彩、镜头之间的视觉连续性和歌词意境。\n\n"
                    + project_text
                )},
            ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        raw = _generate(model, processor, messages, config, torch)
        try:
            return DirectorTreatment.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "上一个结果未通过校验。修正后只返回完整 JSON，不要解释。"
                    f"\n校验错误：{exc}"
                )},
            ])
            corrected = _generate(model, processor, messages, config, torch)
            return DirectorTreatment.model_validate_json(_extract_json(corrected))
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _generate(model, processor, messages: list[dict], config: AIConfig, torch) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generation = {
        "max_new_tokens": config.director_max_new_tokens,
        "do_sample": config.director_temperature > 0,
    }
    if config.director_temperature > 0:
        generation.update({"temperature": config.director_temperature, "top_p": .85})
    with torch.inference_mode():
        output = model.generate(**inputs, **generation)
    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def _extract_json(content: str) -> str:
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def _build_context(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
) -> dict:
    candidate_ids = _candidate_ids(assets, similarities)
    return {
        "song": {
            "duration": analysis.duration,
            "bpm": analysis.bpm,
            "mood": analysis.mood,
            "mood_scores": analysis.mood_scores,
            "average_energy": analysis.average_energy,
            "melodic_motion": analysis.melodic_motion,
            "rhythmic_density": analysis.rhythmic_density,
        },
        "sections": [
            {
                "index": index,
                "label": analysis.section_labels[index] if index < len(analysis.section_labels) else "unknown",
                "start": start,
                "end": end,
                "energy": round(analysis.energy_at((start + end) / 2), 3),
            }
            for index, (start, end) in enumerate(zip(analysis.sections, analysis.sections[1:]))
        ],
        "lyrics": [
            {"start": line.start, "end": line.end, "text": line.text}
            for line in lyrics
        ],
        "assets": [
            {
                "id": asset.id,
                "kind": asset.kind,
                "description": asset.description,
                "tags": asset.tags[:12],
                "mood": asset.mood,
                "quality": asset.quality_score,
                "shot_size": asset.shot_size,
                "camera_motion": asset.camera_motion,
                "dominant_color": asset.dominant_color,
            }
            for asset in assets if asset.id in candidate_ids
        ],
        "instruction": "为每个 section index 提供一项导演策略；素材选择只能使用 assets 中出现的 id。",
    }


def _candidate_ids(assets: list[MediaAsset], similarities: np.ndarray | None, limit: int = 60) -> set[int]:
    if len(assets) <= limit:
        return {asset.id for asset in assets}
    semantic = np.max(similarities, axis=0) if similarities is not None and similarities.size else np.zeros(len(assets))
    ranked = sorted(
        enumerate(assets),
        key=lambda item: float(semantic[item[0]]) * .75 + item[1].quality_score * .25,
        reverse=True,
    )
    return {asset.id for _, asset in ranked[:limit]}


def _build_contact_sheet(
    assets: list[MediaAsset], similarities: np.ndarray | None, cache_dir: Path, limit: int,
    source_starts: np.ndarray | None = None,
) -> Path | None:
    """Build one compact visual reference so the director judges actual footage, not filenames."""
    if limit <= 0 or not assets:
        return None
    semantic = (
        np.max(similarities, axis=0)
        if similarities is not None and similarities.size else np.zeros(len(assets))
    )
    ranked = sorted(
        enumerate(assets),
        key=lambda item: float(semantic[item[0]]) * .7 + item[1].quality_score * .3,
        reverse=True,
    )[:limit]
    still_dir = cache_dir / "director-stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[MediaAsset, Image.Image]] = []
    for asset_column, asset in ranked:
        try:
            source = asset.file
            if asset.kind == "video":
                source = still_dir / f"asset-{asset.id}.jpg"
                if not source.exists():
                    timestamp = max(0.0, asset.duration * .5 - .05)
                    if source_starts is not None and source_starts.size:
                        timestamp = float(np.median(source_starts[:, asset_column]))
                    command([
                        "ffmpeg", "-y", "-v", "error", "-ss", f"{timestamp:.3f}",
                        "-i", str(asset.file), "-frames:v", "1", "-vf", "scale=640:-2",
                        str(source),
                    ])
            with Image.open(source) as opened:
                tiles.append((asset, opened.convert("RGB").copy()))
        except (OSError, RuntimeError, subprocess.SubprocessError):
            continue
    if not tiles:
        return None
    tile_width, tile_height, label_height, columns = 320, 180, 28, 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (tile_width * columns, (tile_height + label_height) * rows), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)
    for index, (asset, image) in enumerate(tiles):
        x = index % columns * tile_width
        y = index // columns * (tile_height + label_height)
        fitted = ImageOps.contain(image, (tile_width, tile_height))
        sheet.paste(fitted, (x + (tile_width - fitted.width) // 2, y + (tile_height - fitted.height) // 2))
        draw.rectangle((x, y, x + 86, y + 24), fill=(0, 0, 0))
        draw.text((x + 7, y + 5), f"ID {asset.id}  {asset.kind}", fill=(255, 220, 72))
    target = cache_dir / "director-contact-sheet.jpg"
    sheet.save(target, quality=90, optimize=True)
    return target


def _sanitize(treatment: DirectorTreatment, section_count: int, asset_ids: set[int]) -> DirectorTreatment:
    treatment.motif_asset_ids = list(dict.fromkeys(x for x in treatment.motif_asset_ids if x in asset_ids))
    seen: set[int] = set()
    valid_sections = []
    for section in treatment.sections:
        if section.section_index >= section_count or section.section_index in seen:
            continue
        seen.add(section.section_index)
        section.preferred_asset_ids = list(dict.fromkeys(x for x in section.preferred_asset_ids if x in asset_ids))
        valid_sections.append(section)
    treatment.sections = valid_sections
    return treatment

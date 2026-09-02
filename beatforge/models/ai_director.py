from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset


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
) -> DirectorTreatment:
    context = _build_context(analysis, lyrics, assets, similarities)
    payload = {
        "model": config.director_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "temperature": config.director_temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "mv_director_treatment",
                "strict": True,
                "schema": DirectorTreatment.model_json_schema(),
            },
        },
    }
    try:
        response = _post(config.director_base_url, payload, config.director_timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        payload["response_format"] = {"type": "json_object"}
        response = _post(config.director_base_url, payload, config.director_timeout_seconds)
    treatment = DirectorTreatment.model_validate_json(_extract_content(response))
    return _sanitize(treatment, len(analysis.sections) - 1, {asset.id for asset in assets})


def _post(base_url: str, payload: dict, timeout: float) -> dict:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("BEATFORGE_DIRECTOR_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as result:
        return json.loads(result.read().decode("utf-8"))


def _extract_content(response: dict) -> str:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
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
        assets,
        key=lambda asset: float(semantic[asset.id]) * .75 + asset.quality_score * .25,
        reverse=True,
    )
    return {asset.id for asset in ranked[:limit]}


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

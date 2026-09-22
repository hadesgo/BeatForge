from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import SUBTITLE_EFFECTS, LyricLine
from beatforge.media import MediaAsset

# Built from the renderer's own list, so the model can only choose effects that exist.
SubtitleEffect = Literal[*SUBTITLE_EFFECTS]


class SectionDirection(BaseModel):
    section_index: int = Field(ge=0)
    narrative_role: str = Field(min_length=1, max_length=160)
    lyric_relation: Literal[
        "literal", "metaphorical", "emotional", "contrast", "abstract"
    ] = "emotional"
    cut_intensity: float = Field(default=0.5, ge=0, le=1)
    preferred_media: Literal["any", "image", "video"] = "any"
    preferred_shot_sizes: list[
        Literal["wide", "medium", "closeup", "detail", "unknown"]
    ] = Field(
        default_factory=list,
        max_length=3,
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
    grade_profile: Literal[
        "energetic",
        "uplifting",
        "melancholic",
        "dreamy",
        "romantic",
        "dark",
        "cinematic",
    ]
    transition_tone: Literal["bright", "dark", "soft", "neutral"] = "neutral"
    sections: list[SectionDirection] = Field(default_factory=list)

    def section(self, index: int) -> SectionDirection | None:
        return next(
            (item for item in self.sections if item.section_index == index), None
        )


SYSTEM_PROMPT = """你是一位经验丰富的音乐录影带导演和剪辑指导。根据已经完成的音乐分析、逐句歌词、素材元数据和视觉检索候选，制定一份可执行的导演方案。
要求：先建立一套克制且统一的视觉圣经，再安排局部变化；保持主体、景别、运动方向、视觉母题和色彩发展的连续性；主歌重视叙事连续性，副歌重复可识别的视觉记忆点，桥段只做一次明确反差，结尾留有呼吸；歌词与画面可以直译、隐喻、情绪呼应或有意对照。color_arc 使用2到4个简短且可执行的色彩阶段，优先使用 natural、warm amber、cold blue、teal orange、dreamy violet、forest green、muted monochrome 等描述，避免每个乐段都换一种无关风格。字幕效果属于同一套设计系统，只有在章节或能量显著变化时才切换。不要虚构不存在的素材 ID；不要输出时间码或 FFmpeg 命令。只返回符合字段说明的 JSON 对象，不要输出解释。"""

# The director prompt is the only place in the pipeline that feeds a very long
# sequence to a language model. Bonsai is mostly linear attention, so the ceiling is
# about KV-cache room rather than a quadratic score matrix: llama.cpp allocates the
# cache for the whole `-c` window up front, which is why the prompt budget and
# `director_llama_args` have to agree.
TREATMENT_SPEC = """返回一个 JSON 对象，字段如下：
- concept: 全片概念，一句话（<=300字）
- narrative_arc: 叙事弧（<=500字）
- visual_style: 统一的视觉风格（<=240字）
- color_arc: 2~4 个色彩阶段描述
- motif_asset_ids: 1~5 个全片复现的视觉母题素材 id
- grade_profile: energetic|uplifting|melancholic|dreamy|romantic|dark|cinematic
- transition_tone: bright|dark|soft|neutral
- sections: 每个乐段一项，字段为
  - section_index: 整数，对应 sections 列表下标
  - narrative_role: 该乐段的叙事作用（<=160字）
  - lyric_relation: literal|metaphorical|emotional|contrast|abstract
  - cut_intensity: 0~1
  - preferred_media: any|image|video
  - preferred_shot_sizes: 0~3 项，wide|medium|closeup|detail|unknown
  - preferred_asset_ids: 0~5 个素材 id
  - subtitle_effect: karaoke|cinematic|bounce|float|glow|typewriter|neon|neon_flicker|shake|wave|punch|glitch|slide|rainbow|flip_in|spotlight
  - transition_tone: bright|dark|soft|neutral
  - edit_intent: continuity|impact|breathe"""

# Prompt variants tried in order until the measured prompt fits the token budget:
# (lyric stride, candidates per lyric, assets). The per-lyric candidate rows are
# the most redundant part, so they go first; sampling the lyrics is the last
# resort because they carry the narrative the director is asked to shape.
#: ``(lyric stride, candidates per lyric, asset cap)``, richest first. The top rungs
#: exist because a 27B ternary checkpoint with a 262k window can read the whole brief,
#: and the earlier ceiling of 24 assets was a KV-cache workaround rather than a
#: judgement about what the director needs.
PROMPT_LADDER: tuple[tuple[int, int, int], ...] = (
    (1, 6, 96),
    (1, 4, 64),
    (1, 3, 40),
    (1, 2, 24),
    (1, 2, 16),
    (1, 1, 16),
    (1, 1, 12),
    (1, 0, 12),
    (1, 0, 8),
    (2, 0, 8),
)
#: How many candidates to build the brief with, before the ladder trims it down. These
#: are the widest rung's numbers; building with the maximum is what lets a big budget
#: actually use it.
CANDIDATE_ASSETS = PROMPT_LADDER[0][2]
CANDIDATE_PER_LYRIC = PROMPT_LADDER[0][1]
#: Tokens kept free on top of ``director_max_new_tokens``. The prompt *and* the answer
#: share the same ``-c`` window, and the measurement can land a token either side of the
#: real request, so the budget stops just short of the edge.
PROMPT_SAFETY_TOKENS = 64


def direct_mv(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    config: AIConfig,
    cache_dir: Path,
    source_starts: np.ndarray | None = None,
) -> DirectorTreatment:
    context = _build_context(analysis, lyrics, assets, similarities, source_starts)
    treatment = _treat_with_llamacpp(context, config, cache_dir)
    return _sanitize(
        treatment, len(analysis.sections) - 1, {asset.id for asset in assets}
    )


def _treat_with_llamacpp(
    context: dict,
    config: AIConfig,
    cache_dir: Path,
) -> DirectorTreatment:
    """Ask a llama-server-backed GGUF checkpoint for the treatment.

    The exchange is: start the server, fit the prompt to the context it really has, ask,
    validate, then give the model exactly one chance to repair its own JSON. The fit
    happens *after* the server is up on purpose - only the server can measure a prompt,
    and only the server knows how much of ``-c`` the answer still needs.
    """
    from beatforge.models.llama_server import open_server

    schema = DirectorTreatment.model_json_schema()
    with open_server(
        binary=config.director_llama_server,
        gguf=config.director_gguf,
        cache_dir=cache_dir,
        repo_id=config.director_model,
        extra_args=list(config.director_llama_args),
        port=config.director_llama_port,
        url=config.director_llama_url or None,
        log=cache_dir / "llama-server.log",
    ) as server:
        budget, budget_note = _prompt_budget(server, config)
        messages, prompt_tokens, trimmed_context = _fit_prompt(
            context,
            config,
            _server_counter(server),
            budget,
        )
        (cache_dir / "director-context.json").write_text(
            json.dumps(trimmed_context, ensure_ascii=False, indent=2),
            "utf-8",
        )
        measured = "服务端实测" if server.tokenizer_available else "字符估算"
        print(
            f"    导演提示词 {prompt_tokens} tokens（{measured}）· {budget_note} · "
            f"llama-server {server.base_url}"
        )
        raw = server.chat(
            messages,
            json_schema=schema,
            temperature=config.director_temperature,
            max_tokens=config.director_max_new_tokens,
        )
        try:
            return DirectorTreatment.model_validate_json(_extract_json(raw))
        except ValidationError as exc:
            corrected = server.chat(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "上一个结果未通过校验。修正后只返回完整 JSON，不要解释。\n"
                            f"校验错误：{exc}"
                        ),
                    },
                ],
                json_schema=schema,
                temperature=config.director_temperature,
                max_tokens=config.director_max_new_tokens,
            )
            return DirectorTreatment.model_validate_json(_extract_json(corrected))


def _server_counter(server):
    """A token counter backed by the server's own tokenizer, with an estimate behind it."""

    def count(messages: list[dict]) -> int:
        if server.tokenizer_available is not False:
            measured = server.count_prompt_tokens(messages)
            if measured is not None:
                return measured
        return _estimated_tokens(messages)

    return count


def _prompt_budget(server, config: AIConfig) -> tuple[int, str]:
    """How long the prompt may be, and why - for the diagnostic line.

    ``-c`` covers the prompt *and* the answer, so the prompt cannot have all of it:
    a 16495-token prompt against a 24576-token context is rejected outright with
    ``exceed_context_size_error``, and the request never even starts. The configured
    ``director_prompt_tokens`` is therefore an upper bound, clamped by what the server
    actually has room for.
    """
    configured = max(256, config.director_prompt_tokens)
    context = server.context_size()
    if context is None:
        return configured, f"提示词上限 {configured}（服务端未报告上下文长度）"
    room = max(256, context - config.director_max_new_tokens - PROMPT_SAFETY_TOKENS)
    if room < configured:
        return room, (
            f"提示词上限 {room}（上下文 {context} − 回复 {config.director_max_new_tokens} "
            f"− 余量 {PROMPT_SAFETY_TOKENS}；被它压低，director_prompt_tokens={configured}）"
        )
    return configured, f"提示词上限 {configured}（上下文 {context}）"


def _fit_prompt(
    context: dict,
    config: AIConfig,
    count_tokens,
    budget: int,
) -> tuple[list[dict], int, dict]:
    """Trim the context until the rendered prompt fits ``budget``.

    ``count_tokens`` is whatever can measure the prompt: the server's own tokenizer when
    the build has one, a character estimate when it does not. The ladder is the same
    either way, and the loop accepts the widest rung that fits.
    """
    budget = max(256, budget)
    messages: list[dict] = []
    tokens = 0
    trimmed = context
    for stride, candidates, assets in PROMPT_LADDER:
        trimmed = _trim_context(
            context,
            lyric_stride=stride,
            candidate_limit=candidates,
            asset_limit=assets,
        )
        messages = _prompt_messages(trimmed)
        tokens = count_tokens(messages)
        if tokens <= budget:
            break
    return messages, tokens, trimmed


def _prompt_messages(context: dict) -> list[dict]:
    project_text = (
        f"{TREATMENT_SPEC}\n\n项目数据:\n{json.dumps(context, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": project_text},
    ]


def _trim_context(
    context: dict,
    *,
    lyric_stride: int,
    candidate_limit: int,
    asset_limit: int,
) -> dict:
    """Drop optional detail so a long song still fits the prompt budget."""
    trimmed = dict(context)
    trimmed["assets"] = list(context.get("assets", []))[:asset_limit]
    lyrics = list(context.get("lyrics", []))
    kept = set(range(0, len(lyrics), lyric_stride))
    trimmed["lyrics"] = [line for index, line in enumerate(lyrics) if index in kept]
    # A row without candidates carries no retrieval signal - the lyric index is
    # already implied by the position in ``lyrics`` - so drop the table instead
    # of paying tokens for bare index bookkeeping.
    trimmed["lyric_candidates"] = [
        {"i": row["i"], "c": row["c"][:candidate_limit]}
        for row in context.get("lyric_candidates", [])
        if candidate_limit and row["i"] in kept
    ]
    if not trimmed["lyric_candidates"]:
        # Keep the brief honest: never point the director at a table that the
        # trimmer just removed.
        trimmed["instruction"] = (
            "lyrics 是 [起始秒, 歌词] 列表。为每个 section index 提供一项导演策略；"
            "素材选择只能使用 assets 中出现的 id。不要求逐句换镜；"
            "应优先用它建立连续的段落叙事和重复母题。"
        )
    return trimmed


def _estimated_tokens(messages: list[dict]) -> int:
    """Conservative estimate, used only when the server cannot measure the prompt.

    Chinese runs about one character per token while the template's English boilerplate
    runs nearer five, so a mixed brief lands on either side of this - which is why the
    ladder prefers the server's own tokenizer and this is the fallback, not the default.
    """
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(
                len(str(item.get("text", "")))
                for item in content
                if isinstance(item, dict)
            )
    return int(total / 1.8) + 64


def _extract_json(content: str) -> str:
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else text


def _build_context(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    source_starts: np.ndarray | None = None,
) -> dict:
    """Build the director brief, at the widest detail the ladder can ask for.

    It is built wide and trimmed down rather than built narrow: the ladder can only take
    detail away, so anything not put in here is detail the director can never use.
    """
    candidates = _ranked_candidates(assets, similarities)
    candidate_ids = {asset.id for asset in candidates}
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
                "label": (
                    analysis.section_labels[index]
                    if index < len(analysis.section_labels)
                    else "unknown"
                ),
                "start": start,
                "end": end,
                "energy": round(analysis.energy_at((start + end) / 2), 3),
            }
            for index, (start, end) in enumerate(
                zip(analysis.sections, analysis.sections[1:])
            )
        ],
        "lyrics": [[round(line.start, 2), line.text] for line in lyrics],
        "assets": [
            {
                "id": asset.id,
                "kind": asset.kind,
                "description": asset.description,
                "mood": asset.mood,
                "shot_size": asset.shot_size,
            }
            for asset in candidates
        ],
        "lyric_candidates": _lyric_candidates(
            lyrics,
            assets,
            similarities,
            source_starts,
            candidate_ids,
            limit=CANDIDATE_PER_LYRIC,
        ),
        "instruction": (
            "lyrics 是 [起始秒, 歌词] 列表；lyric_candidates 的 i 是歌词在 lyrics 中的下标，"
            "c 是 [素材id, 检索得分, 视频取材秒] 列表，只有视频素材带第三项。"
            "为每个 section index 提供一项导演策略；素材选择只能使用 assets 中出现的 id。"
            "不要求逐句换镜；应优先用它建立连续的段落叙事和重复母题。"
        ),
    }


def _lyric_candidates(
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    source_starts: np.ndarray | None,
    candidate_ids: set[int],
    limit: int = 2,
) -> list[dict]:
    if similarities is None or similarities.ndim != 2:
        return []
    row_count = min(len(lyrics), similarities.shape[0])
    column_count = min(len(assets), similarities.shape[1])
    output = []
    for row in range(row_count):
        columns = [
            int(column)
            for column in np.argsort(similarities[row, :column_count])[::-1]
            if assets[int(column)].id in candidate_ids
        ][:limit]
        candidates = []
        for column in columns:
            item = [assets[column].id, round(float(similarities[row, column]), 4)]
            if (
                assets[column].kind == "video"
                and source_starts is not None
                and row < source_starts.shape[0]
                and column < source_starts.shape[1]
            ):
                item.append(round(float(source_starts[row, column]), 3))
            candidates.append(item)
        output.append({"i": row, "c": candidates})
    return output


def _ranked_candidates(
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    limit: int = CANDIDATE_ASSETS,
) -> list[MediaAsset]:
    """Candidates best-first, capped at ``limit``.

    Ranked unconditionally rather than only when the list is long enough to need
    capping: the ladder trims this list with a plain slice, so a list in discovery order
    drops the strongest material first. Ranking only when the cap bites left exactly that
    hole for every project small enough to fit under it - which, with the cap now at 96,
    is most of them.

    ``sorted`` is stable, so equal scores keep discovery order and a project with no
    retrieval signal at all is left alone.
    """
    if similarities is not None and similarities.size:
        semantic = np.max(similarities, axis=0)
    else:
        semantic = np.zeros(len(assets))
    ranked = sorted(
        enumerate(assets),
        key=lambda item: float(semantic[item[0]]) * 0.75 + item[1].quality_score * 0.25,
        reverse=True,
    )
    return [asset for _, asset in ranked[:limit]]


def _sanitize(
    treatment: DirectorTreatment, section_count: int, asset_ids: set[int]
) -> DirectorTreatment:
    treatment.motif_asset_ids = list(
        dict.fromkeys(x for x in treatment.motif_asset_ids if x in asset_ids)
    )
    seen: set[int] = set()
    valid_sections = []
    for section in treatment.sections:
        if section.section_index >= section_count or section.section_index in seen:
            continue
        seen.add(section.section_index)
        section.preferred_asset_ids = list(
            dict.fromkeys(x for x in section.preferred_asset_ids if x in asset_ids)
        )
        valid_sections.append(section)
    treatment.sections = valid_sections
    return treatment

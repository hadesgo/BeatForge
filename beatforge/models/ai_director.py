from __future__ import annotations

import gc
import importlib
import inspect
import json
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field, ValidationError

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import SUBTITLE_EFFECTS, LyricLine
from beatforge.media import MediaAsset
from beatforge.runtime import command


# Built from the renderer's own list, so the model can only choose effects that exist.
SubtitleEffect = Literal[*SUBTITLE_EFFECTS]


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
要求：先建立一套克制且统一的视觉圣经，再安排局部变化；保持主体、景别、运动方向、视觉母题和色彩发展的连续性；主歌重视叙事连续性，副歌重复可识别的视觉记忆点，桥段只做一次明确反差，结尾留有呼吸；歌词与画面可以直译、隐喻、情绪呼应或有意对照。color_arc 使用2到4个简短且可执行的色彩阶段，优先使用 natural、warm amber、cold blue、teal orange、dreamy violet、forest green、muted monochrome 等描述，避免每个乐段都换一种无关风格。字幕效果属于同一套设计系统，只有在章节或能量显著变化时才切换。不要虚构不存在的素材 ID；不要输出时间码或 FFmpeg 命令。只返回符合字段说明的 JSON 对象，不要输出解释。"""

# The director prompt is the only place in the pipeline that feeds a very long
# sequence to a language model, and Spark-X2.5 computes attention with plain
# matmuls: every layer materialises a full [heads, prompt, prompt] score matrix
# (the sliding-window layers mask it afterwards instead of slicing the keys).
# That cost grows with the square of the prompt, so the prompt is capped and the
# GPU budget reserves that memory explicitly.
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
PROMPT_LADDER: tuple[tuple[int, int, int], ...] = (
    (1, 2, 24),
    (1, 2, 16),
    (1, 1, 16),
    (1, 1, 12),
    (1, 0, 12),
    (1, 0, 8),
    (2, 0, 8),
)
# bf16 score matrix plus the fp32 softmax copy and its bf16 cast.
SCORE_BYTES_PER_ELEMENT = 6
# Prompt tokens the JSON-repair round adds on top of the first attempt.
REPAIR_TOKENS = 768
DIRECTOR_OVERHEAD_GB = 1.0
MIN_DIRECTOR_RESERVE_GB = 1.5
RETRY_BUDGET_SCALE = 0.65


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
    context = _build_context(analysis, lyrics, assets, similarities, source_starts)
    visual_reference = None
    if config.director_backend == "multimodal":
        visual_reference = _build_contact_sheet(
            assets, similarities, cache_dir, config.director_contact_sheet_assets, source_starts,
        )
    treatment = _generate_treatment(context, config, device, cache_dir, visual_reference)
    return _sanitize(treatment, len(analysis.sections) - 1, {asset.id for asset in assets})


def _is_cuda(device: str) -> bool:
    """``resolve_device`` can return ``cuda``, ``cpu`` or a user-supplied
    ``cuda:N``, so match on the prefix instead of equality."""
    return device.startswith("cuda")


def _gpu_index(device: str) -> int:
    _, _, index = device.partition(":")
    return int(index) if index.isdigit() else 0


def _generate_treatment(
    context: dict,
    config: AIConfig,
    device: str,
    cache_dir: Path,
    visual_reference: Path | None = None,
) -> DirectorTreatment:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("AI 导演需要 ai 与 ai-cpu/ai-cuda extra") from exc

    _repair_remote_model_code(config.director_model)

    offload_dir = cache_dir / "director-offload"
    model_config = _load_model_config(transformers, config)
    common = {
        "local_files_only": config.offline,
        "trust_remote_code": True,
    }
    if config.director_backend == "text":
        processor = transformers.AutoTokenizer.from_pretrained(config.director_model, **common)
    else:
        processor = transformers.AutoProcessor.from_pretrained(config.director_model, **common)
    messages, prompt_tokens, trimmed_context = _fit_prompt(processor, context, config, visual_reference)
    (cache_dir / "director-context.json").write_text(
        json.dumps(trimmed_context, ensure_ascii=False, indent=2), "utf-8",
    )
    budgets: list[float | None] = [None]
    if _is_cuda(device):
        reserve = _sequence_reserve_gb(config, model_config, prompt_tokens)
        budgets = _gpu_budgets(torch, config, reserve, _gpu_index(device))
        free_gb = torch.cuda.mem_get_info(_gpu_index(device))[0] / 2**30
        print(
            f"    导演提示词约 {prompt_tokens} tokens · 空闲显存 {free_gb:.1f}GiB · "
            f"权重预算 {budgets[0]:.1f}GiB（其余 {reserve:.1f}GiB 留给注意力矩阵、"
            "日志张量和 KV 缓存）"
        )

    model = None
    try:
        for attempt, budget in enumerate(budgets):
            try:
                model = _load_director_model(transformers, config, device, budget, offload_dir)
                model.eval()
                _normalize_generation_config(model)
                raw = _generate(model, processor, messages, config, torch)
                try:
                    return DirectorTreatment.model_validate_json(_extract_json(raw))
                except ValidationError as exc:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": (
                            "上一个结果未通过校验。修正后只返回完整 JSON，不要解释。"
                            f"\n校验错误：{exc}"
                        )},
                    ]
                    corrected = _generate(model, processor, messages, config, torch)
                    return DirectorTreatment.model_validate_json(_extract_json(corrected))
            except RuntimeError as exc:
                del model
                model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt + 1 >= len(budgets) or not _is_out_of_memory(exc):
                    raise
                print(f"    导演阶段显存不足，改用 {budgets[attempt + 1]:.1f}GiB 权重预算重试")
        raise RuntimeError("AI 导演未能加载")
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _load_director_model(
    transformers, config: AIConfig, device: str, budget_gb: float | None, offload_dir: Path,
):
    cuda = _is_cuda(device)
    load_options: dict = {
        "device_map": "auto" if cuda else {"": "cpu"},
        "dtype": "auto",
        "local_files_only": config.offline,
        "low_cpu_mem_usage": True,
    }
    if cuda and budget_gb is not None:
        load_options["max_memory"] = {
            _gpu_index(device): f"{budget_gb:.1f}GiB",
            "cpu": f"{config.director_cpu_memory_gb:.1f}GiB",
        }
        if config.director_offload:
            offload_dir.mkdir(parents=True, exist_ok=True)
            load_options.update({"offload_folder": str(offload_dir), "offload_state_dict": True})
    if config.director_backend == "text":
        return transformers.AutoModelForCausalLM.from_pretrained(
            config.director_model, trust_remote_code=True, **load_options,
        )
    return transformers.AutoModelForMultimodalLM.from_pretrained(
        config.director_model, trust_remote_code=True, **load_options,
    )


def _normalize_generation_config(model) -> None:
    """Spark-X2.5's generation_config uses the legacy top_k=-1 sentinel for
    "no top-k truncation"; transformers>=5 requires a strictly positive
    integer (or None/0 to disable). Normalize so sampling matches intent.
    Remote-code models are not required to expose ``generation_config``.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    if getattr(generation_config, "top_k", None) is not None and generation_config.top_k <= 0:
        generation_config.top_k = None


def _is_out_of_memory(exc: BaseException) -> bool:
    """Match both torch.cuda.OutOfMemoryError and torch.OutOfMemoryError."""
    return "out of memory" in str(exc).lower()


def _load_model_config(transformers, config: AIConfig):
    """Read the model config without loading weights; only used for budgeting."""
    try:
        return transformers.AutoConfig.from_pretrained(
            config.director_model, local_files_only=config.offline, trust_remote_code=True,
        )
    except Exception:
        # A broken or unusual remote config must not stop the director.
        return None


def _dtype_bytes(model_config) -> int:
    """Read the weight dtype. ``dtype`` comes first on purpose: ``torch_dtype`` is
    a deprecated alias on transformers >= 5 and reading it logs a warning."""
    dtype = str(getattr(model_config, "dtype", "") or getattr(model_config, "torch_dtype", ""))
    return 4 if "32" in dtype else 2


def _sequence_reserve_gb(config: AIConfig, model_config, prompt_tokens: int) -> float:
    """Memory one forward pass needs on top of the model weights, in GiB.

    Accelerate's ``max_memory`` only budgets the weights, so everything the
    prefill and the KV cache need has to be subtracted from the GPU budget by
    hand. Assumes an eager attention implementation, which is what the
    remote-code director models ship.
    """
    dtype_bytes = _dtype_bytes(model_config)
    layers = int(getattr(model_config, "num_hidden_layers", 0) or 36)
    heads = int(getattr(model_config, "num_attention_heads", 0) or 16)
    kv_heads = int(getattr(model_config, "num_key_value_heads", 0) or heads)
    hidden = int(getattr(model_config, "hidden_size", 0) or 2560)
    head_dim = int(getattr(model_config, "head_dim", 0) or max(1, hidden // max(heads, 1)))
    intermediate = int(getattr(model_config, "intermediate_size", 0) or hidden * 4)
    vocab = int(getattr(model_config, "vocab_size", 0) or 0)
    window = int(getattr(model_config, "sliding_window", 0) or 0)
    layer_types = list(getattr(model_config, "layer_types", None) or [])
    full_layers = sum(1 for kind in layer_types if str(kind) == "full_attention") or layers
    sliding_layers = max(0, layers - full_layers)

    prefill = prompt_tokens + REPAIR_TOKENS
    sequence = prompt_tokens + config.director_max_new_tokens
    scores = heads * prefill * prefill * SCORE_BYTES_PER_ELEMENT
    masks = 2 * prefill * prefill * dtype_bytes
    # `generate` sets `logits_to_keep=1`, so in practice only the last position
    # reaches the language head and this term is nearly free. Budget the
    # pessimistic version anyway: the remote forward defaults `logits_to_keep` to
    # 0, and under-reserving costs an OOM retry that reloads the whole checkpoint.
    logits = prefill * vocab * dtype_bytes
    kv = 2 * kv_heads * head_dim * dtype_bytes * (
        full_layers * sequence + sliding_layers * min(window or sequence, sequence)
    )
    activations = prefill * intermediate * dtype_bytes * 3
    return max(
        MIN_DIRECTOR_RESERVE_GB,
        DIRECTOR_OVERHEAD_GB + (scores + masks + logits + kv + activations) / 2**30,
    )


def _gpu_budgets(torch, config: AIConfig, reserve_gb: float, index: int = 0) -> list[float]:
    """Weight budgets to try, in GiB: the configured ceiling and one retry."""
    free_gb = torch.cuda.mem_get_info(index)[0] / 2**30
    budget = max(1.0, min(config.director_gpu_memory_gb, free_gb - reserve_gb))
    return [round(budget, 1), round(max(1.0, budget * RETRY_BUDGET_SCALE), 1)]


def _fit_prompt(
    processor, context: dict, config: AIConfig, visual_reference: Path | None,
) -> tuple[list[dict], int, dict]:
    """Trim the context until the rendered prompt fits the configured budget."""
    budget = max(256, config.director_prompt_tokens)
    messages: list[dict] = []
    tokens = 0
    trimmed = context
    for stride, candidates, assets in PROMPT_LADDER:
        trimmed = _trim_context(
            context, lyric_stride=stride, candidate_limit=candidates, asset_limit=assets,
        )
        messages = _prompt_messages(trimmed, visual_reference)
        tokens = _prompt_tokens(processor, messages)
        if tokens <= budget:
            break
    return messages, tokens, trimmed


def _prompt_messages(context: dict, visual_reference: Path | None) -> list[dict]:
    project_text = f"{TREATMENT_SPEC}\n\n项目数据:\n{json.dumps(context, ensure_ascii=False)}"
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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _trim_context(
    context: dict, *, lyric_stride: int, candidate_limit: int, asset_limit: int,
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


def _rendered_token_count(rendered) -> int:
    """Token count of an ``apply_chat_template(..., tokenize=True)`` result.

    The processors this project uses return a ``BatchEncoding`` holding
    ``input_ids`` and ``attention_mask``; its ``len`` is the number of *keys* (2),
    not the number of tokens, so reading it directly silently disabled the prompt
    cap. Bare lists of ids and batched tensors are handled too.
    """
    if isinstance(rendered, Mapping) and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    shape = getattr(rendered, "shape", None)
    if shape:
        return int(shape[-1])
    if rendered and not isinstance(rendered[0], (int, np.integer)):
        # Batched output: either ``[[id, ...]]`` or ``[tensor]``.
        rendered = rendered[0]
        shape = getattr(rendered, "shape", None)
        return int(shape[-1]) if shape else len(rendered)
    return len(rendered)


def _prompt_tokens(processor, messages: list[dict]) -> int:
    """Measure the rendered prompt, falling back to a character estimate."""
    try:
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        )
    except Exception:
        return _estimated_tokens(messages)
    return _rendered_token_count(rendered)


def _estimated_tokens(messages: list[dict]) -> int:
    """Conservative estimate when the chat template cannot be rendered."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(item.get("text", ""))) for item in content if isinstance(item, dict))
    return int(total / 1.8) + 64


def _generate(model, processor, messages: list[dict], config: AIConfig, torch) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        # Spark-X2.5 is an R1-style reasoning model: with thinking enabled it burns
        # the whole token budget on hidden reasoning and never emits the JSON.
        # enable_thinking=False renders "<Bot></think>" so it answers directly.
        enable_thinking=False,
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


TIED_WEIGHTS_LIST = re.compile(r"_tied_weights_keys\s*=\s*\[([^\]]*)\]")
TIED_WEIGHTS_EMBEDDINGS = (
    (r"def get_input_embeddings\s*\(self\):\s*return\s+self\.model\.([\w.]+)", "model."),
    (r"def get_input_embeddings\s*\(self\):\s*return\s+self\.([\w.]+)", ""),
)
MASK_KWARGS_BLOCK = re.compile(r"(?P<indent>[ \t]*)mask_kwargs\s*=\s*\{(?P<body>[^{}]*)\}")
MASK_KWARG_ENTRY = re.compile(r'"(?P<key>\w+)"\s*:\s*(?P<value>[^,\n]+),')
# transformers renamed this argument when the masking helpers moved to
# `transformers.masking_utils`, so the remote call has to follow the installed
# signature instead of the one it was written against.
MASK_KWARG_ALIASES = {"input_embeds": "inputs_embeds", "inputs_embeds": "input_embeds"}


def _mask_kwarg_parameters() -> set[str] | None:
    """Keyword names the installed mask helpers accept.

    The remote code builds one ``mask_kwargs`` dict and feeds it to both
    ``create_causal_mask`` and ``create_sliding_window_causal_mask``, so only the
    intersection of the two signatures can be passed through safely.

    ``None`` means "cannot tell" (missing transformers, or only ``**kwargs``
    signatures): the remote call is then left untouched, because dropping
    arguments we cannot verify would silently change the mask.
    """
    accepted: set[str] | None = None
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        try:
            module = importlib.import_module("transformers.masking_utils")
            parameters = inspect.signature(getattr(module, name)).parameters
        except Exception:
            continue
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            # This helper swallows anything, so it cannot narrow the set.
            continue
        accepted = set(parameters) if accepted is None else accepted & set(parameters)
    return accepted


def _repair_tied_weights(text: str) -> tuple[str, str | None]:
    """transformers >= 5 requires ``_tied_weights_keys`` to be a ``{target: source}``
    mapping, but some remote-code models (e.g. Spark-X2.5) still declare the legacy
    list form, which crashes ``get_expanded_tied_weights_keys`` with
    ``AttributeError: 'list' object has no attribute 'keys'``.

    The embedding parameter name is derived from ``get_input_embeddings``.
    """
    match = TIED_WEIGHTS_LIST.search(text)
    if not match:
        return text, None
    source = ""
    for embed_pattern, prefix in TIED_WEIGHTS_EMBEDDINGS:
        embed_match = re.search(embed_pattern, text)
        if embed_match:
            source = f"{prefix}{embed_match.group(1)}.weight"
            break
    if not source:
        return text, None
    keys = [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]
    if not keys:
        return text, None
    mapping = ", ".join(f'"{key}": "{source}"' for key in keys)
    patched = TIED_WEIGHTS_LIST.sub(f"_tied_weights_keys = {{{mapping}}}", text, count=1)
    return patched, "_tied_weights_keys 改为 dict 映射"


def _repair_mask_kwargs(text: str, accepted: set[str] | None) -> tuple[str, str | None]:
    """Align the remote ``mask_kwargs`` dict with the installed transformers.

    Spark-X2.5 builds it against transformers 4.57 (``input_embeds`` plus
    ``cache_position``), while 5.x renamed the first to ``inputs_embeds`` and
    dropped the second, so the call dies with ``TypeError: create_causal_mask()
    got an unexpected keyword argument 'input_embeds'``. Entries are matched
    against the installed signature, which keeps the same rewrite correct on
    both sides of the rename.
    """
    if accepted is None:
        return text, None
    changes: list[str] = []

    def rebuild(match: re.Match[str]) -> str:
        indent = match.group("indent")
        entries = MASK_KWARG_ENTRY.findall(match.group("body"))
        if not entries:
            return match.group(0)
        lines: list[str] = []
        for key, value in entries:
            if key not in accepted:
                alias = MASK_KWARG_ALIASES.get(key)
                if alias is None or alias not in accepted:
                    changes.append(f"删除 {key}")
                    continue
                changes.append(f"{key} → {alias}")
                key = alias
            lines.append(f'{indent}    "{key}": {value},')
        body = "\n".join(lines)
        return f"{indent}mask_kwargs = {{\n{body}\n{indent}}}"

    patched = MASK_KWARGS_BLOCK.sub(rebuild, text, count=1)
    return patched, ("、".join(changes) if changes else None)


def _remote_module_paths(model_name: str) -> list[Path]:
    """The remote modelling file in the local model directory, plus the copy
    transformers keeps in its module cache."""
    module_paths: list[Path] = []
    local_dir = Path(model_name)
    if (local_dir / "modeling_spark.py").exists():
        module_paths.append(local_dir / "modeling_spark.py")
    try:
        from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_cached_module_file
        module_rel = get_cached_module_file(
            model_name, "modeling_spark.py", local_files_only=True,
        )
        cached = Path(module_rel)
        if not cached.is_absolute():
            cached = Path(HF_MODULES_CACHE) / cached
        if cached not in module_paths:
            module_paths.append(cached)
    except Exception:
        pass
    return module_paths


def _repair_remote_model_code(model_name: str) -> None:
    """Rewrite the remote modelling file on disk so it runs on the transformers
    release that is actually installed.

    Patch the local model file first: transformers re-copies remote code from the
    local model directory into its module cache whenever the two differ, so a
    patch applied only to the cache copy is silently reverted (and the cache
    directory is keyed by a hash of the sources, so it moves on every edit).
    """
    accepted = _mask_kwarg_parameters()
    for module_path in _remote_module_paths(model_name):
        if not module_path.exists():
            continue
        try:
            text = module_path.read_text(encoding="utf-8")
        except OSError:
            continue
        text, tied_note = _repair_tied_weights(text)
        text, mask_note = _repair_mask_kwargs(text, accepted)
        notes = [note for note in (tied_note, mask_note) if note]
        if not notes:
            continue
        try:
            module_path.write_text(text, encoding="utf-8")
        except OSError:
            continue
        print(f"    已修复 {module_path} 的远程代码兼容性：{'；'.join(notes)}")



def _build_context(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    assets: list[MediaAsset],
    similarities: np.ndarray | None,
    source_starts: np.ndarray | None = None,
) -> dict:
    """Build the director brief. Kept compact: every character here becomes
    prompt tokens, and prompt tokens are quadratic in attention memory."""
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
        "lyrics": [[round(line.start, 2), line.text] for line in lyrics],
        "assets": [
            {
                "id": asset.id,
                "kind": asset.kind,
                "description": asset.description,
                "mood": asset.mood,
                "shot_size": asset.shot_size,
                "camera_motion": asset.camera_motion,
            }
            for asset in assets if asset.id in candidate_ids
        ],
        "lyric_candidates": _lyric_candidates(
            lyrics, assets, similarities, source_starts, candidate_ids,
        ),
        "instruction": (
            "lyrics 是 [起始秒, 歌词] 列表；lyric_candidates 的 i 是歌词在 lyrics 中的下标，"
            "c 是 [素材id, 检索得分, 视频取材秒] 列表，只有视频素材带第三项。"
            "为每个 section index 提供一项导演策略；素材选择只能使用 assets 中出现的 id。"
            "不要求逐句换镜；应优先用它建立连续的段落叙事和重复母题。"
        ),
    }


def _lyric_candidates(
    lyrics: list[LyricLine], assets: list[MediaAsset], similarities: np.ndarray | None,
    source_starts: np.ndarray | None, candidate_ids: set[int], limit: int = 2,
) -> list[dict]:
    if similarities is None or similarities.ndim != 2:
        return []
    row_count = min(len(lyrics), similarities.shape[0])
    column_count = min(len(assets), similarities.shape[1])
    output = []
    for row in range(row_count):
        columns = [
            int(column) for column in np.argsort(similarities[row, :column_count])[::-1]
            if assets[int(column)].id in candidate_ids
        ][:limit]
        candidates = []
        for column in columns:
            item = [assets[column].id, round(float(similarities[row, column]), 4)]
            if (
                assets[column].kind == "video" and source_starts is not None
                and row < source_starts.shape[0] and column < source_starts.shape[1]
            ):
                item.append(round(float(source_starts[row, column]), 3))
            candidates.append(item)
        output.append({"i": row, "c": candidates})
    return output


def _candidate_ids(assets: list[MediaAsset], similarities: np.ndarray | None, limit: int = 24) -> set[int]:
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
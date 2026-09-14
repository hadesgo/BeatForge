from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.models.ai_director import (
    DirectorTreatment,
    _build_context,
    _build_contact_sheet,
    _fit_prompt,
    _generate_treatment,
    _gpu_budgets,
    _prompt_messages,
    _prompt_tokens,
    _sequence_reserve_gb,
    _trim_context,
    direct_mv,
)
from beatforge.planner import create_plan


def _analysis() -> AudioAnalysis:
    return AudioAnalysis(
        duration=8, bpm=120, beats=[0, 2, 4, 6, 8], sections=[0, 4, 8],
        energy_times=[0, 4], energy_values=[.3, .8], average_energy=.55,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse", "chorus"],
    )


def _treatment() -> DirectorTreatment:
    return DirectorTreatment.model_validate({
        "concept": "从孤独走向释放",
        "narrative_arc": "封闭空间逐渐过渡到开阔场景",
        "visual_style": "克制的电影感",
        "color_arc": ["cold blue", "warm amber"],
        "motif_asset_ids": [1],
        "grade_profile": "cinematic",
        "transition_tone": "dark",
        "sections": [{
            "section_index": 0,
            "narrative_role": "建立人物处境",
            "cut_intensity": .3,
            "preferred_media": "image",
            "preferred_asset_ids": [1],
            "preferred_shot_sizes": ["closeup"],
            "subtitle_effect": "typewriter",
            "transition_tone": "soft",
            "edit_intent": "continuity",
        }],
    })


def test_director_preferences_influence_shot_selection() -> None:
    assets = [
        MediaAsset(0, Path("a.jpg"), "image", float("inf"), 100, 100, quality_score=.5),
        MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100, quality_score=.5, shot_size="closeup"),
    ]
    similarities = np.array([[.55, .50], [.55, .50]])
    shots = create_plan(
        _analysis(), [LyricLine(0, 4, "独自醒来"), LyricLine(4, 8, "奔向天光")],
        assets, similarities, min_shot=1.5, max_shot=5, treatment=_treatment(),
    )
    assert shots[0].media_id == 1
    assert shots[0].transition_tone == "soft"


def test_director_receives_per_lyric_candidates_and_video_timestamps() -> None:
    assets = [
        MediaAsset(4, Path("portrait.jpg"), "image", float("inf"), 1920, 1080),
        MediaAsset(8, Path("walk.mp4"), "video", 20, 1920, 1080),
    ]
    lyrics = [LyricLine(0, 4, "穿过夜色")]
    context = _build_context(
        _analysis(), lyrics, assets, np.array([[.3, .91]]), np.array([[0, 12.4]]),
    )

    row = context["lyric_candidates"][0]
    assert row["i"] == 0
    assert row["c"][0] == [8, .91, 12.4]
    assert context["lyrics"] == [[0, "穿过夜色"]]


def test_director_response_is_validated_and_sanitized(monkeypatch, tmp_path: Path) -> None:
    treatment_result = _treatment()
    treatment_result.motif_asset_ids = [1, 999, 1]
    treatment_result.sections[0].preferred_asset_ids = [1, 999]
    invalid_section = treatment_result.sections[0].model_copy(update={"section_index": 8})
    treatment_result.sections.append(invalid_section)

    monkeypatch.setattr("beatforge.models.ai_director._generate_treatment", lambda *_: treatment_result)
    assets = [MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100)]
    treatment = direct_mv(
        _analysis(), [LyricLine(0, 4, "独自醒来")], assets, None,
        AIConfig(), "cpu", tmp_path,
    )
    assert treatment.motif_asset_ids == [1]
    assert treatment.sections[0].preferred_asset_ids == [1]
    assert len(treatment.sections) == 1


def test_director_loads_in_process_with_memory_limit_and_releases_cuda(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {"empty": 0, "ipc": 0}
    response = _treatment().model_dump_json()

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.zeros((1, 3), dtype=int))

        def to(self, _device):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model_name, **options):
            calls["processor"] = (model_name, options)
            return cls()

        def apply_chat_template(self, *_args, **_kwargs):
            return Batch()

        def batch_decode(self, *_args, **_kwargs):
            return [response]

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model_name, **options):
            calls["model"] = (model_name, options)
            return cls()

        def eval(self):
            return self

        def generate(self, **_kwargs):
            return np.zeros((1, 4), dtype=int)

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda _index: (12 * 2**30, 12 * 2**30),
        empty_cache=lambda: calls.__setitem__("empty", int(calls["empty"]) + 1),
        ipc_collect=lambda: calls.__setitem__("ipc", int(calls["ipc"]) + 1),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, inference_mode=nullcontext)
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *_a, **_k: _spark_config()),
        AutoModelForCausalLM=Model,
        AutoModelForMultimodalLM=Model,
        AutoProcessor=Processor,
        AutoTokenizer=Processor,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = _generate_treatment(
        {}, AIConfig(offline=True, director_backend="multimodal"), "cuda", tmp_path,
    )

    assert result.concept == _treatment().concept
    options = calls["model"][1]
    assert options["device_map"] == "auto"
    assert options["max_memory"][0] == "9.0GiB"
    assert options["offload_folder"] == str(tmp_path / "director-offload")
    assert calls["empty"] == 1
    assert calls["ipc"] == 1


def test_director_honours_an_explicit_gpu_index(monkeypatch, tmp_path: Path) -> None:
    """`AIConfig.device` is currently limited to auto/cuda/cpu, but the director
    entry points accept a free-form device string, so match on the ``cuda``
    prefix rather than equality: an indexed device must not be downgraded to CPU
    and the free-memory probe must read the card that was asked for."""
    calls: dict[str, object] = {}
    response = _treatment().model_dump_json()

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.zeros((1, 2), dtype=int))

        def to(self, _device):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, _model_name, **_options):
            return cls()

        def apply_chat_template(self, *_args, **_kwargs):
            return Batch()

        def batch_decode(self, *_args, **_kwargs):
            return [response]

    class Model:
        device = "cuda:1"

        @classmethod
        def from_pretrained(cls, _model_name, **options):
            calls["options"] = options
            return cls()

        def eval(self):
            return self

        def generate(self, **_kwargs):
            return np.zeros((1, 2), dtype=int)

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda index: calls.__setitem__("probed", index) or (10 * 2**30, 12 * 2**30),
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, inference_mode=nullcontext)
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *_a, **_k: _spark_config()),
        AutoModelForCausalLM=Model,
        AutoModelForMultimodalLM=Model,
        AutoProcessor=Processor,
        AutoTokenizer=Processor,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = _generate_treatment({}, AIConfig(offline=True), "cuda:1", tmp_path)

    assert result.concept == _treatment().concept
    assert calls["probed"] == 1
    assert calls["options"]["device_map"] == "auto"
    assert 1 in calls["options"]["max_memory"]
    assert 0 not in calls["options"]["max_memory"]


def test_spark_director_uses_text_causal_lm_without_contact_sheet(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    response = _treatment().model_dump_json()

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.zeros((1, 2), dtype=int))

        def to(self, _device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **options):
            calls["tokenizer"] = (model_name, options)
            return cls()

        def apply_chat_template(self, *_args, **_kwargs):
            return Batch()

        def batch_decode(self, *_args, **_kwargs):
            return [response]

    class CausalModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, model_name, **options):
            calls["causal_model"] = (model_name, options)
            return cls()

        def eval(self):
            return self

        def generate(self, **_kwargs):
            return np.zeros((1, 3), dtype=int)

    class Unexpected:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("text director must not use multimodal loaders")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext,
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=CausalModel,
        AutoModelForMultimodalLM=Unexpected,
        AutoProcessor=Unexpected,
        AutoTokenizer=Tokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = _generate_treatment({}, AIConfig(offline=True), "cpu", tmp_path)

    assert result.concept == _treatment().concept
    assert calls["causal_model"][0] == "XHToken/Spark-X2.5-4B"
    assert calls["causal_model"][1]["dtype"] == "auto"
    assert "quantization_config" not in calls["causal_model"][1]


def test_director_contact_sheet_contains_real_candidates(tmp_path: Path) -> None:
    from PIL import Image

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (640, 360), (220, 60, 40)).save(first)
    Image.new("RGB", (360, 640), (30, 80, 180)).save(second)
    assets = [
        MediaAsset(0, first, "image", float("inf"), 640, 360, quality_score=.8),
        MediaAsset(1, second, "image", float("inf"), 360, 640, quality_score=.7),
    ]

    result = _build_contact_sheet(assets, np.array([[.9, .4]]), tmp_path / "cache", 24)

    assert result is not None and result.exists()
    with Image.open(result) as sheet:
        assert sheet.size == (1280, 208)


def _spark_config():
    """The real Spark-X2.5-4B architecture, which drives the memory estimate."""
    return SimpleNamespace(
        torch_dtype="bfloat16", num_hidden_layers=36, num_attention_heads=16,
        num_key_value_heads=4, hidden_size=2560, head_dim=256, intermediate_size=10240,
        vocab_size=131072, sliding_window=512,
        layer_types=["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(36)],
    )


def test_sequence_reserve_grows_quadratically_with_the_prompt() -> None:
    config = AIConfig()
    small = _sequence_reserve_gb(config, _spark_config(), 1500)
    large = _sequence_reserve_gb(config, _spark_config(), 12000)

    # The eager attention of the director model materialises an n x n score
    # matrix per layer, so a long prompt is quadratic in memory: the 12k prompt
    # used before this fix needs more than any consumer card has.
    assert small < 4
    assert large > 20
    assert large / small > 8


def test_gpu_budget_respects_free_memory_instead_of_total_memory() -> None:
    class FakeCuda:
        @staticmethod
        def mem_get_info(_index):
            return 6 * 2**30, 12 * 2**30

    budgets = _gpu_budgets(SimpleNamespace(cuda=FakeCuda), AIConfig(), 2.0)

    # 6GiB free minus the 2GiB the forward pass needs, not the 9GiB ceiling and
    # not the 12GiB card: weights may never fill the card again.
    assert budgets[0] == 4.0
    assert budgets[1] == 2.6
    assert budgets[1] < budgets[0]


def test_gpu_budget_never_exceeds_the_configured_ceiling() -> None:
    class FakeCuda:
        @staticmethod
        def mem_get_info(_index):
            return 24 * 2**30, 24 * 2**30

    budgets = _gpu_budgets(SimpleNamespace(cuda=FakeCuda), AIConfig(), 1.5)
    assert budgets[0] == 9.0


def test_director_prompt_is_trimmed_until_it_fits_the_budget() -> None:
    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            text = messages[-1]["content"]
            ids = list(range(len(text) // 4))
            # Real processors return a BatchEncoding holding input_ids and
            # attention_mask; its len() is the number of *keys*, so treating it
            # as a token count used to make the prompt cap a no-op.
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    context = {
        "song": {"duration": 240.0},
        "sections": [{"index": 0, "label": "verse", "start": 0.0, "end": 240.0, "energy": .5}],
        "lyrics": [[index * 3.8, f"第{index}句歌词"] for index in range(120)],
        "assets": [
            {"id": index, "kind": "image", "description": f"素材 {index} 的详细描述", "mood": "calm"}
            for index in range(24)
        ],
        "lyric_candidates": [
            {"i": index, "c": [[index % 24, .8], [index % 23, .7]]} for index in range(120)
        ],
    }
    config = AIConfig(director_prompt_tokens=1400)

    messages, tokens, trimmed = _fit_prompt(Processor(), context, config, None)

    # The contract is "trim until it fits": the untrimmed brief must overflow the
    # budget, the trimmed one must not, and trimming must actually drop detail
    # (which lever fires depends on the character counts, so don't pin one).
    assert _prompt_tokens(Processor(), _prompt_messages(context, None)) > 1400
    assert tokens <= 1400
    assert (
        len(trimmed["lyrics"]) < len(context["lyrics"])
        or len(trimmed["lyric_candidates"]) < len(context["lyric_candidates"])
        or len(trimmed["assets"]) < len(context["assets"])
    )
    assert "项目数据" in messages[-1]["content"]


def test_dropping_the_candidate_table_also_rewrites_the_instruction() -> None:
    context = {
        "lyrics": [[0.0, "甲"], [3.0, "乙"], [6.0, "丙"]],
        "assets": [{"id": 1}, {"id": 2}],
        "lyric_candidates": [{"i": index, "c": [[1, .9], [2, .8]]} for index in range(3)],
        "instruction": "lyrics 是 [起始秒, 歌词] 列表；lyric_candidates 的 i 是下标。",
    }

    kept = _trim_context(context, lyric_stride=1, candidate_limit=2, asset_limit=2)
    dropped = _trim_context(context, lyric_stride=2, candidate_limit=0, asset_limit=1)

    assert len(kept["lyric_candidates"]) == 3
    assert "lyric_candidates" in kept["instruction"]
    # No candidates at all -> the brief must not point at a table that is gone.
    assert not dropped["lyric_candidates"]
    assert "lyric_candidates" not in dropped["instruction"]
    assert dropped["lyrics"] == [[0.0, "甲"], [6.0, "丙"]]
    assert dropped["assets"] == [{"id": 1}]


def test_director_retries_with_a_smaller_weight_budget_after_oom(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {"budgets": [], "empty": 0}
    response = _treatment().model_dump_json()

    class Batch(dict):
        def __init__(self):
            super().__init__(input_ids=np.zeros((1, 2), dtype=int))

        def to(self, _device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, _model_name, **_options):
            return cls()

        def apply_chat_template(self, *_args, **_kwargs):
            return Batch()

        def batch_decode(self, *_args, **_kwargs):
            return [response]

    class Model:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, _model_name, **options):
            calls["budgets"].append(options["max_memory"][0])
            return cls()

        def eval(self):
            return self

        def generate(self, **_kwargs):
            if len(calls["budgets"]) == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return np.zeros((1, 3), dtype=int)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info(_index):
            return 10 * 2**30, 12 * 2**30

        @staticmethod
        def empty_cache():
            calls["empty"] = int(calls["empty"]) + 1

        @staticmethod
        def ipc_collect():
            return None

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=FakeCuda, inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=Model,
        AutoModelForMultimodalLM=Model,
        AutoProcessor=Tokenizer,
        AutoTokenizer=Tokenizer,
    ))

    result = _generate_treatment({}, AIConfig(offline=True), "cuda", tmp_path)

    assert result.concept == _treatment().concept
    assert len(calls["budgets"]) == 2
    first, second = (float(value.removesuffix("GiB")) for value in calls["budgets"])
    assert first < 10.0        # 空闲显存减去预留，而不是 12GiB 显卡的总量
    assert second < first      # 显存不足后用更小的权重预算重试
    assert int(calls["empty"]) == 2


def test_treatment_spec_documents_every_schema_field() -> None:
    from beatforge.models.ai_director import TREATMENT_SPEC, SectionDirection

    for field in DirectorTreatment.model_fields:
        assert field in TREATMENT_SPEC, field
    for field in SectionDirection.model_fields:
        assert field in TREATMENT_SPEC, field


# Spark-X2.5's published `modeling_spark.py` is written against transformers# 4.57, while the project pins transformers >= 5.16. The repair functions rewrite
# that file in place, so these tests pin down the two rewrites as text.
LEGACY_REMOTE_MODULE = '''class Spark2_5ForCausalLM(Spark2_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def get_input_embeddings(self):
        return self.model.embedding

    def forward(self):
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
'''

# transformers.masking_utils.create_causal_mask, both sides of the rename.
MASK_PARAMETERS_5 = frozenset({
    "config", "inputs_embeds", "attention_mask", "past_key_values", "position_ids",
    "or_mask_function", "and_mask_function", "block_sequence_ids", "layer_idx",
    "allow_is_causal_skip",
})
MASK_PARAMETERS_4 = frozenset({
    "config", "input_embeds", "attention_mask", "cache_position", "past_key_values",
    "position_ids", "or_mask_function", "and_mask_function",
})


def test_remote_mask_kwargs_follow_the_installed_signature() -> None:
    from beatforge.models.ai_director import _repair_mask_kwargs

    patched, note = _repair_mask_kwargs(LEGACY_REMOTE_MODULE, MASK_PARAMETERS_5)

    assert '"inputs_embeds": inputs_embeds,' in patched
    assert '"input_embeds":' not in patched
    assert "cache_position" not in patched
    assert note == "input_embeds → inputs_embeds、删除 cache_position"
    # Everything the mask helpers still accept has to survive untouched.
    assert '"config": self.config,' in patched
    assert '"past_key_values": past_key_values,' in patched
    assert '"position_ids": position_ids,' in patched
    assert "create_causal_mask(**mask_kwargs)" in patched


def test_remote_mask_kwargs_are_left_alone_on_the_old_signature() -> None:
    from beatforge.models.ai_director import _repair_mask_kwargs

    assert _repair_mask_kwargs(LEGACY_REMOTE_MODULE, MASK_PARAMETERS_4) == (
        LEGACY_REMOTE_MODULE, None,
    )


def test_remote_mask_kwargs_are_left_alone_when_the_signature_is_unknown() -> None:
    from beatforge.models.ai_director import _repair_mask_kwargs

    # `**kwargs` signatures cannot be verified, and dropping arguments blindly
    # would change the mask instead of fixing a crash.
    assert _repair_mask_kwargs(LEGACY_REMOTE_MODULE, None) == (LEGACY_REMOTE_MODULE, None)


def test_remote_tied_weights_become_a_mapping() -> None:
    from beatforge.models.ai_director import _repair_tied_weights

    patched, note = _repair_tied_weights(LEGACY_REMOTE_MODULE)

    assert '_tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}' in patched
    assert note == "_tied_weights_keys 改为 dict 映射"


def test_remote_code_repair_rewrites_the_file_only_once(monkeypatch, tmp_path: Path) -> None:
    from beatforge.models.ai_director import _repair_remote_model_code

    module = tmp_path / "modeling_spark.py"
    module.write_text(LEGACY_REMOTE_MODULE, encoding="utf-8")
    monkeypatch.setattr(
        "beatforge.models.ai_director._remote_module_paths", lambda _model_name: [module],
    )
    monkeypatch.setattr(
        "beatforge.models.ai_director._mask_kwarg_parameters", lambda: MASK_PARAMETERS_5,
    )

    _repair_remote_model_code("unused")
    once = module.read_text(encoding="utf-8")

    assert '"inputs_embeds": inputs_embeds,' in once
    assert '_tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}' in once

    # A second run must be a no-op, otherwise every start would rewrite the file
    # and invalidate transformers' module cache.
    _repair_remote_model_code("unused")
    assert module.read_text(encoding="utf-8") == once


def test_rendered_token_count_handles_every_processor_shape() -> None:
    from beatforge.models.ai_director import _rendered_token_count

    # The shape that matters: a BatchEncoding of input_ids + attention_mask, whose
    # own len() is 2 keys while the prompt is 4 tokens long.
    assert _rendered_token_count({"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4}) == 4
    assert _rendered_token_count(np.zeros((1, 7), dtype=int)) == 7
    assert _rendered_token_count([[1, 2, 3]]) == 3
    assert _rendered_token_count([1, 2, 3]) == 3


def test_prompt_tokens_measures_tokens_not_encoding_keys() -> None:
    from beatforge.models.ai_director import _prompt_tokens

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            ids = list(range(len(messages[-1]["content"]) // 4))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    assert _prompt_tokens(Processor(), [{"role": "user", "content": "x" * 400}]) == 100


def test_prompt_tokens_falls_back_to_a_character_estimate() -> None:
    from beatforge.models.ai_director import _prompt_tokens

    class Processor:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("no chat template")

    assert _prompt_tokens(Processor(), [{"role": "user", "content": "x" * 360}]) == int(360 / 1.8) + 64


def test_mask_kwarg_parameters_intersects_both_mask_helpers(monkeypatch) -> None:
    """The remote code shares one `mask_kwargs` dict between both helpers, so a
    name accepted by only one of them must not be passed through."""
    import types

    from beatforge.models.ai_director import _mask_kwarg_parameters

    def create_causal_mask(config, inputs_embeds, attention_mask, cache_position): ...

    def create_sliding_window_causal_mask(config, inputs_embeds, attention_mask): ...

    module = types.ModuleType("transformers.masking_utils")
    module.create_causal_mask = create_causal_mask
    module.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", module)

    assert _mask_kwarg_parameters() == {"config", "inputs_embeds", "attention_mask"}


def test_mask_kwarg_parameters_gives_up_on_kwargs_signatures(monkeypatch) -> None:
    import types

    from beatforge.models.ai_director import _mask_kwarg_parameters

    def create_causal_mask(**kwargs): ...

    def create_sliding_window_causal_mask(**kwargs): ...

    module = types.ModuleType("transformers.masking_utils")
    module.create_causal_mask = create_causal_mask
    module.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", module)

    assert _mask_kwarg_parameters() is None


def test_dtype_bytes_prefers_dtype_over_the_deprecated_alias() -> None:
    from beatforge.models.ai_director import _dtype_bytes

    class Config:
        dtype = "bfloat16"

        @property
        def torch_dtype(self):
            # On transformers >= 5 this alias logs a deprecation warning when read.
            raise AssertionError("the deprecated alias must not be read when dtype exists")

    assert _dtype_bytes(Config()) == 2


def test_dtype_bytes_still_understands_configs_with_only_the_old_alias() -> None:
    from beatforge.models.ai_director import _dtype_bytes

    class LegacyConfig:
        dtype = None
        torch_dtype = "float32"

    assert _dtype_bytes(LegacyConfig()) == 4


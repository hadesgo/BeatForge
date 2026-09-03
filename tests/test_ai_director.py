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
    _generate_treatment,
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

    first = context["lyric_candidates"][0]["candidates"][0]
    assert first == {"asset_id": 8, "score": .91, "source_time": 12.4}


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
        get_device_properties=lambda _index: SimpleNamespace(total_memory=12 * 2**30),
        empty_cache=lambda: calls.__setitem__("empty", int(calls["empty"]) + 1),
        ipc_collect=lambda: calls.__setitem__("ipc", int(calls["ipc"]) + 1),
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, inference_mode=nullcontext)
    fake_transformers = SimpleNamespace(
        AutoModelForMultimodalLM=Model,
        AutoProcessor=Processor,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    result = _generate_treatment(
        {}, AIConfig(offline=True, director_quantization="none"), "cuda", tmp_path,
    )

    assert result.concept == _treatment().concept
    options = calls["model"][1]
    assert options["device_map"] == "auto"
    assert options["max_memory"][0] == "9.0GiB"
    assert options["offload_folder"] == str(tmp_path / "director-offload")
    assert calls["empty"] == 1
    assert calls["ipc"] == 1


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

from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.models.ai_director import (
    DirectorTreatment,
    _build_context,
    _estimated_tokens,
    _fit_prompt,
    _prompt_messages,
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

    monkeypatch.setattr("beatforge.models.ai_director._treat_with_llamacpp", lambda *_: treatment_result)
    assets = [MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100)]
    treatment = direct_mv(
        _analysis(), [LyricLine(0, 4, "独自醒来")], assets, None, AIConfig(), tmp_path,
    )
    assert treatment.motif_asset_ids == [1]
    assert treatment.sections[0].preferred_asset_ids == [1]
    assert len(treatment.sections) == 1


def test_director_prompt_is_trimmed_until_it_fits_the_budget() -> None:
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
    config = AIConfig(director_prompt_tokens=4000)

    messages, tokens, trimmed = _fit_prompt(context, config)

    # The contract is "trim until it fits": the untrimmed brief must overflow the
    # budget, the trimmed one must not, and trimming must actually drop detail
    # (which lever fires depends on the character counts, so don't pin one).
    assert _estimated_tokens(_prompt_messages(context)) > 4000
    assert tokens <= 4000
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


def test_treatment_spec_documents_every_schema_field() -> None:
    from beatforge.models.ai_director import TREATMENT_SPEC, SectionDirection

    for field in DirectorTreatment.model_fields:
        assert field in TREATMENT_SPEC, field
    for field in SectionDirection.model_fields:
        assert field in TREATMENT_SPEC, field


def test_prompt_messages_carry_the_spec_and_the_brief() -> None:
    """llama-server applies the chat template and tokenises server-side, so the client
    sends plain text messages - no image parts, no processor-specific shapes."""
    messages = _prompt_messages({"lyrics": [[0, "甲"]]})

    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(isinstance(message["content"], str) for message in messages)
    assert "项目数据" in messages[1]["content"]


def test_estimated_tokens_counts_characters_across_every_message() -> None:
    """The llama.cpp engine has no tokenizer, so the budget has to run on characters."""
    text = "x" * 360

    assert _estimated_tokens([{"role": "user", "content": text}]) == int(360 / 1.8) + 64
    # Text parts inside a list count too, and the system prompt is not free.
    assert _estimated_tokens([
        {"role": "system", "content": text},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]) == int(2 * 360 / 1.8) + 64

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
        duration=8,
        bpm=120,
        beats=[0, 2, 4, 6, 8],
        sections=[0, 4, 8],
        energy_times=[0, 4],
        energy_values=[0.3, 0.8],
        average_energy=0.55,
        brightness=0.5,
        mood="cinematic",
        mood_scores={"cinematic": 1},
        section_labels=["verse", "chorus"],
    )


def _treatment() -> DirectorTreatment:
    return DirectorTreatment.model_validate(
        {
            "concept": "从孤独走向释放",
            "narrative_arc": "封闭空间逐渐过渡到开阔场景",
            "visual_style": "克制的电影感",
            "color_arc": ["cold blue", "warm amber"],
            "motif_asset_ids": [1],
            "grade_profile": "cinematic",
            "transition_tone": "dark",
            "sections": [
                {
                    "section_index": 0,
                    "narrative_role": "建立人物处境",
                    "cut_intensity": 0.3,
                    "preferred_media": "image",
                    "preferred_asset_ids": [1],
                    "preferred_shot_sizes": ["closeup"],
                    "subtitle_effect": "typewriter",
                    "transition_tone": "soft",
                    "edit_intent": "continuity",
                }
            ],
        }
    )


def _motif_song() -> AudioAnalysis:
    """Long enough that the schedule can actually reserve a motif its appearances."""
    return AudioAnalysis(
        duration=40,
        bpm=120,
        beats=[x / 2 for x in range(81)],
        downbeats=[float(x) for x in range(0, 41, 2)],
        sections=[0, 20, 40],
        energy_times=[0, 40],
        energy_values=[0.5, 0.5],
        average_energy=0.5,
        brightness=0.5,
        mood="cinematic",
        mood_scores={"cinematic": 1},
        section_labels=["verse", "chorus"],
    )


def test_director_motif_is_reserved_and_recurs() -> None:
    """R-03: the director's motif is scheduled into reserved slots ahead of scoring.

    It is not merely a preferred asset any more - a motif that appears once is a shot,
    not a theme. The reservation is what makes a two-shot allusion come back, and the
    same treatment still carries the section's ``transition_tone`` onto the shot.
    """
    assets = [
        MediaAsset(
            0, Path("a.jpg"), "image", float("inf"), 100, 100, quality_score=0.5
        ),
        MediaAsset(
            1,
            Path("b.jpg"),
            "image",
            float("inf"),
            100,
            100,
            quality_score=0.5,
            shot_size="closeup",
        ),
    ]
    lyrics = [LyricLine(index * 4, index * 4 + 4, f"第{index}句") for index in range(10)]
    similarities = np.full((len(lyrics), len(assets)), 0.55)
    shots = create_plan(
        _motif_song(),
        lyrics,
        assets,
        similarities,
        min_shot=1.5,
        max_shot=5,
        treatment=_treatment(),
    )
    motifs = [shot for shot in shots if shot.is_motif]
    assert motifs, "the director's motif was never reserved"
    assert {shot.media_id for shot in motifs} == {1}
    assert len(motifs) >= 3, f"the motif only came back {len(motifs)} time(s)"
    assert shots[0].transition_tone == "soft", "the section's tone was dropped"


def test_director_receives_per_lyric_candidates_and_video_timestamps() -> None:
    assets = [
        MediaAsset(4, Path("portrait.jpg"), "image", float("inf"), 1920, 1080),
        MediaAsset(8, Path("walk.mp4"), "video", 20, 1920, 1080),
    ]
    lyrics = [LyricLine(0, 4, "穿过夜色")]
    context = _build_context(
        _analysis(),
        lyrics,
        assets,
        np.array([[0.3, 0.91]]),
        np.array([[0, 12.4]]),
    )

    row = context["lyric_candidates"][0]
    assert row["i"] == 0
    assert row["c"][0] == [8, 0.91, 12.4]
    assert context["lyrics"] == [[0, "穿过夜色"]]


def test_director_response_is_validated_and_sanitized(
    monkeypatch, tmp_path: Path
) -> None:
    treatment_result = _treatment()
    treatment_result.motif_asset_ids = [1, 999, 1]
    treatment_result.sections[0].preferred_asset_ids = [1, 999]
    invalid_section = treatment_result.sections[0].model_copy(
        update={"section_index": 8}
    )
    treatment_result.sections.append(invalid_section)

    monkeypatch.setattr(
        "beatforge.models.ai_director._treat_with_llamacpp", lambda *_: treatment_result
    )
    assets = [MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100)]
    treatment = direct_mv(
        _analysis(),
        [LyricLine(0, 4, "独自醒来")],
        assets,
        None,
        AIConfig(),
        tmp_path,
    )
    assert treatment.motif_asset_ids == [1]
    assert treatment.sections[0].preferred_asset_ids == [1]
    assert len(treatment.sections) == 1


def test_director_prompt_is_trimmed_until_it_fits_the_budget() -> None:
    context = {
        "song": {"duration": 240.0},
        "sections": [
            {"index": 0, "label": "verse", "start": 0.0, "end": 240.0, "energy": 0.5}
        ],
        "lyrics": [[index * 3.8, f"第{index}句歌词"] for index in range(120)],
        "assets": [
            {
                "id": index,
                "kind": "image",
                "description": f"素材 {index} 的详细描述",
                "mood": "calm",
            }
            for index in range(24)
        ],
        "lyric_candidates": [
            {"i": index, "c": [[index % 24, 0.8], [index % 23, 0.7]]}
            for index in range(120)
        ],
    }
    config = AIConfig(director_prompt_tokens=4000)

    messages, tokens, trimmed = _fit_prompt(context, config, _estimated_tokens, 4000)

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


def test_the_ladder_trims_further_when_the_real_count_is_higher() -> None:
    """真机踩到的那次失败，就是这个差值的形状。

    客户端字符估算给出 14240，服务端实测 16495——**低估 16%**，于是"装得下"的判据通过，
    llama.cpp 却直接拒收整个请求（``exceed_context_size_error``）。所以阶梯必须用
    能拿到的最准确计数：计数更大时，它要一路裁到真正装下为止。
    """
    context = {
        "lyrics": [
            [index * 3.0, f"第{index}句歌词，用来把提示词撑长一点"]
            for index in range(200)
        ],
        "assets": [
            {
                "id": index,
                "kind": "image",
                "description": f"素材 {index} 的详细描述文字",
                "mood": "calm",
            }
            for index in range(40)
        ],
        "lyric_candidates": [
            {"i": index, "c": [[index % 40, 0.8], [index % 39, 0.7]]}
            for index in range(200)
        ],
    }
    config = AIConfig()

    _, _, optimistic = _fit_prompt(context, config, _estimated_tokens, 6000)
    messages, tokens, pessimistic = _fit_prompt(
        context,
        config,
        lambda items: _estimated_tokens(items) * 2,
        6000,
    )

    assert tokens <= 6000, "计数超过预算时必须继续降级"
    assert len(pessimistic["assets"]) <= len(optimistic["assets"])
    assert len(pessimistic["lyrics"]) <= len(optimistic["lyrics"])
    assert "项目数据" in messages[-1]["content"]


def test_the_prompt_budget_leaves_room_for_the_answer() -> None:
    """``-c`` 同时装提示词和回复，把 ``-c`` 当提示词预算就会顶到边界。

    真机报错是 ``request (16495 tokens) exceeds the available context size (24576)`` ——
    提示词预算必须从 ``-c`` 里先扣掉 ``director_max_new_tokens``。
    """
    from beatforge.models.ai_director import PROMPT_SAFETY_TOKENS, _prompt_budget

    class Server:
        @staticmethod
        def context_size() -> int:
            return 24576

    class BlindServer:
        @staticmethod
        def context_size() -> None:
            return None

    budget, note = _prompt_budget(
        Server(),
        AIConfig(director_prompt_tokens=24576, director_max_new_tokens=4096),
    )
    assert budget == 24576 - 4096 - PROMPT_SAFETY_TOKENS
    assert "24576" in note and "4096" in note

    # 配置值更小就按配置值来；服务器没报告上下文就无从压紧。
    assert _prompt_budget(Server(), AIConfig(director_prompt_tokens=2000))[0] == 2000
    assert (
        _prompt_budget(BlindServer(), AIConfig(director_prompt_tokens=8000))[0] == 8000
    )


def test_dropping_the_candidate_table_also_rewrites_the_instruction() -> None:
    context = {
        "lyrics": [[0.0, "甲"], [3.0, "乙"], [6.0, "丙"]],
        "assets": [{"id": 1}, {"id": 2}],
        "lyric_candidates": [
            {"i": index, "c": [[1, 0.9], [2, 0.8]]} for index in range(3)
        ],
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
    assert (
        _estimated_tokens(
            [
                {"role": "system", "content": text},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ]
        )
        == int(2 * 360 / 1.8) + 64
    )

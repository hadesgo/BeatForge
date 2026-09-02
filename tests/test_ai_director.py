import json
from pathlib import Path

import numpy as np

from beatforge.audio import AudioAnalysis
from beatforge.config import AIConfig
from beatforge.lyrics import LyricLine
from beatforge.media import MediaAsset
from beatforge.models.ai_director import DirectorTreatment, direct_mv
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


def test_director_response_is_validated_and_sanitized(monkeypatch) -> None:
    payload = _treatment().model_dump()
    payload["motif_asset_ids"] = [1, 999, 1]
    payload["sections"][0]["preferred_asset_ids"] = [1, 999]
    payload["sections"].append({**payload["sections"][0], "section_index": 8})
    response = {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}

    monkeypatch.setattr("beatforge.models.ai_director._post", lambda *_: response)
    assets = [MediaAsset(1, Path("b.jpg"), "image", float("inf"), 100, 100)]
    treatment = direct_mv(
        _analysis(), [LyricLine(0, 4, "独自醒来")], assets, None,
        AIConfig(director_base_url="http://127.0.0.1:9999/v1"),
    )
    assert treatment.motif_asset_ids == [1]
    assert treatment.sections[0].preferred_asset_ids == [1]
    assert len(treatment.sections) == 1

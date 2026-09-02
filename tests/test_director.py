from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.lyrics import LyricLine
from beatforge.models.ai_director import DirectorTreatment


def test_ai_mood_selects_font_and_effects() -> None:
    analysis = AudioAnalysis(
        duration=6, bpm=130, beats=[], sections=[0, 6],
        energy_times=[0, 3], energy_values=[.2, .9], average_energy=.55,
        brightness=.7, mood="energetic", mood_scores={"energetic": .9},
        melody_times=[0, 3], melody_values=[.2, .8], melodic_motion=.5, rhythmic_density=90,
    )
    config = RenderConfig(subtitle_fonts={"energetic": "My Custom Font"})
    art = create_art_direction(analysis, [LyricLine(0, 2, "慢"), LyricLine(3, 5, "快")], config)
    assert art.font == "My Custom Font"
    assert art.line_effects == ["cinematic", "bounce"]
    assert "saturation=1.20" in art.grade_filter


def test_llm_director_controls_section_subtitle_and_style() -> None:
    analysis = AudioAnalysis(
        duration=6, bpm=100, beats=[], sections=[0, 3, 6],
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=["verse", "chorus"],
    )
    treatment = DirectorTreatment.model_validate({
        "concept": "记忆回环", "narrative_arc": "由现实进入记忆",
        "visual_style": "柔光胶片", "color_arc": ["blue", "amber"],
        "motif_asset_ids": [], "grade_profile": "dreamy", "transition_tone": "soft",
        "sections": [{
            "section_index": 0, "narrative_role": "现实", "subtitle_effect": "typewriter",
        }, {
            "section_index": 1, "narrative_role": "记忆", "subtitle_effect": "glow",
        }],
    })
    art = create_art_direction(
        analysis, [LyricLine(0, 2, "现在"), LyricLine(3, 5, "从前")], RenderConfig(), treatment,
    )
    assert art.line_effects == ["typewriter", "glow"]
    assert art.concept == "记忆回环"
    assert "saturation=.90" in art.grade_filter

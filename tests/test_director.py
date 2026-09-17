from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import PROFILES, create_art_direction
from beatforge.lyrics import SUBTITLE_EFFECTS, LyricLine
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
    # The quiet line holds; the loud one draws from the impact set, which rotates
    # rather than naming a single effect so a chorus does not repeat itself.
    assert art.line_effects[0] == "cinematic"
    assert art.line_effects[1] in {"punch", "shake", "neon_flicker"}
    assert "saturation=1.20" in art.grade_filter


def test_every_subtitle_effect_the_director_picks_can_be_drawn() -> None:
    """The ladder and the renderer have to agree, for every mood and every line.

    A name that only exists in the ladder is not an error anywhere - ``write_ass``
    quietly falls back to the plain fade, and only the lines that happen to land on
    that branch lose their animation. So sweep the whole ladder instead of trusting it.
    """
    for profile in PROFILES:
        analysis = AudioAnalysis(
            duration=12, bpm=120, beats=[0, 12], sections=[0, 4, 8, 12],
            energy_times=[0, 4, 8], energy_values=[.15, .5, .92], average_energy=.5,
            brightness=.5, mood=profile, mood_scores={profile: 1},
            melody_times=[0, 12], melody_values=[.2, .85], melodic_motion=.5,
            rhythmic_density=88, section_labels=["intro", "chorus", "outro"],
        )
        lyrics = [LyricLine(i, i + 1, f"第{i}句") for i in range(6)]
        for effect in (create_art_direction(analysis, lyrics, RenderConfig()).line_effects):
            assert effect in SUBTITLE_EFFECTS, f"{profile} picked unknown effect {effect!r}"


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

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.director import create_art_direction
from beatforge.lyrics import LyricLine


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

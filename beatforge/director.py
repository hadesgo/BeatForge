from __future__ import annotations

from dataclasses import asdict, dataclass

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.lyrics import LyricLine


@dataclass(slots=True)
class ArtDirection:
    mood: str
    font: str
    highlight_color: str
    base_subtitle_effect: str
    line_effects: list[str]
    grade_filter: str
    camera_intensity: float
    transition_tone: str
    grain: float
    vignette: bool

    def as_dict(self) -> dict:
        return asdict(self)


PROFILES = {
    "energetic": ("bounce", "eq=contrast=1.12:saturation=1.20:brightness=0.01", 1.25, "bright"),
    "uplifting": ("karaoke", "eq=contrast=1.05:saturation=1.14:brightness=0.025,colorbalance=rs=.018:bs=-.012", 1.0, "bright"),
    "melancholic": ("cinematic", "eq=contrast=1.07:saturation=.78:brightness=-.025,colorbalance=bs=.035", .65, "dark"),
    "dreamy": ("float", "eq=contrast=.94:saturation=.90:brightness=.018,colorbalance=bs=.02", .55, "soft"),
    "romantic": ("glow", "eq=contrast=1.02:saturation=1.08:brightness=.012,colorbalance=rs=.025:bs=.01", .7, "warm"),
    "dark": ("typewriter", "eq=contrast=1.16:saturation=.72:brightness=-.035,colorbalance=bs=.025", .85, "dark"),
    "cinematic": ("cinematic", "eq=contrast=1.10:saturation=.92,colorbalance=bs=.018:rs=.012", .8, "dark"),
}


def create_art_direction(analysis: AudioAnalysis, lyrics: list[LyricLine], config: RenderConfig) -> ArtDirection:
    mood = analysis.mood if analysis.mood in PROFILES else "cinematic"
    default_effect, grade, camera, tone = PROFILES[mood]
    font = config.subtitle_font if config.subtitle_font != "auto" else config.subtitle_fonts.get(mood, "Microsoft YaHei")
    base_effect = config.subtitle_effect if config.subtitle_effect != "auto" else default_effect
    line_effects = []
    for line in lyrics:
        energy = analysis.energy_at((line.start + line.end) / 2)
        melody = analysis.melody_at((line.start + line.end) / 2)
        if config.subtitle_effect != "auto":
            effect = config.subtitle_effect
        elif energy > .76 or (energy > .58 and analysis.rhythmic_density > 75):
            effect = "bounce"
        elif energy < .24:
            effect = "cinematic" if mood != "dreamy" else "float"
        elif melody > .68:
            effect = "karaoke" if mood not in {"romantic", "dreamy"} else "glow"
        elif mood in {"romantic", "dreamy"}:
            effect = "glow" if mood == "romantic" else "float"
        else:
            effect = base_effect
        line_effects.append(effect)
    return ArtDirection(
        mood=mood, font=font, highlight_color=config.subtitle_highlight_color,
        base_subtitle_effect=base_effect, line_effects=line_effects,
        grade_filter=grade, camera_intensity=round(camera * (.85 + analysis.melodic_motion * .3), 3),
        transition_tone=tone, grain=config.film_grain, vignette=config.vignette,
    )

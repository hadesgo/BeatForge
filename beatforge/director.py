from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from beatforge.audio import AudioAnalysis
from beatforge.config import RenderConfig
from beatforge.fonts import resolve_subtitle_font
from beatforge.lyrics import LyricLine

if TYPE_CHECKING:
    from beatforge.models.ai_director import DirectorTreatment


@dataclass(slots=True)
class ArtDirection:
    concept: str
    narrative_arc: str
    visual_style: str
    color_arc: list[str]
    motifs: list[int]
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


def create_art_direction(
    analysis: AudioAnalysis,
    lyrics: list[LyricLine],
    config: RenderConfig,
    treatment: DirectorTreatment | None = None,
) -> ArtDirection:
    mood = analysis.mood if analysis.mood in PROFILES else "cinematic"
    profile = treatment.grade_profile if treatment else mood
    default_effect, grade, camera, tone = PROFILES[profile]
    requested_font = (
        config.subtitle_font if config.subtitle_font != "auto"
        else config.subtitle_fonts.get(mood, "preset:modern")
    )
    font = resolve_subtitle_font(requested_font, config.subtitle_fonts_dir)
    base_effect = config.subtitle_effect if config.subtitle_effect != "auto" else default_effect
    line_effects = []
    for line in lyrics:
        midpoint = (line.start + line.end) / 2
        energy = analysis.energy_at(midpoint)
        melody = analysis.melody_at(midpoint)
        section, section_index = _section_info(analysis, midpoint)
        section_direction = treatment.section(section_index) if treatment else None
        if config.subtitle_effect != "auto":
            effect = config.subtitle_effect
        elif section_direction:
            effect = section_direction.subtitle_effect
        elif section == "outro":
            effect = "cinematic"
        elif section in {"bridge", "solo"} and energy < .7:
            effect = "glow" if melody > .5 else "cinematic"
        elif section == "chorus" and energy > .58:
            effect = "bounce" if analysis.rhythmic_density > 75 else "karaoke"
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
        concept=treatment.concept if treatment else f"{mood} music video",
        narrative_arc=treatment.narrative_arc if treatment else "Follow the energy and lyrical progression of the song.",
        visual_style=treatment.visual_style if treatment else profile,
        color_arc=(treatment.color_arc or [profile]) if treatment else [profile],
        motifs=treatment.motif_asset_ids if treatment else [],
        mood=mood, font=font, highlight_color=config.subtitle_highlight_color,
        base_subtitle_effect=base_effect, line_effects=line_effects,
        grade_filter=grade, camera_intensity=round(camera * (.85 + analysis.melodic_motion * .3), 3),
        transition_tone=treatment.transition_tone if treatment and treatment.transition_tone != "neutral" else tone,
        grain=config.film_grain, vignette=config.vignette,
    )


def _section_at(analysis: AudioAnalysis, time: float) -> str:
    return _section_info(analysis, time)[0]


def _section_info(analysis: AudioAnalysis, time: float) -> tuple[str, int]:
    for index, (start, end) in enumerate(zip(analysis.sections, analysis.sections[1:])):
        if start <= time < end:
            return (analysis.section_labels[index] if index < len(analysis.section_labels) else "unknown", index)
    return "unknown", max(0, len(analysis.sections) - 2)

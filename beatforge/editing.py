"""Editing craft, written down as configuration.

A cut is not a look. Grade, font and grain belong to the art direction, which reads the
same song; what lives here is the set of decisions an editor makes *about time* - how
long a shot is allowed to be, which musical grid a cut is allowed to land on, whether
the picture is allowed to change shot size between two shots, how loud a transition is
permitted to be. Those are the choices that make one cut feel like a CapCut beat edit
and another like a long-take music video, even when both are cutting the same footage.

Each style below is a coherent set of those habits rather than a menu of independent
knobs, because the habits are what an editor actually has. A卡点 edit is fast, lands on
every beat, changes shot size often and cuts hard; a cinematic edit is long, lands on
bar lines, holds a shot size for a while and dissolves. Picking half of one and half of
the other is how an edit ends up with no point of view.

``edit_style = "auto"`` reads the song and picks, using the same mood analysis the art
direction uses. Naming a style forces it; ``"manual"`` uses the raw render settings.
"""

from __future__ import annotations

from dataclasses import dataclass

# Which musical grid a cut is allowed to land on. This is the single most audible
# editorial decision in the whole file: it is the difference between cutting *with*
# the song and cutting on top of it.
CUT_ALIGNMENTS = ("beat", "downbeat", "phrase", "lyric")

# How loud a visible transition is allowed to be.
TRANSITION_FLAVOURS = ("subtle", "balanced", "impact")


@dataclass(frozen=True, slots=True)
class EditStyle:
    """One editor's habits, as a configuration.

    ``tempo`` is the shot length this style wants at average energy, in seconds, and
    ``energy_gain`` says how hard energy is allowed to pull that around. A style that
    wants long takes keeps ``energy_gain`` low: a long take that shortens on every
    chorus is not a long take any more.
    """

    label: str
    #: What this style is for, in one line. Shown in the log and the README.
    summary: str
    shot_min: float
    shot_max: float
    tempo: float
    energy_gain: float
    cut_alignment: str
    #: How much shorter shots get as a section builds toward its own climax. An editor
    #: tightens toward the drop; the reverse reads as an edit running out of ideas.
    section_speedup: float
    transition_density: float
    transition_flavour: str
    #: How hard the planner pushes back on two consecutive shots of the same size.
    #: Zero lets the picture settle; high values make the cut restless, which is what
    #: fast cutting needs to avoid turning into mush.
    shot_size_contrast: float
    camera_intensity: float
    composite_ratio: float
    subtitle_layout: str


EDIT_STYLES: dict[str, EditStyle] = {
    # ------------------------------------------------------------------ 快剪
    "beat": EditStyle(
        label="卡点快剪",
        summary="每拍一刀，景别频繁变化，冲击转场；短影音最常见的节奏语言",
        shot_min=.55, shot_max=1.9, tempo=1.5, energy_gain=1.9,
        cut_alignment="beat", section_speedup=.18,
        transition_density=.55, transition_flavour="impact",
        # Fast cutting only reads if the shot size keeps changing - otherwise the
        # audience stops being able to tell one cut from the next.
        shot_size_contrast=.16, camera_intensity=1.25, composite_ratio=.30,
        subtitle_layout="free",
    ),
    # ------------------------------------------------------------ 电影感长镜
    "cinematic": EditStyle(
        label="电影感长镜",
        summary="长镜、小节线切入、溶解过渡；让画面有时间自己发生",
        shot_min=3.0, shot_max=6.5, tempo=4.6, energy_gain=.5,
        cut_alignment="downbeat", section_speedup=.04,
        transition_density=.18, transition_flavour="subtle",
        # Long takes need a big size change to feel like a decision rather than a drift.
        shot_size_contrast=.20, camera_intensity=.55, composite_ratio=.12,
        subtitle_layout="band",
    ),
    # ------------------------------------------------------------ 歌词主导
    "lyric": EditStyle(
        label="歌词主导",
        summary="一句一镜，切点跟着歌词走；歌词就是叙事主线时用",
        shot_min=1.6, shot_max=4.6, tempo=3.0, energy_gain=.9,
        cut_alignment="lyric", section_speedup=.08,
        transition_density=.28, transition_flavour="subtle",
        shot_size_contrast=.14, camera_intensity=.75, composite_ratio=.20,
        subtitle_layout="free",
    ),
    # ------------------------------------------------------------ 蒙太奇
    "montage": EditStyle(
        label="蒙太奇叙事",
        summary="景别递进、段落内逐渐加速、叠化承接；靠画面并置表意",
        shot_min=1.2, shot_max=3.6, tempo=2.4, energy_gain=1.1,
        cut_alignment="phrase", section_speedup=.25,
        transition_density=.30, transition_flavour="balanced",
        # Shot-size contrast is the whole grammar here: a wide next to a close-up is a
        # sentence, two mediums next to each other is a list.
        shot_size_contrast=.22, camera_intensity=.95, composite_ratio=.34,
        subtitle_layout="free",
    ),
    # ------------------------------------------------------------ 纪实手持
    "documentary": EditStyle(
        label="纪实手持",
        summary="手持、少转场、弱设计感；让素材看起来是拍到的而不是做出来的",
        shot_min=2.4, shot_max=6.0, tempo=4.0, energy_gain=.7,
        cut_alignment="downbeat", section_speedup=.06,
        # Almost everything is a hard cut. Visible transitions are what make an edit
        # feel authored, which is the opposite of what this style is for.
        transition_density=.10, transition_flavour="subtle",
        shot_size_contrast=.12, camera_intensity=.85, composite_ratio=.05,
        subtitle_layout="band",
    ),
    # ------------------------------------------------------------ 梦幻叠化
    "dream": EditStyle(
        label="梦幻叠化",
        summary="长溶解、极慢运镜、双重曝光；画面之间不划清界限",
        shot_min=2.8, shot_max=6.0, tempo=4.2, energy_gain=.55,
        cut_alignment="phrase", section_speedup=.05,
        transition_density=.42, transition_flavour="subtle",
        shot_size_contrast=.08, camera_intensity=.45, composite_ratio=.36,
        subtitle_layout="free",
    ),
    # ------------------------------------------------------------ 冲击碎剪
    "impact": EditStyle(
        label="冲击碎剪",
        summary="极短镜、故障与闪白、抖动；留给副歌最后一段或 drop",
        shot_min=.35, shot_max=1.2, tempo=1.0, energy_gain=2.4,
        cut_alignment="beat", section_speedup=.30,
        transition_density=.70, transition_flavour="impact",
        shot_size_contrast=.20, camera_intensity=1.4, composite_ratio=.28,
        subtitle_layout="free",
    ),
    # ------------------------------------------------------------ 极简留白
    "minimal": EditStyle(
        label="极简留白",
        summary="少切、远镜、大量负空间；把注意力留给画面本身",
        # Long enough to reach the next phrase unit up: with a phrase grid, two
        # styles that both snap to the same number of bars end up cutting alike no
        # matter what tempo they asked for, and the tempo stops meaning anything.
        shot_min=4.0, shot_max=9.0, tempo=6.5, energy_gain=.35,
        cut_alignment="phrase", section_speedup=.02,
        transition_density=.14, transition_flavour="subtle",
        shot_size_contrast=.10, camera_intensity=.35, composite_ratio=.08,
        subtitle_layout="free",
    ),
}

# Mood to style, for ``edit_style = "auto"``. The mapping follows what the song is
# doing rather than what it sounds like: a melancholic song wants room to breathe, an
# energetic one wants the cut to be part of the percussion.
_MOOD_STYLES = {
    "energetic": "beat",
    "uplifting": "beat",
    "melancholic": "cinematic",
    "dreamy": "dream",
    "romantic": "lyric",
    "dark": "montage",
    "cinematic": "cinematic",
}


def resolve_style(
    requested: str, mood: str, *, average_energy: float = .5, rhythmic_density: float = 50.0,
) -> EditStyle | None:
    """Pick the editing style for a song, or ``None`` for the raw render settings.

    ``"manual"`` is the escape hatch: it means the caller has set shot lengths and
    transition density by hand and does not want a style overriding them.
    """
    if requested == "manual":
        return None
    if requested in EDIT_STYLES:
        return EDIT_STYLES[requested]
    style = EDIT_STYLES[_MOOD_STYLES.get(mood, "cinematic")]
    # A song that is loud and busy the whole way through is not a long-take song, even
    # if its mood says otherwise; the mood describes the colour, not the tempo.
    if style.cut_alignment in {"downbeat", "phrase"} and rhythmic_density > 88 and average_energy > .62:
        return EDIT_STYLES["beat"]
    return style


def style_choices() -> tuple[str, ...]:
    """Every name ``edit_style`` accepts, for validation and for the docs."""
    return ("auto", "manual", *EDIT_STYLES)

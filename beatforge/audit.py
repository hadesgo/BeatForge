"""One place where every "the program decided for the user" is written down.

A render config and an edit style are two answers to the same question - how long a
shot may be, whether the picture may split. When they disagree, the program has to
pick one, and picking silently is how a project ends up shipping a band-layout film to
someone who wrote ``subtitle_layout = "band"`` and got ``free`` anyway.

The rule this module enforces (R-02) is that **nothing is overridden without a trace**:
every decision goes through :meth:`ConfigAudit.record`, which is the single place that
emits the WARNING line, and the same list is written into ``plan.json`` as
``config_audit`` so it can be regressed against. No other module may log an override of
its own - vocal-separation fallback, deprecated keys and the quality gate all call in
here, so "was anything silently changed?" has exactly one answer to read.

Priority is *explicit config first*: a value the user actually chose wins, and the style
only owns the keys the user never picked. "Chose" is decided by
:meth:`beatforge.config.RenderConfig.explicit`, which reads ``model_fields_set`` **and**
compares against the field default - the comparison is required because the shipped
template writes the style-owned keys into every project it creates, so bare presence no
longer means a decision. A user who wants the raw render settings instead of a style
sets ``edit_style = "manual"``; everything a style does take over still lands here as a
WARNING, so no takeover is silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from beatforge.config import RenderConfig
from beatforge.editing import EditStyle

LOGGER = logging.getLogger("beatforge.audit")


def _plain(value: object) -> object:
    """Make a config value safe to drop into ``plan.json`` verbatim."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(slots=True)
class ConfigOverride:
    """One "the program changed / declined to change a setting" decision."""

    key: str
    #: The value that was asked for - the user's explicit value, or the default.
    requested: object
    #: The value that is actually in force downstream.
    effective: object
    #: What made the decision: ``edit_style:卡点快剪``, ``explicit-config``, ...
    overridden_by: str
    #: Whether ``requested`` came from a key the user explicitly wrote.
    explicit: bool = False
    #: Free-text detail, e.g. why a quality gate had to give way.
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "requested": _plain(self.requested),
            "effective": _plain(self.effective),
            "overridden_by": self.overridden_by,
            "explicit": self.explicit,
        }


class ConfigAudit:
    """The single collector every override flows through."""

    def __init__(self) -> None:
        self._items: list[ConfigOverride] = []
        self._seen: set[tuple] = set()

    def record(
        self,
        key: str,
        requested: object,
        effective: object,
        overridden_by: str,
        *,
        explicit: bool = False,
        reason: str = "",
    ) -> None:
        """Note a decision, and shout only when a value actually changed.

        A WARNING is a claim that the film will not be what the config asked for, so it
        is emitted exactly when ``effective`` differs from ``requested``. An explicit
        value that the style merely *disagrees* with is recorded (so the disagreement
        is visible in ``plan.json``) but is not an override and must not warn.
        """
        token = (key, repr(requested), repr(effective), overridden_by, explicit)
        if token in self._seen:
            return
        self._seen.add(token)
        self._items.append(ConfigOverride(
            key=key, requested=requested, effective=effective,
            overridden_by=overridden_by, explicit=explicit, reason=reason,
        ))
        if repr(requested) != repr(effective):
            LOGGER.warning(
                "%s: requested=%r effective=%r overridden_by=%s%s",
                key, requested, effective, overridden_by,
                f" ({reason})" if reason else "",
            )

    def as_list(self) -> list[dict]:
        """The audit trail, in insertion order, ready for ``plan.json``."""
        return [item.as_dict() for item in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)


#: The render keys an :class:`EditStyle` has an opinion about. Each maps to the style
#: attribute of the same job. Anything not named here is never touched by a style.
STYLE_OWNED_KEYS: dict[str, str] = {
    "min_shot_seconds": "shot_min",
    "max_shot_seconds": "shot_max",
    "transition_density": "transition_density",
    "image_composite_ratio": "composite_ratio",
    "subtitle_layout": "subtitle_layout",
}


def apply_style(
    render: RenderConfig, style: EditStyle | None, audit: ConfigAudit,
) -> RenderConfig:
    """Fold a style into a render config, explicit-first, leaving a trail.

    This is the *only* place a style is allowed to touch the render config. For every
    key the style owns: if the user wrote it, the user wins and the style is simply
    noted as disagreeing; if the user never mentioned it, the style takes over and that
    takeover is recorded against the style's label. ``audit.record`` is what turns the
    latter into a WARNING, so a run can be read back and every change explained.
    """
    if style is None:
        return render
    explicit = render.explicit()
    updates: dict[str, object] = {}
    for key, attribute in STYLE_OWNED_KEYS.items():
        value = getattr(style, attribute)
        current = getattr(render, key)
        if key in explicit:
            if current != value:
                audit.record(
                    key, requested=current, effective=current,
                    overridden_by="explicit-config", explicit=True,
                    reason=f"用户显式配置优先于风格 {style.label}（风格想要 {value!r}）",
                )
            continue
        if current != value:
            audit.record(
                key, requested=current, effective=value,
                overridden_by=f"edit_style:{style.label}", explicit=False,
            )
            updates[key] = value
    return render.model_copy(update=updates) if updates else render

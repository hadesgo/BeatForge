import sys
from types import ModuleType, SimpleNamespace

from beatforge.models.music_structure import analyze_beats


def test_allin1_result_is_normalized(monkeypatch) -> None:
    result = SimpleNamespace(
        beats=[.5, 1.0], downbeats=[.5],
        segments=[
            SimpleNamespace(start=0, end=.5, label="start"),
            SimpleNamespace(start=.5, end=2, label="chorus"),
        ],
    )
    monkeypatch.setitem(sys.modules, "allin1_infer", SimpleNamespace(analyze=lambda *args, **kwargs: result))
    output = analyze_beats("song.wav", "allin1", "cpu")
    assert output["downbeats"] == [.5]
    assert output["sections"] == [{"start": .5, "end": 2.0, "label": "chorus"}]


def test_beat_this_backend_returns_beats_and_downbeats(monkeypatch) -> None:
    """``beat-this`` is a real backend choice beside ``allin1`` and had no coverage.

    Its tracker returns beats and downbeats but no section segments, so the result has to
    carry exactly those two keys and nothing else - a stray empty ``sections`` key would
    send ``analyze_music`` down the structure path holding nothing. The tracker is also
    configured here (``final0`` checkpoints, DBN off), so a wrong constructor argument
    fails the test rather than the machine.
    """
    seen: dict = {}

    class File2Beats:
        def __init__(self, *, checkpoint_path: str, device: str, dbn: bool) -> None:
            seen["init"] = (checkpoint_path, device, dbn)

        def __call__(self, audio: str) -> tuple[list[float], list[float]]:
            seen["audio"] = audio
            return [0.5, 1.0004, 1.5], [0.5, 1.5]

    package = ModuleType("beat_this")
    inference = ModuleType("beat_this.inference")
    inference.File2Beats = File2Beats
    package.inference = inference
    monkeypatch.setitem(sys.modules, "beat_this", package)
    monkeypatch.setitem(sys.modules, "beat_this.inference", inference)

    output = analyze_beats("song.wav", "beat-this", "cpu")

    assert seen["init"] == ("final0", "cpu", False)
    assert seen["audio"] == "song.wav"
    assert output == {"beats": [0.5, 1.0, 1.5], "downbeats": [0.5, 1.5]}


def test_a_librosa_backend_reports_no_structure() -> None:
    """``librosa`` is the built-in fallback: it means "no onset model", not "no beats".

    ``analyze_music`` reads a ``None`` here as "leave the heuristic beats in place", so
    returning anything else would silently replace tempo detection with an empty grid.
    """
    assert analyze_beats("song.wav", "librosa", "cpu") is None

import sys
from types import SimpleNamespace

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

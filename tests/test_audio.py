from beatforge.audio import _heuristic_mood


def test_heuristic_mood_is_normalized() -> None:
    scores = _heuristic_mood(.8, .7, 130)
    assert abs(sum(scores.values()) - 1) < 0.0001
    assert max(scores, key=scores.get) in {"energetic", "uplifting"}

"""``classify_music`` sits on the real pipeline path but needs no checkpoint to exercise.

Only the three seams that touch the outside world are faked - the audio loader, the
processor and the model - while ``torch`` stays real, so the normalisation, the blend and
the softmax under test are the ones that actually run.

The fakes are installed on the ``AutoProcessor``/``AutoModel`` *classes* rather than on
the ``transformers`` module: the package is a lazy module, and a
``from transformers import AutoProcessor`` still reaches the real class even after the
module attribute has been replaced.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pytest
import torch
import transformers

from beatforge.models.audio_semantics import MOOD_LABELS, classify_music


class _RecordingProcessor:
    """A stand-in for ``AutoProcessor`` that reports how many clips it received."""

    def __init__(self, seen: dict) -> None:
        self._seen = seen

    def __call__(self, **kwargs) -> dict:
        if "audio" in kwargs:
            clips = kwargs["audio"]
            self._seen["clips"] = clips
            return {"input_features": torch.zeros((len(clips), 4))}
        return {"input_ids": torch.zeros((len(kwargs["text"]), 4))}


class _FakeModel:
    """A stand-in for ``AutoModel`` returning controlled feature tensors."""

    def __init__(self, text_rows, audio_rows) -> None:
        self._text = torch.tensor(text_rows, dtype=torch.float32)
        self._audio = torch.tensor(audio_rows, dtype=torch.float32)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_text_features(self, **_kwargs):
        return SimpleNamespace(pooler_output=self._text)

    def get_audio_features(self, **_kwargs):
        return SimpleNamespace(pooler_output=self._audio)


@pytest.fixture
def fake_backend(monkeypatch) -> dict:
    """Patch the loader, processor and model; hand back the recording dict.

    The model returns seven text features (the identity) against one audio feature that
    matches the first mood, so the blend has a single, checkable winner.
    """
    seen: dict = {}
    model = _FakeModel(np.eye(len(MOOD_LABELS)), [[1, 0, 0, 0, 0, 0, 0]])
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained",
        staticmethod(lambda _name, **_kwargs: _RecordingProcessor(seen)),
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained",
        staticmethod(lambda _name, **_kwargs: model),
    )
    return seen


@pytest.mark.parametrize(
    "seconds,expected_clips",
    [(20.0, 1), (30.0, 1), (45.0, 3)],
)
def test_classify_music_samples_a_long_track_at_its_centre(
    fake_backend, monkeypatch, seconds: float, expected_clips: int,
) -> None:
    """A track over 30s is classified from three centred windows, a short one whole.

    Reading the whole of a long track through the audio encoder is the expensive case the
    windowing exists to avoid; getting the boundary wrong either drops the windows a real
    song needs or pays for them on a clip that fits anyway. The boundary itself is pinned:
    exactly 30s is not "over 30s", so it still goes in whole.
    """
    samples = np.zeros(int(seconds * 48_000), dtype=np.float32)
    monkeypatch.setattr(librosa, "load", lambda *_args, **_kwargs: (samples, 48_000))

    classify_music(Path("song.wav"), model_name="fake", device="cpu", offline=True)

    clips = fake_backend["clips"]
    assert len(clips) == expected_clips
    if expected_clips == 1:
        assert len(clips[0]) == int(seconds * 48_000), "the whole short track must be used"


def test_classify_music_scores_every_mood_and_sums_to_one(fake_backend, monkeypatch) -> None:
    """The return value is a distribution over exactly the seven moods the model knows.

    A missing or extra label would index the label list against the score vector and
    report a mood the model never judged, which every downstream consumer would trust.
    """
    monkeypatch.setattr(librosa, "load", lambda *_a, **_k: (np.zeros(48_000, np.float32), 48_000))

    scores = classify_music(Path("song.wav"), model_name="fake", device="cpu", offline=True)

    assert set(scores) == set(MOOD_LABELS)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)
    # The fixtures make the model clearly favour the first mood; a softmax that collapsed
    # to uniform would still sum to one, so pin the winner as well.
    assert max(scores, key=scores.get) == "energetic"

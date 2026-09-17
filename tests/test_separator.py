"""Vocal separation ahead of transcription: everything except the model itself."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from beatforge.config import AIConfig, ProjectConfig
from beatforge.models.separator import (
    SeparationUnavailable,
    _pick_vocal_stem,
    _stem_path,
    separate_vocals,
)
from beatforge.pipeline import speech_source


def _project(tmp_path: Path, **ai: object) -> ProjectConfig:
    music = tmp_path / "song.wav"
    music.write_bytes(b"RIFF")
    return ProjectConfig.model_validate({
        "root": str(tmp_path),
        "music": str(music),
        "media_dir": str(tmp_path),
        "output": str(tmp_path / "out.mp4"),
        "cache_dir": str(tmp_path / "cache"),
        "ai": {"enabled": True, **ai},
    })


def _hide_package(monkeypatch) -> None:
    """Make ``from audio_separator.separator import Separator`` fail."""
    monkeypatch.setitem(sys.modules, "audio_separator", None)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", None)


# ------------------------------------------------------------------ the cache key


def test_the_stem_cache_is_keyed_on_the_file_and_the_model(tmp_path: Path) -> None:
    """Two checkpoints give two different stems.

    Reusing one for the other would be invisible: both are plausible audio, and the only
    thing that would show up is a slightly different transcript.
    """
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")

    first = _stem_path(audio, tmp_path, "vocals_mel_band_roformer.ckpt")
    again = _stem_path(audio, tmp_path, "vocals_mel_band_roformer.ckpt")
    other = _stem_path(audio, tmp_path, "mel_band_roformer_kim_ft_unwa.ckpt")

    assert first == again, "the same input must resolve to the same cache entry"
    assert first != other, "a different model must not reuse the stem"
    assert first.name.endswith("-vocals.wav")


def test_a_cached_stem_is_returned_without_running_the_model(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = _stem_path(audio, cache, "model.ckpt")
    cached.write_bytes(b"stem")
    _hide_package(monkeypatch)

    assert separate_vocals(audio, cache, model="model.ckpt", device="cpu", offline=True) == cached


# ------------------------------------------------------------------ picking the stem


#: The names a real run produces. The checkpoint is called ``vocals_mel_band_roformer``,
#: so *every* output filename contains the word "vocals" - including the instrumental.
_REAL_NAMES = (
    "clip_(instrumental)_vocals_mel_band_roformer.wav",
    "clip_(vocals)_vocals_mel_band_roformer.wav",
)


def test_the_vocal_stem_is_picked_by_its_marker_not_by_position(tmp_path: Path) -> None:
    """The instrumental's filename contains "vocals" too, and that is the whole trap.

    A loose substring match picks whichever file the package happens to list first, so
    half the time the recogniser is handed the accompaniment - which transcribes to
    nothing at all, and looks like a broken model rather than a broken picker. This was
    a real bug: the first end-to-end run cached the instrumental as the vocal stem.
    """
    for name in _REAL_NAMES:
        (tmp_path / name).write_bytes(b"x")

    for order in (_REAL_NAMES, tuple(reversed(_REAL_NAMES))):
        assert _pick_vocal_stem(tmp_path, list(order)) == tmp_path / _REAL_NAMES[1]


def test_the_vocal_stem_is_found_when_the_package_returns_full_paths(tmp_path: Path) -> None:
    """The package documents fully written paths; a bare filename is also plausible.

    Both have to resolve, or the run fails with "no vocal stem" against a separator that
    worked perfectly.
    """
    for name in _REAL_NAMES:
        (tmp_path / name).write_bytes(b"x")

    assert _pick_vocal_stem(
        tmp_path, [str(tmp_path / name) for name in _REAL_NAMES],
    ) == tmp_path / _REAL_NAMES[1]


def test_no_vocal_stem_is_reported_rather_than_guessed(tmp_path: Path) -> None:
    """If only the accompaniment came back, saying so beats transcribing silence."""
    (tmp_path / _REAL_NAMES[0]).write_bytes(b"x")

    assert _pick_vocal_stem(tmp_path, [_REAL_NAMES[0]]) is None


# ------------------------------------------------------------------ the run itself


def test_separation_runs_the_model_and_keeps_only_the_vocals(tmp_path: Path, monkeypatch) -> None:
    """The flow, with a stand-in for the package: load, separate, keep the vocal, tidy up."""
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    cache = tmp_path / "cache"
    cache.mkdir()
    seen: dict[str, object] = {}

    class FakeSeparator:
        def __init__(self, **options):
            seen["options"] = options

        def load_model(self, **options):
            seen["loaded"] = options

        def separate(self, path):
            seen["source"] = path
            out = Path(seen["options"]["output_dir"])
            (out / "song_(Vocals)_model.wav").write_bytes(b"vocals")
            (out / "song_(Instrumental)_model.wav").write_bytes(b"instrumental")
            # Full paths, the way the package actually returns them.
            return [str(out / "song_(Vocals)_model.wav"), str(out / "song_(Instrumental)_model.wav")]

    monkeypatch.setitem(sys.modules, "audio_separator", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "audio_separator.separator", SimpleNamespace(Separator=FakeSeparator),
    )

    stem = separate_vocals(audio, cache, model="model.ckpt", device="cpu", offline=True)

    assert stem.read_bytes() == b"vocals", "the instrumental was kept instead of the vocal"
    assert seen["loaded"] == {"model_filename": "model.ckpt"}
    assert seen["source"] == str(audio)
    assert not list((cache / "separation").iterdir()), "the scratch directory was not tidied"
    assert not list(stem.parent.glob("*.part*"))


def test_a_missing_package_says_which_extra_to_install(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")
    _hide_package(monkeypatch)

    with pytest.raises(SeparationUnavailable, match="separation"):
        separate_vocals(audio, tmp_path, model="model.ckpt", device="cpu", offline=False)


# ------------------------------------------------------------------ the pipeline path


def test_the_recogniser_is_given_the_vocal_stem(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    stem = tmp_path / "song-vocals.wav"
    stem.write_bytes(b"stem")
    monkeypatch.setattr(
        "beatforge.models.separator.separate_vocals", lambda *_a, **_k: stem,
    )

    source, label = speech_source(project, "cpu")

    assert source == stem
    assert "人声轨" in label


def test_separation_can_be_turned_off(tmp_path: Path) -> None:
    project = _project(tmp_path, separate_vocals=False)

    source, label = speech_source(project, "cpu")

    assert source == project.music
    assert "未做人声分离" in label


def test_a_missing_extra_falls_back_to_the_mix_and_says_so(tmp_path: Path, monkeypatch) -> None:
    """Degrading is fine; degrading quietly is not.

    A transcript taken from the mix is a different and worse transcript, and nothing
    downstream can tell the two apart.
    """
    project = _project(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise SeparationUnavailable("人声分离需要 audio-separator")

    monkeypatch.setattr("beatforge.models.separator.separate_vocals", unavailable)
    source, label = speech_source(project, "cpu")

    assert source == project.music
    assert "人声分离不可用" in label
    assert "audio-separator" in label


def test_separation_is_on_by_default() -> None:
    """The whole point is that the recogniser stops hearing the drum kit."""
    assert AIConfig().separate_vocals is True
    assert AIConfig().separation_model

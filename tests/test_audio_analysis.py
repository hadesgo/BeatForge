from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from beatforge.audio import AudioAnalysis, analyze_music, section_at


def test_audio_analysis_without_ai(tmp_path: Path) -> None:
    sample_rate = 22_050
    samples = librosa.clicks(times=np.arange(0, 4, .5), sr=sample_rate, length=sample_rate * 4)
    audio = tmp_path / "clicks.wav"
    sf.write(audio, samples, sample_rate)
    result = analyze_music(audio)
    assert 3.9 <= result.duration <= 4.1
    assert result.beats
    assert result.sections[0] == 0
    assert result.sections[-1] == result.duration
    assert result.mood in result.mood_scores


def _analysis(sections: list[float], labels: list[str], duration: float) -> AudioAnalysis:
    return AudioAnalysis(
        duration=duration, bpm=120, beats=[0, 1], sections=sections,
        energy_times=[0], energy_values=[.5], average_energy=.5,
        brightness=.5, mood="cinematic", mood_scores={"cinematic": 1},
        section_labels=labels,
    )


def test_a_time_past_the_last_boundary_still_names_the_last_section() -> None:
    """``sections`` spans the whole song, so the final boundary is not a cliff edge.

    The planner and the art director each carried their own copy of this lookup, and
    past the last boundary one answered ``"unknown"`` while the other answered the real
    label - the same timestamp described two different ways. Section labels drive the
    edit intent, the effect family, colour continuity and the motif choice, so that
    disagreement was not cosmetic. They share one function now.
    """
    analysis = _analysis([0.0, 5.0, 10.0], ["intro", "outro"], 10.0)

    assert section_at(analysis, 2.0) == ("intro", 0)
    assert section_at(analysis, 7.0) == ("outro", 1)
    assert section_at(analysis, 10.0) == ("outro", 1), "at the duration, not a cliff"
    assert section_at(analysis, 12.0) == ("outro", 1), "past the duration"


def test_unknown_is_only_for_a_track_with_no_sections() -> None:
    """``"unknown"`` has to mean something, or it stops being worth reporting."""
    assert section_at(_analysis([0.0, 10.0], [], 10.0), 3.0) == ("unknown", 0)
    assert section_at(_analysis([0.0, 10.0], [], 10.0), 11.0) == ("unknown", 0)


def test_a_single_boundary_does_not_index_past_the_end() -> None:
    """``max(0, len(sections) - 2)`` is the guard that keeps the index in range."""
    assert section_at(_analysis([0.0], [], 10.0), 1.0) == ("unknown", 0)

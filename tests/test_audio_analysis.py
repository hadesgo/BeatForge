from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from beatforge.audio import analyze_music


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

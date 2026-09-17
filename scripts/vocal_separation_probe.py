"""Check that vocal separation actually isolated a vocal, and how long it took.

The failure this guards against is quiet: a separator that silently returns the mix, or
a stem-picking bug that keeps the instrumental, produces a perfectly valid WAV that the
recogniser then transcribes into nothing. Listening is the real test, so this writes an
A/B pair to listen to - and it also measures the one property that separates a vocal
from a mix without a reference: **where the energy sits**.

A full mix carries the kick and the bass, so a large share of its energy is below 200 Hz.
A vocal stem has almost none there and almost all of it in the 200 Hz - 4 kHz band where
the voice lives. If that shape is not there, the stem is not a vocal.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.models.separator import SeparationUnavailable, separate_vocals
from beatforge.runtime import command

#: The band a voice lives in, and the band the rhythm section lives in.
_VOICE_BAND = (200.0, 4000.0)
_LOW_BAND = (20.0, 200.0)


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Decode to mono at a fixed rate, so the two files are comparable."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, int(rate) // divisor).astype(np.float32)
    return mono


def band_shares(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Share of total energy in the low band, the voice band, and above."""
    if audio.size < sample_rate:
        return {"low": 0.0, "voice": 0.0, "high": 0.0}
    window = np.hanning(min(len(audio), sample_rate * 30))
    spectrum = np.abs(np.fft.rfft(audio[: len(window)] * window)) ** 2
    freqs = np.fft.rfftfreq(len(window), 1 / sample_rate)
    total = float(spectrum.sum()) or 1.0
    low = float(spectrum[freqs < _LOW_BAND[1]].sum())
    voice = float(spectrum[(freqs >= _VOICE_BAND[0]) & (freqs < _VOICE_BAND[1])].sum())
    high = float(spectrum[freqs >= _VOICE_BAND[1]].sum())
    return {"low": low / total, "voice": voice / total, "high": high / total}


def to_wav(source: Path, target: Path, sample_rate: int) -> Path:
    """Extract a decodable mono WAV so the separator is not the thing being tested."""
    if source.suffix.lower() == ".wav":
        return source
    command([
        "ffmpeg", "-y", "-v", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(sample_rate), str(target),
    ])
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="歌曲或视频文件")
    parser.add_argument("--model", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seconds", type=float, default=90.0,
                        help="只处理前 N 秒，方便快速验证")
    parser.add_argument("--out", type=Path, default=Path(".probe/separation"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("PATH 中缺少 ffmpeg")
    if not args.audio.exists():
        raise SystemExit(f"找不到音频：{args.audio}")

    sample_rate = 44_100
    args.out.mkdir(parents=True, exist_ok=True)
    clip = args.out / "clip.wav"
    if args.seconds > 0:
        command([
            "ffmpeg", "-y", "-v", "error", "-t", f"{args.seconds}", "-i", str(args.audio),
            "-vn", "-ac", "1", "-ar", str(sample_rate), str(clip),
        ])
    else:
        clip = to_wav(args.audio, clip, sample_rate)

    started = time.perf_counter()
    try:
        stem = separate_vocals(
            clip, args.out / "cache", model=args.model,
            device=args.device, offline=args.offline,
        )
    except SeparationUnavailable as exc:
        raise SystemExit(f"分离不可用：{exc}") from exc
    elapsed = time.perf_counter() - started

    import soundfile as sf

    mix_audio = load_mono(clip, sample_rate)
    stem_audio = load_mono(stem, sample_rate)
    mix = band_shares(mix_audio, sample_rate)
    vocal = band_shares(stem_audio, sample_rate)
    info = sf.info(str(stem))
    minutes = len(mix_audio) / sample_rate / 60

    print(f"源：{args.audio.name} · 处理 {minutes:.2f} 分钟 · 耗时 {elapsed:.1f}s "
          f"（{elapsed / max(minutes, .01):.1f}s / 分钟音频）")
    print(f"人声轨：{stem.name} · {info.samplerate} Hz · {info.channels} ch · "
          f"{info.frames / info.samplerate:.1f}s")
    print()
    print(f"{'频段':<14}{'原始混音':>10}{'人声轨':>10}")
    for label, key in (("低频 <200Hz", "low"), ("人声 0.2–4kHz", "voice"), ("高频 >4kHz", "high")):
        print(f"{label:<14}{mix[key]:>9.1%}{vocal[key]:>10.1%}")

    # Listening is the real test, so leave a short A/B pair behind.
    ab = args.out / "ab"
    ab.mkdir(exist_ok=True)
    for label, source in (("mix", clip), ("vocals", stem)):
        command([
            "ffmpeg", "-y", "-v", "error", "-t", "20", "-i", str(source),
            "-ac", "2", "-ar", "44100", str(ab / f"{label}.wav"),
        ])
    print()
    print(f"A/B 试听片段：{ab}")

    lowered = mix["low"] - vocal["low"]
    print()
    if lowered < .05:
        print(f"⚠ 低频占比只降了 {lowered:.1%}——人声轨里仍然带着节奏组，分离可能没生效。")
        return 1
    print(f"✓ 低频占比从 {mix['low']:.1%} 降到 {vocal['low']:.1%}，"
          f"人声频段升到 {vocal['voice']:.1%}——分离确实发生了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

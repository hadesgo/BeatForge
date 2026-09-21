"""MelBand-RoFormer vocal separation, run ahead of transcription.

An ASR model asked to transcribe a full mix is being asked to hear a voice through a
drum kit. The words are there, but they are competing with everything else for the
model's attention, and the failure mode is not silence - it is a confident transcript
with the wrong words in the loud parts. Separating first hands the recogniser the stem
it is actually good at, and the same stem improves forced alignment for free, because
alignment is a question about *when* a word was sung and the answer is much easier to
find without a snare drum on top of it.

Only transcription sees the stem. Beat tracking, energy and section detection stay on
the full mix: those want the drums, and a vocal stem has none.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from beatforge.runtime import release_gpu

#: What the separation model is asked to call the stem we want.
#:
#: The package names its outputs ``<input>_(<stem>)_<model>.wav``, so the match has to
#: be on the parenthesised **stem marker** and not on the filename as a whole. The
#: checkpoint here is called ``vocals_mel_band_roformer``, which means every output file
#: contains the word "vocals" - including the instrumental. Matching loosely picked the
#: accompaniment and fed it to the recogniser, which is a failure that looks like an
#: empty transcript rather than like a bug.
_STEM_MARKER = re.compile(r"_\(([^)]+)\)_")
_VOCAL_STEMS = frozenset({"vocals", "vocal", "voice", "lead"})
_ACCOMPANIMENT_STEMS = frozenset({
    "instrumental", "inst", "accompaniment", "karaoke", "no_vocals", "other", "music",
})

#: Where the checkpoints live under a project's cache directory. Shared with the
#: download step, because two different directories would mean a pre-downloaded
#: checkpoint is never found and the model is fetched a second time on first use - the
#: download step silently not helping, which is the worst version of it failing.
SEPARATOR_SUBDIR = ("models", "separator")


def separator_model_dir(project_cache_dir: Path) -> Path:
    """The directory ``audio-separator`` keeps its checkpoints in."""
    return project_cache_dir.joinpath(*SEPARATOR_SUBDIR)


class SeparationUnavailable(RuntimeError):
    """Raised when the separation extra is not installed or the model cannot load."""


def separate_vocals(
    audio: Path, cache_dir: Path, *, model: str, device: str, offline: bool,
) -> Path:
    """Return a path to the vocal stem for ``audio``, separating it if needed.

    The stem is cached beside the project, keyed on the source file and the model, so a
    re-render does not pay for separation twice. ``offline`` restricts the model to the
    local cache; the first run needs it to be false to download the checkpoint.
    """
    target = _stem_path(audio, cache_dir, model)
    if target.is_file() and target.stat().st_size > 0:
        return target

    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise SeparationUnavailable(
            "人声分离需要 audio-separator；请运行 uv sync --extra ai --extra separation"
        ) from exc

    work = cache_dir / "separation"
    work.mkdir(parents=True, exist_ok=True)
    models = separator_model_dir(cache_dir)
    models.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        log_level=logging.ERROR,
        model_file_dir=str(models),
        output_dir=str(work),
        output_format="WAV",
        use_autocast=device == "cuda",
    )
    try:
        separator.load_model(model_filename=model)
        produced = separator.separate(str(audio))
    except Exception as exc:
        raise SeparationUnavailable(f"MelBand-RoFormer 分离失败：{type(exc).__name__}: {exc}") from exc
    finally:
        del separator
        release_gpu()

    source = _pick_vocal_stem(work, produced)
    if source is None:
        raise SeparationUnavailable(
            f"分离结果里找不到人声轨；模型输出为 {produced!r}"
        )
    temporary = target.with_name(f"{target.stem}.part{target.suffix}")
    temporary.unlink(missing_ok=True)
    shutil.move(str(source), str(temporary))
    temporary.replace(target)
    _discard(work)
    return target


def _stem_path(audio: Path, cache_dir: Path, model: str) -> Path:
    """Where the vocal stem for this file and model lives.

    The model is part of the key: two checkpoints give two different stems, and reusing
    one for the other would be invisible in the transcript.
    """
    try:
        stamp = audio.stat().st_mtime_ns
    except OSError:
        stamp = 0
    digest = hashlib.sha1(f"{audio}:{stamp}:{model}".encode()).hexdigest()[:12]
    return cache_dir / f"{audio.stem}-{digest}-vocals.wav"


def _pick_vocal_stem(work: Path, produced: list[str]) -> Path | None:
    """Find the vocal file among what the separator wrote.

    The package returns fully written paths, but a bare filename is a reasonable thing
    for a version to hand back, so both are resolved. Selection is by the stem marker
    the package puts in the name - never by position in the list, and never by a loose
    substring search, because the checkpoint's own name contains "vocals" and would make
    the accompaniment match too.
    """
    resolved: list[tuple[Path, str]] = []
    for name in produced:
        path = Path(name)
        if not path.is_absolute():
            path = work / path
        if not path.is_file():
            continue
        match = _STEM_MARKER.search(path.name)
        resolved.append((path, match.group(1).casefold() if match else ""))
    for path, stem in resolved:
        if stem in _VOCAL_STEMS:
            return path
    # No recognised marker: take anything that is not explicitly the accompaniment,
    # which still beats guessing by position.
    for path, stem in resolved:
        if stem not in _ACCOMPANIMENT_STEMS:
            return path
    return None


def _discard(work: Path) -> None:
    """Drop the separation scratch directory, keeping any leftovers out of the cache."""
    for leftover in work.iterdir():
        if leftover.is_file():
            leftover.unlink(missing_ok=True)

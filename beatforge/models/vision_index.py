from __future__ import annotations

import hashlib
import gc
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from beatforge.media import MediaAsset, estimate_focus_point
from beatforge.runtime import command


# The Qwen-VL family of processors resizes every image to its own budget before the
# vision tower sees anything - ``max_pixels = 1280 * 28 * 28`` is the usual default,
# about one megapixel. Handing one of them a 24 MP photo therefore buys nothing and
# costs a full-resolution decode, a full-resolution resize, and both alive at once.
# The reranker pays that again for every candidate, and the same asset shows up as
# several candidates, so the waste is multiplied by the length of the shortlist.
#
# Normalising first makes the processor's own resize a no-op and puts a ceiling on
# everything upstream of it. It is a ceiling and not a target: anything already under
# the budget is passed through untouched, so small libraries keep their detail and no
# cache directory is created for them.
DEFAULT_INPUT_PIXELS = 1280 * 28 * 28

# JPEG quality for the cached copies. At ~1 MP this is visually lossless, and the
# alternative - PNG - would make the cache an order of magnitude larger for no gain
# the encoders could see.
_CACHE_QUALITY = 95


def _even(value: float) -> int:
    """Nearest even integer, at least 2. yuv420p and most patch grids want even sizes."""
    return max(2, int(round(value / 2)) * 2)


def _fit_within(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Scale a frame down to ``max_pixels``, preserving its aspect ratio.

    Never upscales: an asset already inside the budget comes back unchanged, which is
    what lets the caller skip the cache entirely. A pixel budget rather than a long
    edge, because the failure mode of a long-edge rule is a tall portrait - 768 wide
    sounds small until it is 768 x 2048 and over the budget anyway.
    """
    if width <= 0 or height <= 0 or width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return _even(width * scale), _even(height * scale)


def _image_size(path: Path) -> tuple[int, int]:
    """Read an image's dimensions from its header, without decoding any pixels."""
    try:
        with Image.open(path) as image:
            return image.size
    except OSError:
        return 0, 0


def _file_digest(path: Path) -> str:
    """Cache key for a file: its path plus its modification time."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return hashlib.sha1(f"{path}:{stamp}".encode()).hexdigest()[:12]


def _load_bounded_image(path: Path, max_pixels: int) -> Image.Image:
    """Decode an image and scale it into the encoder's budget, in one pass."""
    with Image.open(path) as source:
        size = _fit_within(*source.size, max_pixels)
        source.draft("RGB", size)
        image = source.convert("RGB")
    if size == image.size or size[0] <= 0:
        return image
    return image.resize(size, Image.LANCZOS)


_QWEN_RERANKER_CHAT_TEMPLATE = r"""
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if item.type == 'image' or 'image' in item or 'image_url' in item -%}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- elif item.type == 'video' or 'video' in item -%}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' -}}
            {%- elif item.type == 'text' or 'text' in item -%}
                {{- item.text -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- set query = messages | selectattr('role', 'eq', 'query') | first -%}
{%- set document = messages | selectattr('role', 'eq', 'document') | first -%}
{{- '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n' -}}
{{- '<|im_start|>user\n<Query>: ' -}}{{- render_content(query.content) -}}
{{- '\n<Document>: ' -}}{{- render_content(document.content) -}}{{- '<|im_end|>\n' -}}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}
""".strip()


class VisionIndex:
    """WeMM/Qwen multimodal embedding index with a SigLIP2 fallback."""

    # Class-level so an index built without the full constructor - a subclass, or a
    # test that skips model loading - still has a sane input budget.
    input_pixels: int = DEFAULT_INPUT_PIXELS

    def __init__(
        self, model_name: str, device: str, offline: bool, cache_dir: Path, *, backend: str,
        reranker_model: str | None = None, rerank_top_k: int = 0,
        batch_size: int = 4, input_pixels: int = DEFAULT_INPUT_PIXELS,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 AI 依赖；CPU 电脑请运行 uv sync --extra ai --extra ai-cpu") from exc
        self.torch = torch
        self.device = device
        self.backend = backend
        self.offline = offline
        self.reranker_model = reranker_model
        self.rerank_top_k = rerank_top_k
        self.batch_size = batch_size
        self.input_pixels = input_pixels
        self.cache_dir = cache_dir / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = cache_dir / "model-input"
        if backend in {"wemm-embedding", "qwen3-vl-embedding"}:
            self._init_multimodal_embedding(model_name, device, offline)
            return
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=offline,
        ).to(device).eval()

    def similarities(self, texts: list[str], assets: list[MediaAsset], frame_samples: int) -> np.ndarray:
        unique_texts, text_rows = _unique_with_inverse(texts)
        if not assets:
            self.best_source_starts = np.zeros((len(texts), 0), dtype=float)
            return np.zeros((len(texts), 0), dtype=float)
        if self.backend in {"wemm-embedding", "qwen3-vl-embedding"}:
            query_options = {
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            }
            if self.backend == "qwen3-vl-embedding":
                query_options["prompt"] = (
                    "Retrieve the music-video shot that best matches the lyrics, narrative action, "
                    "scene, and emotional atmosphere."
                )
            text_features = np.asarray(self._encode_multimodal(
                unique_texts, query=True, show_progress_bar=_progress_enabled(), **query_options,
            ))
            documents: list[str | Image.Image | dict[str, object]] = []
            spans: list[tuple[int, int, list[Image.Image] | None, np.ndarray | None]] = []
            for asset in tqdm(
                assets, desc="视觉索引 · 素材预处理", unit="个", dynamic_ncols=True, disable=None,
            ):
                start = len(documents)
                frames, sample_times = (
                    (None, None) if asset.kind == "image"
                    else self._video_frame_samples(asset, frame_samples)
                )
                if frames is None:
                    image_document: str | dict[str, object] = self._model_image(asset)
                    if self.backend == "wemm-embedding":
                        image_document = {"image": image_document}
                    documents.append(image_document)
                else:
                    if self.backend == "wemm-embedding":
                        documents.extend({"image": frame} for frame in frames)
                    else:
                        documents.extend(frames)
                spans.append((start, len(documents), frames, sample_times))
            document_features = np.asarray(self._encode_multimodal(
                documents,
                query=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=_progress_enabled(),
            ))
            all_frame_scores = text_features @ document_features.T
            score_columns: list[np.ndarray] = []
            source_columns: list[np.ndarray] = []
            for position, (asset, (start, end, frames, sample_times)) in enumerate(
                tqdm(
                    zip(assets, spans), total=len(assets), desc="视觉索引 · 相似度聚合",
                    unit="个", dynamic_ncols=True, disable=None,
                )
            ):
                frame_scores = all_frame_scores[:, start:end]
                if asset.kind == "video":
                    assert sample_times is not None
                    source_columns.append(sample_times[np.argmax(frame_scores, axis=1)])
                    self._update_video_visuals(asset, frames or [])
                    # Drop the frames now that the only two consumers that needed them
                    # all at once - the encoder and the visual summary - are done. The
                    # reranker reads back the single frame it wants from the cache, so
                    # holding every sampled frame of every video for the rest of the
                    # run would be pure residency.
                    spans[position] = (start, end, None, sample_times)
                    top_count = min(2, frame_scores.shape[1])
                    strongest = np.partition(frame_scores, -top_count, axis=1)[:, -top_count:]
                    score_columns.append(strongest.mean(axis=1))
                else:
                    source_columns.append(np.zeros(len(unique_texts)))
                    score_columns.append(frame_scores[:, 0])
            self.best_source_starts = np.stack(source_columns, axis=1)
            scores = np.stack(score_columns, axis=1)
            if self.reranker_model and self.rerank_top_k > 0:
                scores = self._rerank(unique_texts, assets, scores, frame_samples)
            self.best_source_starts = self.best_source_starts[text_rows]
            return scores[text_rows]
        image_features = np.stack([
            self._asset_embedding(asset, frame_samples)
            for asset in tqdm(
                assets, desc="视觉索引 · 素材编码", unit="个", dynamic_ncols=True, disable=None,
            )
        ])
        text_features = self._text_embeddings(unique_texts)
        return (text_features @ image_features.T)[text_rows]

    def _encode_multimodal(self, inputs, *, query: bool, **kwargs):
        batch_size = min(self.batch_size, max(1, len(inputs)))
        method = self.model.encode
        if getattr(self, "backend", "qwen3-vl-embedding") == "wemm-embedding":
            preferred = "encode_query" if query else "encode_document"
            method = getattr(self.model, preferred, method)
        while True:
            try:
                return method(inputs, batch_size=batch_size, **kwargs)
            except self.torch.OutOfMemoryError:
                if self.device != "cuda" or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.torch.cuda.empty_cache()

    def _encode_qwen(self, inputs, **kwargs):
        """Backward-compatible helper for integrations using the old private method."""
        return self._encode_multimodal(inputs, query=False, **kwargs)

    def _init_multimodal_embedding(
        self, model_name: str, device: str, offline: bool,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("WeMM/Qwen 多模态检索需要 sentence-transformers>=5.7") from exc
        dtype = self.torch.bfloat16 if device == "cuda" else self.torch.float32
        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        load_options = {
            "model_kwargs": model_kwargs,
            "local_files_only": offline,
            "device": device,
        }
        if self.backend == "wemm-embedding":
            load_options["trust_remote_code"] = True
        self.model = SentenceTransformer(model_name, **load_options)
        self.processor = None

    def _rerank(
        self, texts: list[str], assets: list[MediaAsset], base: np.ndarray, frame_samples: int,
    ) -> np.ndarray:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-Reranker 需要 sentence-transformers>=5.4") from exc
        dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        reranker = _load_cross_encoder(
            CrossEncoder, self.reranker_model, model_kwargs,
            device=self.device, offline=self.offline,
        )
        output = base.copy()
        selections: list[tuple[int, np.ndarray]] = []
        pairs: list[tuple[str, str | Image.Image]] = []
        for row, text in enumerate(texts):
            candidates = np.argsort(base[row])[-min(self.rerank_top_k, len(assets)):][::-1]
            selections.append((row, candidates))
            for index in candidates:
                asset = assets[int(index)]
                if asset.kind == "image":
                    # The same downscaled copy the embedding pass used, so the
                    # shortlist does not decode the original once per candidate.
                    document: str | Image.Image = self._model_image(asset)
                else:
                    target_time = float(self.best_source_starts[row, int(index)])
                    document = self._video_frame_at(asset, frame_samples, target_time)
                pairs.append((text, document))
        values = np.asarray(
            reranker.predict(
                pairs, batch_size=self.batch_size,
                show_progress_bar=_progress_enabled(),
                prompt=(
                    "Judge whether the candidate shot is suitable for a polished music video. "
                    "Prioritize lyrical meaning, emotional atmosphere, composition, subject action, "
                    "shot scale, and narrative continuity."
                ),
            )
        ).reshape(-1)
        offset = 0
        for row, candidates in selections:
            end = offset + len(candidates)
            output[row, candidates] = blend_rerank_scores(base[row, candidates], values[offset:end])
            offset = end
        self.model = reranker
        return output

    def _asset_embedding(self, asset: MediaAsset, samples: int) -> np.ndarray:
        images = (
            [_load_bounded_image(asset.file, self.input_pixels)]
            if asset.kind == "image" else self._video_frames(asset, samples)
        )
        vectors: list[np.ndarray] = []
        for offset in range(0, len(images), 8):
            inputs = self.processor(images=images[offset:offset + 8], return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                features = self.torch.nn.functional.normalize(features, dim=-1)
            vectors.extend(features.float().cpu().numpy())
        vector = np.mean(vectors, axis=0)
        return vector / max(np.linalg.norm(vector), 1e-8)

    def _text_embeddings(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for offset in range(0, len(texts), 16):
            inputs = self.processor(text=texts[offset:offset + 16], padding="max_length", return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_text_features(**inputs)
                features = self.torch.nn.functional.normalize(features, dim=-1)
            vectors.extend(features.float().cpu().numpy())
        return np.stack(vectors)

    def _model_image(self, asset: MediaAsset) -> str:
        """Path to a cached, budget-sized copy of an image asset.

        The encoder's own processor resizes to the same budget regardless, so the
        original resolution buys nothing but a bigger decode - and the reranker pays
        it once per shortlist candidate. Anything already inside the budget is passed
        through untouched, so a library of small images never touches the cache.
        """
        width, height = _image_size(asset.file)
        target = _fit_within(width, height, self.input_pixels)
        if target[0] <= 0 or target == (width, height):
            return str(asset.file)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cached = self.model_dir / f"{_file_digest(asset.file)}-{target[0]}x{target[1]}.jpg"
        if cached.is_file():
            return str(cached)
        temporary = cached.with_name(f"{cached.stem}.part.jpg")
        temporary.unlink(missing_ok=True)
        try:
            with Image.open(asset.file) as source:
                # A JPEG can be decoded at 1/2, 1/4 or 1/8 scale for free, which skips
                # the full-resolution decode entirely on exactly the oversized files
                # this path exists for. A no-op on every other format.
                source.draft("RGB", target)
                source.convert("RGB").resize(target, Image.LANCZOS).save(
                    temporary, format="JPEG", quality=_CACHE_QUALITY,
                )
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
        return str(cached)

    def _video_frames(self, asset: MediaAsset, count: int) -> list[Image.Image]:
        frames, _sample_times = self._video_frame_samples(asset, count)
        return frames

    def _video_frame_at(self, asset: MediaAsset, count: int, target_time: float) -> Image.Image:
        """The single cached sample nearest ``target_time``.

        The reranker wants one frame per candidate, and a shortlist usually names the
        same video several times over. Loading the whole sample set to keep one frame
        is the difference between reading one small JPEG and reading a dozen.
        """
        times = self._video_sample_times(asset, count)
        nearest = _nearest_sample_index(times, target_time)
        frames, _actual = self._video_frame_samples(asset, count, indices=[nearest])
        return frames[0]

    def _video_frame_samples(
        self, asset: MediaAsset, count: int, *, indices: list[int] | None = None,
    ) -> tuple[list[Image.Image], np.ndarray]:
        # Decode straight to the size the encoder will use, and put that size in the
        # cache key: changing the budget has to invalidate frames cached at the old
        # one, or a project would silently keep feeding the model the previous size.
        width, height = _fit_within(asset.width, asset.height, self.input_pixels)
        scale = f"scale={width}:{height}" if width > 0 and height > 0 else "scale=768:-2"
        digest = hashlib.sha1(
            f"{asset.file}:{asset.file.stat().st_mtime_ns}:{width}x{height}".encode()
        ).hexdigest()[:12]
        sample_times = self._video_sample_times(asset, count)
        wanted = list(range(len(sample_times))) if indices is None else indices

        frames: list[Image.Image] = []
        actual_times: list[float] = []
        failed_samples = 0
        for index in wanted:
            requested_time = float(sample_times[index])
            decoded = None
            for time in _frame_attempt_times(requested_time, asset.duration):
                if any(abs(time - previous) < .001 for previous in actual_times):
                    continue
                target = self.cache_dir / f"{digest}-{round(time * 1000):012d}.jpg"
                try:
                    decoded = (_open_rgb(target), time) if target.is_file() else None
                except OSError:
                    target.unlink(missing_ok=True)
                if decoded is not None:
                    break
                temporary = target.with_name(f"{target.stem}.part.jpg")
                temporary.unlink(missing_ok=True)
                try:
                    command([
                        "ffmpeg", "-y", "-v", "error", "-ss", f"{time:.3f}",
                        "-i", str(asset.file), "-an", "-sn", "-frames:v", "1",
                        "-vf", scale, str(temporary),
                    ], capture=True)
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        continue
                    frame = _open_rgb(temporary)
                    temporary.replace(target)
                    decoded = (frame, time)
                    break
                except (OSError, subprocess.CalledProcessError):
                    continue
                finally:
                    temporary.unlink(missing_ok=True)
            if decoded is None:
                failed_samples += 1
                continue
            frame, actual_time = decoded
            frames.append(frame)
            actual_times.append(actual_time)
        if not frames:
            raise RuntimeError(
                f"无法从视频抽取任何可用画面：{asset.file}。请检查文件是否损坏、"
                "视频编码是否受当前 FFmpeg 支持。"
            )
        if failed_samples:
            tqdm.write(
                f"警告：{asset.file.name} 有 {failed_samples}/{len(wanted)} 个采样点无法解码，"
                f"已使用其余 {len(frames)} 帧继续分析。"
            )
        return frames, np.asarray(actual_times, dtype=float)

    @staticmethod
    def _video_sample_times(asset: MediaAsset, count: int) -> np.ndarray:
        return np.linspace(0.1, max(0.1, asset.duration - 0.1), count)

    @staticmethod
    def _update_video_visuals(asset: MediaAsset, frames: list[Image.Image] | list[str]) -> None:
        colors = []
        focus_points = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                continue
            thumbnail = frame.copy()
            thumbnail.thumbnail((96, 96))
            colors.append(np.median(np.asarray(thumbnail).reshape(-1, 3), axis=0))
            focus_points.append(estimate_focus_point(frame))
        if colors and asset.dominant_color == [128, 128, 128]:
            asset.dominant_color = np.median(np.stack(colors), axis=0).astype(int).tolist()
        if focus_points and asset.focus_point == [.5, .5]:
            asset.focus_point = np.median(np.asarray(focus_points), axis=0).round(4).tolist()


def blend_rerank_scores(base: np.ndarray, reranked: np.ndarray) -> np.ndarray:
    """Blend broad embedding recall with precise pairwise judgement."""
    if reranked.size == 0:
        return base
    low, high = float(reranked.min()), float(reranked.max())
    normalized = (reranked - low) / max(high - low, 1e-8)
    return base * .3 + normalized * .7


def _nearest_sample_index(sample_times: np.ndarray, target: float) -> int:
    if sample_times.size == 0:
        return 0
    return int(np.argmin(np.abs(sample_times - target)))


def _progress_enabled() -> bool:
    """Only render model-internal progress bars in an interactive terminal."""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _frame_attempt_times(requested: float, duration: float) -> list[float]:
    """Try the requested position first, then progressively safer earlier positions."""
    safe_end = max(0.0, duration - .5)
    candidates = [
        requested,
        min(requested, safe_end),
        requested - .25,
        requested - .75,
        requested - 1.5,
        requested - 3.0,
        0.0,
    ]
    output: list[float] = []
    for candidate in candidates:
        value = round(float(np.clip(candidate, 0, max(duration, 0))), 3)
        if value not in output:
            output.append(value)
    return output


def _open_rgb(path: Path) -> Image.Image:
    """Load and detach a cached frame so the underlying file can be replaced safely."""
    with Image.open(path) as source:
        frame = source.convert("RGB")
        frame.load()
    return frame


def _load_cross_encoder(CrossEncoder, model_name: str, model_kwargs: dict, *, device: str, offline: bool):
    """Load Qwen VL rerankers explicitly to bypass incompatible saved ST module metadata."""
    if "qwen3-vl-reranker" not in model_name.casefold():
        options = {"model_kwargs": model_kwargs, "local_files_only": offline}
        if "device_map" not in model_kwargs:
            options["device"] = device
        return CrossEncoder(model_name, **options)

    try:
        from sentence_transformers.cross_encoder.modules import LogitScore, Transformer

        shared = {"local_files_only": offline}
        transformer = Transformer(
            model_name,
            transformer_task="any-to-any",
            model_kwargs={**shared, **model_kwargs},
            processor_kwargs={**shared, "chat_template": _QWEN_RERANKER_CHAT_TEMPLATE},
            config_kwargs=shared.copy(),
        )
        true_token_id = transformer.tokenizer.convert_tokens_to_ids("yes")
        false_token_id = transformer.tokenizer.convert_tokens_to_ids("no")
        if not isinstance(true_token_id, int) or not isinstance(false_token_id, int):
            raise ValueError("tokenizer 没有单独的 yes/no token")
        options = {
            "modules": [
                transformer,
                LogitScore(true_token_id=true_token_id, false_token_id=false_token_id),
            ],
        }
        if "device_map" in model_kwargs:
            class DeviceMappedCrossEncoder(CrossEncoder):
                def to(self, *_args, **_kwargs):
                    return self

            return DeviceMappedCrossEncoder(**options)
        options["device"] = device
        return CrossEncoder(**options)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "无法按 Transformer(any-to-any) + LogitScore 加载 Qwen3-VL-Reranker。"
            "请确认 sentence-transformers>=5.4、transformers 和模型快照完整。"
        ) from exc


def _unique_with_inverse(values: list[str]) -> tuple[list[str], np.ndarray]:
    """Return stable unique values and rows that restore the original order."""
    unique: list[str] = []
    positions: dict[str, int] = {}
    inverse = np.empty(len(values), dtype=np.intp)
    for row, value in enumerate(values):
        position = positions.get(value)
        if position is None:
            position = len(unique)
            positions[value] = position
            unique.append(value)
        inverse[row] = position
    return unique, inverse

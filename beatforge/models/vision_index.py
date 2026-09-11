from __future__ import annotations

import hashlib
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from beatforge.media import MediaAsset, estimate_focus_point
from beatforge.models.quantization import QuantizationMode, quantized_load_kwargs
from beatforge.runtime import command


class VisionIndex:
    """WeMM/Qwen multimodal embedding index with a SigLIP2 fallback."""

    def __init__(
        self, model_name: str, device: str, offline: bool, cache_dir: Path, *, backend: str,
        reranker_model: str | None = None, rerank_top_k: int = 0,
        quantization: QuantizationMode = "none",
        batch_size: int = 4,
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
        self.quantization = quantization
        self.batch_size = batch_size
        self.cache_dir = cache_dir / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if backend in {"wemm-embedding", "qwen3-vl-embedding"}:
            self._init_multimodal_embedding(model_name, device, offline, quantization)
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
                    image_document: str | dict[str, object] = str(asset.file)
                    if self.backend == "wemm-embedding":
                        image_document = {"image": str(asset.file)}
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
            asset_spans = zip(assets, spans)
            for asset, (start, end, frames, sample_times) in tqdm(
                asset_spans, total=len(assets), desc="视觉索引 · 相似度聚合", unit="个",
                dynamic_ncols=True, disable=None,
            ):
                frame_scores = all_frame_scores[:, start:end]
                if asset.kind == "video":
                    assert sample_times is not None
                    source_columns.append(sample_times[np.argmax(frame_scores, axis=1)])
                    self._update_video_visuals(asset, frames or [])
                    top_count = min(2, frame_scores.shape[1])
                    strongest = np.partition(frame_scores, -top_count, axis=1)[:, -top_count:]
                    score_columns.append(strongest.mean(axis=1))
                else:
                    source_columns.append(np.zeros(len(unique_texts)))
                    score_columns.append(frame_scores[:, 0])
            self.best_source_starts = np.stack(source_columns, axis=1)
            scores = np.stack(score_columns, axis=1)
            if self.reranker_model and self.rerank_top_k > 0:
                scores = self._rerank(
                    unique_texts, assets, scores, frame_samples,
                    asset_frames=[frames for _start, _end, frames, _times in spans],
                    asset_frame_times=[times for _start, _end, _frames, times in spans],
                )
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
        self, model_name: str, device: str, offline: bool, quantization: QuantizationMode,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("WeMM/Qwen 多模态检索需要 sentence-transformers>=5.7") from exc
        dtype = self.torch.bfloat16 if device == "cuda" else self.torch.float32
        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        model_kwargs.update(quantized_load_kwargs(quantization, self.torch, device))
        if quantization != "none" and device == "cuda":
            model_kwargs["device_map"] = "auto"
        load_options = {
            "model_kwargs": model_kwargs,
            "local_files_only": offline,
        }
        if "device_map" not in model_kwargs:
            load_options["device"] = device
        if self.backend == "wemm-embedding":
            load_options["trust_remote_code"] = True
        self.model = SentenceTransformer(model_name, **load_options)
        self.processor = None

    def _rerank(
        self, texts: list[str], assets: list[MediaAsset], base: np.ndarray, frame_samples: int,
        *, asset_frames: list[list[Image.Image] | None] | None = None,
        asset_frame_times: list[np.ndarray | None] | None = None,
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
        model_kwargs.update(quantized_load_kwargs(self.quantization, self.torch, self.device))
        if self.quantization != "none" and self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        reranker = CrossEncoder(
            self.reranker_model,
            device=self.device,
            model_kwargs=model_kwargs,
            local_files_only=self.offline,
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
                    document: str | Image.Image = str(asset.file)
                else:
                    frames = asset_frames[int(index)] if asset_frames is not None else None
                    sample_times = asset_frame_times[int(index)] if asset_frame_times is not None else None
                    if frames is None:
                        frames, sample_times = self._video_frame_samples(asset, frame_samples)
                    if sample_times is None:
                        sample_times = self._video_sample_times(asset, len(frames))
                    target_time = float(self.best_source_starts[row, int(index)])
                    document = frames[_nearest_sample_index(sample_times, target_time)]
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
        images = [Image.open(asset.file).convert("RGB")] if asset.kind == "image" else self._video_frames(asset, samples)
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

    def _video_frames(self, asset: MediaAsset, count: int) -> list[Image.Image]:
        frames, _sample_times = self._video_frame_samples(asset, count)
        return frames

    def _video_frame_samples(self, asset: MediaAsset, count: int) -> tuple[list[Image.Image], np.ndarray]:
        digest = hashlib.sha1(f"{asset.file}:{asset.file.stat().st_mtime_ns}".encode()).hexdigest()[:12]
        frames: list[Image.Image] = []
        actual_times: list[float] = []
        failed_samples = 0
        for requested_time in self._video_sample_times(asset, count):
            decoded = None
            for time in _frame_attempt_times(float(requested_time), asset.duration):
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
                        "-vf", "scale=768:-2", str(temporary),
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
                f"警告：{asset.file.name} 有 {failed_samples}/{count} 个采样点无法解码，"
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

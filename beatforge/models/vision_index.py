from __future__ import annotations

import hashlib
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from beatforge.media import MediaAsset, estimate_focus_point
from beatforge.models.quantization import QuantizationMode, quantized_load_kwargs
from beatforge.runtime import command


class VisionIndex:
    """Qwen3-VL-Embedding index with a SigLIP2 compatibility backend."""

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
        if backend == "qwen3-vl-embedding":
            self._init_qwen(model_name, device, offline, quantization)
            return
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=offline,
        ).to(device).eval()

    def similarities(self, texts: list[str], assets: list[MediaAsset], frame_samples: int) -> np.ndarray:
        if self.backend == "qwen3-vl-embedding":
            text_features = np.asarray(self._encode_qwen(
                texts,
                prompt="Retrieve the music-video shot that best matches the lyrics, narrative action, scene, and emotional atmosphere.",
                normalize_embeddings=True,
                convert_to_numpy=True,
            ))
            documents: list[str | Image.Image] = []
            spans: list[tuple[int, int, list[Image.Image] | None]] = []
            for asset in assets:
                start = len(documents)
                frames = None if asset.kind == "image" else self._video_frames(asset, frame_samples)
                documents.extend([str(asset.file)] if frames is None else frames)
                spans.append((start, len(documents), frames))
            document_features = np.asarray(self._encode_qwen(
                documents,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ))
            score_columns: list[np.ndarray] = []
            source_columns: list[np.ndarray] = []
            for asset, (start, end, frames) in zip(assets, spans):
                vectors = document_features[start:end]
                frame_scores = text_features @ vectors.T
                if asset.kind == "video":
                    sample_times = self._video_sample_times(asset, frame_samples)
                    source_columns.append(sample_times[np.argmax(frame_scores, axis=1)])
                    self._update_video_visuals(asset, frames or [])
                    top_count = min(2, frame_scores.shape[1])
                    strongest = np.partition(frame_scores, -top_count, axis=1)[:, -top_count:]
                    score_columns.append(strongest.mean(axis=1))
                else:
                    source_columns.append(np.zeros(len(texts)))
                    score_columns.append(frame_scores[:, 0])
            self.best_source_starts = np.stack(source_columns, axis=1)
            scores = np.stack(score_columns, axis=1)
            if self.reranker_model and self.rerank_top_k > 0:
                scores = self._rerank(texts, assets, scores, frame_samples)
            return scores
        image_features = np.stack([self._asset_embedding(asset, frame_samples) for asset in assets])
        text_features = self._text_embeddings(texts)
        return text_features @ image_features.T

    def _encode_qwen(self, inputs, **kwargs):
        batch_size = min(self.batch_size, max(1, len(inputs)))
        while True:
            try:
                return self.model.encode(inputs, batch_size=batch_size, **kwargs)
            except self.torch.OutOfMemoryError:
                if self.device != "cuda" or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                self.torch.cuda.empty_cache()

    def _init_qwen(
        self, model_name: str, device: str, offline: bool, quantization: QuantizationMode,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-Embedding 需要 sentence-transformers>=5.4") from exc
        dtype = self.torch.bfloat16 if device == "cuda" else self.torch.float32
        model_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        model_kwargs.update(quantized_load_kwargs(quantization, self.torch, device))
        if quantization != "none" and device == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            local_files_only=offline,
        )
        self.processor = None

    def _rerank(self, texts: list[str], assets: list[MediaAsset], base: np.ndarray, frame_samples: int) -> np.ndarray:
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
        for row, text in enumerate(texts):
            candidates = np.argsort(base[row])[-min(self.rerank_top_k, len(assets)):][::-1]
            documents: list[str | Image.Image] = []
            for index in candidates:
                asset = assets[int(index)]
                if asset.kind == "image":
                    documents.append(str(asset.file))
                else:
                    frames = self._video_frames(asset, frame_samples)
                    sample_times = self._video_sample_times(asset, len(frames))
                    target_time = float(self.best_source_starts[row, int(index)])
                    documents.append(frames[_nearest_sample_index(sample_times, target_time)])
            values = reranker.predict(
                [(text, document) for document in documents], batch_size=self.batch_size,
                prompt=(
                    "Judge whether the candidate shot is suitable for a polished music video. "
                    "Prioritize lyrical meaning, emotional atmosphere, composition, subject action, "
                    "shot scale, and narrative continuity."
                ),
            )
            output[row, candidates] = blend_rerank_scores(base[row, candidates], np.asarray(values).reshape(-1))
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
        digest = hashlib.sha1(f"{asset.file}:{asset.file.stat().st_mtime_ns}".encode()).hexdigest()[:12]
        frames: list[Image.Image] = []
        for index, time in enumerate(self._video_sample_times(asset, count)):
            target = self.cache_dir / f"{digest}-{index}.jpg"
            if not target.exists():
                command([
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{time:.3f}", "-i", str(asset.file),
                    "-frames:v", "1", "-vf", "scale=768:-2", str(target),
                ])
            frames.append(Image.open(target).convert("RGB"))
        return frames

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

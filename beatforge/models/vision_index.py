from __future__ import annotations

import hashlib
import gc
from pathlib import Path

import numpy as np
from PIL import Image

from beatforge.media import MediaAsset
from beatforge.runtime import command


class VisionIndex:
    """Qwen3-VL-Embedding index with a SigLIP2 compatibility backend."""

    def __init__(
        self, model_name: str, device: str, offline: bool, cache_dir: Path, *, backend: str,
        reranker_model: str | None = None, rerank_top_k: int = 0,
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
        self.cache_dir = cache_dir / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if backend == "qwen3-vl-embedding":
            self._init_qwen(model_name, device, offline)
            return
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=offline,
        ).to(device).eval()

    def similarities(self, texts: list[str], assets: list[MediaAsset], frame_samples: int) -> np.ndarray:
        if self.backend == "qwen3-vl-embedding":
            text_features = np.asarray(self.model.encode(
                texts,
                prompt="检索与歌词意境、人物、场景和情绪最匹配的音乐视频画面。",
                normalize_embeddings=True,
                convert_to_numpy=True,
            ))
            asset_features = []
            for asset in assets:
                documents = [str(asset.file)] if asset.kind == "image" else self._video_frames(asset, frame_samples)
                vectors = np.asarray(self.model.encode(
                    documents, normalize_embeddings=True, convert_to_numpy=True,
                ))
                vector = vectors.mean(axis=0)
                asset_features.append(vector / max(np.linalg.norm(vector), 1e-8))
            scores = text_features @ np.stack(asset_features).T
            if self.reranker_model and self.rerank_top_k > 0:
                scores = self._rerank(texts, assets, scores, frame_samples)
            return scores
        image_features = np.stack([self._asset_embedding(asset, frame_samples) for asset in assets])
        text_features = self._text_embeddings(texts)
        return text_features @ image_features.T

    def _init_qwen(self, model_name: str, device: str, offline: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-Embedding 需要 sentence-transformers>=5.4") from exc
        dtype = self.torch.bfloat16 if device == "cuda" else self.torch.float32
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
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
        reranker = CrossEncoder(
            self.reranker_model,
            device=self.device,
            model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
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
                    documents.append(frames[len(frames) // 2])
            values = reranker.predict([(text, document) for document in documents])
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
        for index, time in enumerate(np.linspace(0.1, max(0.1, asset.duration - 0.1), count)):
            target = self.cache_dir / f"{digest}-{index}.jpg"
            if not target.exists():
                command([
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{time:.3f}", "-i", str(asset.file),
                    "-frames:v", "1", "-vf", "scale=768:-2", str(target),
                ])
            frames.append(Image.open(target).convert("RGB"))
        return frames


def blend_rerank_scores(base: np.ndarray, reranked: np.ndarray) -> np.ndarray:
    """Blend broad embedding recall with precise pairwise judgement."""
    if reranked.size == 0:
        return base
    low, high = float(reranked.min()), float(reranked.max())
    normalized = (reranked - low) / max(high - low, 1e-8)
    return base * .3 + normalized * .7

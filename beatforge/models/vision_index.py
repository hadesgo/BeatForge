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
            query_inputs = [{
                "text": text,
                "instruction": "检索与歌词意境、人物、场景和情绪最匹配的音乐视频画面。",
            } for text in texts]
            text_features = self._qwen_process(query_inputs)
            asset_features = []
            for asset in assets:
                if asset.kind == "image":
                    item = {"image": str(asset.file)}
                else:
                    item = {"image": self._video_frames(asset, frame_samples)}
                asset_features.append(self._qwen_process([item])[0])
            scores = text_features @ np.stack(asset_features).T
            if self.reranker_model and self.rerank_top_k > 0:
                scores = self._rerank(texts, assets, scores, frame_samples)
            return scores
        image_features = np.stack([self._asset_embedding(asset, frame_samples) for asset in assets])
        text_features = self._text_embeddings(texts)
        return text_features @ image_features.T

    def _init_qwen(self, model_name: str, device: str, offline: bool) -> None:
        try:
            try:
                from qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError:
                from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as exc:
            raise RuntimeError("缺少 Qwen3-VL-Embedding 官方实现，请安装 uv 的 qwen extra") from exc
        dtype = self.torch.bfloat16 if device == "cuda" else self.torch.float32
        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=offline,
            max_length=4096,
            max_frames=12,
        )
        self.processor = None

    def _qwen_process(self, inputs: list[dict]) -> np.ndarray:
        with self.torch.inference_mode():
            output = self.model.process(inputs)
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "float"):
            output = output.float().cpu().numpy()
        output = np.asarray(output)
        return output / np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1e-8)

    def _rerank(self, texts: list[str], assets: list[MediaAsset], base: np.ndarray, frame_samples: int) -> np.ndarray:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        try:
            from src.models.qwen3_vl_reranker import Qwen3VLReranker
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL-Reranker 官方实现不可用，请重新安装 qwen extra") from exc
        dtype = self.torch.bfloat16 if self.device == "cuda" else self.torch.float32
        reranker = Qwen3VLReranker(
            model_name_or_path=self.reranker_model,
            torch_dtype=dtype, attn_implementation="sdpa",
            local_files_only=self.offline, max_length=4096, max_frames=12,
        )
        output = base.copy()
        for row, text in enumerate(texts):
            candidates = np.argsort(base[row])[-min(self.rerank_top_k, len(assets)):][::-1]
            documents = []
            for index in candidates:
                asset = assets[int(index)]
                documents.append(
                    {"image": str(asset.file)} if asset.kind == "image"
                    else {"image": self._video_frames(asset, frame_samples)}
                )
            values = reranker.process({
                "instruction": "判断候选画面与歌词意境、叙事动作和情绪是否适合作为精良 MV 镜头。",
                "query": {"text": text}, "documents": documents,
            })
            if hasattr(values, "float"):
                values = values.float().cpu().numpy()
            output[row, candidates] = blend_rerank_scores(base[row, candidates], np.asarray(values).reshape(-1))
        self.model = reranker
        return output


def blend_rerank_scores(base: np.ndarray, reranked: np.ndarray) -> np.ndarray:
    """Blend broad embedding recall with precise pairwise judgement."""
    if reranked.size == 0:
        return base
    low, high = float(reranked.min()), float(reranked.max())
    normalized = (reranked - low) / max(high - low, 1e-8)
    return base * .3 + normalized * .7

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

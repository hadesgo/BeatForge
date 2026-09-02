from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from beatforge.media import MediaAsset
from beatforge.runtime import command


class VisionIndex:
    """SigLIP2 image/text embedding index, loaded for one pipeline stage only."""

    def __init__(self, model_name: str, device: str, offline: bool, cache_dir: Path) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 AI 依赖；CPU 电脑请运行 uv sync --extra ai --extra ai-cpu") from exc
        self.torch = torch
        self.device = device
        self.cache_dir = cache_dir / "frames"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=offline)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=offline,
        ).to(device).eval()

    def similarities(self, texts: list[str], assets: list[MediaAsset], frame_samples: int) -> np.ndarray:
        image_features = np.stack([self._asset_embedding(asset, frame_samples) for asset in assets])
        text_features = self._text_embeddings(texts)
        return text_features @ image_features.T

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

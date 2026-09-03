from pathlib import Path

import numpy as np

from beatforge.media import MediaAsset
from beatforge.models.vision_index import VisionIndex, blend_rerank_scores


class FakeSentenceTransformer:
    def encode(self, documents, **kwargs):
        if kwargs.get("prompt"):
            return np.array([[1.0, 0.0], [0.0, 1.0]])
        vectors = []
        for document in documents:
            vectors.append([1.0, 0.0] if "sunset" in str(document) else [0.0, 1.0])
        return np.asarray(vectors)


def test_sentence_transformer_ranks_images_for_lyrics(tmp_path: Path) -> None:
    index = VisionIndex.__new__(VisionIndex)
    index.backend = "qwen3-vl-embedding"
    index.model = FakeSentenceTransformer()
    index.reranker_model = None
    index.rerank_top_k = 0
    assets = [
        MediaAsset(0, tmp_path / "sunset.jpg", "image", float("inf"), 100, 100),
        MediaAsset(1, tmp_path / "city.jpg", "image", float("inf"), 100, 100),
    ]

    scores = index.similarities(["夕阳", "城市"], assets, frame_samples=3)

    assert scores.shape == (2, 2)
    assert np.argmax(scores[0]) == 0
    assert np.argmax(scores[1]) == 1


def test_reranker_blend_preserves_shape_and_uses_pairwise_scores() -> None:
    base = np.array([.8, .7, .6])
    result = blend_rerank_scores(base, np.array([.1, .9, .4]))
    assert result.shape == base.shape
    assert np.argmax(result) == 1

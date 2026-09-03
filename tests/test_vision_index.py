from pathlib import Path

import numpy as np

from beatforge.media import MediaAsset
from beatforge.models.vision_index import VisionIndex, _nearest_sample_index, blend_rerank_scores


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
    index.batch_size = 4
    index.reranker_model = None
    index.rerank_top_k = 0
    assets = [
        MediaAsset(0, tmp_path / "sunset.jpg", "image", float("inf"), 100, 100),
        MediaAsset(1, tmp_path / "city.jpg", "image", float("inf"), 100, 100),
    ]

    scores = index.similarities(["夕阳", "城市"], assets, frame_samples=3)

    assert scores.shape == (2, 2)
    assert index.best_source_starts.shape == (2, 2)
    assert np.argmax(scores[0]) == 0
    assert np.argmax(scores[1]) == 1


def test_reranker_blend_preserves_shape_and_uses_pairwise_scores() -> None:
    base = np.array([.8, .7, .6])
    result = blend_rerank_scores(base, np.array([.1, .9, .4]))
    assert result.shape == base.shape
    assert np.argmax(result) == 1


def test_reranker_uses_frame_nearest_to_lyric_match() -> None:
    assert _nearest_sample_index(np.array([.1, 5.0, 9.9]), 8.7) == 2


def test_visual_encoding_reduces_batch_after_cuda_oom() -> None:
    class OutOfMemoryError(Exception):
        pass

    class Model:
        def __init__(self):
            self.batches = []

        def encode(self, inputs, *, batch_size, **_kwargs):
            self.batches.append(batch_size)
            if batch_size > 2:
                raise OutOfMemoryError
            return np.ones((len(inputs), 2))

    index = VisionIndex.__new__(VisionIndex)
    index.model = Model()
    index.batch_size = 4
    index.device = "cuda"
    index.torch = type("Torch", (), {
        "OutOfMemoryError": OutOfMemoryError,
        "cuda": type("Cuda", (), {"empty_cache": staticmethod(lambda: None)})(),
    })()

    result = index._encode_qwen(["a", "b", "c", "d"])

    assert result.shape == (4, 2)
    assert index.model.batches == [4, 2]

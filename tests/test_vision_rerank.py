import numpy as np

from beatforge.models.vision_index import blend_rerank_scores


def test_reranker_can_correct_embedding_order() -> None:
    blended = blend_rerank_scores(np.array([.8, .7]), np.array([.1, .9]))
    assert blended[1] > blended[0]

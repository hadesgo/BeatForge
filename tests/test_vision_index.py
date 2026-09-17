import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import beatforge.models.vision_index as vision_index_module
from beatforge.media import MediaAsset
from beatforge.models.vision_index import (
    DEFAULT_INPUT_PIXELS,
    VisionIndex,
    _load_cross_encoder,
    _nearest_sample_index,
    _unique_with_inverse,
    blend_rerank_scores,
)


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


def test_wemm_uses_query_document_interfaces_and_explicit_image_inputs(tmp_path: Path) -> None:
    class FakeWeMM:
        def __init__(self):
            self.documents = None

        def encode_query(self, inputs, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.array([[1.0, 0.0]])

        def encode_document(self, inputs, **_kwargs):
            self.documents = inputs
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def encode(self, *_args, **_kwargs):
            raise AssertionError("WeMM should use encode_query/encode_document")

    index = VisionIndex.__new__(VisionIndex)
    index.backend = "wemm-embedding"
    index.model = FakeWeMM()
    index.batch_size = 2
    index.reranker_model = None
    index.rerank_top_k = 0
    assets = [
        MediaAsset(0, tmp_path / "match.jpg", "image", float("inf"), 100, 100),
        MediaAsset(1, tmp_path / "other.jpg", "image", float("inf"), 100, 100),
    ]

    scores = index.similarities(["歌词"], assets, frame_samples=3)

    assert np.argmax(scores[0]) == 0
    assert index.model.documents == [
        {"image": str(assets[0].file)}, {"image": str(assets[1].file)},
    ]


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


def test_duplicate_lyrics_are_encoded_once_and_restored(tmp_path: Path) -> None:
    class FakeWeMM:
        def __init__(self):
            self.queries = None

        def encode_query(self, inputs, **_kwargs):
            self.queries = inputs
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def encode_document(self, _inputs, **_kwargs):
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def encode(self, *_args, **_kwargs):
            raise AssertionError("WeMM should use encode_query/encode_document")

    index = VisionIndex.__new__(VisionIndex)
    index.backend = "wemm-embedding"
    index.model = FakeWeMM()
    index.batch_size = 4
    index.reranker_model = None
    index.rerank_top_k = 0
    assets = [
        MediaAsset(0, tmp_path / "first.jpg", "image", float("inf"), 100, 100),
        MediaAsset(1, tmp_path / "second.jpg", "image", float("inf"), 100, 100),
    ]

    scores = index.similarities(["副歌", "主歌", "副歌"], assets, frame_samples=3)

    assert index.model.queries == ["副歌", "主歌"]
    assert scores.shape == (3, 2)
    np.testing.assert_array_equal(scores[0], scores[2])
    np.testing.assert_array_equal(index.best_source_starts[0], index.best_source_starts[2])


def test_unique_with_inverse_preserves_first_seen_order() -> None:
    unique, inverse = _unique_with_inverse(["b", "a", "b", "c", "a"])

    assert unique == ["b", "a", "c"]
    np.testing.assert_array_equal(inverse, [0, 1, 0, 2, 1])


def test_reranker_reads_only_the_frame_it_needs(tmp_path: Path, monkeypatch) -> None:
    """The shortlist names the same video several times; loading all its samples for
    each of those names is the waste this avoids."""
    predictions = []
    created_options = {}

    class FakeCrossEncoder:
        def __init__(self, *_args, **kwargs):
            created_options.update(kwargs)

        def predict(self, pairs, **_kwargs):
            predictions.append(pairs)
            return np.array([.2, .8])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    index = VisionIndex.__new__(VisionIndex)
    index.model = object()
    index.device = "cuda"
    index.torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    index.reranker_model = "fake-reranker"
    index.rerank_top_k = 1
    index.batch_size = 2
    index.offline = False
    index.best_source_starts = np.array([[.1], [9.9]])
    asset = MediaAsset(0, tmp_path / "clip.mp4", "video", 10.0, 1920, 1080)
    wanted: list[float] = []

    def fake_frame_at(_asset, _count, target_time):
        wanted.append(target_time)
        return Image.new("RGB", (8, 8), "red")

    monkeypatch.setattr(index, "_video_frame_at", fake_frame_at)
    monkeypatch.setattr(
        index, "_video_frame_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the reranker must not load the whole sample set")
        ),
    )
    result = index._rerank(["first", "second"], [asset], np.array([[.7], [.6]]), 3)

    assert result.shape == (2, 1)
    assert wanted == [.1, 9.9], "each row must ask for its own best frame"
    assert len(predictions) == 1
    assert [pair[0] for pair in predictions[0]] == ["first", "second"]
    assert created_options["device"] == "cuda"
    assert created_options["model_kwargs"]["dtype"] == "bfloat16"
    assert "quantization_config" not in created_options["model_kwargs"]


def test_oversized_images_are_cached_at_the_encoder_budget(tmp_path: Path) -> None:
    """The processor resizes to its own budget anyway, so a 24 MP original only buys
    a bigger decode - once per shortlist candidate, not once per asset."""
    index = VisionIndex.__new__(VisionIndex)
    index.input_pixels = DEFAULT_INPUT_PIXELS
    index.model_dir = tmp_path / "model-input"

    big = tmp_path / "big.jpg"
    Image.new("RGB", (6000, 4000), (120, 140, 160)).save(big)
    asset = MediaAsset(0, big, "image", float("inf"), 6000, 4000)

    cached = Path(index._model_image(asset))
    assert cached != big and cached.is_file()
    with Image.open(cached) as image:
        assert image.width * image.height <= DEFAULT_INPUT_PIXELS
        assert image.width / image.height == pytest.approx(6000 / 4000, rel=.01)
    assert Path(index._model_image(asset)) == cached, "a second pass must reuse the copy"


def test_images_inside_the_budget_are_passed_through_untouched(tmp_path: Path) -> None:
    """No cache directory, no re-encode: a small library keeps exactly what it had."""
    index = VisionIndex.__new__(VisionIndex)
    index.input_pixels = DEFAULT_INPUT_PIXELS
    index.model_dir = tmp_path / "model-input"

    small = tmp_path / "small.jpg"
    Image.new("RGB", (400, 300), (120, 140, 160)).save(small)
    asset = MediaAsset(0, small, "image", float("inf"), 400, 300)

    assert index._model_image(asset) == str(small)
    assert not index.model_dir.exists()


def test_video_frames_are_decoded_to_the_budget_and_keyed_by_it(tmp_path: Path, monkeypatch) -> None:
    """Changing the budget has to invalidate frames cached at the old size, or a
    project would quietly keep feeding the model the previous one."""
    index = VisionIndex.__new__(VisionIndex)
    index.cache_dir = tmp_path / "frames"
    index.cache_dir.mkdir()
    index.input_pixels = DEFAULT_INPUT_PIXELS
    asset = MediaAsset(0, tmp_path / "clip.mp4", "video", 10.0, 6000, 4000)
    asset.file.write_bytes(b"x")
    commands: list[list[str]] = []

    def fake_command(args, **_kwargs):
        commands.append(args)
        target = Path(args[-1])
        Image.new("RGB", (8, 8), "red").save(target)

    monkeypatch.setattr("beatforge.models.vision_index.command", fake_command)
    frames, times = index._video_frame_samples(asset, 2)

    assert len(frames) == 2 and times.shape == (2,)
    scale = next(arg for arg in commands[0] if arg.startswith("scale="))
    width, height = (int(value) for value in scale.removeprefix("scale=").split(":"))
    assert width * height <= DEFAULT_INPUT_PIXELS
    assert width / height == pytest.approx(6000 / 4000, rel=.01)

    index.input_pixels = DEFAULT_INPUT_PIXELS // 4
    index._video_frame_samples(asset, 2)
    assert commands[-1] != commands[0], "a different budget must not reuse the old frames"
    assert Path(commands[-1][-1]).name.split("-")[0] != Path(commands[0][-1]).name.split("-")[0]


def test_video_frame_extraction_retries_when_ffmpeg_creates_no_output(
    tmp_path: Path, monkeypatch,
) -> None:
    index = VisionIndex.__new__(VisionIndex)
    index.cache_dir = tmp_path / "frames"
    index.cache_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-placeholder")
    asset = MediaAsset(0, video, "video", 10.0, 1920, 1080)
    attempts = []

    def fake_command(args, *, capture=False):
        assert capture is True
        attempts.append(float(args[args.index("-ss") + 1]))
        if len(attempts) == 2:
            Image.new("RGB", (8, 8), "blue").save(args[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vision_index_module, "command", fake_command)

    frames, sample_times = index._video_frame_samples(asset, 1)

    assert attempts == [.1, 0.0]
    assert len(frames) == 1
    np.testing.assert_array_equal(sample_times, [0.0])


def test_embedding_uses_native_bfloat16_without_quantization(monkeypatch) -> None:
    created = {}

    class FakeSentenceTransformer:
        def __init__(self, _model_name, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    index = VisionIndex.__new__(VisionIndex)
    index.backend = "wemm-embedding"
    index.torch = SimpleNamespace(bfloat16="bfloat16", float32="float32")

    index._init_multimodal_embedding("fake-model", "cuda", True)

    assert created["device"] == "cuda"
    assert created["model_kwargs"]["dtype"] == "bfloat16"
    assert "quantization_config" not in created["model_kwargs"]


def test_qwen_reranker_uses_explicit_multimodal_module_chain(monkeypatch) -> None:
    created = {}

    class FakeTransformer:
        def __init__(self, model_name, **kwargs):
            created["transformer"] = (model_name, kwargs)
            self.tokenizer = SimpleNamespace(
                convert_tokens_to_ids=lambda token: {"yes": 9693, "no": 2152}[token],
            )

    class FakeLogitScore:
        def __init__(self, **kwargs):
            created["logit_score"] = kwargs

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            created["cross_encoder"] = (args, kwargs)
            self.to("cuda")

        def to(self, *_args, **_kwargs):
            created["moved_after_device_map"] = True
            return self

    module = SimpleNamespace(Transformer=FakeTransformer, LogitScore=FakeLogitScore)
    monkeypatch.setitem(sys.modules, "sentence_transformers.cross_encoder.modules", module)

    result = _load_cross_encoder(
        FakeCrossEncoder,
        "/models/Qwen3-VL-Reranker-2B",
        {"device_map": "auto", "dtype": "bfloat16"},
        device="cuda",
        offline=True,
    )

    assert isinstance(result, FakeCrossEncoder)
    assert created["transformer"][1]["transformer_task"] == "any-to-any"
    assert created["transformer"][1]["model_kwargs"]["local_files_only"] is True
    chat_template = created["transformer"][1]["processor_kwargs"]["chat_template"]
    assert "selectattr('role', 'eq', 'query')" in chat_template
    assert "selectattr('role', 'eq', 'document')" in chat_template
    assert "<|image_pad|>" in chat_template
    assert "<|im_start|>assistant" in chat_template
    assert created["logit_score"] == {"true_token_id": 9693, "false_token_id": 2152}
    assert created["cross_encoder"][0] == ()
    assert "device" not in created["cross_encoder"][1]
    assert "moved_after_device_map" not in created

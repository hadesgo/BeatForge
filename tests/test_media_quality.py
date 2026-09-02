from pathlib import Path

from PIL import Image

from beatforge.media import _visual_quality


def test_visual_quality_extracts_color_and_score(tmp_path: Path) -> None:
    image = tmp_path / "red.jpg"
    Image.new("RGB", (1920, 1080), (200, 20, 20)).save(image)
    score, color = _visual_quality(image, "image", 1920, 1080)
    assert 0 <= score <= 1
    assert color[0] > color[1]

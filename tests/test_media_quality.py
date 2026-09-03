from pathlib import Path

from PIL import Image

from beatforge.media import _visual_quality, estimate_focus_point


def test_visual_quality_extracts_color_and_score(tmp_path: Path) -> None:
    image = tmp_path / "red.jpg"
    Image.new("RGB", (1920, 1080), (200, 20, 20)).save(image)
    score, color, focus = _visual_quality(image, "image", 1920, 1080)
    assert 0 <= score <= 1
    assert color[0] > color[1]
    assert focus == [.5, .5]


def test_focus_estimation_stays_conservative_for_off_center_detail() -> None:
    image = Image.new("RGB", (400, 200), (20, 20, 20))
    for x in range(280, 350):
        for y in range(40, 160):
            image.putpixel((x, y), (235, 220, 60))
    focus = estimate_focus_point(image)
    assert .5 < focus[0] <= .8
    assert .18 <= focus[1] <= .82

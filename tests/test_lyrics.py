from beatforge.lyrics import parse_lrc, srt_timestamp


def test_parse_lrc() -> None:
    lines = parse_lrc("[00:01.20]日落海边\n[00:04.50][00:08.00]一起奔跑", 10)
    assert [(x.start, x.end, x.text) for x in lines] == [
        (1.2, 4.5, "日落海边"), (4.5, 8.0, "一起奔跑"), (8.0, 10, "一起奔跑")
    ]
    assert srt_timestamp(61.234) == "00:01:01,234"


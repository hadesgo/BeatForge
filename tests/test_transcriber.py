from dataclasses import dataclass

from beatforge.models.transcriber import group_aligned_tokens


@dataclass
class Item:
    text: str
    start_time: float
    end_time: float


def test_qwen_alignment_is_grouped_into_lines() -> None:
    lines = group_aligned_tokens([
        Item("你", 0, .3), Item("好", .3, .6), Item("。", .6, .7),
        Item("远", 1.6, 1.9), Item("方", 1.9, 2.2),
    ])
    assert [line.text for line in lines] == ["你好。", "远方"]
    assert lines[0].tokens[1].start == .3


def test_qwen_alignment_accepts_transformers_dict_output() -> None:
    lines = group_aligned_tokens([
        {"text": "星", "start_time": 0.0, "end_time": 0.4},
        {"text": "光", "start_time": 0.4, "end_time": 0.8},
    ])
    assert lines[0].text == "星光"
    assert lines[0].tokens[-1].end == .8

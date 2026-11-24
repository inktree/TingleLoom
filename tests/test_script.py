# -*- coding: utf-8 -*-
"""台本解析器测试。"""
import pytest

from tingleloom import config
from tingleloom.script import (
    BedChange,
    Pause,
    ScriptError,
    SfxEvent,
    TextSegment,
    parse_duration,
    parse_pan,
    parse_sfx,
    parse_script,
)

DEMO = """# 雨夜图书馆
@preset whisper_cn
@seed 1234
@bed rain_light

欢迎来到今晚的放松时间。请找一个舒服的位置，把耳机戴好。
(pause 6)
接下来的时间里，你不需要思考什么，只要听着就好。(sfx page_turn @3.2)
慢慢地，你会觉得身体越来越沉。

(sfx ear_cleaning 3m)
(sfx breath_soft 3m)
(pause 10)

感觉怎么样？有没有舒服一点。
"""


def test_duration_parsing():
    assert parse_duration("90") == 90.0
    assert parse_duration("90s") == 90.0
    assert parse_duration("3m") == 180.0
    assert parse_duration("1m30s") == 90.0
    assert parse_duration("0.5s") == 0.5
    with pytest.raises(ValueError):
        parse_duration("abc")


def test_pan_parsing():
    assert parse_pan("C").value == 0.0
    assert parse_pan("L").value == pytest.approx(-0.7)
    assert parse_pan("R").value == pytest.approx(0.7)
    assert parse_pan("L70").value == pytest.approx(-0.7)
    assert parse_pan("R30").value == pytest.approx(0.3)
    assert parse_pan("LR").swap_interval == config.PAN_SWAP_INTERVAL
    assert parse_pan("LR").explicit is True
    assert parse_pan("LR12").swap_interval == 12.0
    assert parse_pan("lr1.5").swap_interval == 1.5
    with pytest.raises(ValueError):
        parse_pan("LR0")
    directive = parse_sfx("气泡水 44s LR15", 1)
    assert directive.description == "气泡水"
    assert directive.duration == 44.0
    assert directive.pan.swap_interval == 15.0
    with pytest.raises(ValueError):
        parse_pan("X9")


@pytest.mark.parametrize("line", ["(sfx 气泡水 44s LR0)", "你好。(pan LR0)", "你好。 LR0"])
def test_invalid_swap_interval_reports_script_line(line):
    with pytest.raises(ScriptError, match="LR 轮换周期必须为正数") as exc:
        parse_script("# 测试\n" + line)
    assert exc.value.line_no == 2


def test_parse_sfx_full():
    d = parse_sfx("耳语 @3.5 90s 大 L80", 1)
    assert d.description == "耳语"
    assert d.at == 3.5
    assert d.duration == 90.0
    assert d.level == "大"
    assert d.pan.value == pytest.approx(-0.8)


def test_parse_sfx_description_only():
    d = parse_sfx("page_turn", 1)
    assert d.description == "page_turn"
    assert d.at is None and d.duration is None and d.level == "中"


def test_parse_sfx_multiword_description():
    d = parse_sfx("glass tapping @1.5", 1)
    assert d.description == "glass tapping"
    assert d.at == 1.5


def test_parse_sfx_errors():
    with pytest.raises(ScriptError):
        parse_sfx("", 1)
    with pytest.raises(ScriptError):
        parse_sfx("@2.0", 1)          # 没有描述
    with pytest.raises(ScriptError):
        parse_sfx("翻书 @x", 1)        # 坏参数


def test_full_script():
    s = parse_script(DEMO)
    assert s.title == "雨夜图书馆"
    assert s.seed == 1234
    assert s.default_preset == "whisper_cn"
    assert [type(it).__name__ for it in s.items] == [
        "BedChange", "TextSegment", "Pause", "TextSegment",
        "TextSegment", "SfxEvent", "SfxEvent", "Pause", "TextSegment",
    ]
    assert isinstance(s.items[0], BedChange) and s.items[0].name == "rain_light"
    assert s.items[2].seconds == 6.0
    assert s.items[7].seconds == 10.0

    segs = s.text_segments
    assert len(segs) == 4
    assert segs[0].text.startswith("欢迎来到今晚")
    # 第二句带内联触发音，且描述/偏移正确
    assert "sfx" not in segs[1].text
    assert len(segs[1].sfx) == 1
    assert segs[1].sfx[0].description == "page_turn"
    assert segs[1].sfx[0].at == 3.2

    events = s.sfx_events
    assert len(events) == 2
    assert events[0].directive.description == "ear_cleaning"
    assert events[0].directive.duration == 180.0


def test_inline_pan_and_bare_trailing_pan():
    s = parse_script("你好呀。(pan L70)")
    assert s.text_segments[0].pan.value == pytest.approx(-0.7)
    assert s.text_segments[0].text == "你好呀。"

    s2 = parse_script("你好呀。 R30")
    assert s2.text_segments[0].pan.value == pytest.approx(0.3)
    assert s2.text_segments[0].text == "你好呀。"


def test_midscript_preset_switch():
    text = "@preset whisper_cn\n你好。\n@preset whisper_ja\nこんにちは。\n"
    s = parse_script(text)
    segs = s.text_segments
    assert segs[0].language == "zh" and segs[0].preset == "whisper_cn"
    assert segs[1].language == "ja" and segs[1].preset == "whisper_ja"
    assert s.default_preset == "whisper_cn"


def test_meta_title_and_pan_swap():
    s = parse_script("@title 测试\n@pan_swap 150\n@seed 42\n你好。")
    assert s.title == "测试"
    assert s.pan_swap == 150.0
    assert s.seed == 42


def test_errors_have_line_numbers():
    with pytest.raises(ScriptError) as e:
        parse_script("@preset nonexistent")
    assert e.value.line_no == 1

    with pytest.raises(ScriptError) as e2:
        parse_script("你好。\n(pause 0)")
    assert e2.value.line_no == 2

    with pytest.raises(ScriptError) as e3:
        parse_script("你好。\n(sfx 翻书 @bad)")
    assert e3.value.line_no == 2


def test_unknown_metadata_rejected():
    with pytest.raises(ScriptError, match="未知元信息：@unknown") as error:
        parse_script("你好。\n@unknown 1")
    assert error.value.line_no == 2

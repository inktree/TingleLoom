"""工作台处理器数据流与 Gradio 构建测试，不联网、不启动 TTS。"""
import json
import os
from pathlib import Path
from unittest.mock import Mock

import main as root_main

import gradio as gr
import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from tingleloom import cli, config, llm, materials, mine, pipeline, presets, webui
from tingleloom.mix import MixReport
from tingleloom.timeline import Timeline


@pytest.fixture
def catalog(tmp_path):
    items = []
    for mat_id, desc, tags in (
        ("unknown_trigger_01_L", "待确认", []),
        ("brush_01_R", "刷子", ["待标注", "轻刷"]),
        ("paper_01_C", "翻书", ["纸张"]),
    ):
        path = tmp_path / f"{mat_id}.wav"
        sf.write(path, np.zeros((1600, 2), dtype=np.float32), 16000)
        items.append(materials.Material(
            id=mat_id, file=str(path), desc=desc, tags=tags,
            duration_sec=0.1, sample_rate=16000, channels=2, base_lufs=-32.0,
            source="original.mp4 @ 35:40-36:15 (2140-2175s)",
        ))
    index = tmp_path / "index.yaml"
    materials.save_index(items, index)
    return index, items


def test_material_listing_inspection_and_relabel(catalog, monkeypatch):
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args: learned.append(args))
    index, items = catalog
    rows, choices = webui.list_material_rows(index_path=index)
    assert choices == [m.id for m in items]
    assert rows[2] == [items[2].id, "翻书", "纸张", items[2].source, 0.1, -32.0, items[2].file]
    pending_rows, pending_choices = webui.list_material_rows(True, index)
    assert pending_choices == [items[0].id, items[1].id]
    assert len(pending_rows) == 2
    audio, mat_id, desc, tags, metadata = webui.inspect_material(items[0].id, index)
    assert (audio, mat_id, desc, tags) == (items[0].file, items[0].id, "待确认", "")
    assert items[0].source in metadata
    assert "16000Hz" in metadata and "2" in metadata and "-32.0 LUFS" in metadata

    status, refreshed_rows, refreshed_choices = webui.apply_material_relabel(
        items[0].id, " cotton_swab ", " 棉签擦耳 ", " 棉签, 轻擦, ", True, index,
    )
    assert "cotton_swab_01_L" in status
    assert refreshed_choices == [items[1].id]
    assert len(refreshed_rows) == 1
    assert not items[0].path.exists()
    renamed = materials.load_index(index)[0]
    assert renamed.path.is_file() and renamed.id == "cotton_swab_01_L"
    assert renamed.desc == "棉签擦耳" and renamed.tags == ["棉签", "轻擦"]
    assert learned == [(renamed.path, "cotton_swab", "棉签擦耳", renamed.tags)]
    assert webui.inspect_material(renamed.id, index)[0] == str(renamed.path)


def test_inspect_missing_audio_and_unknown_id(catalog):
    index, items = catalog
    items[0].path.unlink()
    assert webui.inspect_material(items[0].id, index)[0] is None
    with pytest.raises(ValueError, match="未找到素材"):
        webui.inspect_material("missing", index)


@pytest.mark.parametrize("send_audio", [True, False])
def test_ai_assist_backfills_fields(catalog, tmp_path, monkeypatch, send_audio):
    index, items = catalog
    cfg_path = tmp_path / "llm.yaml"
    webui.update_llm_settings("https://api.example/v1", "test-key", "text", "ears", 0.4, cfg_path)
    calls = []

    def completion(messages, model, cfg):
        calls.append((messages, model, cfg))
        return json.dumps({"prefix": "cotton_swab", "desc": "棉签擦耳",
                           "tags": ["棉签", "轻擦"], "reason": "细密摩擦声"}, ensure_ascii=False)

    monkeypatch.setattr(llm, "chat_completion", completion)
    assert webui.run_ai_assist_identify(items[0].id, "轻微纤维声", send_audio, index, cfg_path) == (
        "cotton_swab", "棉签擦耳", "棉签,轻擦", "细密摩擦声",
    )
    messages, model, cfg = calls[0]
    assert model == ("ears" if send_audio else "text")
    assert cfg.api_key == "test-key" and cfg.temperature == 0.4
    content = messages[0]["content"]
    if send_audio:
        hint = content[0]["text"]
        assert content[1]["input_audio"]["format"] == "wav"
        assert content[1]["input_audio"]["data"]
    else:
        hint = content
    for text in (items[0].id, items[0].desc, "tags:", items[0].source, "duration_sec: 0.1", "base_lufs: -32.0", "轻微纤维声"):
        assert text in hint
    assert materials.load_index(index)[0].id == items[0].id  # AI 仅建议，不擅自落库。


def test_settings_templates_and_generation(catalog, tmp_path, monkeypatch):
    index, items = catalog
    cfg_path = tmp_path / "secrets" / "llm.yaml"
    status = webui.update_llm_settings("https://api.example/v1", "secret", "writer", "ears", 0, cfg_path)
    assert str(cfg_path) in status
    assert status == f"API 配置已保存：{cfg_path}（密钥状态：****cret）"
    assert llm.load_llm_config(cfg_path) == llm.LLMConfig("https://api.example/v1", "secret", "writer", "ears", 0)
    templates_path = tmp_path / "templates.yaml"
    original = llm.ScriptTemplate("sleep", "旧名称", "说明", "旧规则", "{{user_prompt}}")
    llm.save_templates([original], templates_path)
    webui.update_template_item("sleep", "助眠", "新说明", "只输出台本",
                               "需求 {{user_prompt}}\n素材 {{materials_catalog}}", templates_path)
    webui.update_template_item("custom", "自由", "自定义", "自由规则", "{{user_prompt}}", templates_path)
    saved = llm.load_templates(templates_path)
    assert [t.id for t in saved] == ["sleep", "custom"]
    assert saved[0].name == "助眠" and saved[0].description == "新说明"
    calls = []

    def completion(messages, model, cfg):
        calls.append((messages, model, cfg))
        return "```asmr\n@title 夜晚\n@preset whisper_cn\n晚安。\n```"

    monkeypatch.setattr(llm, "chat_completion", completion)
    text = webui.run_script_generation("三分钟翻书助眠", "sleep", index, cfg_path, templates_path)
    assert text == "@title 夜晚\n@preset whisper_cn\n晚安。"
    messages, model, cfg = calls[0]
    assert model == "writer" and cfg.temperature == 0
    assert messages[0]["content"] == "只输出台本"
    assert "三分钟翻书助眠" in messages[1]["content"]
    assert items[2].id in messages[1]["content"] and "翻书" in messages[1]["content"]
    assert "{{materials_catalog}}" not in messages[1]["content"]


@pytest.mark.parametrize("profile", ["high", "bed"])
@pytest.mark.parametrize("probe_only", [True, False])
def test_mine_result_projection(catalog, monkeypatch, probe_only, profile):
    _, items = catalog
    report = mine.AudioQualityReport(16000, 2, 60, -65, -40, -30, 35, -1, 0, 7500, 0.2,
                                     (mine.TimelineHint(0, 30, "刷麦", "brush"), mine.TimelineHint(30, 60, "说话", exclude=True)))
    candidate = mine.MinedCandidate(
        5, 20, "L", 0.9, -32.5, 12, 0.1, 0.1, 0.8, 0.4,
        vad_ratio=0.001, vad_max_run_sec=0.03, trajectory_5s="左→右",
        clap_slug="brush", clap_desc="刷子", clap_score=0.55, clap_voice_score=0.1,
        timeline_hint=report.timeline_hints[0],
    )
    calls = []

    def ingest(path, **kwargs):
        calls.append((path, kwargs))
        return report, [(candidate, None if probe_only else items[0])]

    monkeypatch.setattr(mine, "mine_and_ingest", ingest)
    summary, rows, audio = webui.run_mine_task("source.wav", profile, 35, 10, 5, "", "CC0", probe_only, "00:00-00:30 刷麦\n00:30-01:00 说话")
    assert "60.0s" in summary and "7500Hz" in summary
    assert "1 个正向章节、1 个跳过区间" in summary
    assert ("仅探测，未更新原型库" if probe_only else "已入库为待标注素材，确认英文前缀后更新原型库") in summary
    if probe_only:
        assert "仅探测模式（未入库）" in summary
        assert "取消勾选“仅探测候选”" in summary
    else:
        target_dir = config.BED_DIR if profile == "bed" else config.SFX_DIR
        assert f"已自动切分写入 `{target_dir}`" in summary
        assert f"登记入库 `{config.MATERIALS_INDEX}`" in summary
        assert "点击下方表格任意行可直接试听" in summary
    assert rows == [[1, "00:05-00:20 (5-20s)", "L", "-32.5", "0.10% / 0.03s",
                     "刷子 (0.55 / 耳语0.10)", "左→右", "brush" if probe_only else items[0].id]]
    assert audio == (None if probe_only else str(items[0].path))
    assert calls == [(Path("source.wav"), {
        "prefix": "auto", "profile": profile, "slice_seconds": 35, "min_slice_seconds": 10,
        "top_n": 5, "license": "CC0", "probe_only": probe_only, "hints_text": "00:00-00:30 刷麦\n00:30-01:00 说话",
    })]
    with pytest.raises(ValueError, match="请提供源音频路径"):
        webui.run_mine_task("", "high", 35, 10, 5, "auto", "CC0", False)


def test_synthesize_beds_handler(catalog, tmp_path, monkeypatch):
    index, _ = catalog
    out_dir = tmp_path / "bed"
    created = [materials.Material("room_tone", str(out_dir / "room_tone.wav"))]
    synthesize = Mock(side_effect=[created, []])
    monkeypatch.setattr(presets, "synthesize_beds", synthesize, raising=False)
    assert webui.run_synthesize_beds(True, index, out_dir) == "已合成并入库 1 条基础床音：room_tone"
    synthesize.assert_called_with(out_dir=out_dir, index_path=index, overwrite=True)
    assert webui.run_synthesize_beds(False, index, out_dir) == "基础床音已齐全（room_tone, rain_soft, rain_hard），无需重复生成"
    synthesize.assert_called_with(out_dir=out_dir, index_path=index, overwrite=False)


def test_preview_mined_rows(catalog):
    index, items = catalog
    rows = [[1, "0-10s", items[0].id], [2, "10-20s", items[2].id], [3, "20-30s", "brush"]]
    assert webui.preview_mined_row(rows, 0, index) == str(items[0].path)
    assert webui.preview_mined_row(rows, 1, index) == str(items[2].path)
    items[2].path.unlink()
    assert webui.preview_mined_row(rows, 1, index) is None
    assert webui.preview_mined_row(rows, 2, index) is None


def test_save_and_render_script(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXAMPLES_DIR", tmp_path / "examples")
    monkeypatch.setenv("TINGLELOOM_TTS_PROVIDERS", "cpu")
    calls = []
    output = tmp_path / "finished.wav"
    mp3_output = output.with_suffix(".mp3")

    def render(script_path):
        calls.append(script_path)
        assert script_path.read_text(encoding="utf-8") == "@title 夜晚\n晚安。"
        assert config.tts_providers()[0][0] == "CPUExecutionProvider"
        report = MixReport(output, mp3_output, 12, 44100, 0.2, -40, -40, -60, 22, -2, 0, 1, 0, 0)
        return pipeline.RunResult(script_path, output, mp3_output, Timeline(), [], [], ["刷子"], report, ["测试警告"])

    monkeypatch.setattr(pipeline, "run_pipeline", render)
    path, mp3_path, summary = webui.save_and_render_script("@title 夜晚\n晚安。", "custom.txt", "cpu")
    assert calls == [config.EXAMPLES_DIR / "custom.asmr"]
    assert path == str(output)
    assert mp3_path == str(mp3_output)
    assert f"成品 WAV：{output}" in summary
    assert f"成品 MP3：{mp3_output}" in summary
    assert "12.0s" in summary and "刷子" in summary and "测试警告" in summary


@pytest.mark.parametrize("stored_key", ["${OPENAI_API_KEY}", "sk-12345678"])
def test_settings_blank_key_preserves_configuration(tmp_path, monkeypatch, stored_key):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-12345678")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump({"api_key": stored_key, "model": "old-model"}), encoding="utf-8")
    status = webui.update_llm_settings("https://api.example/v1", "", "new-model", "ears", 0.4, path)
    assert "****5678" in status
    assert "sk-12345678" not in status
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["api_key"] == stored_key
    assert llm.load_llm_config(path).model == "new-model"


def test_build_webui(monkeypatch, catalog):
    key = "sk-browser-must-not-receive-12345678"
    monkeypatch.setattr(llm, "load_llm_config", lambda: llm.LLMConfig(api_key=key))
    index, items = catalog
    list_rows = webui.list_material_rows
    monkeypatch.setattr(webui, "list_material_rows",
                        lambda flag=False, index_path=index, category="all": list_rows(flag, index_path, category))
    list_categories = webui.list_material_categories
    monkeypatch.setattr(webui, "list_material_categories",
                        lambda flag=False, index_path=index: list_categories(flag, index_path))
    relabel_view = webui._relabel_material_view
    monkeypatch.setattr(webui, "_relabel_material_view",
                        lambda mat_id, prefix, desc, tags, flag, index_path, category:
                        relabel_view(mat_id, prefix, desc, tags, flag, index, category))
    monkeypatch.setattr(mine, "record_prototype", lambda *args: None)
    mined_rows = [[1, "0-10s", items[0].id], [2, "10-20s", items[2].id]]
    run_mine = Mock(return_value=("完成", mined_rows, str(items[0].path)))
    monkeypatch.setattr(webui, "run_mine_task", run_mine)
    # 事件处理器使用真实素材索引，但不依赖项目素材库。
    original_preview = webui.preview_mined_row
    preview_row = Mock(side_effect=lambda rows, idx: original_preview(rows, idx, index))
    monkeypatch.setattr(webui, "preview_mined_row", preview_row)
    app = webui.build_webui()
    try:
        assert isinstance(app, gr.Blocks)
        components = app.config["components"]
        functions = {item.fn.__name__: item.fn for item in app.fns.values() if item.fn is not None}
        for block_fn in app.fns.values():
            if block_fn.fn is not None and block_fn.fn.__name__ == "select_material":
                assert len(block_fn.outputs) == 7
            if block_fn.fn is not None and block_fn.fn.__name__ == "relabel_and_refresh":
                assert len(block_fn.outputs) == 13
        for event_index in ([1, 3], 1):
            event = gr.SelectData(None, {"index": event_index, "value": items[2].id})
            assert functions["on_mine_row_select"](mined_rows, event) == str(items[2].path)
        refreshed = functions["mine_and_refresh"]("source.wav", "high", 35, 10, 15, "auto", "CC0", False, True, "00:00-00:30 刷麦")
        assert refreshed[:3] == ("完成", mined_rows, str(items[0].path))
        assert refreshed[3] == list_rows(True, index)[0]
        assert refreshed[4].choices == [(m.id, m.id) for m in items[:2]]
        assert refreshed[4].value is None
        assert refreshed[5].value == "all"
        filtered = functions["refresh_materials"](False, "paper")
        assert filtered[0] == list_rows(False, index, "paper")[0]
        assert filtered[9].value == "paper"
        reset = functions["refresh_materials"](True, "paper")
        assert reset[9].value == "all"
        rows_after_relabel = Mock(wraps=webui.list_material_rows)
        categories_after_relabel = Mock(wraps=webui.list_material_categories)
        monkeypatch.setattr(webui, "list_material_rows", rows_after_relabel)
        monkeypatch.setattr(webui, "list_material_categories", categories_after_relabel)
        renamed = functions["relabel_and_refresh"](items[2].id, "new_paper", "翻书", "纸张", False, "paper")
        rows_after_relabel.assert_called_once_with(False, index, "all")
        categories_after_relabel.assert_called_once_with(False, index)
        assert renamed[10].value == "all"
        assert len(renamed[1]) == 3
        kept = functions["mine_and_refresh"]("source.wav", "high", 35, 10, 15, "auto", "CC0", False, False, "", "new_paper")
        assert kept[5].value == "new_paper"
        assert len(kept[3]) == 1
        run_mine.assert_any_call("source.wav", "high", 35, 10, 15, "auto", "CC0", False, "00:00-00:30 刷麦")
        upload_props = next(item["props"] for item in components if item["type"] == "file")
        assert upload_props["file_types"] == ["audio", "video"]
        key_props = next(item["props"] for item in components
                         if item["props"].get("label", "").startswith("api_key"))
        assert key_props["value"] == ""
        assert "****5678" in key_props["placeholder"]
        assert key not in json.dumps(app.config, default=str)
        api_info = app.get_api_info()
        assert "named_endpoints" in api_info
        with TestClient(gr.routes.App.create_app(app)) as client:
            response = client.get("/")
        assert response.status_code == 200
    finally:
        app.close()


@pytest.mark.parametrize("argv,port,inbrowser,tts_dir", [
    ([], 7860, True, None),
    (["--port", "7861", "--no-browser"], 7861, False, None),
    (["--tts-dir", "/tmp/tts_deploy", "--port", "7861", "--no-browser"], 7861, False, "/tmp/tts_deploy"),
])
def test_root_launcher_routes_to_webui(monkeypatch, argv, port, inbrowser, tts_dir):
    monkeypatch.delenv("TINGLELOOM_TTS_DIR", raising=False)
    launch = Mock()
    monkeypatch.setattr(webui, "launch_webui", launch)
    assert root_main.main(argv) == 0
    launch.assert_called_once_with(host="127.0.0.1", port=port, share=False, inbrowser=inbrowser)
    assert os.environ.get("TINGLELOOM_TTS_DIR") == tts_dir


def test_root_launcher_passes_through_subcommand(monkeypatch):
    list_materials = Mock(return_value=0)
    launch = Mock()
    monkeypatch.setattr(cli, "cmd_material_list", list_materials)
    monkeypatch.setattr(webui, "launch_webui", launch)
    assert root_main.main(["--tts-dir=/tmp/tts_deploy", "material", "list"]) == 0
    list_materials.assert_called_once()
    args = list_materials.call_args.args[0]
    assert args.command == "material" and args.tts_dir == "/tmp/tts_deploy"
    launch.assert_not_called()


def test_launch_webui_opens_browser(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("no_proxy", "example.com")
    app = Mock()
    monkeypatch.setattr(webui, "build_webui", lambda: app)
    assert webui.launch_webui() is app.queue.return_value.launch.return_value
    for key in ("NO_PROXY", "no_proxy"):
        assert os.environ[key].split(",") == ["example.com", "127.0.0.1", "localhost"]
    app.queue.assert_called_once_with()
    app.queue.return_value.launch.assert_called_once_with(
        server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True,
    )


@pytest.mark.parametrize("new_tags", ["", "待标注,轻刷"])
def test_relabel_unedited_tags_exits_pending(catalog, new_tags, monkeypatch):
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args: learned.append(args))
    index, items = catalog
    _, _, choices = webui.apply_material_relabel(
        items[1].id, "soft_brush", items[1].desc, new_tags, True, index,
    )
    updated = materials.load_index(index)[1]
    assert updated.tags == ["轻刷"]
    assert updated.id not in choices
    assert learned == [(updated.path, "soft_brush", updated.desc, ["轻刷"])]


def test_relabel_requires_selection(catalog):
    index, _ = catalog
    with pytest.raises(ValueError, match="请先选择素材"):
        webui.apply_material_relabel(None, "brush", "", "", index_path=index)


@pytest.mark.parametrize("template_id,name", [(" ", "名称"), ("valid", " ")])
def test_template_requires_id_and_name(tmp_path, template_id, name):
    with pytest.raises(ValueError, match="模板 ID 和模板名称不能为空"):
        webui.update_template_item(template_id, name, "", "", "", tmp_path / "absent.yaml")


@pytest.mark.parametrize("probe_only,material_idx,has_hint,expected", [
    (True, 2, True, "仅探测，未更新原型库"),
    (False, 2, True, "已更新本地声学原型库"),
    (False, 1, True, "已入库为待标注素材，确认英文前缀后更新原型库"),
    (False, 2, False, "无正向素材入库，未更新原型库"),
])
def test_supervision_learning_status_uses_pending_semantics(catalog, monkeypatch, probe_only, material_idx, has_hint, expected):
    _, items = catalog
    hint = mine.TimelineHint(0, 60, "人类章节")
    report = mine.AudioQualityReport(16000, 2, 60, -65, -40, -30, 35, -1, 0, 7500, 0.2, (hint,))
    candidate = mine.MinedCandidate(0, 20, "C", 0.9, -40, 0, 0.1, 0.1, 0.2, 0.3,
                                    timeline_hint=hint if has_hint else None)
    monkeypatch.setattr(mine, "mine_and_ingest", lambda *args, **kwargs: (report, [(candidate, None if probe_only else items[material_idx])]))
    summary, _, _ = webui.run_mine_task("source.wav", "high", 35, 10, 5, "auto", "CC0", probe_only, "00:00-01:00 人类章节")
    assert f"（{expected}）" in summary



def test_material_category_choices_and_filter(catalog, monkeypatch):
    index, items = catalog
    monkeypatch.setattr(materials, "category_descriptions", lambda: {"paper": "翻书", "brush": "刷子"})
    choices = webui.list_material_categories(index_path=index)
    assert choices == [("全部类别 (3)", "all"), ("刷子 · brush (1)", "brush"),
                       ("翻书 · paper (1)", "paper"), ("unknown_trigger · unknown_trigger (1)", "unknown_trigger")]
    assert [value for _, value in webui.list_material_categories(True, index)] == ["all", "brush", "unknown_trigger"]
    rows, ids = webui.list_material_rows(False, index, "paper")
    assert ids == [items[2].id] and len(rows) == 1
    assert webui.list_material_rows(True, index, "paper") == ([], [])


@pytest.mark.parametrize("page,expected_page,length", [(1, 1, 10), (2, 2, 10), (3, 3, 3), (-1, 1, 10), (100, 3, 3)])
def test_paginate_material_rows(page, expected_page, length):
    rows = [[str(i)] for i in range(23)]
    page_rows, current, total_pages, label = webui.paginate_material_rows(rows, page)
    assert current == expected_page and total_pages == 3
    assert len(page_rows) == length
    assert page_rows == rows[(current - 1) * 10:current * 10]
    assert label == f"第 {current} / 3 页（共 23 条，每页 10 条）"
    assert webui.paginate_material_rows([], 99) == ([], 1, 1, "第 1 / 1 页（共 0 条，每页 10 条）")


def test_webui_material_pagination_and_row_selection(monkeypatch):
    rows = [[f"brush_{i}", "刷子", "", "", 10, -20, "brush.wav"] for i in range(23)]
    monkeypatch.setattr(webui, "list_material_rows", lambda *args, **kwargs: (rows, [row[0] for row in rows]))
    monkeypatch.setattr(webui, "list_material_categories", lambda *args: [("全部类别", "all")])
    monkeypatch.setattr(webui, "inspect_material", lambda mat_id: (f"{mat_id}.wav", mat_id, "刷子", "轻柔", "信息"))
    app = webui.build_webui()
    try:
        functions = {item.fn.__name__: item.fn for item in app.fns.values() if item.fn is not None}
        table = next(item for item in app.config["components"] if item["type"] == "dataframe" and item["props"]["headers"][0] == "ID")
        assert len(table["props"]["value"]["data"]) == 10
        page_rows, page, label = functions["change_material_page"](False, "all", 1, 1)
        assert page_rows == rows[10:20] and page == 2
        assert "第 2 / 3 页" in label
        event = gr.SelectData(None, {"index": [3, 2], "value": "标签"})
        result = functions["on_material_row_select"](page_rows, event)
        assert result == ("brush_13", "brush_13.wav", "brush_13", "刷子", "轻柔", "信息", "", "")
        refreshed = functions["refresh_materials"](False, "all", 100)
        assert refreshed[0] == rows[20:] and refreshed[-2] == 3
        batch = Mock(return_value=("已批量翻译归类 15 条待标注素材", rows, [row[0] for row in rows], [("全部类别", "all")], "all"))
        monkeypatch.setattr(webui, "run_batch_pending_relabel", batch)
        updated = functions["batch_relabel_and_refresh"](False, "unknown_trigger", 2)
        batch.assert_called_once_with(False, category="unknown_trigger")
        assert updated[0] == "已批量翻译归类 15 条待标注素材"
        assert updated[1] == rows[10:20] and updated[-2] == 2
        assert updated[2].value is None and updated[3] is None
        assert updated[10].value == "all"
        button = next(item for item in app.config["components"] if item["props"].get("value") == "一键 AI 批量翻译归类待标注素材")
        event = next(item for item in app.fns.values() if item.fn is not None and item.fn.__name__ == "batch_relabel_and_refresh")
        assert len(event.outputs) == 13 and button["type"] == "button"
    finally:
        app.close()


def test_webui_provider_tab_switch_and_save(tmp_path, monkeypatch):
    path = tmp_path / "llm.yaml"
    load_config = llm.load_llm_config
    inspect_provider = webui.inspect_llm_provider
    update_settings = webui.update_llm_settings
    for provider in ("anthropic", "gemini"):
        update_settings(f"https://{provider}.example", f"{provider}-private-key", f"{provider}-text", f"{provider}-audio", 0.3, path, provider)
    monkeypatch.setattr(webui, "inspect_llm_provider", lambda provider: inspect_provider(provider, path))
    monkeypatch.setattr(llm, "load_llm_config", lambda config_path=path, provider=None: load_config(config_path, provider))
    monkeypatch.setattr(webui, "update_llm_settings", lambda base, key, model, audio, temp, provider="openai": update_settings(base, key, model, audio, temp, path, provider))
    app = webui.build_webui()
    try:
        tabs = [item for item in app.config["components"] if item["type"] == "tabitem"]
        assert [tab["props"]["label"].split(".")[0] for tab in tabs] == ["1", "2", "3", "4"]
        assert tabs[-1]["props"]["label"] == "4. API 与多协议模型配置"
        functions = {item.fn.__name__: item.fn for item in app.fns.values() if item.fn is not None}
        switched = functions["inspect_provider"]("anthropic")
        assert switched[0:2] == ("anthropic", "https://anthropic.example")
        assert switched[2].value == "" and "****-key" in switched[2].placeholder
        assert switched[3:] == ("anthropic-text", "anthropic-audio", 0.3)
        status, key_update = functions["save_provider_settings"]("https://claude-new.example", "", "claude-new", "claude-audio", 0.2, "anthropic")
        assert "[anthropic]" in status and key_update.value == ""
        assert load_config(path).provider == "anthropic"
        assert load_config(path, "anthropic").api_key == "anthropic-private-key"
        gemini = functions["inspect_provider"]("gemini")
        assert gemini[1] == "https://gemini.example" and gemini[3] == "gemini-text"
        serialized = json.dumps(app.config, default=str) + status + str(key_update.placeholder)
        for provider in ("anthropic", "gemini"):
            assert f"{provider}-private-key" not in serialized
    finally:
        app.close()


@pytest.mark.parametrize("listing", [webui.list_material_rows, webui.list_material_categories])
def test_material_listing_prunes_deleted_files(catalog, listing):
    index, items = catalog
    items[0].path.unlink()
    items[2].path.unlink()
    listing(index_path=index)
    assert materials.load_index(index) == items[1:2]
    rows, choices = webui.list_material_rows(True, index)
    assert choices == [items[1].id] and [row[0] for row in rows] == choices
    categories = webui.list_material_categories(True, index)
    assert categories[0] == ("全部类别 (1)", "all")
    assert len(categories) == 2 and categories[1][1] == "brush"
    assert categories[1][0].endswith("(1)")
    mat_id, seq, dest = materials.next_available_id("unknown_trigger", "L", materials.load_index(index), items[0].path.parent)
    assert (mat_id, seq, dest) == (items[0].id, 1, items[0].path)


def test_batch_pending_handler_refreshes_categories_and_selection(catalog, monkeypatch, tmp_path):
    index, items = catalog
    cfg_path = tmp_path / "llm.yaml"
    monkeypatch.setattr(mine, "record_prototype", lambda *args: None)

    def relabel(index_path, llm_config_path):
        assert index_path == index and llm_config_path == cfg_path
        return [materials.relabel_material(m.id, "wood_tapping", desc="木块轻敲", tags=["木块"], index_path=index) for m in items[:2]]

    monkeypatch.setattr(llm, "batch_relabel_pending_materials", relabel)
    status, rows, choices, categories, category = webui.run_batch_pending_relabel(True, index, "unknown_trigger", cfg_path)
    assert status == "已批量翻译归类 2 条待标注素材"
    assert rows == [] and choices == []
    assert categories == [("全部类别 (0)", "all")] and category == "all"
    assert all(not m.is_pending and m.path.is_file() for m in materials.load_index(index))

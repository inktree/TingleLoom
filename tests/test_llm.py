"""LLM 配置、模板、生成与多模态请求的公开契约测试（不联网）。"""
import base64
import io
import json
import urllib.error

import numpy as np
import pytest
import soundfile as sf
import yaml

from tingleloom import config, llm, script
from tingleloom.materials import Material


@pytest.fixture
def cfg():
    return llm.LLMConfig(base_url="https://llm.example/v1/", api_key="test-key")


@pytest.fixture
def capture_http(monkeypatch):
    def install(content):
        requests = []

        def urlopen(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"))

        monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
        return requests

    return install


def test_config_environment_and_roundtrip(tmp_path, monkeypatch):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_AUDIO_MODEL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "secrets" / "llm.yaml"
    assert llm.load_llm_config(path) == llm.LLMConfig()
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "text-model")
    monkeypatch.setenv("OPENAI_AUDIO_MODEL", "audio-model")
    path.parent.mkdir()
    path.write_text('api_key: "${OPENAI_API_KEY}"\nbase_url: ""\nmodel: ""\ntemperature: 0\n', encoding="utf-8")
    loaded = llm.load_llm_config(path)
    assert loaded.api_key == "from-env"
    assert loaded.base_url == "https://proxy.example/v1"
    assert loaded.model == "text-model"
    assert loaded.audio_model == "audio-model"
    assert loaded.temperature == 0
    saved = tmp_path / "saved" / "llm.yaml"
    assert llm.save_llm_config(loaded, saved) == saved
    assert llm.load_llm_config(saved) == loaded


@pytest.mark.parametrize("key,expected", [
    ("", ""), ("   ", ""), ("abcd", "****"),
    ("sk-12345678", "****5678"), ("${OPENAI_API_KEY}", "<env:OPENAI_API_KEY>"),
])
def test_redact_api_key(key, expected):
    assert llm.redact_api_key(key) == expected


@pytest.mark.parametrize("stored_key,submitted_key", [
    ("${OPENAI_API_KEY}", ""), ("${OPENAI_API_KEY}", "****5678"),
    ("sk-12345678", ""), ("sk-12345678", "****5678"),
    ("${OPENAI_API_KEY}", "<env:OPENAI_API_KEY>"),
])
def test_save_preserves_original_api_key(tmp_path, monkeypatch, stored_key, submitted_key):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-plaintext")
    path = tmp_path / "llm.yaml"
    path.write_text(yaml.safe_dump({"api_key": stored_key}), encoding="utf-8")
    llm.save_llm_config(llm.LLMConfig(api_key=submitted_key, model="new-model"), path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["api_key"] == stored_key
    assert saved["model"] == "new-model"


def test_builtin_templates_roundtrip(tmp_path):
    templates = llm.load_templates()
    assert [template.id for template in templates] == [
        "sleep_coaching", "ear_care_3dio", "bilingual_whisper", "custom_freeform",
    ]
    path = tmp_path / "templates.yaml"
    assert llm.save_templates(templates, path) == path
    assert llm.load_templates(path) == templates


def test_generate_script_injects_catalog_and_strips_fence(cfg, capture_http, tmp_path):
    body = "@title 夜晚\n@preset whisper_cn\n@seed 42\n你好，晚安。\n(pause 4s)\n(sfx brush_soft 20s C)\n(pause 20s)"
    requests = capture_http(f"```asmr\n{body}\n```")
    path = tmp_path / "brush.wav"
    path.touch()
    material = Material(id="brush_soft", file=str(path), desc="柔软毛刷", duration_sec=20, tags=["毛刷", "轻柔"])
    result = llm.generate_script("轻柔刷耳", [material], cfg=cfg)
    assert result == body
    parsed = script.parse_script(result)
    assert parsed.title == "夜晚"
    request = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://llm.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_method() == "POST"
    assert payload["model"] == cfg.model
    assert payload["temperature"] == cfg.temperature
    prompt = payload["messages"][1]["content"]
    for expected in ("轻柔刷耳", "brush_soft", "柔软毛刷", "duration_sec: 20", "毛刷, 轻柔"):
        assert expected in prompt
    assert "{{materials_catalog}}" not in prompt
    assert "{{user_prompt}}" not in prompt
    assert llm.format_materials_catalog([]) == "（当前素材库为空）"


def test_catalog_separates_beds_from_trigger_materials(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    beds_path = tmp_path / "beds.yaml"
    beds_path.write_text(yaml.safe_dump({
        "room_quiet": {"offset_db": 24, "layers": [{"material": "room_tone"}]},
        "library": {"layers": [{"material": "room_tone"}, {"material": "ambience"}]},
        "rain_light": {"layers": [{"material": "rain_soft"}]},
        "pending_bed": {"layers": [{"material": "unknown_bed"}]},
        "empty": {"layers": []},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "BEDS_FILE", beds_path)
    items = [
        Material("room_tone", "sfx/room.wav", duration_sec=90),
        Material("ambience", "bed/ambience.wav", duration_sec=90),
        Material("tagged_01_C", "sfx/tagged.wav", tags=["床音"], duration_sec=90),
        Material("brush_01_C", "sfx/brush.wav", duration_sec=20),
        Material("rain_soft", "bed/missing.wav"),
        Material("unknown_bed", "bed/pending.wav"),
    ]
    for item in items:
        if item.id != "rain_soft":
            item.path.parent.mkdir(parents=True, exist_ok=True)
            item.path.touch()
    text = llm.format_materials_catalog(items)
    triggers, beds = text.split("[可用环境床音预设 (@bed", 1)
    assert "brush_01_C" in triggers
    assert all(item.id not in triggers for item in items if item.id != "brush_01_C")
    assert "@bed room_quiet（层素材: room_tone, 低于节目 24dB）" in beds
    assert "@bed library" in beds
    assert all(name not in beds for name in ("@bed rain_light", "@bed pending_bed", "@bed empty"))
    assert [m.id for m in llm.retrieve_materials_for_script(items)] == ["brush_01_C"]
    bed_only = llm.format_materials_catalog(items[:3])
    assert "@bed room_quiet" in bed_only and "- id:" not in bed_only
    assert llm.format_available_beds([], beds_path) == ""


def test_generate_missing_key_and_unknown_template():
    with pytest.raises(ValueError, match="未配置 LLM API Key"):
        llm.generate_script("晚安", [], cfg=llm.LLMConfig())
    with pytest.raises(ValueError, match="未知台本模板：missing"):
        llm.generate_script("晚安", [], template_id="missing")


def test_http_error_reports_response(cfg, monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match=r"LLM 请求失败 \(401\)：bad key"):
        llm.chat_completion([], cfg.model, cfg)


IDENTIFICATION = {"prefix": "soft_brush", "desc": "柔软毛刷", "tags": ["轻柔", "毛刷"], "reason": "细密的摩擦声"}


def test_identify_audio_payload(cfg, capture_http, tmp_path):
    requests = capture_http(json.dumps(IDENTIFICATION, ensure_ascii=False))
    path = tmp_path / "stereo.wav"
    # 外围两秒为另一电平，确保发送中间十秒，右声道有声而左声道静音。
    data = np.zeros((14 * 16000, 2), dtype=np.float32)
    data[:, 1] = 0.8
    data[2 * 16000:12 * 16000, 1] = 0.2
    sf.write(path, data, 16000, subtype="PCM_16")
    assert llm.assist_identify_material(path, "CLAP 候选：毛刷", cfg=cfg) == IDENTIFICATION
    payload = json.loads(requests[0].data)
    assert payload["model"] == cfg.audio_model
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "CLAP 候选：毛刷" in content[0]["text"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    wav = io.BytesIO(base64.b64decode(content[1]["input_audio"]["data"], validate=True))
    info = sf.info(wav)
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (16000, 1, 160000, "PCM_16")
    wav.seek(0)
    decoded, _ = sf.read(wav)
    assert np.allclose(decoded, 0.2, atol=1 / 32768)


def test_identify_text_only_does_not_read_audio(cfg, capture_http, tmp_path):
    requests = capture_http("```json\n" + json.dumps(IDENTIFICATION) + "\n```")
    # 不存在的音频路径仍可用于仅依据声学提示的请求。
    assert llm.assist_identify_material(tmp_path / "absent.wav", "CLAP：摩擦；低频能量弱", send_audio=False, cfg=cfg) == IDENTIFICATION
    payload = json.loads(requests[0].data)
    assert payload["model"] == cfg.model
    content = payload["messages"][0]["content"]
    assert isinstance(content, str)
    assert "CLAP：摩擦；低频能量弱" in content
    assert "未上传音频" in content


def test_identify_rejects_missing_result_keys(cfg, capture_http, tmp_path):
    capture_http('{"prefix":"brush"}')
    with pytest.raises(ValueError, match="prefix、desc、tags、reason"):
        llm.assist_identify_material(tmp_path / "absent.wav", send_audio=False, cfg=cfg)


@pytest.mark.parametrize("value,expected", [("null", 0.7), ('""', 0.7), ('"0.25"', 0.25), ("0", 0.0)])
def test_config_temperature_boundary(tmp_path, value, expected):
    path = tmp_path / "llm.yaml"
    path.write_text(f"temperature: {value}\n", encoding="utf-8")
    assert llm.load_llm_config(path).temperature == expected


def test_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层必须是映射对象"):
        llm.load_llm_config(path)


@pytest.mark.parametrize("error", [urllib.error.URLError("connection refused"), TimeoutError("timed out")])
def test_network_error_reports_failure(cfg, monkeypatch, error):
    def urlopen(request, timeout):
        raise error

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="LLM 请求失败") as caught:
        llm.chat_completion([], cfg.model, cfg)
    assert caught.value.__cause__ is error


def test_identify_json_error_and_numeric_prefix(cfg, capture_http, tmp_path):
    capture_http("not json")
    with pytest.raises(ValueError, match="素材鉴别返回了非法 JSON"):
        llm.assist_identify_material(tmp_path / "absent.wav", send_audio=False, cfg=cfg)
    result = {**IDENTIFICATION, "prefix": "3dio_ear_care"}
    capture_http(json.dumps(result))
    assert llm.assist_identify_material(tmp_path / "absent.wav", send_audio=False, cfg=cfg) == result


def test_natural_language_timeline_fallback(monkeypatch):
    from tingleloom import llm, mine
    cfg = llm.LLMConfig(api_key="key")
    calls = []
    def completion(messages, model, config):
        calls.append(messages)
        return "00:00-02:00 说话跳过\n35:00-40:00 气泡水"
    monkeypatch.setattr(llm, "load_llm_config", lambda: cfg)
    monkeypatch.setattr(llm, "chat_completion", completion)
    hints = mine.resolve_timeline_hints("前两分钟说话跳过，35分左右五分钟气泡水", 3000)
    assert len(calls) == 1
    assert [(h.start_sec, h.end_sec, h.exclude) for h in hints] == [(0, 120, True), (2100, 2400, False)]
    mine.resolve_timeline_hints("00:00-02:00 说话跳过", 3000)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="未能从时间戳提示"):
        llm.parse_timeline_hints_text("前面说话", 3000, llm.LLMConfig())



def test_script_retrieval_quotas_diversity_and_pending(monkeypatch, tmp_path):
    from collections import Counter
    monkeypatch.setattr(llm.materials_mod, "category_descriptions", lambda: {"ear": "采耳", "water": "气泡水"})
    items = [Material(f"{cat}_{i:02d}_{ch}", "", desc=desc, duration_sec=100-i)
             for cat, desc in [("ear", "采耳"), ("water", "气泡水")] + [(f"other{n}", "木质") for n in range(10)]
             for i, ch in enumerate(["L", "L", "L", "R", "C"], 1)]
    items += [Material("unknown_trigger_01_C", ""), Material("known_01_C", "", tags=["待标注"])]
    for material in items:
        path = tmp_path / f"{material.id}.wav"
        path.touch()
        material.file = str(path)
    missing = Material("missing_01_C", str(tmp_path / "missing.wav"))
    assert llm.retrieve_materials_for_script([missing]) == []
    selected = llm.retrieve_materials_for_script(items + [missing], "采耳和气泡水助眠")
    assert len(selected) == 24
    counts = Counter(m.category for m in selected)
    assert max(counts.values()) == 3
    assert [m.category for m in selected[:6]] == ["ear"] * 3 + ["water"] * 3
    assert len(counts) == 12
    assert {m.id.rsplit("_", 1)[-1] for m in selected[:3]} == {"L", "R", "C"}
    assert not any(m.is_pending for m in selected)
    catalog = llm.format_materials_catalog(items, "采耳和气泡水助眠")
    assert "12 类 60 条可用素材" in catalog
    assert "[采耳 (ear)]" in catalog and "[气泡水 (water)]" in catalog
    assert catalog.count("- id:") == 24
    assert "unknown_trigger" not in catalog and "known_01_C" not in catalog
    assert llm.format_materials_catalog([items[-1]]) == "（当前素材库为空）"
    assert len({m.category for m in llm.retrieve_materials_for_script(items, max_items=12)}) == 12


@pytest.mark.parametrize("provider,response,expected", [
    ("openai-responses", {"output": [{"type": "message", "content": [{"type": "output_text", "text": "结果"}]}]}, "结果"),
    ("openai-responses", {"output_text": "直返结果"}, "直返结果"),
    ("anthropic", {"content": [{"type": "text", "text": "结果"}]}, "结果"),
    ("gemini", {"candidates": [{"content": {"parts": [{"thought": True, "text": "隐藏思考"}, {"text": "结果"}]}}]}, "结果"),
])
def test_native_protocol_requests(monkeypatch, provider, response, expected):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    cfg = llm.LLMConfig(base_url=llm.PROVIDER_PRESETS[provider]["base_url"], api_key="secret", provider=provider)
    messages = [{"role": "system", "content": "规则"}, {"role": "user", "content": [
        {"type": "text", "text": "提示"},
        {"type": "input_audio", "input_audio": {"data": "encoded-wav", "format": "wav"}},
    ]}]
    assert llm.chat_completion(messages, "test-model", cfg) == expected
    request = requests[0]
    payload = json.loads(request.data)
    if provider == "openai-responses":
        assert request.full_url.endswith("/v1/responses")
        assert request.get_header("Authorization") == "Bearer secret"
        assert payload["instructions"] == "规则"
        assert payload["input"] == [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "提示"},
            {"type": "input_audio", "data": "encoded-wav", "format": "wav"},
        ]}]
    elif provider == "anthropic":
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        assert request.get_header("X-api-key") == "secret"
        assert request.get_header("Anthropic-version") == "2023-06-01"
        assert payload["system"] == "规则"
        assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "提示"}]}]
        assert payload["max_tokens"] == 4096
    else:
        assert request.full_url.endswith("/v1beta/models/test-model:generateContent")
        assert request.get_header("X-goog-api-key") == "secret"
        assert payload["systemInstruction"] == {"parts": [{"text": "规则"}]}
        assert payload["contents"] == [{"role": "user", "parts": [
            {"text": "提示"}, {"inlineData": {"mimeType": "audio/wav", "data": "encoded-wav"}},
        ]}]
        assert payload["generationConfig"]["temperature"] == cfg.temperature


@pytest.mark.parametrize("provider", ["deepseek", "anthropic"])
def test_identify_text_only_providers_do_not_read_audio(monkeypatch, tmp_path, provider):
    response = json.dumps({"prefix": "brush", "desc": "刷子", "tags": [], "reason": "仅按提示判断"})
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        body = {"choices": [{"message": {"content": response}}]} if provider == "deepseek" else {"content": [{"type": "text", "text": response}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    cfg = llm.LLMConfig(base_url=llm.PROVIDER_PRESETS[provider]["base_url"], api_key="key", provider=provider)
    assert llm.assist_identify_material(tmp_path / "not-read.wav", send_audio=True, cfg=cfg)["prefix"] == "brush"
    assert "input_audio" not in requests[0].data.decode()
    payload = json.loads(requests[0].data)
    content = payload["messages"][0]["content"]
    if provider == "anthropic":
        content = content[0]["text"]
    assert "本次未上传音频" in content
    assert payload["model"] == cfg.model


def test_gemini_identify_sends_native_wav(tmp_path, monkeypatch):
    path = tmp_path / "brush.wav"
    sf.write(path, np.ones((16000, 1)) * 0.1, 16000)
    requests = []
    result = {"prefix": "brush", "desc": "刷子", "tags": [], "reason": "听音判断"}

    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(result)}]}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    cfg = llm.LLMConfig(base_url=llm.PROVIDER_PRESETS["gemini"]["base_url"], api_key="key", provider="gemini")
    assert llm.assist_identify_material(path, cfg=cfg) == result
    inline = json.loads(requests[0].data)["contents"][0]["parts"][1]["inlineData"]
    assert inline["mimeType"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(base64.b64decode(inline["data"])))
    assert sr == 16000 and len(data) == 16000


def test_provider_configs_are_independent(tmp_path, monkeypatch):
    path = tmp_path / "llm.yaml"
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env")
    path.write_text(yaml.safe_dump({"api_key": "${OPENAI_API_KEY}", "model": "legacy-model", "openai": {"api_key": ""}}), encoding="utf-8")
    anthropic = llm.LLMConfig("https://claude.example", "${ANTHROPIC_API_KEY}", "claude-custom", "claude-audio", 0, "anthropic")
    gemini = llm.LLMConfig("https://gemini.example", "gemini-secret", "gemini-custom", "gemini-audio", 0.2, "gemini")
    llm.save_llm_config(anthropic, path)
    llm.save_llm_config(gemini, path)
    assert llm.load_llm_config(path) == gemini
    loaded = llm.load_llm_config(path, "anthropic")
    assert (loaded.base_url, loaded.model, loaded.api_key, loaded.temperature) == (anthropic.base_url, anthropic.model, "anthropic-env", 0)
    loaded.api_key = "****-env"
    llm.save_llm_config(loaded, path)
    assert llm.load_llm_config(path, "gemini") == gemini
    assert llm.load_llm_config(path, "openai").model == "legacy-model"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["providers"]["anthropic"]["api_key"] == "${ANTHROPIC_API_KEY}"
    assert raw["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"


def test_provider_block_precedence_and_environment(tmp_path, monkeypatch):
    path = tmp_path / "llm.yaml"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path.write_text(yaml.safe_dump({"provider": "anthropic", "model": "flat-model", "anthropic": {"base_url": "https://native.example"}}), encoding="utf-8")
    assert llm.load_llm_config(path).base_url == "https://native.example"
    assert llm.load_llm_config(path).model == "flat-model"
    gemini = llm.load_llm_config(path, "gemini")
    assert gemini.model == "gemini-env"
    assert gemini.api_key == "fallback-key"
    with pytest.raises(ValueError, match="不支持的 LLM 协议格式"):
        llm.load_llm_config(path, "unknown")


def test_classify_timeline_labels_batches_context_and_validates(cfg, capture_http, monkeypatch):
    monkeypatch.setattr(llm.materials_mod, "category_descriptions", lambda: {"wood_tapping": "敲击木块", "unknown_trigger": "待标注"})
    result = {
        "木製マカロン": {"slug": "wood_tapping", "desc": "木块敲击摩擦", "tags": ["木块", "轻敲"], "exclude": False},
        "invalid slug": {"slug": "../bad", "desc": "错误", "tags": []},
        "invalid desc": {"slug": "paint", "desc": "English only", "tags": []},
        "invalid tags": {"slug": "paint", "desc": "颜料涂刷", "tags": "涂刷"},
        "unknown": {"slug": "unknown_trigger", "desc": "未识别", "tags": []},
        "プレビュー": {"slug": "unknown_trigger", "desc": "片头预览", "tags": [], "exclude": True},
    }
    requests = capture_http(json.dumps(result, ensure_ascii=False))
    classified = llm.classify_timeline_labels([" 木製マカロン ", "木製マカロン", "", "invalid slug", "invalid desc", "invalid tags", "unknown", "プレビュー"], "木头和剃须膏制作零食", cfg)
    assert set(classified) == {"木製マカロン", "プレビュー"}
    assert classified["プレビュー"]["exclude"] is True
    assert len(requests) == 1
    body = json.loads(requests[0].data)
    prompt = json.loads(body["messages"][1]["content"])
    assert prompt["labels"].count("木製マカロン") == 1
    assert prompt["source_hint"] == "木头和剃须膏制作零食"
    assert prompt["known_categories"] == {"wood_tapping": "敲击木块"}
    assert llm.classify_timeline_labels([], cfg=cfg) == {}
    assert llm.classify_timeline_labels(["木製マカロン"], cfg=llm.LLMConfig()) == {}
    assert len(requests) == 1


def test_batch_relabel_pending_preserves_numbering_and_skips_exclusions(tmp_path, monkeypatch, cfg, capture_http):
    from tingleloom import materials, mine

    index = tmp_path / "index.yaml"
    items = []
    for mat_id, desc in (("wood_tapping_01_C", "已有木块"), ("unknown_trigger_04_C", "木製マカロン04_双耳均衡"),
                         ("unknown_trigger_05_R", "木製マカロン05_右耳"), ("unknown_trigger_06_C", "プレビュー06_双耳均衡")):
        path = tmp_path / f"{mat_id}.wav"
        sf.write(path, np.zeros(160), 16000)
        items.append(materials.Material(mat_id, str(path), desc=desc, tags=["待标注", "原标签", "木块"] if mat_id.startswith("unknown_trigger") else [],
                                        source="木头和剃须膏制作零食.mp4 @ 01:00-02:00 (60-120s)"))
    materials.save_index(items, index)
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args: learned.append(args))
    monkeypatch.setattr(llm, "load_llm_config", lambda path: cfg)
    requests = capture_http(json.dumps({
        "木製マカロン": {"slug": "wood_tapping", "desc": "木块敲击摩擦", "tags": ["木块", "轻敲"], "exclude": False},
        "プレビュー": {"slug": "preview", "desc": "片头预览", "tags": [], "exclude": True},
    }))
    updated = llm.batch_relabel_pending_materials(index)
    assert [m.id for m in updated] == ["wood_tapping_02_C", "wood_tapping_03_R"]
    assert [m.desc for m in updated] == ["木块敲击摩擦02_双耳均衡", "木块敲击摩擦03_右耳"]
    assert all(m.path.is_file() and not m.is_pending for m in updated)
    assert all(m.tags == ["木块", "轻敲", "原标签"] for m in updated)
    assert not items[1].path.exists() and not items[2].path.exists()
    saved = materials.load_index(index)
    assert saved[0] == items[0] and saved[3] == items[3]
    assert len(requests) == 1 and len(learned) == 2
    assert all(args[2] == "木块敲击摩擦" for args in learned)
    monkeypatch.setattr(llm, "load_llm_config", lambda path: llm.LLMConfig())
    with pytest.raises(ValueError, match="未配置 LLM API Key，无法执行批量翻译归类"):
        llm.batch_relabel_pending_materials(index)

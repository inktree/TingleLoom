# -*- coding: utf-8 -*-
"""素材清单 / 触发音匹配 / 床音渲染测试（合成素材，不联网）。"""
import numpy as np
import pytest
import soundfile as sf

from tingleloom import config
from tingleloom import llm
from tingleloom import presets as presets_mod
from tingleloom import sfx as sfx_mod
from tingleloom.materials import (
    Material,
    find_best,
    load_index,
    next_available_id,
    prune_missing,
    register,
    register_and_save,
    relabel_material,
    save_index,
)
from tingleloom.script import Pan, SfxDirective
from tingleloom.sfx import SfxOccurrence


def _write(path, seconds=1.0, sr=44100, freq=440.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def test_register_and_match(tmp_path):
    path = _write(tmp_path / "page_turn_01.wav")
    materials = []
    m = register(path, materials, mat_id="page_turn_01", desc="翻书",
                 tags=["纸质", "翻页"], source="freesound", license="CC0")
    assert m.duration_sec == pytest.approx(1.0, abs=0.02)
    assert m.base_lufs < 0
    assert m.sample_rate == 44100
    assert find_best(materials, "翻书").id == "page_turn_01"
    assert find_best(materials, "page_turn").id == "page_turn_01"
    assert find_best(materials, "page").id == "page_turn_01"        # id 子串
    assert find_best(materials, "翻页").id == "page_turn_01"        # 命中 tag
    assert find_best(materials, "完全不相关") is None


def test_index_roundtrip(tmp_path):
    index_path = tmp_path / "index.yaml"
    path = _write(tmp_path / "a.wav")
    register_and_save(path, index_path=index_path, mat_id="a", desc="A", tags=["t"])
    mats = load_index(index_path)
    assert len(mats) == 1
    assert mats[0].id == "a"
    assert mats[0].tags == ["t"]


def test_load_index_discovers_builtin_beds_when_default_index_missing(tmp_path, monkeypatch):
    index_path = tmp_path / "index.yaml"
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    monkeypatch.setattr(config, "MATERIALS_INDEX", index_path)
    bed_dir = tmp_path / "bed"
    bed_dir.mkdir()
    (bed_dir / "523389_leaves_rustle.wav").touch()
    (bed_dir / "room_tone.wav").touch()

    materials = load_index()
    assert [(m.id, m.base_lufs, m.license) for m in materials] == [
        ("room_tone", -43.29, "CC0"),
        ("leaves_rustle", -53.6, "CC0"),
    ]
    assert load_index(index_path) == materials
    assert "@bed room_quiet" in llm.format_available_beds(materials)
    assert load_index(tmp_path / "independent.yaml") == []

    custom = Material(id="room_tone", file="bed/room_tone.wav", desc="自定义底噪", base_lufs=-40.0)
    save_index([custom], index_path)
    materials = load_index()
    assert [m.id for m in materials] == ["room_tone", "leaves_rustle"]
    assert materials[0] == custom


def test_gain_and_pan():
    m = Material(id="x", file="sfx/x.wav", base_lufs=-30.0)
    ref = config.SFX_REFERENCE_LUFS          # 基准是配置项，别写死
    assert sfx_mod.material_gain_db(m, "中") == pytest.approx(ref + 30.0)
    assert sfx_mod.material_gain_db(m, "小") == pytest.approx(ref + 30.0 - 6.0)
    assert sfx_mod.material_gain_db(m, "大") == pytest.approx(ref + 30.0 + 4.0)
    unmeasured = Material(id="unmeasured", file="sfx/unmeasured.wav")
    assert unmeasured.base_lufs is None
    assert sfx_mod.material_gain_db(unmeasured, "中") == pytest.approx(0.0)
    measured_zero = Material(id="zero", file="sfx/zero.wav", base_lufs=0.0)
    assert sfx_mod.material_gain_db(measured_zero, "中") == pytest.approx(ref)

    data = np.ones((10, 2), dtype=np.float32)
    left = sfx_mod.apply_pan(data, Pan("L70", -0.7))
    assert left[0, 0] == pytest.approx(1.0)
    assert left[0, 1] == pytest.approx(0.3)
    center = sfx_mod.apply_pan(data, Pan("C", 0.0))
    assert center[0, 0] == pytest.approx(1.0) and center[0, 1] == pytest.approx(1.0)
    right = sfx_mod.apply_pan(data, Pan("R30", 0.3))
    assert right[0, 0] == pytest.approx(0.7) and right[0, 1] == pytest.approx(1.0)


def test_prepare_sfx_single_and_continuous(tmp_path):
    path = _write(tmp_path / "tap.wav", seconds=12.0)
    materials = []
    m = register(path, materials, mat_id="tap", desc="敲击", tags=["敲击"])

    single = SfxOccurrence(SfxDirective("敲击", duration=None), 1, "standalone")
    seg = sfx_mod.prepare_sfx(single, m, sr=8000, rng=np.random.default_rng(0))
    assert seg.shape[1] == 2
    assert seg.shape[0] <= int(config.SINGLE_SFX_MAX * 8000) + 1
    assert np.abs(seg).max() > 0

    cont = SfxOccurrence(SfxDirective("敲击", duration=30.0), 1, "standalone")
    seg2 = sfx_mod.prepare_sfx(cont, m, sr=8000)
    assert seg2.shape[0] == 30 * 8000


def test_render_bed(tmp_path):
    path = _write(tmp_path / "room.wav", seconds=2.0, freq=200)
    materials = []
    register(path, materials, mat_id="room_tone", desc="底噪")

    preset = presets_mod.BedPreset("room_quiet", 0.5, [presets_mod.BedLayer("room_tone", 0.0)])
    bed = presets_mod.render_bed(preset, 5.0, materials, sr=8000)
    assert bed.shape == (5 * 8000, 2)
    assert np.abs(bed).max() > 0

    warns = []
    missing = presets_mod.BedPreset("x", 1.0, [presets_mod.BedLayer("nope", 0.0)])
    zero = presets_mod.render_bed(missing, 1.0, materials, sr=8000, warnings=warns)
    assert np.allclose(zero, 0)
    assert warns


def test_synthesize_beds_and_preset_rendering(tmp_path):
    options = dict(duration_sec=4.0, sr=8000, out_dir=tmp_path / "bed",
                   index_path=tmp_path / "index.yaml")
    created = presets_mod.synthesize_beds(**options)
    assert [m.id for m in created] == ["room_tone", "rain_soft", "rain_hard"]
    original = []
    for material, rms in zip(created, (0.005, 0.008, 0.014)):
        assert material.path.parent == tmp_path / "bed"
        wave, sr = sf.read(material.path, always_2d=True)
        assert wave.shape == (4 * 8000, 2) and sr == 8000
        assert np.sqrt(np.mean(wave ** 2)) == pytest.approx(rms, abs=0.0001)
        assert not np.array_equal(wave[:, 0], wave[:, 1])
        # 50ms 微观动态保留雨滴颗粒与微风呼吸，避免静止底噪。
        frame_db = 10 * np.log10(np.mean(wave.reshape(-1, 400, 2) ** 2, axis=(1, 2)) + 1e-20)
        grain_snr = np.percentile(frame_db, 95) - np.percentile(frame_db, 10)
        assert grain_snr > 3.5
        assert material.channels == 2
        assert material.base_lufs is not None and np.isfinite(material.base_lufs)
        assert material.source == "procedural_dsp" and material.license == "CC0"
        assert material.commercial_ok
        original.append(wave)
    assert presets_mod.synthesize_beds(**options, seed=1) == []
    for material, wave in zip(created, original):
        assert np.array_equal(sf.read(material.path, always_2d=True)[0], wave)
    replaced = presets_mod.synthesize_beds(**options, seed=1, overwrite=True)
    assert len(replaced) == 3
    assert not np.array_equal(sf.read(replaced[0].path, always_2d=True)[0], original[0])
    materials = load_index(options["index_path"])
    assert len(materials) == 3
    beds = presets_mod.load_beds()
    # 下载床音与程序化基础床音来源不同；离线渲染用本地音频补齐真实录音层。
    for mat_id in ("leaves_rustle", "campfire_ember", "calm_waves"):
        register(_write(tmp_path / f"{mat_id}.wav", seconds=4.0, sr=8000),
                 materials, mat_id=mat_id, tags=["环境音"])
    for name in beds:
        warns = []
        bed = presets_mod.render_bed(beds[name], 6.0, materials, sr=8000, warnings=warns)
        assert bed.shape == (6 * 8000, 2) and np.any(bed)
        assert warns == []
    with pytest.raises(ValueError, match="未知合成床音类型"):
        presets_mod.synthesize_beds(["missing"], **options)
    for duration in (0, -1):
        with pytest.raises(ValueError, match="床音时长必须为正数"):
            presets_mod.synthesize_beds(duration_sec=duration, out_dir=options["out_dir"],
                                       index_path=options["index_path"])


def test_beds_yaml_loads():
    beds = presets_mod.load_beds()
    assert "rain_light" in beds
    assert beds["rain_light"].layers  # 至少有一层


def test_relabel_material(tmp_path, monkeypatch, capsys):
    from tingleloom import mine
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args: learned.append(args))
    from tingleloom.cli import main

    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    index = tmp_path / "index.yaml"
    monkeypatch.setattr(config, "MATERIALS_INDEX", index)
    old = _write(tmp_path / "unknown_trigger_01_L.wav")
    existing = _write(tmp_path / "wood_tapping_01_L.wav")
    save_index([
        Material(old.stem, old.name, desc="待标注触发音01_左耳", tags=["待标注"], source="原来源"),
        Material(existing.stem, existing.name),
    ], index)
    assert main(["material", "label"]) == 0
    assert old.name in capsys.readouterr().out
    updated = relabel_material(old.stem, "wood_tapping", tags=["木块"], index_path=index)
    assert updated.id == "wood_tapping_02_L"
    assert updated.desc == "敲击木块02_左耳"
    assert updated.path.is_file() and not old.exists()
    assert updated.source == "原来源"
    assert learned == [(updated.path, "wood_tapping", "敲击木块", ["木块"])]
    assert load_index(index)[0] == updated
    assert main(["material", "label", "--id", updated.id, "--desc", "自定义敲击", "--tags", ""]) == 0
    saved = load_index(index)[0]
    assert saved.desc == "自定义敲击" and saved.tags == []
    with pytest.raises(ValueError, match="未找到素材"):
        relabel_material("missing", index_path=index)


@pytest.mark.parametrize("tags", [None, ["待标注"], []])
def test_relabel_confirms_pending_and_preserves_explicit_tags(tmp_path, tags, monkeypatch):
    from tingleloom import mine
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args: learned.append(args))
    old = _write(tmp_path / "unknown_trigger_01_LR.wav")
    index = tmp_path / "index.yaml"
    save_index([Material(old.stem, str(old), desc="待标注触发音01_左右交替", tags=["待标注"])], index)
    occupied = _write(tmp_path / "wood_tapping_01_LR.wav")
    original_audio = occupied.read_bytes()
    updated = relabel_material(old.stem, " wood_tapping ", desc="待标注触发音01_左右交替",
                               tags=tags, index_path=index)
    assert updated.id == "wood_tapping_02_LR"
    assert updated.desc == "敲击木块02_左右交替"
    assert not updated.is_pending
    with np.load(config.CLAP_PROMPTS_NPZ) as prompts:
        tags_map = dict(zip(prompts["slugs"].tolist(), prompts["tags"].tolist()))
    assert updated.tags == ([] if tags == [] else [t for t in tags_map["wood_tapping"].split(",") if t])
    assert occupied.read_bytes() == original_audio
    assert learned == [(updated.path, "wood_tapping", "敲击木块", updated.tags)]


@pytest.mark.parametrize("old_desc,new_desc,expected", [
    ("木製マカロン04_双耳均衡", None, "敲击木块02_右耳"),
    ("木製マカロン04_双耳均衡", "木块摩擦01_双耳均衡", "木块摩擦02_右耳"),
    ("木製マカロン04_双耳均衡", "棉签擦耳", "棉签擦耳"),
])
def test_relabel_formats_description_with_allocated_id(tmp_path, monkeypatch, old_desc, new_desc, expected):
    from tingleloom import mine
    monkeypatch.setattr(mine, "record_prototype", lambda *args: None)
    old = _write(tmp_path / "unknown_trigger_04_R.wav")
    _write(tmp_path / "wood_tapping_01_C.wav")
    index = tmp_path / "index.yaml"
    save_index([Material(old.stem, str(old), desc=old_desc, tags=["待标注"])], index)
    updated = relabel_material(old.stem, "wood_tapping", desc=new_desc, index_path=index)
    assert updated.id == "wood_tapping_02_R"
    assert updated.desc == expected


@pytest.mark.parametrize("prefix", ["", "   ", "../invalid"])
def test_relabel_rejects_invalid_prefix(tmp_path, prefix):
    index = tmp_path / "index.yaml"
    save_index([Material("unknown_trigger_01_C", "unused.wav")], index)
    with pytest.raises(ValueError, match="素材前缀只能包含"):
        relabel_material("unknown_trigger_01_C", prefix, index_path=index)


@pytest.mark.parametrize("mat_id,category", [
    ("water_spray_01_R", "water_spray"), ("page_turn_01", "page_turn"),
    ("wood_block", "wood_block"), ("custom_sound_LR", "custom_sound"),
])
def test_material_category(mat_id, category):
    assert Material(mat_id, "").category == category


def test_category_descriptions_and_source_spans(tmp_path, monkeypatch):
    from tingleloom.materials import category_descriptions, parse_source_span, existing_source_spans
    prompts, prototypes = tmp_path / "prompts.npz", tmp_path / "prototypes.npz"
    np.savez(prompts, slugs=["water"], descs=["水声"])
    monkeypatch.setattr(config, "CLAP_PROMPTS_NPZ", prompts)
    assert category_descriptions(prototypes) == {
        "water": "水声", "unknown_trigger": "待标注触发音",
        "ambient_bed": "稳态环境床音", "room_tone": "林间轻柔微风",
        "rain_soft": "细雨", "rain_hard": "大雨",
    }
    np.savez(prototypes, proto_slugs=["water", "custom"], proto_descs=["气泡水", "自定义"])
    assert category_descriptions(prototypes)["water"] == "水声"
    assert category_descriptions(prototypes)["custom"] == "自定义"
    source = "original.mp4 @ 35:40-36:15 (2140-2175s)"
    assert parse_source_span(source) == ("original.mp4", 2140, 2175)
    for invalid in ("original.mp4", "x @ 00:10-00:10 (10-10s)", "x @ 00:20-00:10 (20-10s)"):
        assert parse_source_span(invalid) is None
    present = tmp_path / "present.wav"
    present.touch()
    items = [Material("water_01_C", str(present), source=source),
             Material("water_02_C", str(tmp_path / "missing.wav"), source=source),
             Material("water_03_C", str(present), source="other.mp4 @ 00:00-00:10 (0-10s)")]
    assert existing_source_spans(items, " original.mp4 ") == [(2140, 2175)]
    assert parse_source_span(" @ 00:00-00:10 (0-10s)") is None


def test_prune_missing_and_reuse_deleted_material_id(tmp_path):
    index = tmp_path / "index.yaml"
    deleted = _write(tmp_path / "unknown_trigger_01_C.wav")
    existing = _write(tmp_path / "unknown_trigger_02_R.wav")
    items = [Material(deleted.stem, str(deleted)), Material(existing.stem, str(existing))]
    save_index(items, index)
    deleted.unlink()
    # 编号分配不依赖展示刷新先清理索引。
    assert next_available_id("unknown_trigger", "C", items, tmp_path) == (
        "unknown_trigger_01_C", 1, deleted,
    )
    assert prune_missing(index) == items[1:]
    assert load_index(index) == items[1:]
    assert existing.is_file()
    assert prune_missing(index) == items[1:]


@pytest.mark.parametrize("command", ["list", "label"])
def test_cli_listing_prunes_deleted_materials(tmp_path, monkeypatch, capsys, command):
    from tingleloom.cli import main

    index = tmp_path / "index.yaml"
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    monkeypatch.setattr(config, "MATERIALS_INDEX", index)
    deleted = _write(tmp_path / "unknown_trigger_01_C.wav")
    existing = _write(tmp_path / "unknown_trigger_02_R.wav")
    items = [Material(deleted.stem, str(deleted)), Material(existing.stem, str(existing))]
    save_index(items, index)
    deleted.unlink()
    assert main(["material", command]) == 0
    output = capsys.readouterr().out
    assert items[0].id not in output and items[1].id in output
    assert load_index(index) == items[1:]


@pytest.mark.parametrize("original", [False, True])
def test_cli_fetch_registers_bed(tmp_path, monkeypatch, original):
    import urllib.error
    from tingleloom.cli import main

    monkeypatch.syspath_prepend(str(config.SCRIPTS_DIR))
    import freesound_search as fs

    bed_dir = tmp_path / "bed"
    index = tmp_path / "index.yaml"
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path)
    monkeypatch.setattr(config, "BED_DIR", bed_dir)
    monkeypatch.setattr(config, "MATERIALS_INDEX", index)
    sound = {"id": 123, "name": "Gentle rain", "username": "recordist",
             "license": "https://creativecommons.org/publicdomain/zero/1.0/",
             "tags": ["rain", "quiet"]}
    monkeypatch.setattr(fs, "load_token", lambda: "offline-token")

    def search(token, query, **kwargs):
        assert token == "offline-token" and query == "gentle rain"
        assert kwargs["cc0"] is True
        return [sound]

    def download(results, out_dir):
        assert results == [sound] and out_dir == str(bed_dir)
        bed_dir.mkdir(exist_ok=True)
        return [(sound, _write(bed_dir / "123_rain.wav"))]

    def failed_original(results, out_dir):
        raise urllib.error.URLError("original unavailable")

    monkeypatch.setattr(fs, "search", search)
    monkeypatch.setattr(fs, "download_previews", download)
    monkeypatch.setattr(fs, "download_original", failed_original)
    args = ["material", "fetch", "gentle rain", "--type", "bed", "--cc0",
            "--mat-id", "gentle_rain", "--desc", "柔和雨声"]
    if original:
        args.append("--original")
    assert main(args) == 0
    material, = load_index(index)
    assert material.id == "gentle_rain" and material.desc == "柔和雨声"
    assert material.path == bed_dir / "123_rain.wav"
    assert material.base_lufs is not None and np.isfinite(material.base_lufs)
    assert material.duration_sec == pytest.approx(1.0)
    assert material.sample_rate == 44100
    assert "环境音" in material.tags
    assert material.source == "freesound" and material.author == "recordist"
    assert material.source_url == "https://freesound.org/s/123/"
    assert material.license == "CC0" and material.commercial_ok
    assert not material.attribution_required
    assert llm.retrieve_materials_for_script([material]) == []


def test_freesound_access_token_refresh(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(config.SCRIPTS_DIR))
    import freesound_oauth as oauth

    secret = tmp_path / "oauth.json"
    monkeypatch.setattr(oauth, "SECRET_FILE", secret)
    monkeypatch.setattr(oauth.time, "time", lambda: 1000)
    oauth.save({"client_id": "client", "client_secret": "secret",
                "access_token": "expired", "refresh_token": "refresh", "expires_at": 900})
    calls = []

    def post_form(url, payload):
        calls.append(payload)
        assert payload["grant_type"] == "refresh_token"
        assert payload["refresh_token"] == "refresh"
        return {"access_token": "renewed", "refresh_token": "rotated", "expires_in": 3600}

    monkeypatch.setattr(oauth, "post_form", post_form)
    assert oauth.ensure_access_token() == "renewed"
    saved = json.loads(secret.read_text(encoding="utf-8"))
    assert saved["refresh_token"] == "rotated" and saved["expires_at"] == 4600
    assert oauth.ensure_access_token() == "renewed"
    assert len(calls) == 1

# -*- coding: utf-8 -*-
"""TTS 缓存 / 批处理测试（用假引擎，不加载真模型）。"""
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tingleloom import config
from tingleloom.script import Pan, TextSegment
from tingleloom.tts import SynthOptions, TTSCache, make_key, tts_fingerprint
from tingleloom.tts.base import TTSResult
from tingleloom.tts.batch import synthesize_all


def _seg(text, preset="whisper_cn", lang="zh"):
    return TextSegment(text, preset, lang, [], Pan("C", 0.0), 1)


class FakeEngine:
    engine_id = "fake"
    sample_rate = 32000

    def __init__(self, outdir):
        self.outdir = outdir
        self.calls = 0

    def synthesize(self, text, language, preset, seed=1234, **options):
        self.calls += 1
        path = self.outdir / ("syn-%d.wav" % self.calls)
        sr = 32000
        data = np.zeros(int(sr * 0.2), dtype=np.float32)
        sf.write(str(path), data, sr, subtype="PCM_16")
        return TTSResult(path, sr, 0.2, {"duration_seconds": 0.2, "sample_rate": sr, "text": text})


def test_tts_dir_resolution_and_fingerprint(tmp_path, monkeypatch):
    default_dir = tmp_path / "default"
    file_dir = tmp_path / "configured"
    env_dir = tmp_path / "environment"
    config_file = tmp_path / "tts_dir.txt"
    monkeypatch.setattr(config, "DEFAULT_TTS_DIR", default_dir)
    monkeypatch.setattr(config, "TTS_DIR_FILE", config_file)
    monkeypatch.delenv("TINGLELOOM_TTS_DIR", raising=False)
    assert config.tts_dir() == default_dir.resolve()

    config_file.write_text("\n  # deployment path\n  \n", encoding="utf-8")
    assert config.tts_dir() == default_dir.resolve()
    config_file.write_text("# deployment path\n  %s  \n%s\n" % (file_dir, default_dir), encoding="utf-8")
    assert config.tts_dir() == file_dir.resolve()
    monkeypatch.setenv("TINGLELOOM_TTS_DIR", "  %s  " % env_dir)
    assert config.tts_dir() == env_dir.resolve()
    monkeypatch.setenv("TINGLELOOM_TTS_DIR", "  ")
    assert config.tts_dir() == file_dir.resolve()

    manifest_path = file_dir / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {"package": "custom_onnx", "sampling_rate": 24000}
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    assert tts_fingerprint() == {
        "engine_id": "custom_onnx:%s" % digest,
        "sample_rate": 24000,
        "deploy_dir": str(file_dir.resolve()),
    }

    del manifest["package"]
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    assert tts_fingerprint(file_dir)["engine_id"] == "%s:%s" % (config.TTS_ENGINE_ID, digest)


def test_key_deterministic_and_sensitive():
    o = SynthOptions()
    k1 = make_key("你好", "whisper_cn", 1, o, "eid", 32000, "cpu")
    k2 = make_key("你好", "whisper_cn", 1, o, "eid", 32000, "cpu")
    assert k1 == k2
    assert make_key("你好", "whisper_cn", 1, SynthOptions(noise_scale=0.7), "eid", 32000, "cpu") != k1
    assert make_key("你好", "whisper_cn", 2, o, "eid", 32000, "cpu") != k1
    assert make_key("你好", "whisper_cn", 1, o, "eid", 32000, "cuda") != k1
    assert make_key("你好！", "whisper_cn", 1, o, "eid", 32000, "cpu") != k1
    assert make_key("你好", "normal_cn", 1, o, "eid", 32000, "cpu") != k1


def test_batch_then_resume_hits_cache(tmp_path):
    cache = TTSCache(tmp_path / "cache")
    outdir = tmp_path / "out"
    outdir.mkdir()
    segs = [_seg("第一句。"), _seg("第二句。"), _seg("第三句。")]

    created = {"n": 0}

    def make_engine():
        created["n"] += 1
        return FakeEngine(outdir)

    r1 = synthesize_all(segs, SynthOptions(), 7, cache, "eid", 32000, "cpu", make_engine)
    assert created["n"] == 1
    assert len(r1) == 3
    assert all(not r.from_cache for r in r1)
    assert all(r.duration == pytest.approx(0.2) for r in r1)

    def make_engine_forbidden():
        raise AssertionError("全部命中缓存时不应重建引擎")

    r2 = synthesize_all(segs, SynthOptions(), 7, cache, "eid", 32000, "cpu", make_engine_forbidden)
    assert all(r.from_cache for r in r2)
    assert [r.key for r in r2] == [r.key for r in r1]
    assert all(r.duration == pytest.approx(0.2) for r in r2)

    # 音频仍在但元数据缺失：重新合成，不能以 0 秒时长静默命中。
    cache.meta_path(r1[0].key).unlink()
    r3 = synthesize_all(segs, SynthOptions(), 7, cache, "eid", 32000, "cpu", make_engine)
    assert created["n"] == 2
    assert [r.from_cache for r in r3] == [False, True, True]
    assert all(r.duration == pytest.approx(0.2) for r in r3)


def test_partial_resume(tmp_path):
    cache = TTSCache(tmp_path / "cache")
    outdir = tmp_path / "out"
    outdir.mkdir()
    make_engine = lambda: FakeEngine(outdir)

    synthesize_all([_seg("A。"), _seg("B。")], SynthOptions(), 1, cache, "eid", 32000, "cpu", make_engine)
    # 再跑 A B C：A/B 命中，只有 C 需要合成
    created = {"n": 0}

    def counting():
        created["n"] += 1
        return FakeEngine(outdir)

    r = synthesize_all(
        [_seg("A。"), _seg("B。"), _seg("C。")],
        SynthOptions(), 1, cache, "eid", 32000, "cpu", counting,
    )
    assert created["n"] == 1
    assert [x.from_cache for x in r] == [True, True, False]

# -*- coding: utf-8 -*-
"""流水线端到端测试（假引擎，不加载真模型）。"""
import json

import numpy as np
import soundfile as sf

from tingleloom import config
from tingleloom.materials import Material, save_index
from tingleloom.pipeline import run_pipeline
from tingleloom.tts.base import TTSResult


def _wav(path, seconds=1.0, sr=8000, freq=220.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


class FakeEngine:
    engine_id = "fake"
    sample_rate = 8000

    def __init__(self, outdir):
        self.outdir = outdir
        self.calls = 0

    def synthesize(self, text, language, preset, seed=1234, **options):
        self.calls += 1
        path = self.outdir / ("s%d.wav" % self.calls)
        data = (0.2 * np.sin(2 * np.pi * 220 * np.arange(8000, dtype=np.float32) / 8000)).astype(np.float32)
        sf.write(str(path), data, 8000, subtype="PCM_16")
        return TTSResult(path, 8000, 1.0, {"duration_seconds": 1.0, "sample_rate": 8000})


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(config, "MATERIALS_INDEX", tmp_path / "materials" / "index.yaml")
    monkeypatch.setattr(config, "BEDS_FILE", tmp_path / "presets" / "beds.yaml")
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "TTS_CACHE_DIR", tmp_path / "cache" / "tts")
    (tmp_path / "presets").mkdir(parents=True, exist_ok=True)


def test_pipeline_end_to_end_and_resume(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)

    tap = _wav(tmp_path / "tap.wav")
    save_index(
        [Material(id="tap", file=str(tap), desc="敲击", base_lufs=-20.0,
                  duration_sec=1.0, sample_rate=8000, channels=2)],
        config.MATERIALS_INDEX,
    )

    script_path = tmp_path / "demo.asmr"
    script_path.write_text(
        "# 测试\n@preset whisper_cn\n@seed 5\n\n"
        "你好，世界。(sfx 敲击 @0.3)\n"
        "(pause 1)\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "tts_out"
    outdir.mkdir()
    result = run_pipeline(
        script_path,
        out_path=tmp_path / "out.wav",
        engine_factory=lambda: FakeEngine(outdir),
        fingerprint={"engine_id": "fake", "sample_rate": 8000},
    )
    assert result.output_path.exists()
    assert result.mp3_path.exists()
    assert result.mp3_path == result.output_path.with_suffix(".mp3")
    assert result.gaps == []
    assert len(result.segment_audios) == 1
    assert not result.segment_audios[0].from_cache
    assert result.mix_report.vocals == 1
    assert result.mix_report.sfx == 1

    # build 产物
    assert (config.BUILD_DIR / "timeline.json").exists()
    timeline_payload = json.loads((config.BUILD_DIR / "timeline.json").read_text(encoding="utf-8"))
    assert timeline_payload["output_mp3"] == str(result.mp3_path)
    assert (config.BUILD_DIR / "events.json").exists()

    # 二跑：全部命中缓存，不应再建引擎
    def forbidden():
        raise AssertionError("全部命中缓存时不应重建引擎")

    result2 = run_pipeline(
        script_path,
        out_path=tmp_path / "out2.wav",
        engine_factory=forbidden,
        fingerprint={"engine_id": "fake", "sample_rate": 8000},
    )
    assert result2.segment_audios[0].from_cache
    assert result2.output_path.exists()

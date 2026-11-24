# -*- coding: utf-8 -*-
"""时间轴与混音器测试（合成素材，不加载 TTS）。"""
import av
import numpy as np
import pytest
import soundfile as sf

from tingleloom import mix as mix_mod
from tingleloom import presets as presets_mod
from tingleloom.materials import register
from tingleloom.script import parse_script
from tingleloom.sfx import collect_occurrences, resolve_sfx
from tingleloom.timeline import build_timeline
from tingleloom.tts.batch import SegmentAudio


def _wav(path, seconds, sr=8000, freq=300.0):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def _segment_audio(segment, duration, path):
    return SegmentAudio(
        key="k", order=0, segment=segment, wav_path=path, duration=duration,
        sample_rate=8000, from_cache=True,
    )


def test_cursor_model(tmp_path):
    script = parse_script(
        "@preset whisper_cn\n"
        "@bed room_quiet\n"
        "第一句。\n"
        "(pause 2)\n"
        "第二句。(sfx 敲击 @1.0)\n"
        "(sfx 环境 5s)\n"
    )
    segs = script.text_segments
    audios = [
        _segment_audio(segs[0], 3.0, tmp_path / "a.wav"),
        _segment_audio(segs[1], 4.0, tmp_path / "b.wav"),
    ]
    timeline = build_timeline(script, audios)

    assert timeline.segments[0].start == 0.0
    assert timeline.segments[1].start == pytest.approx(5.0)     # 3 + pause2
    assert timeline.beds[0].start == 0.0 and timeline.beds[0].name == "room_quiet"

    # 内联触发音：段起点 + @1.0
    assert timeline.sfx[0].start == pytest.approx(6.0)
    assert timeline.sfx[0].duration is None
    # 独立持续触发音：游标处启动，不推进游标
    assert timeline.sfx[1].start == pytest.approx(9.0)
    assert timeline.sfx[1].duration == 5.0
    assert timeline.total_seconds == pytest.approx(9.0)
    assert [placed.occurrence for placed in timeline.sfx] == collect_occurrences(script)


@pytest.mark.parametrize("count", [0, 2])
def test_timeline_rejects_audio_count_mismatch(tmp_path, count):
    script = parse_script("你好。")
    segment = _segment_audio(script.text_segments[0], 1.0, tmp_path / "v.wav")
    with pytest.raises(ValueError, match="文本段数量与 TTS 产物数量不一致"):
        build_timeline(script, [segment] * count)


def test_mix_render(tmp_path):
    sr = 44100
    materials = []
    register(_wav(tmp_path / "tap.wav", 1.0), materials, mat_id="tap", desc="敲击")
    register(_wav(tmp_path / "env.wav", 2.0), materials, mat_id="env", desc="环境")
    register(_wav(tmp_path / "room.wav", 2.0), materials, mat_id="room_tone", desc="底噪")

    script = parse_script(
        "@preset whisper_cn\n"
        "@bed room_quiet\n"
        "你好，世界。(sfx 敲击 @0.5)\n"
        "(pause 1)\n"
        "(sfx 环境 2s)\n"
    )
    segs = script.text_segments
    vocal_wav = _wav(tmp_path / "vocal.wav", 1.5)
    audios = [_segment_audio(segs[0], 1.5, vocal_wav)]

    resolved, gaps = resolve_sfx(script, materials)
    assert gaps == []
    assert all(r.resolved for r in resolved)

    timeline = build_timeline(script, audios)
    beds = {"room_quiet": presets_mod.BedPreset(
        "room_quiet", 0.3, [presets_mod.BedLayer("room_tone", 0.0)])}

    out = tmp_path / "mix.wav"
    report = mix_mod.render(
        timeline, resolved, beds, materials, sr=sr, output_path=out)

    assert report.output_path.exists()
    assert report.mp3_path.exists()
    assert report.mp3_path == out.with_suffix(".mp3")
    with av.open(str(report.mp3_path)) as container:
        stream = container.streams.audio[0]
        assert stream.sample_rate == sr
        assert stream.channels == 2
        assert stream.bit_rate == 320000
        decoded = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
        assert decoded.shape[1] > 0
        assert np.abs(decoded).max() > 0
    assert report.vocals == 1
    assert report.sfx == 2
    assert report.peak <= 1.0
    # 人声 1.5 + pause 1 + 环境 2 = 4.5，加尾巴
    assert report.duration_seconds >= 4.5
    data, out_sr = sf.read(str(out))
    assert out_sr == sr
    assert data.shape[1] == 2
    assert np.abs(data).max() > 0


def test_mix_reports_gap(tmp_path):
    sr = 8000
    materials = []
    script = parse_script("@preset whisper_cn\n你好。(sfx 不存在的声音 @0.2)\n")
    resolved, gaps = resolve_sfx(script, materials)
    assert gaps == ["不存在的声音"]
    assert resolved[0].material is None

    segs = script.text_segments
    audios = [_segment_audio(segs[0], 1.0, _wav(tmp_path / "v.wav", 1.0))]
    timeline = build_timeline(script, audios)
    report = mix_mod.render(
        timeline, resolved, {}, materials, sr=sr, output_path=tmp_path / "m.wav")
    assert report.output_path.exists()
    assert report.sfx == 0     # 缺口事件被跳过，不崩


def test_sfx_explicit_pan_applied_once(tmp_path):
    sr = 8000
    materials = []
    register(_wav(tmp_path / "tap.wav", 1.0), materials, mat_id="tap", desc="敲击")
    script = parse_script("(sfx 敲击 L70)")
    resolved, gaps = resolve_sfx(script, materials)
    assert not gaps
    timeline = build_timeline(script, [])
    out = tmp_path / "pan.wav"
    mix_mod.render(timeline, resolved, {}, materials, sr=sr, target_lufs=-40.0,
                   pan_swap=150, output_path=out)
    data, _ = sf.read(str(out))
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    assert rms[1] / rms[0] == pytest.approx(0.3, abs=0.002)


def test_bed_gain_offset():
    assert mix_mod.bed_gain_db(-20.0, -50.0, 22.0) == pytest.approx(8.0)
    assert mix_mod.bed_gain_db(-20.0, -20.0, 22.0) == pytest.approx(-22.0)
    assert mix_mod.bed_gain_db(-20.0, float("-inf"), 22.0) == 0.0


def test_bed_sits_below_program(tmp_path):
    from tingleloom import config

    sr = 8000
    materials = []
    register(_wav(tmp_path / "room.wav", 4.0), materials, mat_id="room_tone", desc="底噪")
    script = parse_script("@preset whisper_cn\n@bed room_quiet\n你好。\n(pause 2)\n")
    segs = script.text_segments
    audios = [_segment_audio(segs[0], 1.0, _wav(tmp_path / "v.wav", 1.0))]
    timeline = build_timeline(script, audios)
    beds = {"room_quiet": presets_mod.BedPreset(
        "room_quiet", 0.3, [presets_mod.BedLayer("room_tone", 0.0)])}
    report = mix_mod.render(
        timeline, [], beds, materials, sr=sr, output_path=tmp_path / "m.wav")
    # 床音应坐在节目底下约 BED_OFFSET_DB
    assert report.bed_lufs == pytest.approx(
        report.program_lufs - config.BED_OFFSET_DB, abs=1.0)

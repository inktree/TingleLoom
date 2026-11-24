import json
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

from tingleloom import config, llm, materials, mine
from tingleloom.cli import build_parser
from tingleloom.mine import (
    MinedCandidate, ClapTagger, SileroVad, _vad_params, classify_channel_tag, extract_slice, format_trajectory_5s,
    mine_and_ingest, mine_candidates, scan_audio,
)


@pytest.mark.parametrize("diff,tag", [
    ([-12] * 10, "L"), ([12] * 10, "R"), ([0] * 10, "C"),
    ([-12] * 5 + [12] * 5, "LR"), ([12] * 5 + [-12] * 5, "RL"),
])
def test_channel_tags(diff, tag):
    assert classify_channel_tag(np.asarray(diff)) == tag


@pytest.fixture
def source(tmp_path):
    sr = 24000
    t = np.arange(sr * 24) / sr
    data = np.zeros((len(t), 2), dtype=np.float32)
    data[:sr * 4] = (0.0001 * np.sin(2 * np.pi * 100 * t[:sr * 4]))[:, None]
    data[sr * 4:sr * 12, 0] = 0.025 * np.sin(2 * np.pi * 6000 * t[sr * 4:sr * 12])
    data[sr * 4:sr * 12, 1] = data[sr * 4:sr * 12, 0] * 0.1
    data[sr * 12:sr * 20, 1] = 0.025 * np.sin(2 * np.pi * 90 * t[sr * 12:sr * 20])
    data[sr * 12:sr * 20, 0] = data[sr * 12:sr * 20, 1] * 0.1
    data[sr * 20:] = data[sr * 4:sr * 8]
    data[sr * 21, 0] = 1.0
    path = tmp_path / "source.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    return path


def test_scan_and_rank_exclude_clipping(source):
    report, features = scan_audio(source)
    assert features.shape == (24, 9)
    assert report.sample_rate == 24000
    assert report.channels == 2
    assert report.duration_sec == 24
    assert report.total_clip_samples == 1
    assert report.cutoff_hz >= 6000
    assert features[21, 3] == 1
    for profile, start, tag in [("high", 4, "L"), ("low", 12, "R")]:
        candidates = mine_candidates(source, features, report, profile=profile,
                                     slice_seconds=4, step_seconds=4, top_n=3,
                                     min_gap_seconds=4)
        assert candidates[0].start_sec == start
        assert candidates[0].channel_tag == tag
        assert all(c.end_sec <= 20 for c in candidates)
        assert candidates[0].trajectory_5s.startswith(tag + "(")
    with pytest.raises(ValueError, match="profile"):
        mine_candidates(source, features, report, profile="other")


def test_mine_bed_profile_selects_steady_centered_ambience(tmp_path, monkeypatch):
    sr = 8000
    rng = np.random.default_rng(83)
    data = rng.uniform(-0.025, 0.025, (24 * sr, 2)).astype(np.float32)
    data[:4 * sr, 1] *= 0.1  # 偏左，不可作为居中床音。
    data[5 * sr] = 1.0  # 削波。
    data[9 * sr] = 0.4  # 未削波，但峰值因子过高。
    for second, gain in zip(range(12, 16), (0.4, 2.5, 0.4, 2.5)):
        data[second * sr:(second + 1) * sr] *= gain
    path = tmp_path / "ambience.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    report, features = scan_audio(path)
    options = dict(profile="bed", slice_seconds=4, step_seconds=4, top_n=2,
                   min_gap_seconds=4)
    candidates = mine_candidates(path, features, report, **options)
    assert {c.start_sec for c in candidates} == {16, 20}
    for cand in candidates:
        f = features[cand.start_sec:cand.end_sec]
        assert cand.channel_tag == "C"
        assert np.std(f[:, 0]) <= 2.5
        assert np.max(f[:, 1] - f[:, 0]) <= 14.0
        assert cand.mean_abs_rl_db <= 3.5

    class QuietVad:
        def detect(self, seg, sr, *args):
            return 0.0, 0.0, 0.0

    class AmbientClap:
        trigger_embeds = np.eye(2, dtype=np.float32)

        def analyze(self, seg, sr, baseline=None):
            return np.array([1.0, 0.0]), "unknown_trigger", "待标注触发音", ("待标注",), 0.01, 0.0

    monkeypatch.setattr(mine, "SileroVad", QuietVad)
    monkeypatch.setattr(mine, "ClapTagger", AmbientClap)
    monkeypatch.setattr(config, "BED_DIR", tmp_path / "bed")
    _, results = mine_and_ingest(path, profile="bed", slice_seconds=4, step_seconds=4,
                                 top_n=1, index_path=tmp_path / "index.yaml")
    assert len(results) == 1
    cand, material = results[0]
    assert cand.start_sec in (16, 20)
    assert material.id == "ambient_bed_01_C"
    assert material.path.parent == tmp_path / "bed"
    assert material.path.is_file() and not material.is_pending
    assert material.tags == ["环境音", "床音"]
    wave, _ = sf.read(material.path, always_2d=True)
    assert np.all(wave[0] == 0) and np.all(wave[-1] == 0)
    # 已识别的环境类别保留语义，人声过滤不随低分豁免而放松。
    class WaterClap(AmbientClap):
        def analyze(self, seg, sr, baseline=None):
            return np.array([1.0, 0.0]), "water_liquid", "水声", ("水",), 0.2, 0.0

    tagged = mine_candidates(path, features, report, clap=WaterClap(), **options)
    assert all(c.clap_slug == "water_liquid" and "环境音" in c.clap_tags for c in tagged)

    class VoiceClap(AmbientClap):
        def analyze(self, seg, sr, baseline=None):
            return np.array([1.0, 0.0]), "unknown_trigger", "待标注触发音", ("待标注",), 0.01, 0.4

    assert mine_candidates(path, features, report, clap=VoiceClap(), **options) == []


def test_vad_real_models_and_short_tail():
    vad = SileroVad()
    assert vad.detect(np.zeros((16000, 2), dtype=np.float32), 16000) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(42)
    for length in (300, 16031):
        seg = rng.normal(0, 0.01, (length, 2)).astype(np.float32)
        ratio, max_run, peak = vad.detect(seg, 16000)
        assert 0.0 <= max_run <= np.ceil(length / 512) * 0.032
        assert 0.0 <= ratio <= 1.0
        assert 0.0 <= peak <= 1.0

    # 验证跨 5 秒归一化块边界时重置 LSTM state，避免前一块宽频摩擦饱和状态压制后续人声
    reset_frames = []
    orig_run = vad.dec.run

    def spy_dec_run(output_names, input_feed):
        if not np.any(input_feed["state"]):
            reset_frames.append(len(reset_frames))
        return orig_run(output_names, input_feed)

    vad.dec.run = spy_dec_run
    vad.detect(rng.normal(0, 0.05, (16000 * 11, 1)).astype(np.float32), 16000)
    assert len(reset_frames) == 3


def test_ingest_and_probe(source, tmp_path, monkeypatch):
    learned = []
    monkeypatch.setattr(mine, "record_prototype", lambda *args, **kwargs: learned.append(args))
    out_dir, index = tmp_path / "sfx", tmp_path / "index.yaml"
    options = dict(prefix="clean", slice_seconds=4, step_seconds=4, top_n=1,
                   out_dir=out_dir, index_path=index, tags=("触发音",),
                   desc_prefix="刮擦", source="合成", license="CC0",
                   hints_text="00:04-00:12 刮擦")
    _, probes = mine_and_ingest(source, probe_only=True, **options)
    assert len(probes) == 1
    assert probes[0][1] is None
    assert not out_dir.exists()
    assert not index.exists()
    assert not learned
    _, results = mine_and_ingest(source, **options)
    cand, material = results[0]
    assert cand.channel_tag == "L"
    assert (out_dir / "clean_01_L.wav").is_file()
    registered = materials.load_index(index)
    assert len(registered) == 1
    assert registered[0].id == material.id == "clean_01_L"
    assert registered[0].desc == "刮擦01_左耳"
    assert registered[0].base_lufs is not None
    assert registered[0].sample_rate == config.SAMPLE_RATE
    assert registered[0].duration_sec == 4
    assert registered[0].source == f"合成 @ {mine.format_span(cand.start_sec, cand.end_sec)}"
    assert learned == [(material.path, "clean", "刮擦", tuple(material.tags))]
    assert cand.timeline_hint.label == "刮擦"
    assert registered[0].tags == list(dict.fromkeys((*cand.clap_tags, "触发音")))
    first_path = material.path
    original_audio = first_path.read_bytes()
    _, repeated = mine_and_ingest(source, **options)
    assert repeated[0][1].id == "clean_02_L"
    assert repeated[0][1].path.is_file()
    assert first_path.read_bytes() == original_audio
    assert [m.id for m in materials.load_index(index)] == ["clean_01_L", "clean_02_L"]


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".ogg", ".m4a"])
def test_streaming_slice_matches_decoded_timeline(tmp_path, suffix):
    sr = 24000
    t = np.arange(sr * 3) / sr
    data = np.column_stack((0.02 * np.sin(2 * np.pi * (300 * t + 200 * t ** 2)),
                            0.03 * np.sin(2 * np.pi * 1000 * t))).astype(np.float32)
    path = tmp_path / ("timeline" + suffix)
    if suffix == ".m4a":
        with av.open(str(path), "w") as container:
            stream = container.add_stream("aac", rate=sr)
            stream.layout = "stereo"
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data.T), format="fltp", layout="stereo")
            frame.sample_rate = sr
            frame.pts = 0
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    else:
        sf.write(path, data, sr)
    whole, _ = extract_slice(path, 0, 3, target_sr=sr)
    cut, rate = extract_slice(path, 0.75, 1.85, target_sr=sr)
    assert rate == sr
    assert cut.shape == (round(1.1 * sr), 2)
    assert cut.dtype == np.float32
    np.testing.assert_allclose(cut, whole[round(0.75 * sr):round(1.85 * sr)], atol=1e-6)
    report, features = scan_audio(path)
    assert report.duration_sec >= 3
    assert len(features) >= 3


def test_video_only_container_rejects_missing_audio(tmp_path):
    path = tmp_path / "video_only.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = stream.height = 16
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with pytest.raises(ValueError, match="文件中没有音频流"):
        scan_audio(path)
    with pytest.raises(ValueError, match="文件中没有音频流"):
        extract_slice(path, 0, 1)


def test_short_scan_and_trajectory(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros((100, 2)), 16000)
    with pytest.raises(ValueError, match="音频时长不足 1 秒"):
        scan_audio(path)
    left = np.tile([0.02, 0.002], (500, 1)).astype(np.float32)
    assert format_trajectory_5s(np.concatenate((left, left[:, ::-1])), 100) == "L(-20) -> R(+20)"
    args = build_parser().parse_args(["material", "mine", str(path), "--probe-only"])
    assert args.profile == "high"
    assert args.probe_only
    assert args.min_dur is None
    assert args.max_vad_run == 0.25
    assert args.prefix == "auto"
    assert args.max_clap_voice == 0.42
    assert args.top == 15
    assert args.min_clap_score == 0.04
    assert args.diversity_sim == 0.90
    args = build_parser().parse_args(["material", "mine", str(path),
                                      "--min-dur", "15", "--max-vad-run", "0.2", "--min-clap-score", "0.08"])
    assert args.min_dur == 15
    assert args.max_vad_run == 0.2
    assert args.min_clap_score == 0.08


@pytest.fixture
def vad_with_probs(monkeypatch):
    """保留真实分帧和游程统计，仅替换 ONNX 的逐帧概率。"""
    def create(*channel_probs):
        probabilities = iter(channel_probs)

        class Encoder:
            def run(self, outputs, inputs):
                probs = next(probabilities)
                assert len(probs) == len(inputs["input"])
                return [np.asarray(probs, dtype=np.float32)[:, None, None]]

        class Decoder:
            def run(self, outputs, inputs):
                return inputs["input"], inputs["state"]

        sessions = iter((Encoder(), Decoder()))
        monkeypatch.setattr(mine.ort, "InferenceSession", lambda *args, **kwargs: next(sessions))
        return SileroVad()

    return create


def test_vad_debounce_and_channel_maxima(vad_with_probs):
    left = [0.9] * 3 + [0.1] * 7 + [0.8] + [0.1] * 6 + [0.7] * 3
    right = [0.1] * 5 + [0.6] * 4 + [0.1] * 11
    seg = np.full((20 * 512, 2), 0.01, dtype=np.float32)
    assert vad_with_probs(left, left).detect(seg, 16000) == pytest.approx((0, 0, 0.9))
    assert vad_with_probs(left, right).detect(seg, 16000) == pytest.approx((0.2, 0.128, 0.9))


@pytest.mark.parametrize("formant,high,flatness,expected", [
    (0.6, 0.2, 0.2, (0.22, 2)),
    (0.2, 0.6, 0.5, (0.4, 3)),
    (0.3, 0.3, 0.3, (0.4, 3)),
])
def test_acoustic_vad_params(formant, high, flatness, expected):
    cand = MinedCandidate(0, 35, "C", 1, -35, 0, 0.1, formant, high, flatness)
    assert _vad_params(cand) == expected


def test_continuous_voice_rejected_despite_low_ratio(tmp_path, vad_with_probs):
    sr = 16000
    data = np.random.default_rng(7).normal(0, 0.01, (35 * sr, 2)).astype(np.float32)
    data[:4 * sr] *= 0.01
    path = tmp_path / "voice.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    report, features = scan_audio(path)
    probs = np.full(int(np.ceil(len(data) / 512)), 0.1)
    probs[100:108] = 0.9
    ratio, max_run, _ = vad_with_probs(probs, probs).detect(data, sr)
    assert ratio < 0.015
    assert max_run == pytest.approx(0.256)
    options = dict(slice_seconds=35, min_slice_seconds=35, top_n=1)
    assert mine_candidates(path, features, report, vad=vad_with_probs(probs, probs), **options) == []
    candidates = mine_candidates(path, features, report, vad=vad_with_probs(probs, probs),
                                 max_vad_run_sec=0.3, **options)
    assert len(candidates) == 1
    assert candidates[0].vad_max_run_sec == pytest.approx(max_run)


def test_short_windows_recover_between_interruptions(tmp_path):
    sr = 8000
    data = np.random.default_rng(8).normal(0, 0.01, (125 * sr, 2)).astype(np.float32)
    data[:15 * sr] *= 0.01
    interruptions = tuple(range(0, 121, 20))
    for second in interruptions:
        data[second * sr] = 1.0
    path = tmp_path / "interrupted.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    report, features = scan_audio(path)
    options = dict(slice_seconds=35, step_seconds=5, top_n=3)
    assert mine_candidates(path, features, report, min_slice_seconds=35, **options) == []
    candidates = mine_candidates(path, features, report, min_slice_seconds=15, **options)
    assert len(candidates) >= 2
    assert all(c.end_sec - c.start_sec == 15 for c in candidates)
    assert all(not any(c.start_sec <= t < c.end_sec for t in interruptions) for c in candidates)
    assert all(a.start_sec >= b.end_sec + 10 or b.start_sec >= a.end_sec + 10
               for i, a in enumerate(candidates) for b in candidates[i + 1:])
    with pytest.raises(ValueError, match="最短切片时长"):
        mine_candidates(path, features, report, min_slice_seconds=36, **options)


@pytest.mark.parametrize("length", [24000, 264000])
def test_clap_real_onnx(length, tmp_path):
    tagger = ClapTagger(prototypes_path=tmp_path / "prototypes.npz")
    t = np.arange(length) / 24000
    mono = (0.01 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    emb, slug, desc, tags, trigger_sim, voice_sim = tagger.analyze(
        np.column_stack((mono, mono * 0.2)), 24000)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-6)
    assert slug in tagger.slugs
    best = list(tagger.slugs).index(slug)
    assert desc == tagger.descs[best]
    assert tags == tuple(str(tagger.tags[best]).split(","))
    assert -1 <= trigger_sim <= 1
    assert -1 <= voice_sim <= 1
    assert tagger.mel_filters.shape == (513, 64)
    assert tagger.trigger_embeds.shape == (26, 512)
    assert tagger.voice_embeds.shape == (3, 512)


@pytest.fixture
def ranked_clap_source(source, monkeypatch):
    report, _ = scan_audio(source)
    features = np.tile([-35, -20, 0, 0, 0.1, 0.2, 0.4, 0.3, 0.5], (12, 1))
    features[:, 8] = np.repeat([0.6, 0.5, 0.4], 4)

    def extract(path, start, end):
        return np.full(((end - start) * 100, 2), start, dtype=np.float32), 100

    monkeypatch.setattr(mine, "extract_slice", extract)
    return source, features, report


@pytest.mark.parametrize("trigger,voice,accepted", [
    (0.5, 0.43, False), (0.36, 0.38, False), (0.5, 0.4, True),
])
def test_clap_whisper_filter(ranked_clap_source, trigger, voice, accepted):
    class Tagger:
        trigger_embeds = np.eye(1, 512, dtype=np.float32)

        def analyze(self, seg, sr, baseline=None):
            emb = np.eye(1, 512, dtype=np.float32)[0]
            return emb, "wood_tapping", "小木块敲击", ("木块",), trigger, voice

    candidates = mine_candidates(*ranked_clap_source, slice_seconds=4,
                                 step_seconds=4, top_n=1, min_gap_seconds=4, clap=Tagger())
    assert bool(candidates) == accepted
    if accepted:
        assert candidates[0].clap_score == trigger
        assert candidates[0].clap_voice_score == voice


@pytest.mark.parametrize("score,min_score,with_baseline,accepted", [
    (0.0, 0.04, True, False),
    (0.04, 0.04, True, True),
    (0.045, 0.06, True, False),
    (0.119, 0.04, False, False),
    (0.12, 0.04, False, True),
])
def test_clap_minimum_trigger_score(ranked_clap_source, score, min_score, with_baseline, accepted):
    path, features, report = ranked_clap_source
    if not with_baseline:
        features = features[:4]

    class Tagger:
        trigger_embeds = np.eye(1, 512, dtype=np.float32)

        def analyze(self, seg, sr, baseline=None):
            return self.trigger_embeds[0], "unknown_trigger", "待标注触发音", ("待标注",), score, 0.0

    candidates = mine_candidates(path, features, report, slice_seconds=4,
                                 step_seconds=4, min_gap_seconds=4,
                                 clap=Tagger(), min_clap_score=min_score)
    assert bool(candidates) == accepted
    if accepted:
        assert all(c.clap_score == score and c.clap_slug == "unknown_trigger" for c in candidates)


def test_adaptive_rms_preserves_quiet_dynamic_audio(ranked_clap_source):
    path, features, report = ranked_clap_source
    report = replace(report, noise_floor_db=-70.0, median_db=-56.0, active_db=-49.0)
    features[:, 0] = -55.0
    features[8:, 0] = -70.0
    candidates = mine_candidates(path, features, report, slice_seconds=4,
                                 step_seconds=4, min_gap_seconds=4)
    assert [c.start_sec for c in candidates] == [0, 4]
    assert all(c.mean_rms_db == -55.0 for c in candidates)


@pytest.mark.parametrize("top_n,starts", [(2, [0, 8]), (3, [0, 8, 4])])
def test_clap_diversity_before_backfill(ranked_clap_source, top_n, starts):
    class Tagger:
        trigger_embeds = np.eye(1, 512, dtype=np.float32)

        def analyze(self, seg, sr, baseline=None):
            emb = np.zeros(512, dtype=np.float32)
            emb[int(seg[0, 0] == 8)] = 1.0
            return emb, "wood_tapping", "小木块敲击", ("木块",), 0.5, 0.2

    candidates = mine_candidates(*ranked_clap_source, slice_seconds=4,
                                 step_seconds=4, top_n=top_n, min_gap_seconds=4,
                                 clap=Tagger(), diversity_max_sim=0.9)
    assert [c.start_sec for c in candidates] == starts
    assert all(a.end_sec <= b.start_sec or b.end_sec <= a.start_sec
               for i, a in enumerate(candidates) for b in candidates[i + 1:])


def test_auto_ingest_semantics(source, tmp_path, monkeypatch):
    candidates = [
        MinedCandidate(4, 8, "L", 1, -35, 20, 0.1, 0.2, 0.7, 0.5,
                       clap_slug="wood_tapping", clap_desc="小木块敲击", clap_tags=("敲击", "木块")),
        MinedCandidate(12, 16, "R", 1, -35, 20, 0.1, 0.2, 0.7, 0.5,
                       clap_slug="mic_brushing", clap_desc="麦克风刷擦", clap_tags=("刷擦",)),
        MinedCandidate(16, 20, "C", 1, -35, 0, 0.1, 0.2, 0.7, 0.5,
                       clap_slug="wood_tapping", clap_desc="小木块敲击", clap_tags=("敲击", "木块")),
    ]
    monkeypatch.setattr(mine, "mine_candidates", lambda *args, **kwargs: candidates)
    out_dir, index = tmp_path / "auto_sfx", tmp_path / "auto_index.yaml"
    _, results = mine_and_ingest(source, prefix="auto", top_n=3,
                                 out_dir=out_dir, index_path=index, tags=("木块", "触发音"))
    expected = [
        ("wood_tapping_01_L", "小木块敲击01_左耳", ["敲击", "木块", "触发音"]),
        ("mic_brushing_01_R", "麦克风刷擦01_右耳", ["刷擦", "木块", "触发音"]),
        ("wood_tapping_02_C", "小木块敲击02_双耳均衡", ["敲击", "木块", "触发音"]),
    ]
    assert len(results) == 3
    for (_, material), (stem, desc, tags) in zip(results, expected):
        assert (out_dir / f"{stem}.wav").is_file()
        assert material.id == stem
        assert material.desc == desc
        assert material.tags == tags
    first_audio = {material.path: material.path.read_bytes() for _, material in results}
    _, repeated = mine_and_ingest(source, prefix="auto", top_n=3,
                                  out_dir=out_dir, index_path=index, tags=("木块", "触发音"))
    assert [material.id for _, material in repeated] == [
        "wood_tapping_03_L", "mic_brushing_02_R", "wood_tapping_04_C",
    ]
    assert len(materials.load_index(index)) == 6
    for path, original_audio in first_audio.items():
        assert path.read_bytes() == original_audio


@pytest.fixture
def clap_with_embeddings(monkeypatch):
    def create(embs):
        class Session:
            def run(self, outputs, inputs):
                assert inputs["input_features"].shape == (len(embs), 1, 1001, 64)
                return [embs]

        monkeypatch.setattr(mine.ort, "InferenceSession", lambda *args, **kwargs: Session())
        return ClapTagger()
    return create


def test_clap_full_window_voice_maximum(clap_with_embeddings):
    sr = 16000
    seg = np.zeros((35 * sr, 2), dtype=np.float32)
    seg[:5 * sr] = (0.02 * np.sin(2 * np.pi * 300 * np.arange(5 * sr) / sr))[:, None]
    seg[15 * sr:25 * sr] = np.random.default_rng(1).normal(0, 0.06, (10 * sr, 2))
    # 首、尾子窗口分别携带人声，验证最大池化不被中间窗口均值稀释。
    embs = np.eye(4, 512, dtype=np.float32)
    tagger = clap_with_embeddings(embs)
    tagger.voice_embeds = embs[[0, 3]]
    emb, _, _, _, _, voice = tagger.analyze(seg, sr)
    assert voice == pytest.approx(1.0)
    np.testing.assert_allclose(emb[:4], [0.5] * 4)


def test_clap_baseline_ranking_and_unknown(clap_with_embeddings):
    embs = np.eye(1, 512, dtype=np.float32)
    tagger = clap_with_embeddings(embs)
    tagger.trigger_embeds = np.zeros((2, 512), dtype=np.float32)
    tagger.trigger_embeds[:, 0] = [0.6, 0.4]
    tagger.slugs = np.array(["hub", "wood_tapping"])
    tagger.descs = np.array(["背景", "敲击木块"])
    tagger.tags = np.array(["背景", "敲击,木块"])
    seg = np.full((16000, 2), 0.01, dtype=np.float32)
    assert tagger.analyze(seg, 16000)[1] == "hub"
    result = tagger.analyze(seg, 16000, baseline=np.array([0.59, 0.2]))
    assert result[1:4] == ("wood_tapping", "敲击木块", ("敲击", "木块"))
    assert result[4] == pytest.approx(0.2)
    result = tagger.analyze(seg, 16000, baseline=np.array([0.59, 0.39]))
    assert result[1:4] == ("unknown_trigger", "待标注触发音", ("待标注",))


def test_vad_local_normalization_preserves_soft_opening(monkeypatch):
    sr = 16000
    seg = np.zeros((35 * sr, 2), dtype=np.float32)
    t = np.arange(5 * sr) / sr
    seg[:5 * sr] = (0.02 * np.sin(2 * np.pi * 300 * t))[:, None]
    seg[15 * sr:25 * sr] = np.random.default_rng(2).normal(0, 0.06, (10 * sr, 2))

    class Encoder:
        def run(self, outputs, inputs):
            frames = inputs["input"]
            # 在真实分帧边界验证开头轻声被独立放大，而空档保持静音。
            assert np.sqrt(np.mean(frames[1:100, 64:] ** 2)) == pytest.approx(0.07, abs=0.001)
            assert np.max(np.abs(frames[160:300])) == 0
            probs = np.full(len(frames), 0.1, dtype=np.float32)
            probs[:100] = 0.9
            return [probs[:, None, None]]

    class Decoder:
        def run(self, outputs, inputs):
            return inputs["input"], inputs["state"]

    sessions = iter((Encoder(), Decoder()))
    monkeypatch.setattr(mine.ort, "InferenceSession", lambda *args, **kwargs: next(sessions))
    ratio, run, peak = SileroVad().detect(seg, sr)
    assert run == pytest.approx(3.2)
    assert ratio > 0.08
    assert peak == pytest.approx(0.9)


@pytest.mark.parametrize("text,start,end", [
    ("01:40--14:21 耳边吹气", 100, 861),
    ("14:22 - 18:37 长火柴燃烧", 862, 1117),
    ("35:00~42:00 气泡水", 2100, 2520),
    ("35分到42分 气泡水", 2100, 2520),
    ("100s-200s 刷麦", 100, 200),
    ("1:00:00～1:02:00 custom prop", 3600, 3720),
    ("1分20秒至120秒 刷麦", 80, 120),
    ("100s to 200s 刷麦", 100, 200),
])
def test_timeline_range_formats(text, start, end):
    hint, = mine.parse_timeline_hints(text, 4000)
    assert (hint.start_sec, hint.end_sec) == (start, end)
    assert not hint.exclude


def test_timeline_chapters_exclusions_and_labels():
    hints = mine.parse_timeline_hints("00:06 预览\n01:40 耳边吹气\n14:22 长火柴燃烧\n35分 气泡水\n42分 忽略说话", 2600)
    assert [(h.start_sec, h.end_sec) for h in hints] == [(6, 100), (100, 862), (862, 2100), (2100, 2520), (2520, 2600)]
    assert [h.exclude for h in hints] == [True, False, False, False, True]
    assert hints[1].slug
    assert mine.parse_timeline_hints("00:00-00:10 zyzzyva prop", 20)[0].slug == "zyzzyva_prop"
    assert mine.parse_timeline_hints("00:00-00:10 芊芊陌陌", 20)[0].slug == ""
    assert mine.format_span(2140, 2175) == "35:40-36:15 (2140-2175s)"
    assert mine.format_span(3599, 3601) == "59:59-1:00:01 (3599-3601s)"


def test_supervised_round_robin_keeps_voice_and_exclusion_guards(tmp_path, monkeypatch):
    path = tmp_path / "chapters.wav"
    sf.write(path, np.full((200 * 100, 2), 0.02, dtype=np.float32), 100)
    features = np.tile([-40, -20, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.3], (200, 1))
    features[:100, 8] = 0.8  # First chapter scores much higher.
    report = mine.AudioQualityReport(100, 2, 200, -65, -40, -30, 35, -20, 0, 50, 1)

    class Clap:
        trigger_embeds = np.array([[1., 0.]])
        slugs = np.array(["water"])
        tags = np.array(["水声,气泡"])
        def analyze(self, seg, sr, baseline=None):
            return np.array([1., 0.]), "wood_tapping", "敲击木块", ("木块", "轻敲"), -0.2, 0.1

    class Vad:
        def detect(self, seg, sr, *args):
            return 0., 0., 0.

    hints = [mine.TimelineHint(0, 100, "自定义道具", "custom_prop"),
             mine.TimelineHint(100, 200, "气泡水", "water"),
             mine.TimelineHint(0, 20, "预览", exclude=True)]
    candidates = mine.mine_candidates(path, features, report, slice_seconds=20, min_slice_seconds=20,
                                      step_seconds=10, top_n=4, hints=hints, clap=Clap(), vad=Vad())
    assert [c.clap_slug for c in candidates] == ["custom_prop", "water", "custom_prop", "water"]
    assert all(c.start_sec >= 20 for c in candidates)
    assert candidates[0].clap_desc == "自定义道具"
    assert candidates[0].clap_tags == ("自定义道具",)
    assert candidates[1].clap_tags == ("水声", "气泡", "气泡水")
    assert candidates[0].score == pytest.approx(0.2 * 2.5 + 0.8 * 1.5 - 0.1 * 0.8)
    class VoiceClap(Clap):
        def analyze(self, seg, sr, baseline=None):
            emb, slug, desc, tags, score, _ = super().analyze(seg, sr, baseline)
            return emb, slug, desc, tags, score, 0.8
    assert not mine.mine_candidates(path, features, report, slice_seconds=20, min_slice_seconds=20,
                                    hints=hints, clap=VoiceClap())
    class SpeechVad(Vad):
        def detect(self, seg, sr, *args):
            return 0.5, 2., 0.9
    assert not mine.mine_candidates(path, features, report, slice_seconds=20, min_slice_seconds=20,
                                    hints=hints, clap=Clap(), vad=SpeechVad())


def test_prototypes_incremental_custom_class_and_builtin_fusion(tmp_path, monkeypatch):
    path = tmp_path / "confirmed.wav"
    sf.write(path, np.full((1000, 2), 0.02, dtype=np.float32), 1000)
    prompts = tmp_path / "prompts.npz"
    memory = tmp_path / "prototypes.npz"
    basis = np.eye(3, 512, dtype=np.float32)
    np.savez(prompts, mel_filters=np.zeros((513, 64)), slugs=["builtin"], descs=["内置"],
             tags=["旧标签"], trigger_embeds=basis[:1], voice_embeds=basis[2:])
    monkeypatch.setattr(mine.ort, "InferenceSession", lambda *args, **kwargs: None)
    current = [basis[1]]
    monkeypatch.setattr(ClapTagger, "_embed_chunks", lambda self, seg, sr: np.array(current))
    tagger = ClapTagger(prompts_path=prompts, prototypes_path=memory)
    seg, sr = mine.extract_slice(path, 0, 1)
    assert tagger.analyze(seg, sr)[1] == "unknown_trigger"
    mine.record_prototype(path, "new_prop", "新道具", ["甲"], memory, tagger)
    current[0] = (basis[0] + basis[1]) / np.sqrt(2)
    mine.record_prototype(path, "new_prop", "新描述", ["乙"], memory, tagger)
    mine.record_prototype(path, "builtin", "确认内置", ["监督"], memory, tagger)
    learned = ClapTagger(prompts_path=prompts, prototypes_path=memory)
    expected = basis[1] + current[0]
    expected /= np.linalg.norm(expected)
    with np.load(memory) as data:
        assert data["proto_slugs"].tolist() == ["new_prop", "builtin"]
        assert data["proto_descs"][0] == "新描述"
        assert data["proto_tags"][0] == "甲,乙"
        np.testing.assert_allclose(data["proto_embeds"][0], expected, atol=1e-6)
        assert data["proto_counts"].tolist() == [2, 1]
    # 已存中心已归一化；后续更新按样本计数对中心加权。
    current[0] = basis[2]
    mine.record_prototype(path, "new_prop", "新描述", ["乙"], memory, tagger)
    expected_third = (2 * expected + basis[2]) / 3
    expected_third /= np.linalg.norm(expected_third)
    with np.load(memory) as data:
        assert data["proto_counts"].tolist() == [3, 1]
        np.testing.assert_allclose(data["proto_embeds"][0], expected_third, atol=1e-6)
    current[0] = expected
    assert learned.analyze(seg, sr)[1:4] == ("new_prop", "新描述", ("甲", "乙"))
    fused = 0.4 * basis[0] + 0.6 * (basis[0] + basis[1]) / np.sqrt(2)
    np.testing.assert_allclose(learned.trigger_embeds[0], fused / np.linalg.norm(fused), atol=1e-6)
    assert learned.descs[0] == "内置"
    assert learned.tags[0] == "旧标签,监督"
    monkeypatch.setattr(config, "CLAP_PROMPTS_NPZ", prompts)
    monkeypatch.setattr(config, "PROTOTYPES_NPZ", memory)
    hints = mine.parse_timeline_hints("00:00-00:01 内置\n00:01-00:02 监督\n00:02-00:03 新描述", 3)
    assert [hint.slug for hint in hints] == ["builtin", "builtin", "new_prop"]


def test_cli_reads_hints_file(tmp_path, monkeypatch):
    from tingleloom import cli
    hints_file = tmp_path / "chapters.txt"
    hints_file.write_text("00:00-01:00 刷麦", encoding="utf-8")
    calls = []
    report = mine.AudioQualityReport(100, 2, 60, -65, -40, -30, 35, -20, 0, 50, 1)
    def ingest(path, **kwargs):
        calls.append(kwargs)
        return report, []
    monkeypatch.setattr(mine, "mine_and_ingest", ingest)
    assert cli.main(["material", "mine", "source.mp4", "--hints", str(hints_file)]) == 0
    assert calls[0]["hints_text"] == "00:00-01:00 刷麦"


@pytest.mark.parametrize("prefix", ["auto", "custom_prop"])
def test_unnamed_human_chapter_does_not_learn_guessed_category(source, tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(llm, "load_llm_config", lambda: llm.LLMConfig())
    memory = tmp_path / "prototypes.npz"
    emb = np.eye(1, 512, 1, dtype=np.float32)[0]

    class Clap:
        slugs = np.array(["wood_tapping"])
        tags = np.array(["木块,轻敲"])
        trigger_embeds = emb[None, :]
        def analyze(self, seg, sr, baseline=None):
            return emb, "wood_tapping", "敲击木块", ("木块", "轻敲"), 0.6, 0.1
        def _embed(self, seg, sr):
            return emb

    class Vad:
        def detect(self, seg, sr, *args):
            return 0., 0., 0.

    monkeypatch.setattr(mine, "ClapTagger", Clap)
    monkeypatch.setattr(mine, "SileroVad", Vad)
    record = mine.record_prototype
    learned = []
    def learn(path, slug, desc, tags, **kwargs):
        learned.append(slug)
        record(path, slug, desc, tags, prototypes_path=memory, **kwargs)
    monkeypatch.setattr(mine, "record_prototype", learn)
    report, results = mine.mine_and_ingest(
        source, prefix=prefix, hints_text="00:04-00:12 芊芊陌陌",
        slice_seconds=4, step_seconds=4, top_n=1, tags=("附加", "待标注"),
        out_dir=tmp_path / "sfx", index_path=tmp_path / "index.yaml",
    )
    assert report.timeline_hints[0].slug == ""
    cand, material = results[0]
    assert cand.clap_slug == "unknown_trigger"
    assert cand.clap_tags == ("待标注", "芊芊陌陌")
    assert cand.clap_desc == "芊芊陌陌"
    assert "木块" not in material.tags and "轻敲" not in material.tags
    if prefix == "auto":
        assert material.id.startswith("unknown_trigger_")
        assert material.is_pending
        assert material.tags == ["待标注", "芊芊陌陌", "附加"]
        assert learned == [] and not memory.exists()
    else:
        assert material.id.startswith("custom_prop_")
        assert not material.is_pending
        assert material.tags == ["芊芊陌陌", "附加"]
        assert learned == ["custom_prop"]
        with np.load(memory) as data:
            assert data["proto_slugs"].tolist() == ["custom_prop"]
            assert data["proto_descs"].tolist() == ["芊芊陌陌"]
            assert data["proto_tags"].tolist() == ["芊芊陌陌,附加"]
            np.testing.assert_array_equal(data["proto_embeds"][0], emb)



def test_repeated_ingest_avoids_source_spans_but_probe_keeps_them(source, tmp_path, monkeypatch):
    monkeypatch.setattr(mine, "record_prototype", lambda *args, **kwargs: None)
    options = dict(prefix="clean", slice_seconds=4, step_seconds=4, top_n=1,
                   out_dir=tmp_path / "sfx", index_path=tmp_path / "index.yaml",
                   hints_text="00:04-00:12 刮擦", source="original")
    _, first = mine_and_ingest(source, **options)
    _, second = mine_and_ingest(source, **options)
    assert len(first) == len(second) == 1
    a, b = first[0][0], second[0][0]
    assert not (a.start_sec < b.end_sec and b.start_sec < a.end_sec)
    _, third = mine_and_ingest(source, **options)
    assert third == []
    _, probes = mine_and_ingest(source, probe_only=True, **options)
    assert [(c.start_sec, c.end_sec) for c, _ in probes] == [(a.start_sec, a.end_sec)]


def test_timeline_local_matching_includes_learned_categories(tmp_path, monkeypatch):
    path = tmp_path / "prototypes.npz"
    np.savez(path, proto_slugs=np.array(["custom_shaving_foam"]), proto_descs=np.array(["剃须膏奶油搅拌"]), proto_tags=np.array(["剃须膏,奶油搅拌"]))
    monkeypatch.setattr(config, "PROTOTYPES_NPZ", path)
    hint, = mine.parse_timeline_hints("00:00-00:20 剃须膏奶油搅拌", 20)
    assert hint.slug == "custom_shaving_foam"


def test_resolve_foreign_timeline_labels_and_exclusions(monkeypatch):
    calls = []

    def classify(labels, source_hint=""):
        calls.append((labels, source_hint))
        return {
            "木製マカロン": {"slug": "wood_tapping", "desc": "木块敲击摩擦", "tags": ["木块", "轻敲", "木块"], "exclude": False},
            "プレビュー": {"slug": "preview", "desc": "预览", "tags": [], "exclude": True},
        }

    monkeypatch.setattr(llm, "classify_timeline_labels", classify)
    hints = mine.resolve_timeline_hints("00:00 木製マカロン\n00:10 プレビュー\n00:20 custom prop\n00:30 跳过说话", 40, "木头零食")
    assert calls == [(["木製マカロン", "プレビュー", "custom prop"], "木头零食")]
    assert hints[0].slug == "wood_tapping" and hints[0].label == "木块敲击摩擦"
    assert hints[0].tags == ("木块", "轻敲", "木製マカロン")
    assert hints[1].exclude and hints[3].exclude
    assert hints[2].slug == "custom_prop"
    monkeypatch.setattr(llm, "parse_timeline_hints_text", lambda text, duration: "00:00-00:10 木製マカロン")
    assert mine.resolve_timeline_hints("从开头十秒是日语木制点心", 10)[0].slug == "wood_tapping"


@pytest.mark.parametrize("error", [ValueError("invalid JSON"), RuntimeError("API failed")])
def test_timeline_classification_failure_keeps_local_hints(monkeypatch, error):
    def classify(*args, **kwargs):
        raise error

    monkeypatch.setattr(llm, "classify_timeline_labels", classify)
    hints = mine.resolve_timeline_hints("00:00-00:10 木製マカロン", 10)
    assert hints == mine.parse_timeline_hints("00:00-00:10 木製マカロン", 10)


def test_foreign_timeline_ingest_propagates_translation_and_source(source, tmp_path, monkeypatch):
    calls, learned = [], []

    def classify(labels, source_hint=""):
        calls.append((labels, source_hint))
        return {"木製マカロン": {"slug": "wood_tapping", "desc": "木块敲击摩擦", "tags": ["木块", "轻敲"], "exclude": False}}

    monkeypatch.setattr(llm, "classify_timeline_labels", classify)
    monkeypatch.setattr(mine, "record_prototype", lambda *args, **kwargs: learned.append(args))
    report, results = mine.mine_and_ingest(source, hints_text="00:04-00:12 木製マカロン", source="木头和剃须膏制作零食",
                                          slice_seconds=4, step_seconds=4, top_n=1, out_dir=tmp_path / "sfx", index_path=tmp_path / "index.yaml")
    assert calls == [(["木製マカロン"], "木头和剃须膏制作零食")]
    assert report.timeline_hints[0].slug == "wood_tapping"
    cand, material = results[0]
    assert cand.clap_slug == "wood_tapping" and cand.clap_desc == "木块敲击摩擦"
    assert {"木块", "轻敲", "木製マカロン"}.issubset(material.tags)
    assert not material.is_pending and material.path.is_file()
    assert learned[0][2] == "木块敲击摩擦"


def test_prototype_export_verify_and_merge_lifecycle(tmp_path, monkeypatch):
    from tingleloom import cli

    source = tmp_path / "confirmed.wav"
    samples = np.full((16000, 2), 0.1, dtype=np.float32)
    samples[:1000] = 1.0
    samples[6000:] = 0.4
    sf.write(source, samples, 1000, subtype="FLOAT")
    prompts = tmp_path / "prompts.npz"
    memory = tmp_path / "prototypes.npz"
    index = tmp_path / "index.yaml"
    out = tmp_path / "contributions"
    basis = np.eye(4, 512, dtype=np.float32)
    np.savez(prompts, mel_filters=np.zeros((513, 64)), slugs=["builtin"], descs=["内置"],
             tags=["旧标签"], trigger_embeds=basis[:1], voice_embeds=basis[3:], extra=np.array([42]))
    materials.save_index([materials.Material(id="new_prop_01_C", file=str(source), desc="新道具01_双耳均衡",
                                             tags=["新标签"], duration_sec=16)], index)
    monkeypatch.setattr(config, "CLAP_PROMPTS_NPZ", prompts)
    monkeypatch.setattr(mine.ort, "InferenceSession", lambda *args, **kwargs: None)
    current = [basis[1]]
    monkeypatch.setattr(ClapTagger, "_embed_chunks", lambda self, seg, sr: np.array(current))
    tagger = ClapTagger(prompts_path=prompts, prototypes_path=memory)

    class Vad:
        ratio = 0.0
        def detect(self, seg, sr):
            return self.ratio, self.ratio * 2, 0.0

    vad = Vad()
    pkg = mine.export_prototype("new_prop", out_dir=out, prototypes_path=memory,
                                index_path=index, clap=tagger)
    proof = pkg / "proof_10s.wav"
    data, sr = sf.read(proof)
    assert len(data) / sr == 10
    assert np.max(np.abs(data)) < 0.99
    assert np.max(np.abs(data[0])) == 0
    payload_path = pkg / "prototype.json"
    original = json.loads(payload_path.read_text(encoding="utf-8"))
    assert original["desc"] == "新道具" and original["tags"] == ["新标签"]
    assert original["sample_count"] == 1
    np.testing.assert_allclose(original["embedding"], basis[1])
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert report.passed and report.top1_slug == "new_prop"
    assert report.repro_cosine == pytest.approx(1) and report.margin == pytest.approx(1)

    matched = materials.load_index(index)
    matched.append(materials.Material(id="new_prop_02_C", file=str(source), desc="新道具02_双耳均衡",
                                      tags=["新标签", "补充标签"], duration_sec=16))
    materials.save_index(matched, index)
    multi_pkg = mine.export_prototype("new_prop", out_dir=tmp_path / "multi", prototypes_path=memory,
                                      index_path=index, clap=tagger)
    multi = json.loads((multi_pkg / "prototype.json").read_text(encoding="utf-8"))
    assert multi["desc"] == "新道具" and multi["tags"] == ["新标签", "补充标签"]
    assert multi["sample_count"] == 2

    # 默认核验使用不加载本地 prototypes 的基线。
    original_tagger = mine.ClapTagger
    def baseline(**kwargs):
        assert kwargs["prototypes_path"] == Path("__nonexistent__")
        return original_tagger(**kwargs)
    monkeypatch.setattr(mine, "ClapTagger", baseline)
    assert mine.verify_prototype(pkg, prompts_path=prompts, vad=vad).passed
    monkeypatch.setattr(mine, "ClapTagger", original_tagger)

    forged = dict(original, embedding=basis[2].tolist())
    payload_path.write_text(json.dumps(forged), encoding="utf-8")
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert not report.passed and any("向量复现" in r for r in report.reasons)
    with pytest.raises(ValueError, match="原型校验未通过"):
        mine.merge_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert proof.is_file()
    payload_path.write_text(json.dumps(original), encoding="utf-8")
    vad.ratio = 0.2
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert not report.passed and any("人声活动" in r for r in report.reasons)
    vad.ratio = 0.0
    payload_path.write_text(json.dumps(dict(original, embedding=basis[0].tolist())), encoding="utf-8")
    current[0] = basis[0]
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert not report.passed and report.margin == pytest.approx(0)
    assert any("区分度不足" in r for r in report.reasons)
    current[0] = basis[3]
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert any("负向人声" in r for r in report.reasons)
    payload_path.write_text(json.dumps(original), encoding="utf-8")
    current[0] = basis[1]
    clipped = data.copy()
    clipped[100] = 1.0
    sf.write(proof, clipped, sr, subtype="FLOAT")
    report = mine.verify_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad)
    assert report.clip_samples == 2 and not report.passed
    sf.write(proof, data, sr, subtype="FLOAT")

    assert mine.merge_prototype(pkg, prompts_path=prompts, clap=tagger, vad=vad).passed
    assert not proof.exists()
    with np.load(prompts) as merged:
        assert merged["slugs"].tolist() == ["builtin", "new_prop"]
        assert merged["sample_counts"].tolist() == [1, 1]
        assert merged["extra"].tolist() == [42]
        np.testing.assert_allclose(merged["trigger_embeds"][1], basis[1])

    # 已有类别强化：验证和归并使用相同计数权重。
    current[0] = (basis[0] + basis[1]) / np.sqrt(2)
    np.savez(memory, proto_slugs=["builtin"], proto_descs=["强化"], proto_tags=["新标签"],
             proto_embeds=np.array(current), proto_counts=np.array([3], dtype=np.int32))
    builtin_pkg = mine.export_prototype("builtin", out_dir=out, proof_audio=source,
                                        prototypes_path=memory, index_path=index, clap=tagger)
    assert mine.merge_prototype(builtin_pkg, prompts_path=prompts, remove_proof=False,
                                clap=tagger, vad=vad).passed
    assert (builtin_pkg / "proof_10s.wav").is_file()
    expected = basis[0] + 3 * current[0]
    expected /= np.linalg.norm(expected)
    with np.load(prompts) as merged:
        assert merged["sample_counts"].tolist() == [4, 1]
        assert merged["descs"][0] == "强化" and merged["tags"][0] == "旧标签,新标签"
        np.testing.assert_allclose(merged["trigger_embeds"][0], expected, atol=1e-6)

    # CLI 仍调用真实流水线，只将依赖定向到临时库。
    export, verify, merge = mine.export_prototype, mine.verify_prototype, mine.merge_prototype
    def cli_export(slug, **kwargs):
        return export(slug, prototypes_path=memory, index_path=index, clap=tagger, **kwargs)
    def cli_verify(pkg_dir, **kwargs):
        kwargs.update(prompts_path=prompts, clap=tagger, vad=vad)
        return verify(pkg_dir, **kwargs)
    def cli_merge(pkg_dir, **kwargs):
        return merge(pkg_dir, prompts_path=prompts, clap=tagger, vad=vad, **kwargs)
    monkeypatch.setattr(mine, "export_prototype", cli_export)
    monkeypatch.setattr(mine, "verify_prototype", cli_verify)
    monkeypatch.setattr(mine, "merge_prototype", cli_merge)
    monkeypatch.setattr(config, "CONTRIBUTIONS_DIR", out)
    current[0] = basis[2]
    assert cli.main(["material", "proto-export", "cli_prop", "--audio", str(source), "--dur", "5"]) == 0
    cli_pkg = out / "cli_prop"
    assert sf.info(cli_pkg / "proof_10s.wav").duration == 5
    assert cli.main(["material", "proto-verify", str(cli_pkg)]) == 0
    vad.ratio = 0.1
    assert cli.main(["material", "proto-verify", str(cli_pkg)]) == 1
    vad.ratio = 0.0
    assert cli.main(["material", "proto-merge", str(cli_pkg), "--keep-proof"]) == 0
    assert (cli_pkg / "proof_10s.wav").is_file()
    assert cli.main(["material", "proto-merge", str(cli_pkg)]) == 0
    assert not (cli_pkg / "proof_10s.wav").exists()


@pytest.mark.parametrize("changes", [
    {"slug": "../escape"}, {"slug": "unknown_trigger"}, {"desc": 42}, {"tags": "tag"},
    {"sample_count": 0}, {"sample_count": True}, {"embedding": [0.] * 511},
    {"embedding": [0.] * 512}, {"embedding": [float("nan")] * 512},
])
def test_prototype_rejects_invalid_contribution_metadata(tmp_path, changes):
    payload = dict(slug="new_prop", desc="新道具", tags=[], sample_count=1,
                   embedding=np.eye(1, 512, dtype=np.float32)[0].tolist())
    payload.update(changes)
    (tmp_path / "prototype.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        mine.verify_prototype(tmp_path)

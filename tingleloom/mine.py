"""流式声学体检、时间轴监督归类、触发音挖掘与素材入库。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import json
import re

import av
import numpy as np
import onnxruntime as ort
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from . import audio, config, materials as materials_mod


def format_timestamp(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    h, rem = divmod(sec, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def format_span(start_sec: float, end_sec: float) -> str:
    return f"{format_timestamp(start_sec)}-{format_timestamp(end_sec)} ({int(round(start_sec))}-{int(round(end_sec))}s)"


@dataclass
class TimelineHint:
    start_sec: int
    end_sec: int
    label: str
    slug: str = ""
    exclude: bool = False
    tags: Tuple[str, ...] = ()


_TIME = r"(?:\d+:\d{2}(?::\d{2})?|\d+\s*分(?:\s*\d+\s*秒)?|\d+\s*(?:秒|s))"
_HINT_LINE = re.compile(rf"^\s*({_TIME})(?:\s*(?:--|-|~|～|—|至|到|to)\s*({_TIME}))?\s*(.*)$", re.I)
_SKIP_WORDS = ("跳过", "忽略", "排除", "说话", "人声", "聊天", "预览", "片头", "片尾", "广告", "skip", "ignore", "intro", "outro", "talk")


def _time_seconds(value: str) -> int:
    if ":" in value:
        total = 0
        for part in value.split(":"):
            total = total * 60 + int(part)
        return total
    nums = [int(n) for n in re.findall(r"\d+", value)]
    return nums[0] * 60 + (nums[1] if len(nums) > 1 else 0) if "分" in value else nums[0]


def parse_timeline_hints(text: str, duration_sec: float) -> List[TimelineHint]:
    rows = []
    for line in text.splitlines():
        match = _HINT_LINE.match(line)
        if match and match[3].strip():
            rows.append((_time_seconds(match[1]), _time_seconds(match[2]) if match[2] else None, match[3].strip()))
    rows.sort(key=lambda row: row[0])
    if not rows:
        return []
    with np.load(config.CLAP_PROMPTS_NPZ) as prompts:
        by_slug = {str(slug): materials_mod.Material(id=str(slug), file="", desc=str(desc), tags=str(tags).split(","))
                   for slug, desc, tags in zip(prompts["slugs"], prompts["descs"], prompts["tags"])}
    if config.PROTOTYPES_NPZ.is_file():
        with np.load(config.PROTOTYPES_NPZ) as prototypes:
            for slug, desc, tags in zip(prototypes["proto_slugs"], prototypes["proto_descs"], prototypes["proto_tags"]):
                if str(slug) in by_slug:
                    material = by_slug[str(slug)]
                    material.tags = list(dict.fromkeys([*material.tags, *str(tags).split(",")]))
                else:
                    by_slug[str(slug)] = materials_mod.Material(id=str(slug), file="", desc=str(desc), tags=str(tags).split(","))
    classes = list(by_slug.values())
    hints = []
    for i, (start, end, label) in enumerate(rows):
        end = min(int(duration_sec), end if end is not None else rows[i + 1][0] if i + 1 < len(rows) else int(duration_sec))
        if start >= end:
            continue
        exclude = any(word in label.lower() for word in _SKIP_WORDS)
        slug = ""
        if not exclude:
            best = max(classes, key=lambda m: materials_mod.score_material(m, label))
            slug = best.id if materials_mod.score_material(best, label) > 0 else "_".join(re.findall(r"[a-z]+", label.lower()))
        hints.append(TimelineHint(start, end, label, slug, exclude))
    return hints


def resolve_timeline_hints(text: str, duration_sec: float, source_hint: str = "") -> List[TimelineHint]:
    from . import llm

    hints = parse_timeline_hints(text, duration_sec)
    if text.strip() and not hints:
        hints = parse_timeline_hints(llm.parse_timeline_hints_text(text, duration_sec), duration_sec)
        if not hints:
            raise ValueError("未能从时间戳提示中解析出有效时间区间，请使用 MM:SS-MM:SS 描述 格式或配置 LLM API Key")
    to_classify = [h for h in hints if not h.exclude and (
        not h.slug or re.search(r"[\u3040-\u30ff]", h.label) or not re.search(r"[\u4e00-\u9fff]", h.label)
    )]
    if to_classify:
        try:
            classified = llm.classify_timeline_labels([h.label for h in to_classify], source_hint=source_hint)
        except (ValueError, RuntimeError):
            return hints
        for hint in to_classify:
            item = classified.get(hint.label)
            if item is not None:
                orig_label = hint.label
                hint.slug, hint.label = item["slug"], item["desc"]
                hint.tags = tuple(dict.fromkeys((*item["tags"], orig_label)))
                hint.exclude = bool(item.get("exclude", False))
    return hints


@dataclass
class AudioQualityReport:
    sample_rate: int
    channels: int
    duration_sec: float
    noise_floor_db: float
    median_db: float
    active_db: float
    snr_db: float
    max_peak_db: float
    total_clip_samples: int
    cutoff_hz: float
    mean_lr_corr: float
    timeline_hints: Tuple[TimelineHint, ...] = ()


@dataclass
class MinedCandidate:
    start_sec: int
    end_sec: int
    channel_tag: str
    score: float
    mean_rms_db: float
    mean_abs_rl_db: float
    sub_ratio: float
    formant_ratio: float
    high_ratio: float
    flatness: float
    vad_ratio: float = 0.0
    vad_max_run_sec: float = 0.0
    vad_peak: float = 0.0
    trajectory_5s: str = ""
    clap_slug: str = ""
    clap_desc: str = ""
    clap_tags: Tuple[str, ...] = ()
    clap_score: float = 0.0
    clap_voice_score: float = 0.0
    timeline_hint: Optional[TimelineHint] = None


CHANNEL_DESC = materials_mod.CHANNEL_DESC
_SF_SUFFIXES = {".wav", ".flac", ".ogg"}
_EPS = 1e-12
_VAD_FRAME_SECONDS = 512 / 16000
_VAD_BLOCK_FRAMES = (5 * 16000) // 512
_VAD_PROB_THRESHOLD = 0.4
_VAD_DEBOUNCE_FRAMES = 3
_CLAP_SR = 48000
_CLAP_CHUNK_SAMPLES = 480000
_CLAP_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)).astype(np.float32)


class ClapTagger:
    """纯 NumPy 前处理与 ONNX 音频编码；词表向量随包预计算。"""

    def __init__(self, model_path: Path = config.CLAP_AUDIO_ONNX,
                 prompts_path: Path = config.CLAP_PROMPTS_NPZ,
                 prototypes_path: Path = config.PROTOTYPES_NPZ):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        with np.load(prompts_path) as prompts:
            self.mel_filters = prompts["mel_filters"]
            self.slugs = prompts["slugs"]
            self.descs = prompts["descs"]
            self.tags = prompts["tags"]
            self.trigger_embeds = prompts["trigger_embeds"]
            self.voice_embeds = prompts["voice_embeds"]

        if prototypes_path.is_file():
            with np.load(prototypes_path) as prototypes:
                slugs, descs, tags = self.slugs.tolist(), self.descs.tolist(), self.tags.tolist()
                embeds = list(self.trigger_embeds)
                for slug, desc, tag, emb in zip(prototypes["proto_slugs"], prototypes["proto_descs"], prototypes["proto_tags"], prototypes["proto_embeds"]):
                    if slug in slugs:
                        idx = slugs.index(slug)
                        fused = 0.4 * embeds[idx] + 0.6 * emb
                        embeds[idx] = fused / max(float(np.linalg.norm(fused)), _EPS)
                        tags[idx] = ",".join(dict.fromkeys((*str(tags[idx]).split(","), *str(tag).split(","))))
                    else:
                        slugs.append(str(slug))
                        descs.append(str(desc))
                        tags.append(str(tag))
                        embeds.append(emb)
                self.slugs, self.descs, self.tags = np.asarray(slugs), np.asarray(descs), np.asarray(tags)
                self.trigger_embeds = np.asarray(embeds)

    def _embed_chunks(self, seg: np.ndarray, sr: int) -> np.ndarray:
        rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2, axis=0))
        if np.sum(rms) < 1e-4:
            mono = np.zeros(len(seg), dtype=np.float32)
        else:
            mono = ((rms[0] * seg[:, 0] + rms[1] * seg[:, 1]) / np.sum(rms)).astype(np.float32)
        mono_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        if mono_rms >= 1e-4:
            mono = np.clip(mono / mono_rms * 0.08, -0.99, 0.99).astype(np.float32)
        mono = audio.resample(mono[:, None], sr, _CLAP_SR)[:, 0]
        if len(mono) <= _CLAP_CHUNK_SAMPLES:
            reps = _CLAP_CHUNK_SAMPLES // len(mono)
            tiled = np.tile(mono, reps)[:_CLAP_CHUNK_SAMPLES]
            chunks = [np.pad(tiled, (0, _CLAP_CHUNK_SAMPLES - len(tiled)))]
        else:
            starts = np.unique(np.linspace(
                0, len(mono) - _CLAP_CHUNK_SAMPLES,
                num=max(2, int(np.ceil(len(mono) / _CLAP_CHUNK_SAMPLES))), dtype=int))
            chunks = [mono[start:start + _CLAP_CHUNK_SAMPLES] for start in starts]
        log_mels = []
        for chunk in chunks:
            padded = np.pad(chunk, (512, 512), mode="reflect")
            frames = sliding_window_view(padded, 1024)[::480] * _CLAP_WINDOW
            power = np.abs(np.fft.rfft(frames, n=1024, axis=1)) ** 2
            mel = (power @ self.mel_filters).astype(np.float32)
            log_mels.append(10.0 * np.log10(np.maximum(mel, 1e-10)))
        batch_log_mel = np.stack(log_mels)[:, None, :, :].astype(np.float32)
        embs = self.session.run(None, {"input_features": batch_log_mel})[0].astype(np.float32)
        embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), _EPS)
        return embs

    def _embed(self, seg: np.ndarray, sr: int) -> np.ndarray:
        emb = np.mean(self._embed_chunks(seg, sr), axis=0)
        return emb / max(float(np.linalg.norm(emb)), _EPS)

    def analyze(self, seg: np.ndarray, sr: int, baseline: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, str, str, Tuple[str, ...], float, float]:
        embs = self._embed_chunks(seg, sr)
        voice_score = float(np.max(embs @ self.voice_embeds.T))
        emb = np.mean(embs, axis=0)
        emb /= max(float(np.linalg.norm(emb)), _EPS)
        raw_sims = self.trigger_embeds @ emb
        rank_sims = raw_sims if baseline is None else raw_sims - baseline
        best = int(np.argmax(rank_sims))
        score = float(rank_sims[best])
        if score < (0.15 if baseline is None else 0.05):
            return emb, "unknown_trigger", "待标注触发音", ("待标注",), score, voice_score
        return (emb, str(self.slugs[best]), str(self.descs[best]),
                tuple(str(self.tags[best]).split(",")), score, voice_score)


def record_prototype(audio_path: Path, slug: str, desc: str, tags: Sequence[str],
                     prototypes_path: Path = config.PROTOTYPES_NPZ,
                     clap: Optional[ClapTagger] = None) -> None:
    seg, sr = audio.read_audio(audio_path)
    emb = (clap if clap is not None else ClapTagger())._embed(seg, sr)
    slugs, descs, all_tags, embeds, counts = [], [], [], [], []
    if prototypes_path.is_file():
        with np.load(prototypes_path) as data:
            slugs, descs, all_tags = data["proto_slugs"].tolist(), data["proto_descs"].tolist(), data["proto_tags"].tolist()
            embeds = list(data["proto_embeds"])
            counts = data["proto_counts"].tolist() if "proto_counts" in data else [1] * len(slugs)
    if slug in slugs:
        idx = slugs.index(slug)
        n = int(counts[idx])
        emb = (n * embeds[idx] + emb) / (n + 1)
        counts[idx] = n + 1
        emb /= max(float(np.linalg.norm(emb)), _EPS)
        descs[idx] = desc
        all_tags[idx] = ",".join(dict.fromkeys((*all_tags[idx].split(","), *tags)))
        embeds[idx] = emb
    else:
        slugs.append(slug)
        descs.append(desc)
        all_tags.append(",".join(tags))
        embeds.append(emb)
        counts.append(1)
    prototypes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prototypes_path, proto_slugs=slugs, proto_descs=descs, proto_tags=all_tags, proto_embeds=np.asarray(embeds, dtype=np.float32),
             proto_counts=np.asarray(counts, dtype=np.int32))


def export_prototype(
    slug: str,
    out_dir: Path = config.CONTRIBUTIONS_DIR,
    proof_audio: Optional[Path] = None,
    proof_seconds: float = 10.0,
    prototypes_path: Path = config.PROTOTYPES_NPZ,
    index_path: Path = config.MATERIALS_INDEX,
    clap: Optional[ClapTagger] = None,
) -> Path:
    """导出本地原型与代表性证明音频，供社区 PR 核验。"""
    _validate_prototype_slug(slug)
    if not 5.0 <= proof_seconds <= 10.0:
        raise ValueError("证明音频目标时长必须在 5 至 10 秒之间")
    if proof_audio is not None and not Path(proof_audio).is_file():
        raise FileNotFoundError(f"证明音频不存在：{proof_audio}")
    matched = sorted((m for m in materials_mod.load_index(index_path)
                      if m.category == slug and not m.is_pending and m.path.is_file()),
                     key=lambda m: m.duration_sec, reverse=True)
    if proof_audio is None and not matched:
        raise ValueError(f"未找到类别 {slug} 的本地可用音频，请通过 --audio 指定证明音频")
    src_audio = Path(proof_audio) if proof_audio is not None else matched[0].path
    data, sr = audio.read_audio(src_audio)
    if len(data) / sr < 5.0:
        raise ValueError("证明音频时长不能少于 5 秒")
    win = int(round(proof_seconds * sr))
    if len(data) > win:
        # 前缀和使每个候选窗口只需常数时间：优先最少削波，再取最高能量。
        energy = np.concatenate(([0.0], np.cumsum(np.sum(data.astype(np.float64) ** 2, axis=1))))
        clips = np.concatenate(([0], np.cumsum(np.sum(np.abs(data) >= 0.99, axis=1))))
        starts = range(0, len(data) - win + 1, max(1, sr // 10))
        best_start = max(starts, key=lambda s: (-(clips[s + win] - clips[s]), energy[s + win] - energy[s]))
        seg = data[best_start:best_start + win]
    else:
        seg = data
    seg = audio.fade_edges(seg, sr, config.SFX_CUT_FADE, config.SFX_CUT_FADE)

    emb = None
    sample_count = None
    if prototypes_path.is_file():
        with np.load(prototypes_path) as prototypes:
            slugs = prototypes["proto_slugs"].tolist()
            if slug in slugs:
                idx = slugs.index(slug)
                desc = str(prototypes["proto_descs"][idx])
                tags_list = [t for t in str(prototypes["proto_tags"][idx]).split(",") if t]
                sample_count = int(prototypes["proto_counts"][idx]) if "proto_counts" in prototypes else 1
                if sample_count > 1:
                    emb = np.asarray(prototypes["proto_embeds"][idx], dtype=np.float32)
    if emb is None:
        tagger = clap if clap is not None else ClapTagger()
    if sample_count is None:
        desc = re.sub(r"\d+_(?:左耳|右耳|双耳均衡|左右交替|右左交替)$", "", matched[0].desc) if matched else ""
        desc = desc or materials_mod.category_descriptions().get(slug, slug)
        tags_list = list(dict.fromkeys(t for m in matched for t in m.tags))
        matched_paths = [m.path for m in matched] if matched else [src_audio]
        sample_count = len(matched_paths)
        if sample_count > 1:
            embs = []
            for path in matched_paths:
                samples, sample_rate = audio.read_audio(path)
                embs.append(tagger._embed(samples, sample_rate))
            emb = np.mean(embs, axis=0)
            emb /= max(float(np.linalg.norm(emb)), _EPS)
    pkg_dir = Path(out_dir) / slug
    pkg_dir.mkdir(parents=True, exist_ok=True)
    proof_path = pkg_dir / "proof_10s.wav"
    audio.write_audio(proof_path, seg, sr)
    if sample_count == 1:
        proof, proof_sr = audio.read_audio(proof_path)
        emb = tagger._embed(proof, proof_sr)
    payload = {"slug": slug, "desc": desc, "tags": tags_list, "sample_count": sample_count,
               "embedding": [round(float(x), 7) for x in emb]}
    (pkg_dir / "prototype.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pkg_dir


def _validate_prototype_slug(slug: str) -> None:
    if not isinstance(slug, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug) is None:
        raise ValueError("类别 slug 只能包含英文字母、数字、下划线和连字符")
    if slug == "unknown_trigger":
        raise ValueError("待标注类别 unknown_trigger 不能贡献原型")


def _read_prototype_payload(pkg_dir: Path) -> Tuple[dict, np.ndarray]:
    payload = json.loads((pkg_dir / "prototype.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(k in payload for k in ("slug", "desc", "tags", "sample_count", "embedding")):
        raise ValueError("prototype.json 缺少原型字段")
    _validate_prototype_slug(payload["slug"])
    if not isinstance(payload["desc"], str) or not isinstance(payload["tags"], list) or not all(isinstance(t, str) for t in payload["tags"]):
        raise ValueError("原型 desc 必须为字符串，tags 必须为字符串列表")
    if type(payload["sample_count"]) is not int or payload["sample_count"] < 1:
        raise ValueError("原型 sample_count 必须为正整数")
    emb = np.asarray(payload["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if emb.shape != (512,) or not 0.99 <= norm <= 1.01:
        raise ValueError("原型 embedding 必须是长度 512、L2 范数在 [0.99, 1.01] 内的向量")
    return payload, emb / norm


@dataclass
class PrototypeVerifyReport:
    slug: str
    passed: bool
    repro_cosine: float
    vad_ratio: float
    vad_max_run_sec: float
    voice_score: float
    clip_samples: int
    top1_slug: str
    margin: float
    reasons: Tuple[str, ...] = ()


def verify_prototype(
    pkg_dir: Path,
    prompts_path: Path = config.CLAP_PROMPTS_NPZ,
    clap: Optional[ClapTagger] = None,
    vad: Optional[SileroVad] = None,
) -> PrototypeVerifyReport:
    """以不含本地原型的基线核验贡献包的复现性、纯净度和区分度。"""
    pkg_dir = Path(pkg_dir)
    payload, e_sub = _read_prototype_payload(pkg_dir)
    slug, sample_count = payload["slug"], payload["sample_count"]
    proof_path = pkg_dir / "proof_10s.wav"
    if not proof_path.is_file():
        raise FileNotFoundError(proof_path)
    seg, sr = audio.read_audio(proof_path)
    if not np.isfinite(seg).all():
        raise ValueError("证明音频必须只包含有限采样值")
    reasons = []
    if not 4.5 <= len(seg) / sr <= 10.5:
        reasons.append("证明音频时长必须在 4.5 至 10.5 秒之间")
    tagger = clap if clap is not None else ClapTagger(prompts_path=prompts_path, prototypes_path=Path("__nonexistent__"))
    detector = vad if vad is not None else SileroVad()
    embs = tagger._embed_chunks(seg, sr)
    e_proof = np.mean(embs, axis=0)
    e_proof /= max(float(np.linalg.norm(e_proof)), _EPS)
    repro_cosine = float(e_sub @ e_proof)
    min_repro = 0.999 if sample_count == 1 else 0.85
    if repro_cosine < min_repro:
        reasons.append(f"向量复现余弦相似度不足：{repro_cosine:.4f} < {min_repro}")
    vad_ratio, vad_max_run_sec, _ = detector.detect(seg, sr)
    voice_score = float(np.max(embs @ tagger.voice_embeds.T))
    clip_samples = int(np.sum(np.abs(seg) >= 0.99))
    if vad_ratio > 0.0 or vad_max_run_sec > 0.0:
        reasons.append(f"检测到人声活动：vad_ratio={vad_ratio:.2%}, max_run={vad_max_run_sec:.2f}s")
    if voice_score >= 0.35:
        reasons.append(f"CLAP 负向人声/耳语得分过高：{voice_score:.3f} >= 0.35")
    if clip_samples > 0:
        reasons.append(f"证明音频存在削波样本：{clip_samples}")
    with np.load(prompts_path) as prompts:
        existing_slugs = prompts["slugs"].tolist()
        trigger_embeds = prompts["trigger_embeds"]
        existing_counts = prompts["sample_counts"] if "sample_counts" in prompts else np.ones(len(existing_slugs), dtype=np.int32)
        if slug not in existing_slugs:
            other_sims = trigger_embeds @ e_proof
            best_idx = int(np.argmax(other_sims))
            best_other = float(other_sims[best_idx])
            best_other_slug = existing_slugs[best_idx]
            margin = repro_cosine - best_other
            top1_slug = slug if margin >= 0 else best_other_slug
            if margin <= 0.03:
                reasons.append(f"新增类别区分度不足（最高撞车类别 {best_other_slug}）：margin={margin:.4f} <= 0.03")
        else:
            idx = existing_slugs.index(slug)
            fused = existing_counts[idx] * trigger_embeds[idx] + sample_count * e_sub
            trigger_embeds[idx] = fused / max(float(np.linalg.norm(fused)), _EPS)
            sims = trigger_embeds @ e_proof
            top1_slug = existing_slugs[int(np.argmax(sims))]
            other_mask = np.arange(len(existing_slugs)) != idx
            best_other = float(np.max(sims[other_mask])) if np.any(other_mask) else -1.0
            margin = float(sims[idx]) - best_other
            if top1_slug != slug or margin <= 0.0:
                reasons.append(f"既有类别强化后未命中自身（top1={top1_slug}, margin={margin:.4f}）")
    return PrototypeVerifyReport(slug, not reasons, repro_cosine, vad_ratio, vad_max_run_sec,
                                 voice_score, clip_samples, top1_slug, margin, tuple(reasons))


def merge_prototype(
    pkg_dir: Path,
    prompts_path: Path = config.CLAP_PROMPTS_NPZ,
    remove_proof: bool = True,
    clap: Optional[ClapTagger] = None,
    vad: Optional[SileroVad] = None,
) -> PrototypeVerifyReport:
    """核验后按样本数归并内置词表，保留词表中的其他数组。"""
    pkg_dir = Path(pkg_dir)
    report = verify_prototype(pkg_dir, prompts_path=prompts_path, clap=clap, vad=vad)
    if not report.passed:
        raise ValueError("原型校验未通过：" + "；".join(report.reasons))
    payload, e_sub = _read_prototype_payload(pkg_dir)
    with np.load(prompts_path) as prompts:
        arrays = {key: prompts[key] for key in prompts.files}
    slugs, descs, tags = (arrays[key].tolist() for key in ("slugs", "descs", "tags"))
    embeds = list(arrays["trigger_embeds"])
    counts = arrays["sample_counts"].tolist() if "sample_counts" in arrays else [1] * len(slugs)
    slug, new_n = payload["slug"], payload["sample_count"]
    if slug in slugs:
        idx = slugs.index(slug)
        old_n = int(counts[idx])
        fused = (old_n * embeds[idx] + new_n * e_sub) / (old_n + new_n)
        embeds[idx] = fused / max(float(np.linalg.norm(fused)), _EPS)
        counts[idx] = old_n + new_n
        tags[idx] = ",".join(dict.fromkeys((*str(tags[idx]).split(","), *payload["tags"])))
        if payload["desc"]:
            descs[idx] = payload["desc"]
    else:
        slugs.append(slug)
        descs.append(payload["desc"])
        tags.append(",".join(payload["tags"]))
        embeds.append(e_sub)
        counts.append(new_n)
    arrays.update(slugs=np.asarray(slugs), descs=np.asarray(descs), tags=np.asarray(tags),
                  trigger_embeds=np.asarray(embeds, dtype=np.float32), sample_counts=np.asarray(counts, dtype=np.int32))
    np.savez(prompts_path, **arrays)
    proof_path = pkg_dir / "proof_10s.wav"
    if remove_proof and proof_path.is_file():
        proof_path.unlink()
    return report


def _blocks(path: Path, start_sec: float = 0.0,
            end_sec: Optional[float] = None) -> Iterator[Tuple[np.ndarray, int, int]]:
    """只保留当前解码块；容器时间戳以音轨起点为零。"""
    path = Path(path)
    if path.suffix.lower() in _SF_SUFFIXES:
        with sf.SoundFile(path) as reader:
            sr, channels = reader.samplerate, reader.channels
            reader.seek(min(round(start_sec * sr), len(reader)))
            stop = len(reader) if end_sec is None else min(round(end_sec * sr), len(reader))
            while reader.tell() < stop:
                data = reader.read(min(sr, stop - reader.tell()), dtype="float32", always_2d=True)
                yield audio.to_stereo(data), sr, channels
    else:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"文件中没有音频流：{path}")
            stream = container.streams.audio[0]
            sr = stream.codec_context.sample_rate
            channels = stream.codec_context.channels
            origin = stream.start_time if stream.start_time is not None else 0
            if start_sec > 0:
                # 有损编码需要预卷恢复重叠窗/位储备；输出仍严格裁到请求区间。
                seek_sec = max(0.0, start_sec - 1.0)
                container.seek(origin + int(seek_sec / stream.time_base), stream=stream, backward=True)
            converter = av.AudioResampler(format="fltp")
            start_sample = round(start_sec * sr)
            stop_sample = None if end_sec is None else round(end_sec * sr)
            for frame in container.decode(stream):
                for converted in converter.resample(frame):
                    offset = round((converted.pts * converted.time_base - origin * stream.time_base) * sr)
                    if stop_sample is not None and offset >= stop_sample:
                        return
                    left = max(0, start_sample - offset)
                    right = converted.samples if stop_sample is None else min(converted.samples, stop_sample - offset)
                    if right > left:
                        data = converted.to_ndarray().T[left:right]
                        yield audio.to_stereo(data).astype(np.float32, copy=False), sr, channels


def extract_slice(path: Path, start_sec: float, end_sec: float,
                  target_sr: int = config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    if not 0 <= start_sec < end_sec:
        raise ValueError("切片时间必须满足 0 <= start < end")
    chunks = []
    sr = target_sr
    for block, sr, _ in _blocks(path, start_sec, end_sec):
        chunks.append(block)
    if not chunks:
        raise ValueError("切片区间内没有音频")
    seg = np.concatenate(chunks)
    if sr != target_sr:
        seg = audio.resample(seg, sr, target_sr)
    return np.ascontiguousarray(seg, dtype=np.float32), target_sr


def scan_audio(path: Path) -> Tuple[AudioQualityReport, np.ndarray]:
    rows, correlations = [], []
    pending = np.empty((0, 2), dtype=np.float32)
    total_samples, clips, max_peak = 0, 0, 0.0
    active_count = 0
    for block, sr, channels in _blocks(path):
        if total_samples == 0:
            window = np.hanning(sr)[:, None]
            freq = np.fft.rfftfreq(sr, 1 / sr)
            formant_mask = (freq >= 400) & (freq < 2800)
            band_masks = (
                (freq >= 0) & (freq < 180),
                formant_mask,
                (freq >= 2800) & (freq < 7000),
                (freq >= 7000) & (freq < 16000),
            )
        total_samples += len(block)
        clips += int(np.sum(np.abs(block) >= 0.99))
        max_peak = max(max_peak, float(np.max(np.abs(block))))
        pending = np.concatenate((pending, block))
        while len(pending) >= sr:
            seg, pending = pending[:sr], pending[sr:]
            rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2, axis=0))
            rms_db = 20 * np.log10(max(np.sqrt(np.mean(rms ** 2)), _EPS))
            peak_db = 20 * np.log10(max(float(np.max(np.abs(seg))), _EPS))
            diff = 20 * np.log10(max(rms[1], _EPS) / max(rms[0], _EPS))
            spectrum = np.mean(np.abs(np.fft.rfft(seg * window, axis=0)) ** 2, axis=1)
            energy = max(float(np.sum(spectrum)), _EPS)
            ratios = [float(np.sum(spectrum[mask]) / energy) for mask in band_masks]
            formant = spectrum[formant_mask]
            flatness = float(np.exp(np.mean(np.log(formant + _EPS))) / (np.mean(formant) + _EPS))
            rows.append([rms_db, peak_db, diff, np.sum(np.abs(seg) >= 0.99), *ratios, flatness])
            centered = seg.astype(np.float64) - np.mean(seg, axis=0)
            denom = np.sqrt(np.sum(centered[:, 0] ** 2) * np.sum(centered[:, 1] ** 2))
            correlations.append(float(np.sum(centered[:, 0] * centered[:, 1]) / max(denom, _EPS)))
            if rms_db > -45.0:
                if active_count == 0:
                    active_spectrum = np.zeros_like(spectrum)
                active_spectrum += spectrum
                active_count += 1
    if not rows:
        raise ValueError("音频时长不足 1 秒")
    features = np.asarray(rows, dtype=np.float64)
    noise, median, active = np.percentile(features[:, 0], [10, 50, 90])
    cutoff = 0.0
    if active_count:
        active_spectrum /= active_count
        bins = np.flatnonzero(active_spectrum > np.max(active_spectrum) * 10 ** (-75 / 10))
        cutoff = float(freq[bins[-1]])
    report = AudioQualityReport(sr, channels, total_samples / sr, float(noise), float(median),
                                float(active), float(active - noise),
                                float(20 * np.log10(max(max_peak, _EPS))), clips,
                                cutoff, float(np.mean(correlations)))
    return report, features


def classify_channel_tag(rl_diff_db: np.ndarray) -> str:
    w = len(rl_diff_db)
    l_mask, r_mask = rl_diff_db < -4.0, rl_diff_db > 4.0
    l_cnt, r_cnt = int(np.sum(l_mask)), int(np.sum(r_mask))
    if l_cnt > w * 0.65:
        return "L"
    if r_cnt > w * 0.65:
        return "R"
    if l_cnt >= max(2, int(w * 0.14)) and r_cnt >= max(2, int(w * 0.14)):
        return "LR" if np.flatnonzero(l_mask)[0] <= np.flatnonzero(r_mask)[0] else "RL"
    return "C"


def format_trajectory_5s(seg: np.ndarray, sr: int) -> str:
    parts = []
    for t in range(0, len(seg), 5 * sr):
        rms = np.sqrt(np.mean(seg[t:t + 5 * sr].astype(np.float64) ** 2, axis=0))
        diff = float(20 * np.log10(max(rms[1], _EPS) / max(rms[0], _EPS)))
        tag = "L" if diff < -4 else "R" if diff > 4 else "C"
        parts.append(f"{tag}({diff:+.0f})")
    return " -> ".join(parts)


class SileroVad:
    def __init__(self, encoder_path: Path = config.SILERO_ENCODER_ONNX,
                 decoder_path: Path = config.SILERO_DECODER_ONNX):
        self.enc = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    def detect(self, seg: np.ndarray, sr: int,
               prob_threshold: float = _VAD_PROB_THRESHOLD,
               debounce_frames: int = _VAD_DEBOUNCE_FRAMES) -> Tuple[float, float, float]:
        active_ratio, max_run_sec, peak_prob = 0.0, 0.0, 0.0
        for mono in seg.T:
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            if rms < 1e-4:
                continue
            blocks = []
            for i in range(0, len(mono), 5 * sr):
                sub = mono[i:i + 5 * sr]
                sub_rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
                blocks.append(sub / sub_rms * 0.07
                              if sub_rms >= max(1e-4, rms * 0.15) else sub * 0.0)
            norm_mono = np.clip(np.concatenate(blocks), -0.99, 0.99)
            mono = audio.resample(norm_mono[:, None], sr, 16000)[:, 0]
            # 保留不足一帧的尾部，避免短切片或尾部人声被遗漏。
            padded = np.pad(mono, (64, (-len(mono)) % 512))
            frames = np.ascontiguousarray(sliding_window_view(padded, 576)[::512], dtype=np.float32)
            features = self.enc.run(None, {"input": frames})[0][:, :, 0]
            probs = []
            for idx, feature in enumerate(features):
                if idx % _VAD_BLOCK_FRAMES == 0:
                    state = np.zeros((2, 1, 128), dtype=np.float32)
                prob, state = self.dec.run(None, {"input": feature[None, :], "state": state})
                probs.append(float(prob.item()))
            probs = np.asarray(probs)
            active = probs > prob_threshold
            edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            for start, end in zip(starts, ends):
                if end - start <= debounce_frames:
                    active[start:end] = False
            valid_runs = [int(end - start) for start, end in zip(starts, ends)
                          if end - start > debounce_frames]
            active_ratio = max(active_ratio, float(np.mean(active)))
            max_run_sec = max(max_run_sec, float(max(valid_runs, default=0) * _VAD_FRAME_SECONDS))
            peak_prob = max(peak_prob, float(np.max(probs)))
        return active_ratio, max_run_sec, peak_prob


def _vad_params(cand: MinedCandidate) -> Tuple[float, int]:
    if cand.formant_ratio > 0.48 and cand.flatness < 0.28:
        return 0.22, 2
    return _VAD_PROB_THRESHOLD, _VAD_DEBOUNCE_FRAMES


def mine_candidates(path: Path, features: np.ndarray, report: AudioQualityReport,
                    profile: str = "high", slice_seconds: int = 35, step_seconds: int = 10,
                    top_n: int = 15, max_vad_ratio: float = 0.015,
                    min_gap_seconds: Optional[int] = None,
                    vad: Optional[SileroVad] = None, max_vad_run_sec: float = 0.25,
                    min_slice_seconds: Optional[int] = None,
                    clap: Optional[ClapTagger] = None, max_clap_voice: float = 0.42,
                    diversity_max_sim: float = 0.90,
                    min_clap_score: float = 0.04, hints: Sequence[TimelineHint] = (),
                    existing_spans: Sequence[Tuple[int, int]] = ()) -> List[MinedCandidate]:
    positive_hints = [h for h in hints if not h.exclude]

    if profile not in ("high", "low", "balanced", "bed"):
        raise ValueError("profile 仅支持 high、low、balanced、bed")
    if slice_seconds <= 0 or step_seconds <= 0 or top_n <= 0:
        raise ValueError("切片时长、步长和数量必须为正数")
    if min_slice_seconds is None:
        min_slice_seconds = min(slice_seconds, 15)
    if min_slice_seconds <= 0 or min_slice_seconds > slice_seconds:
        raise ValueError("最短切片时长必须满足 0 < min_slice_seconds <= slice_seconds")
    win_schedule = list(range(slice_seconds, min_slice_seconds, -10)) + [min_slice_seconds]
    silence_floor = min(max(report.noise_floor_db + 1.5, -68.0), report.median_db - 1.5)
    min_rms = min(-50.0, max(silence_floor + 3.0, report.median_db - 2.0))
    selected = []
    selected_embs = []
    baseline = None
    if clap is not None:
        sample_win = min(slice_seconds, len(features))
        active_starts = np.flatnonzero(features[:len(features) - sample_win + 1, 0] >= silence_floor)
        if len(active_starts) > 1:
            starts = active_starts[np.unique(np.linspace(0, len(active_starts) - 1,
                                                         min(12, len(active_starts)), dtype=int))]
            sample_embs = []
            for start in starts:
                seg, sr = extract_slice(path, int(start), int(start) + sample_win)
                sample_embs.append(clap.analyze(seg, sr)[0])
            mean_emb = np.mean(sample_embs, axis=0)
            baseline = clap.trigger_embeds @ (mean_emb / max(float(np.linalg.norm(mean_emb)), _EPS))

    def conflicts(cand: MinedCandidate) -> bool:
        if any(cand.start_sec < end and start < cand.end_sec for start, end in existing_spans):
            return True
        if min_gap_seconds is not None:
            return any((cand.start_sec < c.end_sec and c.start_sec < cand.end_sec)
                       or abs(cand.start_sec - c.start_sec) < min_gap_seconds for c in selected)
        return any(cand.start_sec < c.end_sec + 10 and c.start_sec < cand.end_sec + 10
                   for c in selected)

    for win in win_schedule:
        if len(selected) == top_n:
            break
        ranked = []
        for t in range(0, len(features) - win + 1, step_seconds):
            if any(h.exclude and t < h.end_sec and h.start_sec < t + win for h in hints):
                continue
            hint = next((h for h in positive_hints if h.start_sec <= t + win / 2 < h.end_sec), None)
            if positive_hints and hint is None:
                continue
            f = features[t:t + win]
            if np.sum(f[:, 3]) > 0 or np.max(f[:, 1]) > -1.5:
                continue
            mean_rms = float(np.mean(f[:, 0]))
            if mean_rms < min_rms or mean_rms > -26.0:
                continue
            if np.sum(f[:, 0] < silence_floor) > max(2, win // 5):
                continue
            m_sub, m_for, m_sibil, m_air, m_flat = np.mean(f[:, 4:], axis=0)
            m_high = m_sibil + m_air
            m_abs_rl = float(np.mean(np.abs(f[:, 2])))
            if profile == "bed":
                rms_std = float(np.std(f[:, 0]))
                crest = float(np.max(f[:, 1] - f[:, 0]))
                if rms_std > 2.5 or crest > 14.0:
                    continue
                if m_abs_rl > 3.5 or classify_channel_tag(f[:, 2]) != "C":
                    continue
                score = float(m_flat * 2.0 + (m_sub + m_high) * 0.8 - m_for * 1.2
                              - rms_std * 0.6 - crest * 0.08 - m_abs_rl * 0.15)
            elif profile == "high":
                score = m_high * 2.5 + m_flat * 1.5 - m_for * 0.8 + m_abs_rl / 30.0
            elif profile == "low":
                score = m_sub * 1.8 + m_abs_rl / 20.0 + m_flat * 1.2 - m_for * 0.6
            else:
                score = m_flat * 1.8 + m_high + m_sub - m_for * 0.6 + m_abs_rl / 30.0
            ranked.append(MinedCandidate(t, t + win, classify_channel_tag(f[:, 2]), float(score),
                                         mean_rms, m_abs_rl, float(m_sub), float(m_for),
                                         float(m_high), float(m_flat), timeline_hint=hint))
        deferred = []
        ordered = sorted(ranked, key=lambda c: c.score, reverse=True)
        groups = ([deque(c for c in ordered if c.timeline_hint is h) for h in positive_hints]
                  if positive_hints else [deque(ordered)])
        chapter_counts = [sum(c.timeline_hint is h for c in selected) for h in positive_hints] if positive_hints else [0]
        while any(groups):
            group_idx = min((i for i, group in enumerate(groups) if group), key=lambda i: chapter_counts[i])
            cand = groups[group_idx].popleft()
            hint = cand.timeline_hint
            if conflicts(cand):
                continue
            seg, sr = extract_slice(path, cand.start_sec, cand.end_sec)
            if vad is not None:
                cand.vad_ratio, cand.vad_max_run_sec, cand.vad_peak = vad.detect(seg, sr, *_vad_params(cand))
                if cand.vad_ratio > max_vad_ratio or cand.vad_max_run_sec > max_vad_run_sec:
                    continue
            if clap is not None:
                emb, slug, desc, c_tags, c_score, v_score = clap.analyze(seg, sr, baseline=baseline)
                cand.clap_slug, cand.clap_desc, cand.clap_tags = slug, desc, c_tags
                cand.clap_score, cand.clap_voice_score = c_score, v_score
                if v_score > max_clap_voice or (v_score > 0.35 and v_score > c_score):
                    continue
                min_trigger_score = min_clap_score if baseline is not None else max(min_clap_score, 0.12)
                if hint is None:
                    if profile == "bed":
                        if cand.clap_slug == "unknown_trigger":
                            cand.clap_slug, cand.clap_desc, cand.clap_tags = (
                                "ambient_bed", "稳态环境床音", ("环境音", "床音"),
                            )
                        else:
                            cand.clap_tags = tuple(dict.fromkeys((*cand.clap_tags, "环境音")))
                    elif c_score < min_trigger_score:
                        continue
            if hint is not None:
                if hint.slug:
                    cand.clap_slug = hint.slug
                    slug_list = list(clap.slugs) if clap is not None else []
                    base_tags = tuple(str(clap.tags[slug_list.index(hint.slug)]).split(",")) if hint.slug in slug_list else ()
                    cand.clap_tags = tuple(dict.fromkeys((*base_tags, *hint.tags, hint.label)))
                else:
                    cand.clap_slug = "unknown_trigger"
                    cand.clap_tags = ("待标注", hint.label)
                cand.clap_desc = hint.label
            cand.trajectory_5s = format_trajectory_5s(seg, sr)
            if clap is not None:
                if hint is None and any(float(emb @ prev) > diversity_max_sim for prev in selected_embs):
                    deferred.append(cand)
                    continue
                selected_embs.append(emb)
            selected.append(cand)
            chapter_counts[group_idx] += 1
            if len(selected) == top_n:
                break
        for cand in deferred:
            if len(selected) == top_n:
                break
            if not conflicts(cand):
                selected.append(cand)
    return selected


def mine_and_ingest(path: Path, prefix: str = "auto", profile: str = "high",
                    slice_seconds: int = 35, step_seconds: int = 10, top_n: int = 15,
                    max_vad_ratio: float = 0.015, out_dir: Optional[Path] = None,
                    desc_prefix: str = "", tags: Tuple[str, ...] = (), source: str = "",
                    license: str = "", commercial_ok: bool = True, probe_only: bool = False,
                    index_path: Path = config.MATERIALS_INDEX,
                    max_vad_run_sec: float = 0.25, min_slice_seconds: Optional[int] = None,
                    max_clap_voice: float = 0.42, diversity_max_sim: float = 0.90,
                    min_clap_score: float = 0.04, hints_text: str = ""
                    ) -> Tuple[AudioQualityReport, List[Tuple[MinedCandidate, Optional[materials_mod.Material]]]]:
    report, features = scan_audio(path)
    path = Path(path)
    report.timeline_hints = tuple(resolve_timeline_hints(hints_text, report.duration_sec, source_hint=source or path.stem)) if hints_text.strip() else ()
    origin_name = source or path.name
    if probe_only:
        existing_spans = ()
    else:
        materials = materials_mod.load_index(index_path)
        existing_spans = materials_mod.existing_source_spans(materials, origin_name)
    clap = ClapTagger()
    candidates = mine_candidates(path, features, report, profile, slice_seconds, step_seconds,
                                 top_n, max_vad_ratio, vad=SileroVad(),
                                 max_vad_run_sec=max_vad_run_sec, min_slice_seconds=min_slice_seconds,
                                 clap=clap, hints=report.timeline_hints, max_clap_voice=max_clap_voice,
                                 diversity_max_sim=diversity_max_sim, min_clap_score=min_clap_score,
                                 existing_spans=existing_spans)
    if probe_only:
        return report, [(c, None) for c in candidates]
    out_dir = Path(out_dir) if out_dir is not None else (config.BED_DIR if profile == "bed" else config.SFX_DIR)
    results = []
    for cand in candidates:
        seg, sr = extract_slice(path, cand.start_sec, cand.end_sec)
        fade_sec = min(config.BED_FADE, (cand.end_sec - cand.start_sec) / 4.0) if profile == "bed" else config.SFX_CUT_FADE
        audio.fade_edges(seg, sr, fade_sec, fade_sec)
        stem_prefix = cand.clap_slug if prefix == "auto" else prefix
        mat_id, seq, dest = materials_mod.next_available_id(
            stem_prefix, cand.channel_tag, materials, out_dir,
        )
        audio.write_audio(dest, seg, sr)
        base_desc = desc_prefix or cand.clap_desc
        desc = f"{base_desc}{seq:02d}_{CHANNEL_DESC[cand.channel_tag]}"
        raw_tags = dict.fromkeys((*cand.clap_tags, *tags))
        if stem_prefix == "unknown_trigger":
            merged_tags = tuple(dict.fromkeys(("待标注", *raw_tags)))
        else:
            merged_tags = tuple(t for t in raw_tags if t != "待标注")
        origin = f"{origin_name} @ {format_span(cand.start_sec, cand.end_sec)}"
        material = materials_mod.register(dest, materials, mat_id=mat_id, desc=desc, tags=merged_tags, source=origin,
                                          license=license, commercial_ok=commercial_ok)
        if stem_prefix != "unknown_trigger" and cand.timeline_hint is not None:
            record_prototype(dest, stem_prefix, base_desc, merged_tags, clap=clap)
        results.append((cand, material))
    materials_mod.save_index(materials, index_path)
    return report, results

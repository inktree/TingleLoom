# -*- coding: utf-8 -*-
"""流水线编排：台本 → TTS → 素材匹配 → 时间轴 → 混音 → 成品 + build 产物。

产物目录：
    build/timeline.json  冻结时间轴（各段落起止、事件落点、成品信息）
    build/events.json    触发音匹配结果 + 缺口清单
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from . import config
from . import mix as mix_mod
from .materials import load_index
from .presets import load_beds
from .script import load_script
from .sfx import ResolvedSfx, resolve_sfx
from .timeline import Timeline, build_timeline
from .tts import (
    OnnxDeployEngine,
    SynthOptions,
    TTSCache,
    provider_signature,
    tts_fingerprint,
    synthesize_all,
)
from .tts.batch import SegmentAudio


@dataclass
class RunResult:
    script_path: Path
    output_path: Path
    mp3_path: Path
    timeline: Timeline
    segment_audios: List[SegmentAudio]
    resolved: List[ResolvedSfx]
    gaps: List[str]
    mix_report: mix_mod.MixReport
    warnings: List[str] = field(default_factory=list)


def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    script_path,
    out_path=None,
    *,
    options: Optional[SynthOptions] = None,
    seed: Optional[int] = None,
    target_lufs: Optional[float] = None,
    sr: Optional[int] = None,
    tts_dir=None,
    strict: bool = False,
    progress: Optional[Callable] = None,
    fingerprint: Optional[dict] = None,
    engine_factory: Optional[Callable] = None,
) -> RunResult:
    config.ensure_dirs()
    sr = sr or config.SAMPLE_RATE
    script_path = Path(script_path)
    script = load_script(script_path)
    warnings: List[str] = list(script.warnings)

    materials = load_index()
    beds = load_beds()
    resolved, gaps = resolve_sfx(script, materials)

    options = options or SynthOptions()
    seed = script.seed if seed is None else seed
    target_lufs = config.TARGET_LUFS if target_lufs is None else target_lufs

    fingerprint = fingerprint or tts_fingerprint(tts_dir)
    providers = provider_signature()
    cache = TTSCache()

    tts_tmp = config.BUILD_DIR / "tts_tmp"

    def make_engine():
        if engine_factory is not None:
            return engine_factory()
        return OnnxDeployEngine(
            deploy=tts_dir,
            output_dir=tts_tmp,
        )

    segment_audios = synthesize_all(
        script.text_segments, options, seed, cache,
        fingerprint["engine_id"], fingerprint["sample_rate"], providers,
        make_engine, progress=progress,
    )
    # 引擎产物已拷进内容寻址缓存，清掉临时目录
    shutil.rmtree(tts_tmp, ignore_errors=True)

    timeline = build_timeline(script, segment_audios)

    if out_path is None:
        out_path = config.OUTPUT_DIR / (script_path.stem + ".wav")

    mix_report = mix_mod.render(
        timeline, resolved, beds, materials,
        pan_swap=script.pan_swap, sr=sr, target_lufs=target_lufs, seed=seed,
        output_path=out_path, warnings=warnings,
    )

    _write_build(script_path, resolved, gaps, timeline, mix_report)

    if strict and gaps:
        raise RuntimeError("有未匹配的触发音（--strict）：%s" % ", ".join(gaps))

    return RunResult(script_path, mix_report.output_path, mix_report.mp3_path, timeline,
                     segment_audios, resolved, gaps, mix_report, warnings)


def _write_build(script_path, resolved, gaps, timeline, mix_report) -> None:

    timeline_payload = {
        "pipeline_version": config.PIPELINE_VERSION,
        "script": str(script_path),
        "sample_rate": mix_report.sample_rate,
        "total_seconds": mix_report.duration_seconds,
        "segments": [
            {
                "order": p.audio.order, "start": round(p.start, 3), "duration": round(p.audio.duration, 3),
                "key": p.audio.key, "preset": p.audio.segment.preset, "text": p.audio.segment.text,
                "from_cache": p.audio.from_cache,
            }
            for p in timeline.segments
        ],
        "sfx": [
            {
                "start": round(p.start, 3), "duration": p.duration,
                "description": p.occurrence.directive.description,
                "level": p.occurrence.directive.level, "pan": p.occurrence.directive.pan.raw,
                "material": (r.material.id if r.material else None), "resolved": r.resolved,
                "line": p.occurrence.line_no, "origin": p.occurrence.origin,
            }
            for p, r in zip(timeline.sfx, resolved)
        ],
        "beds": [{"start": round(b.start, 3), "name": b.name} for b in timeline.beds],
        "output": str(mix_report.output_path),
        "output_mp3": str(mix_report.mp3_path),
        "loudness_lufs": round(mix_report.loudness_lufs, 2),
        "program_lufs": round(mix_report.program_lufs, 2),
        "bed_lufs": round(mix_report.bed_lufs, 2),
        "bed_offset_db": mix_report.bed_offset_db,
        "bed_gain_db": round(mix_report.bed_gain_db, 2),
        "program_gain_db": round(mix_report.program_gain_db, 2),
        "peak": round(mix_report.peak, 4),
    }
    _write_json(config.BUILD_DIR / "timeline.json", timeline_payload)

    _write_json(config.BUILD_DIR / "events.json", {
        "gaps": gaps,
        "events": [
            {
                "description": r.occurrence.directive.description,
                "material": (r.material.id if r.material else None),
                "line": r.occurrence.line_no, "origin": r.occurrence.origin,
            }
            for r in resolved
        ],
    })

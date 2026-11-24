# -*- coding: utf-8 -*-
"""混音器（numpy）：把时间轴、人声、触发音、床音叠成一条成品。

音量关系（依 ASMR 行业经验）：
    节目（人声 + 触发音）→ 归一化到 TARGET_LUFS（约 −20 LUFS）
    环境床音             → **按预设**放到「节目 − offset_db」之下
        · 纯底噪（room_quiet）压很深（~24 dB）
        · 雨声这类"听得见"的铺底压得浅（~14 dB）

床音**不参与节目归一**：逐段渲染 → 逐段定位 → 叠加。总线最后整体软限幅。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import audio, config
from . import presets as presets_mod
from . import sfx as sfx_mod
from .presets import BedPreset
from .sfx import ResolvedSfx, apply_pan
from .timeline import Timeline


@dataclass
class MixReport:
    output_path: Path
    mp3_path: Path
    duration_seconds: float
    sample_rate: int
    peak: float
    loudness_lufs: float
    program_lufs: float
    bed_lufs: float
    bed_offset_db: float
    bed_gain_db: float
    program_gain_db: float
    vocals: int
    sfx: int
    beds: int
    warnings: List[str] = field(default_factory=list)


def bed_gain_db(program_lufs: float, bed_lufs: float, offset_db: float) -> float:
    """把床音放到「节目响度 − offset_db」所需的增益（dB）。"""
    if not np.isfinite(bed_lufs):
        return 0.0
    return (program_lufs - offset_db) - bed_lufs


def _add_at(canvas: np.ndarray, start_frame: int, buffer: np.ndarray) -> None:
    if len(buffer) == 0:
        return
    n = canvas.shape[0]
    end = min(n, start_frame + len(buffer))
    if end <= start_frame:
        return
    canvas[start_frame:end] += buffer[:end - start_frame]


def render(
    timeline: Timeline,
    resolved: List[ResolvedSfx],
    bed_presets: Dict[str, BedPreset],
    materials: List,
    pan_swap: Optional[float] = None,
    sr: int = config.SAMPLE_RATE,
    target_lufs: float = config.TARGET_LUFS,
    seed: int = 1234,
    output_path=None,
    warnings: Optional[List[str]] = None,
) -> MixReport:
    warns = warnings if warnings is not None else []

    # 1. 人声：读入 → 归一化到 VOCAL_REFERENCE_LUFS → 声像
    vocal_events = []
    for placed in timeline.segments:
        data, _ = audio.read_audio(placed.audio.wav_path, target_sr=sr)
        pan = placed.audio.segment.pan.for_start(placed.start, pan_swap)
        gain = config.VOCAL_GAIN_DB
        seg_lufs = audio.loudness_lufs(data, sr)
        if np.isfinite(seg_lufs):
            gain += config.VOCAL_REFERENCE_LUFS - seg_lufs
        data = audio.apply_gain_db(data, gain)
        vocal_events.append((placed.start, apply_pan(data, pan, sr=sr)))

    # 2. 触发音：按素材准备后，在时间轴位置统一应用声像
    sfx_events = []
    for index, (placed, res) in enumerate(zip(timeline.sfx, resolved)):
        if res.material is None:
            continue
        rng = np.random.default_rng(seed + 1000 + index)
        buffer = sfx_mod.prepare_sfx(res.occurrence, res.material, sr=sr, rng=rng)
        pan = placed.occurrence.directive.pan.for_start(placed.start, pan_swap)
        sfx_events.append((placed.start, apply_pan(buffer, pan, sr=sr)))

    # 3. 内容长度（不含末尾空白）
    content_total = timeline.total_seconds
    for start, buffer in vocal_events:
        content_total = max(content_total, start + len(buffer) / sr)
    for start, buffer in sfx_events:
        content_total = max(content_total, start + len(buffer) / sr)
    n = max(int(round(content_total * sr)), 1)

    # 4. 节目（人声 + 触发音）画布
    program = np.zeros((n, 2), dtype=np.float32)
    for start, buffer in vocal_events:
        _add_at(program, int(round(start * sr)), buffer)
    for start, buffer in sfx_events:
        _add_at(program, int(round(start * sr)), buffer)

    # 5. 先把节目归一化到目标响度（床音要相对它定位）
    program_lufs = audio.loudness_lufs(program, sr)
    program_gain = 0.0
    if audio.peak(program) > 0 and np.isfinite(program_lufs):
        program_gain = target_lufs - program_lufs
        program = audio.apply_gain_db(program, program_gain)
        program_lufs = audio.loudness_lufs(program, sr)
    else:
        program_lufs = target_lufs

    # 6. 床音：逐段渲染 → 逐段定位到「节目 − offset」→ 叠加
    bed_canvas = np.zeros((n, 2), dtype=np.float32)
    bed_gain = 0.0
    bed_offset = config.BED_OFFSET_DB
    beds = sorted(timeline.beds, key=lambda b: b.start)
    for index, bed in enumerate(beds):
        preset = bed_presets.get(bed.name)
        if preset is None:
            warns.append("未知床音预设：%s" % bed.name)
            continue
        next_start = beds[index + 1].start if index + 1 < len(beds) else content_total
        seg_start = max(0.0, bed.start - preset.xfade if index > 0 else bed.start)
        seg_end = next_start + preset.xfade if index + 1 < len(beds) else next_start
        duration = max(0.0, seg_end - seg_start)
        buffer = presets_mod.render_bed(
            preset, duration, materials, sr=sr, warnings=warns, apply_fades=False
        )
        fade_in = preset.xfade if index > 0 else 0.0
        fade_out = preset.xfade if index + 1 < len(beds) else 0.0
        if fade_in or fade_out:
            buffer = audio.fade_edges(buffer, sr, fade_in, fade_out)

        offset = preset.offset_db if preset.offset_db is not None else config.BED_OFFSET_DB
        if audio.peak(buffer) > 0:
            seg_lufs = audio.loudness_lufs(buffer, sr)
            gain = bed_gain_db(program_lufs, seg_lufs, offset)
            buffer = audio.apply_gain_db(buffer, gain)
            if index == 0:
                bed_gain = gain
                bed_offset = offset
        _add_at(bed_canvas, int(round(seg_start * sr)), buffer)

    bed_lufs = audio.loudness_lufs(bed_canvas, sr) if audio.peak(bed_canvas) > 0 else 0.0

    # 7. 总线：叠加 → 软限幅（不再整体归一，否则会把床音又提上来）
    final = audio.soft_limit(program + bed_canvas)
    # 末尾淡出，再留一段空白收尾
    if config.OUTRO_FADE_SECONDS > 0:
        final = audio.fade_edges(final, sr, 0.0, config.OUTRO_FADE_SECONDS)
    tail_frames = int(round(config.TAIL_SECONDS * sr))
    if tail_frames > 0:
        tail = np.zeros((tail_frames, final.shape[1]), dtype=np.float32)
        final = np.concatenate([final, tail], axis=0)
    final_lufs = audio.loudness_lufs(final, sr)

    out = (Path(output_path) if output_path else (config.OUTPUT_DIR / "mix.wav")).with_suffix(".wav")
    mp3_out = out.with_suffix(".mp3")
    audio.write_audio(out, final, sr, subtype=config.OUTPUT_SUBTYPE)
    audio.write_mp3(mp3_out, final, sr)

    return MixReport(
        output_path=out,
        mp3_path=mp3_out,
        duration_seconds=len(final) / sr,
        sample_rate=sr,
        peak=audio.peak(final),
        loudness_lufs=final_lufs,
        program_lufs=program_lufs,
        bed_lufs=bed_lufs,
        bed_offset_db=bed_offset,
        bed_gain_db=bed_gain,
        program_gain_db=program_gain,
        vocals=len(vocal_events),
        sfx=len(sfx_events),
        beds=len(timeline.beds),
        warnings=warns,
    )

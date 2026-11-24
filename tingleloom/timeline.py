# -*- coding: utf-8 -*-
"""时间轴：游标模型。台本只描述事件，不描述等待关系。

    人声行        → 从游标 t 起播，播完 t 前进该段实际时长
    (pause N)    → t 前进 N 秒
    (sfx foo 时长) → 在 t 处启动，自己播完，t 不动
    @bed 名       → 在 t 处切换环境音

节目名义总长 = 游标终点；触发音造成的额外尾巴由混音器按实际长度补齐。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .script import BedChange, Pause, Script, SfxEvent, TextSegment
from .sfx import SfxOccurrence, collect_occurrences
from .tts.batch import SegmentAudio


@dataclass
class PlacedSegment:
    start: float
    duration: float
    audio: SegmentAudio


@dataclass
class PlacedSfx:
    start: float
    duration: Optional[float]   # None = 单次（长度取决于素材）
    occurrence: SfxOccurrence


@dataclass
class PlacedBed:
    start: float
    name: str


@dataclass
class Timeline:
    segments: List[PlacedSegment] = field(default_factory=list)
    sfx: List[PlacedSfx] = field(default_factory=list)
    beds: List[PlacedBed] = field(default_factory=list)
    total_seconds: float = 0.0


def build_timeline(script: Script, segment_audios: List[SegmentAudio]) -> Timeline:
    occ_iter = iter(collect_occurrences(script))
    if len(script.text_segments) != len(segment_audios):
        raise ValueError("文本段数量与 TTS 产物数量不一致")
    timeline = Timeline()
    cursor = 0.0
    seg_iter = iter(segment_audios)

    for item in script.items:
        if isinstance(item, TextSegment):
            audio_seg = next(seg_iter)
            timeline.segments.append(PlacedSegment(cursor, audio_seg.duration, audio_seg))
            for directive in item.sfx:
                occ = next(occ_iter)
                timeline.sfx.append(PlacedSfx(cursor + (directive.at or 0.0), directive.duration, occ))
            cursor += audio_seg.duration
        elif isinstance(item, Pause):
            cursor += item.seconds
        elif isinstance(item, SfxEvent):
            directive = item.directive
            start = cursor + (directive.at or 0.0)
            occ = next(occ_iter)
            timeline.sfx.append(PlacedSfx(start, directive.duration, occ))
        elif isinstance(item, BedChange):
            timeline.beds.append(PlacedBed(cursor, item.name))

    timeline.total_seconds = cursor
    return timeline

# -*- coding: utf-8 -*-
"""逐段 TTS 批处理：一段一请求 + 内容寻址缓存 + 续跑。

一段一请求是刻意的：引擎对每个 fragment 用 seed+index 采样，单请求单 fragment
时 index 恒为 0，插入新段不影响任何旧段，缓存粒度 = 请求粒度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..script import TextSegment
from .base import SynthOptions, TTSEngine
from .cache import TTSCache, make_key


@dataclass
class SegmentAudio:
    key: str
    order: int
    segment: TextSegment
    wav_path: Path
    duration: float
    sample_rate: int
    from_cache: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.segment.text


def synthesize_all(
    segments: Sequence[TextSegment],
    options: SynthOptions,
    seed: int,
    cache: TTSCache,
    engine_id: str,
    sample_rate: int,
    providers: str,
    make_engine: Callable[[], TTSEngine],
    progress: Optional[Callable[[int, int, TextSegment], None]] = None,
) -> List[SegmentAudio]:
    """对全部文本段合成，产物落缓存。

    命中缓存直接复用（不创建引擎）；未命中才调用 make_engine()。
    """
    results: List[SegmentAudio] = []
    pending: List[tuple] = []

    for order, seg in enumerate(segments):
        key = make_key(seg.text, seg.preset, seed, options, engine_id, sample_rate, providers)
        cached = cache.get(key)
        if cached is not None:
            wav, meta = cached
            results.append(SegmentAudio(
                key=key, order=order, segment=seg, wav_path=wav,
                duration=float(meta["duration_seconds"]),
                sample_rate=int(meta["sample_rate"]),
                from_cache=True, metadata=meta,
            ))
        else:
            pending.append((order, seg, key))

    if pending:
        engine = make_engine()
        for done, (order, seg, key) in enumerate(pending, 1):
            if progress is not None:
                progress(done, len(pending), seg)
            result = engine.synthesize(
                text=seg.text, language=seg.language, preset=seg.preset,
                seed=seed, **options.engine_kwargs(),
            )
            metadata = dict(result.metadata)
            metadata["duration_seconds"] = result.duration_seconds
            metadata["sample_rate"] = result.sample_rate
            metadata["text"] = seg.text
            metadata["preset"] = seg.preset
            metadata["language"] = seg.language
            dest = cache.store(key, result.path, metadata)
            results.append(SegmentAudio(
                key=key, order=order, segment=seg, wav_path=dest,
                duration=float(result.duration_seconds), sample_rate=result.sample_rate,
                from_cache=False, metadata=metadata,
            ))

    results.sort(key=lambda r: r.order)
    return results

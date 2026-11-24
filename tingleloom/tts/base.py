# -*- coding: utf-8 -*-
"""TTS 引擎抽象：结果结构、合成参数、引擎协议。

不同引擎（本地 ONNX 部署包、未来的其他）都实现 TTSEngine 协议，
批处理层只依赖这个协议，方便替换。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol

from .. import config


@dataclass
class TTSResult:
    path: Path
    sample_rate: int
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthOptions:
    """一次合成的全部采样 / 切分参数。

    这些项**都会进入缓存键**——少一项就会出现「参数变了键没变」的脏命中。
    """

    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    fragment_interval: float = config.DEFAULT_FRAGMENT_INTERVAL
    fragment_strategy: str = "sentence"
    max_fragment_characters: int = config.DEFAULT_MAX_FRAGMENT_CHARACTERS
    rng_strategy: str = "per_fragment"
    whisper_sampling: str = "default"

    def engine_kwargs(self) -> Dict[str, Any]:
        """传给引擎 synthesize 的关键字参数。"""
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "noise_scale": self.noise_scale,
            "fragment_interval": self.fragment_interval,
            "fragment_strategy": self.fragment_strategy,
            "max_fragment_characters": self.max_fragment_characters,
            "rng_strategy": self.rng_strategy,
            "whisper_sampling": self.whisper_sampling,
        }


class TTSEngine(Protocol):
    engine_id: str
    sample_rate: int

    def synthesize(self, text: str, language: str, preset: str,
                   seed: int = 1234, **options) -> TTSResult:
        ...

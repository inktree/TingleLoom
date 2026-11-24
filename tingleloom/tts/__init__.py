# -*- coding: utf-8 -*-
"""TTS 子包：引擎抽象、本地 ONNX 部署包适配、内容寻址缓存、逐段批处理。"""
from .base import SynthOptions, TTSEngine, TTSResult
from .batch import SegmentAudio, synthesize_all
from .cache import TTSCache, make_key
from .onnx_deploy import OnnxDeployEngine, provider_signature, tts_fingerprint

__all__ = [
    "TTSResult", "SynthOptions", "TTSEngine",
    "TTSCache", "make_key",
    "SegmentAudio", "synthesize_all",
    "OnnxDeployEngine", "tts_fingerprint", "provider_signature",
]

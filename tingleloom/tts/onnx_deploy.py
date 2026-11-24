# -*- coding: utf-8 -*-
"""本地 ONNX TTS 部署包适配。

部署包目录通过配置或参数指定，只在需要合成且缓存未命中时实例化引擎。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .. import config
from .base import TTSResult


def deploy_dir(path: Optional[Path] = None) -> Path:
    return Path(path).expanduser().resolve() if path else config.tts_dir()


def tts_fingerprint(path: Optional[Path] = None) -> Dict[str, Any]:
    """只读 manifest，拿到 engine_id（含 manifest 哈希）与采样率，不加载模型。

    用于在「是否要加载引擎」之前就算出缓存键。
    """
    directory = deploy_dir(path)
    manifest_path = directory / "artifacts" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("找不到 TTS manifest：%s" % manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    manifest = json.loads(manifest_bytes)
    package_id = manifest.get("package") or config.TTS_ENGINE_ID
    return {
        "engine_id": "%s:%s" % (package_id, digest),
        "sample_rate": int(manifest["sampling_rate"]),
        "deploy_dir": str(directory),
    }


def provider_signature() -> str:
    """把 providers 组合成缓存键里用的字符串。"""
    providers, decoder_providers = config.tts_providers()
    return "|".join(providers) + "/" + "|".join(decoder_providers)


class OnnxDeployEngine:
    engine_id = config.TTS_ENGINE_ID
    sample_rate = 32000

    def __init__(
        self,
        deploy: Optional[Path] = None,
        providers: Optional[Sequence[str]] = None,
        decoder_providers: Optional[Sequence[str]] = None,
        output_dir: Optional[Path] = None,
    ):
        directory = deploy_dir(deploy)
        if not directory.is_dir():
            raise FileNotFoundError("找不到 TTS 部署包：%s" % directory)

        if providers is None:
            providers, decoder_providers = config.tts_providers()
        elif decoder_providers is None:
            decoder_providers = providers
        provider_list = tuple(providers)
        decoder_list = tuple(decoder_providers)

        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        from runtime import OnnxSynthesisEngine  # 延迟导入：避免未用到时加载 ORT

        self._engine = OnnxSynthesisEngine(
            package_dir=directory,
            providers=provider_list,
            decoder_providers=decoder_list,
            output_dir=output_dir,
        )
        fingerprint = tts_fingerprint(directory)
        self.engine_id = fingerprint["engine_id"]
        self.sample_rate = fingerprint["sample_rate"]

    def synthesize(self, text: str, language: str, preset: str,
                   seed: int = 1234, **options) -> TTSResult:
        result = self._engine.synthesize(
            text=text, language=language, preset=preset, seed=seed, **options
        )
        return TTSResult(
            path=result.path,
            sample_rate=int(result.sample_rate),
            duration_seconds=float(result.duration_seconds),
            metadata=dict(result.metadata),
        )

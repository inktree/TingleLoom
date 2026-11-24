# -*- coding: utf-8 -*-
"""内容寻址缓存。

缓存键由「所有影响输出字节的输入」决定，与它在台本里的位置无关。
同样的输入 → 同样的键 → 直接复用；任何一项变了 → 新键 → 重新合成。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .. import config
from .base import SynthOptions


def make_key(text: str, preset: str, seed: int, options: SynthOptions,
             engine_id: str, sample_rate: int, providers: str) -> str:
    """由内容生成缓存键（sha1 前 20 位）。"""
    fields = {
        "text": text,
        "preset": preset,
        "seed": int(seed),
        "engine_id": engine_id,
        "sample_rate": int(sample_rate),
        "providers": providers,
        **options.engine_kwargs(),
    }
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


class TTSCache:
    """cache/tts/<key>.wav + <key>.json。"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.dir = Path(cache_dir) if cache_dir else config.TTS_CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def wav_path(self, key: str) -> Path:
        return self.dir / ("%s.wav" % key)

    def meta_path(self, key: str) -> Path:
        return self.dir / ("%s.json" % key)

    def get(self, key: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
        wav = self.wav_path(key)
        meta = self.meta_path(key)
        if not (wav.is_file() and wav.stat().st_size > 0 and meta.is_file()):
            return None
        return wav, json.loads(meta.read_text(encoding="utf-8"))

    def store(self, key: str, source_wav, metadata: Dict[str, Any]) -> Path:
        """先写元数据，再原子发布 WAV，避免命中缺少元数据的音频。"""
        dest = self.wav_path(key)
        tmp = self.dir / ("%s.tmp%d" % (key, os.getpid()))
        shutil.copyfile(str(source_wav), str(tmp))
        self.meta_path(key).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(str(tmp), str(dest))
        return dest

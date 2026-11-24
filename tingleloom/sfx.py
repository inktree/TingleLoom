# -*- coding: utf-8 -*-
"""触发音匹配与预处理。

台本里只写「要什么声音」（翻书 / page_turn），这里去素材清单里找，
找到就绑定，找不到就记进缺口清单。匹配到之后按指令准备音频：
限时长 / 循环适配 / 档位增益。声像由混音器在时间轴放置时统一应用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import audio, config, materials as materials_mod
from .materials import Material
from .script import Pan, SfxDirective, SfxEvent, TextSegment


@dataclass
class SfxOccurrence:
    directive: SfxDirective
    line_no: int
    origin: str                       # "standalone" | "inline"
    text_index: Optional[int] = None  # inline 时：所属文本段序号


@dataclass
class ResolvedSfx:
    occurrence: SfxOccurrence
    material: Optional[Material]

    @property
    def directive(self) -> SfxDirective:
        return self.occurrence.directive

    @property
    def resolved(self) -> bool:
        return self.material is not None


def collect_occurrences(script) -> List[SfxOccurrence]:
    """按台本顺序收集所有触发音（standalone + 文本行内联）。"""
    occurrences: List[SfxOccurrence] = []
    text_index = -1
    for item in script.items:
        if isinstance(item, TextSegment):
            text_index += 1
            for directive in item.sfx:
                occurrences.append(SfxOccurrence(directive, item.line_no, "inline", text_index))
        elif isinstance(item, SfxEvent):
            occurrences.append(SfxOccurrence(item.directive, item.line_no, "standalone", None))
    return occurrences


def resolve_sfx(script, materials: List[Material]):
    """把台本里的触发音绑到素材。返回 (resolved 列表, 缺口描述列表)。"""
    resolved: List[ResolvedSfx] = []
    gaps: List[str] = []
    seen: set = set()
    for occ in collect_occurrences(script):
        material = materials_mod.find_best(materials, occ.directive.description)
        resolved.append(ResolvedSfx(occ, material))
        if material is None and occ.directive.description not in seen:
            seen.add(occ.directive.description)
            gaps.append(occ.directive.description)
    return resolved, gaps


# ---------------------------------------------------------------- 增益 / 声像
def material_gain_db(material: Material, level: str = "中",
                     reference: float = config.SFX_REFERENCE_LUFS) -> float:
    """把素材基准响度拉到 reference，再叠加台本档位。"""
    level_db = config.GAIN_LEVELS_DB[level]
    base = material.base_lufs if material.base_lufs is not None else reference
    return (reference - base) + level_db


def _collapse_to_mono(data: np.ndarray) -> np.ndarray:
    """按声道 RMS 加权汇聚，保持同相信号及单侧音源的原始振幅。"""
    left = data[:, 0]
    right = data[:, 1]
    rms_l = float(np.sqrt(np.mean(left ** 2)))
    rms_r = float(np.sqrt(np.mean(right ** 2)))
    total = rms_l + rms_r
    if total == 0.0:
        return left.copy()
    return (rms_l / total) * left + (rms_r / total) * right


def apply_pan(data: np.ndarray, pan: Pan, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """保留原生声场，或将汇聚后的音源固定定向、段内平滑左右轮换。"""
    if len(data) == 0 or (pan.swap_interval is None and pan.value == 0.0):
        return data.astype(np.float32, copy=True)

    if pan.swap_interval is not None:
        n = len(data)
        step = max(1, int(round(pan.swap_interval * sr)))
        mono = np.empty(n, dtype=np.float32)
        alpha = np.empty(n, dtype=np.float32)
        for k, start in enumerate(range(0, n, step)):
            end = min(n, start + step)
            mono[start:end] = _collapse_to_mono(data[start:end])
            alpha[start:end] = k % 2
        half_xf = min(int(round(config.PAN_SWAP_XFADE * sr / 2.0)), step // 2)
        if half_xf:
            for k, cut in enumerate(range(step, n, step), 1):
                start = cut - half_xf
                end = min(n, cut + half_xf)
                # 末尾可能截在过渡中间，保留完整淡化曲线的对应前缀。
                alpha[start:end] = np.linspace((k - 1) % 2, k % 2, 2 * half_xf)[:end - start]
        amount = 0.75
        left = mono * (1.0 - amount * alpha)
        right = mono * ((1.0 - amount) + amount * alpha)
    else:
        mono = _collapse_to_mono(data)
        p = pan.value
        if p < 0:
            left = mono
            right = mono * (1.0 + p)
        else:
            left = mono * (1.0 - p)
            right = mono
    return np.stack([left, right], axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------- 预处理
def prepare_sfx(occurrence: SfxOccurrence, material: Material,
                sr: int = config.SAMPLE_RATE, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """把素材处理成可以直接混入的事件音频。"""
    data, _ = audio.read_audio(material.path, target_sr=sr)
    directive = occurrence.directive

    if directive.duration:
        # 持续触发音：适配成指定时长（不够长循环 + 接缝交叉淡化）
        seg = audio.fit_length(data, sr, directive.duration, xfade=2.0)
        seg = audio.fade_edges(seg, sr, config.BED_FADE, config.BED_FADE)
    else:
        # 单次触发音：太长则取一段（随机起点，同一素材重复用时不重样）
        max_len = int(config.SINGLE_SFX_MAX * sr)
        if len(data) > max_len:
            if rng is not None:
                start = int(rng.integers(0, len(data) - max_len))
            else:
                start = 0
            seg = data[start:start + max_len].copy()
        else:
            seg = data.copy()
        seg = audio.fade_edges(seg, sr, config.SFX_CUT_FADE, config.SFX_CUT_FADE)

    return audio.apply_gain_db(seg, material_gain_db(material, directive.level))

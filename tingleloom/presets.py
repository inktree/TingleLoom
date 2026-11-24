# -*- coding: utf-8 -*-
"""环境音（床音）预设：presets/beds.yaml 的读取与渲染。

预设名 → 一组素材（各带音量、循环参数）。层素材缺失时跳过并告警。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from scipy import signal

from . import audio, config, materials as materials_mod
from .materials import Material


SYNTH_BED_SPECS = {
    "room_tone": ("林间轻柔微风", ("环境音", "微风", "树叶")),
    "rain_soft": ("树叶微雨（物理合成）", ("环境音", "雨滴", "细雨")),
    "rain_hard": ("屋檐夜雨（物理合成）", ("环境音", "雨滴", "夜雨")),
}


def _synth_bed_wave(bed_id: str, duration_sec: float, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(1, int(round(duration_sec * sr)))
    xf = min(int(round(2.0 * sr)), n // 4) if n >= 4 else 0
    total = n + xf
    if bed_id == "room_tone":
        # 低频阵风只调制响度，不引入可听的纯音；双耳树叶纹理独立呼吸。
        common = rng.standard_normal((total, 1))
        indep = rng.standard_normal((total, 2))
        leaves = 0.25 * common + np.sqrt(1.0 - 0.25 ** 2) * indep
        leaves = signal.sosfilt(signal.butter(1, 380.0, fs=sr, output="sos"), leaves, axis=0)
        leaves = signal.sosfilt(signal.butter(2, [140.0, 2400.0], btype="bandpass", fs=sr, output="sos"), leaves, axis=0)
        t_norm = np.arange(total, dtype=np.float64) / n
        c1, c2, c3 = max(1, round(duration_sec / 19.0)), max(1, round(duration_sec / 11.0)), max(1, round(duration_sec / 7.0))
        gust_l = 0.35 + 0.30 * np.sin(2 * np.pi * c1 * t_norm) + 0.20 * np.sin(2 * np.pi * c2 * t_norm + 0.8) + 0.15 * np.cos(2 * np.pi * c3 * t_norm)
        gust_r = 0.35 + 0.30 * np.sin(2 * np.pi * c1 * t_norm + 0.6) + 0.20 * np.sin(2 * np.pi * c2 * t_norm + 2.1) + 0.15 * np.cos(2 * np.pi * c3 * t_norm + 1.2)
        # 阵风调幅限制在约 10.6 dB，避免波谷退成近乎静音。
        envelope = 0.55 + (0.30 / 0.65) * (np.stack([gust_l, gust_r], axis=1) - 0.35)
        data = leaves * envelope
        target_rms = 0.005
    else:
        # 固体表面微冲击：远景密集微雨珠 + 近景宽频轻拍，完全不含正弦音源。
        patter_rate, drop_rate, patter_weight, target_rms = (
            (220.0, 18.0, 0.28, 0.008) if bed_id == "rain_soft" else (480.0, 42.0, 0.42, 0.014)
        )
        p_len = max(8, int(round(0.004 * sr)))
        p_t = np.arange(p_len, dtype=np.float64) / sr
        p_env = (p_t / 0.0009) * np.exp(1.0 - p_t / 0.0009)
        p_env *= np.minimum(1.0, (p_len - 1 - np.arange(p_len)) / max(1, p_len // 4))
        p_count = rng.poisson(patter_rate * total / sr)
        p_imp = np.zeros((total, 2), dtype=np.float64)
        p_pos = rng.integers(0, total, p_count)
        p_amp = rng.lognormal(mean=-0.6, sigma=0.4, size=p_count)
        p_pan = rng.uniform(0.2, 0.8, size=p_count)
        np.add.at(p_imp[:, 0], p_pos, p_amp * p_pan)
        np.add.at(p_imp[:, 1], p_pos, p_amp * (1.0 - p_pan))
        p_noise = signal.sosfilt(signal.butter(1, 550.0, fs=sr, output="sos"), rng.standard_normal((total, 2)), axis=0)
        p_noise = signal.sosfilt(signal.butter(2, [220.0, 2600.0], btype="bandpass", fs=sr, output="sos"), p_noise, axis=0)
        patter = np.stack([
            signal.fftconvolve(p_imp[:, 0], p_env, mode="full")[:total],
            signal.fftconvolve(p_imp[:, 1], p_env, mode="full")[:total],
        ], axis=1) * p_noise
        patter /= max(float(np.sqrt(np.mean(patter ** 2))), 1e-12)

        templates = []
        for center_hz, length in zip(np.geomspace(480.0, 2100.0, 24), rng.uniform(0.006, 0.018, 24)):
            m_len = max(8, int(round(length * sr)))
            t = np.arange(m_len, dtype=np.float64) / sr
            tau = length / 7.2
            env = (t / tau) * np.exp(1.0 - t / tau)
            env *= np.minimum(1.0, (m_len - 1 - np.arange(m_len)) / max(1, int(0.002 * sr)))
            burst = signal.sosfilt(signal.butter(1, 500.0, fs=sr, output="sos"), rng.standard_normal(m_len))
            lo = max(140.0, center_hz * 0.45)
            hi = min(center_hz * 1.65, sr * 0.42)
            burst = signal.sosfilt(signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), burst)
            burst /= max(float(np.sqrt(np.mean(burst ** 2))), 1e-12)
            templates.append(env * burst)
        drops = np.zeros((total, 2), dtype=np.float64)
        count = rng.poisson(drop_rate * total / sr)
        positions = rng.integers(0, total, count)
        kinds = rng.integers(0, len(templates), count)
        amplitudes = rng.lognormal(mean=-0.35, sigma=0.45, size=count)
        pans = rng.uniform(-0.75, 0.75, count)
        for pos, kind, amp, pan in zip(positions, kinds, amplitudes, pans):
            kernel = templates[kind][:total - pos]
            weights = np.sqrt([(1.0 - pan) / 2.0, (1.0 + pan) / 2.0])
            drops[pos:pos + len(kernel)] += kernel[:, None] * (amp * weights)
        drops /= max(float(np.sqrt(np.mean(drops ** 2))), 1e-12)
        data = patter_weight * patter + (1.0 - patter_weight) * drops
    if xf > 0:
        ramp = np.linspace(0.0, 1.0, xf)[:, None]
        data[:xf] = data[n:n + xf] * (1.0 - ramp) + data[:xf] * ramp
    data = data[:n]
    data *= target_rms / np.sqrt(np.mean(data ** 2))
    return np.clip(data, -0.95, 0.95).astype(np.float32)


def synthesize_beds(
    bed_ids: Optional[List[str]] = None,
    duration_sec: float = 90.0,
    sr: int = config.SAMPLE_RATE,
    seed: int = 20261006,
    out_dir: Path = config.BED_DIR,
    index_path: Path = config.MATERIALS_INDEX,
    overwrite: bool = False,
) -> List[Material]:
    """合成可循环的基础环境床音，并通过统一素材入口测量、注册。"""
    if duration_sec <= 0:
        raise ValueError("床音时长必须为正数")
    targets = list(SYNTH_BED_SPECS) if bed_ids is None else list(dict.fromkeys(bed_ids))
    for bed_id in targets:
        if bed_id not in SYNTH_BED_SPECS:
            raise ValueError(f"未知合成床音类型：{bed_id}")
    items = materials_mod.load_index(index_path)
    out_dir = Path(out_dir)
    created = []
    for idx, bed_id in enumerate(targets):
        if not overwrite:
            existing = materials_mod.find_index(items, bed_id)
            if existing is not None and existing.path.is_file():
                continue
        wave = _synth_bed_wave(bed_id, duration_sec, sr, seed + idx * 101)
        dest = out_dir / f"{bed_id}.wav"
        audio.write_audio(dest, wave, sr)
        desc, tags = SYNTH_BED_SPECS[bed_id]
        created.append(materials_mod.register(
            dest, items, mat_id=bed_id, desc=desc, tags=list(tags),
            source="procedural_dsp", license="CC0", commercial_ok=True,
        ))
    if created:
        materials_mod.save_index(items, index_path)
    return created


@dataclass
class BedLayer:
    material: str
    gain_db: float = 0.0


@dataclass
class BedPreset:
    name: str
    xfade: float = 2.0
    layers: List[BedLayer] = field(default_factory=list)
    offset_db: Optional[float] = None   # 放到「节目 − offset_db」之下；None 用 config 默认


def load_beds(path: Optional[Path] = None) -> Dict[str, BedPreset]:
    path = Path(path) if path else config.BEDS_FILE
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("beds.yaml 顶层必须是映射")
    presets: Dict[str, BedPreset] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"预设 {name} 必须是映射：{spec!r}")
        layers = []
        for layer in spec.get("layers", []):
            if not isinstance(layer, dict) or "material" not in layer:
                raise ValueError(f"预设 {name} 的 layer 缺少 material：{layer!r}")
            layers.append(BedLayer(str(layer["material"]), float(layer.get("gain_db", 0.0))))
        offset = spec.get("offset_db")
        presets[str(name)] = BedPreset(
            str(name), float(spec.get("xfade", 2.0)), layers,
            float(offset) if offset is not None else None,
        )
    return presets


def render_bed(preset: Optional[BedPreset], duration: float, materials: List[Material],
               sr: int = config.SAMPLE_RATE, warnings: Optional[List[str]] = None,
               apply_fades: bool = True) -> np.ndarray:
    """把床音预设渲染成长度 duration 的立体声浮点。

    混音器自己做段落交叉淡化时传 apply_fades=False。
    """
    n = int(round(duration * sr))
    canvas = np.zeros((max(n, 0), 2), dtype=np.float32)
    if preset is None or n <= 0:
        return canvas

    for layer in preset.layers:
        material = materials_mod.find_index(materials, layer.material)
        if material is None or not material.path.is_file():
            if warnings is not None:
                warnings.append("床音层素材缺失：%s（预设 %s）" % (layer.material, preset.name))
            continue
        data, _ = audio.read_audio(material.path, target_sr=sr)
        seg = audio.fit_length(data, sr, duration, xfade=preset.xfade)
        base = material.base_lufs if material.base_lufs is not None else config.BED_LAYER_REFERENCE_LUFS
        gain = (config.BED_LAYER_REFERENCE_LUFS - base) + layer.gain_db
        canvas += audio.apply_gain_db(seg, gain)

    # 低切去隆隆声；绝对音量由混音器统一放到「节目 − BED_OFFSET_DB」。
    canvas = audio.highpass(canvas, sr, config.BED_HIGHPASS_HZ)

    if apply_fades:
        canvas = audio.fade_edges(canvas, sr, config.BED_FADE, config.BED_FADE)
    return canvas

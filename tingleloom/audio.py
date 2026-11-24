# -*- coding: utf-8 -*-
"""音频基础工具：读写、重采样、淡入淡出、循环适配、响度、软限幅。

混音器、素材预处理、质检共用这一层。剪切逻辑从原 scripts/trim_audio.py
整理成可复用函数，行为保持一致：
  - 任何剪切都带淡入淡出（硬切有咔哒声）
  - 床音不够长时循环拼接，接缝交叉淡化
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import av
import numpy as np
import soundfile as sf
from scipy import signal

from . import config


# ---------------------------------------------------------------- 读取 / 写入
def to_stereo(data: np.ndarray) -> np.ndarray:
    """把 (n, ch) 规整成 (n, 2)。单声道复制，多声道取前两路。"""
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    if channels == 1:
        return np.repeat(data, 2, axis=1)
    if channels >= 2:
        return data[:, :2]
    raise ValueError("音频没有声道")


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """多相重采样。"""
    if sr_in == sr_out:
        return data
    ratio = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    out = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=0)
    return out.astype(np.float32, copy=False)


def read_audio(path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """读取音频 → (float32 (n,2), sr)。给了 target_sr 且不同则重采样。"""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    data = to_stereo(data)
    if target_sr is not None and sr != target_sr:
        data = resample(data, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), sr


def write_audio(path, data: np.ndarray, sr: int, subtype: str = config.OUTPUT_SUBTYPE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def write_mp3(path, data: np.ndarray, sr: int, bitrate: str = config.OUTPUT_MP3_BITRATE) -> Path:
    """将 float32 音频直接编码为 320kbps 双耳独立立体声（True Stereo）MP3。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.ascontiguousarray(to_stereo(data).T, dtype=np.float32)
    with av.open(str(path), mode="w", format="mp3") as container:
        stream = container.add_stream(
            "libmp3lame",
            rate=sr,
            options={"b": bitrate, "compression_level": "0", "joint_stereo": "0"},
        )
        stream.layout = "stereo"
        if stereo.shape[1] > 0:
            frame = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def probe(path):
    """只读元信息，不加载音频。"""
    return sf.info(str(path))


# ---------------------------------------------------------------- 剪辑
def fade_edges(seg: np.ndarray, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    """给分段加淡入淡出。seg: (n, ch)。原地修改并返回。"""
    n = len(seg)
    fin = min(int(fade_in * sr), n // 2)
    fout = min(int(fade_out * sr), n // 2)
    if fin > 0:
        ramp = np.linspace(0.0, 1.0, fin, dtype=np.float32)[:, None]
        seg[:fin] *= ramp
    if fout > 0:
        ramp = np.linspace(1.0, 0.0, fout, dtype=np.float32)[:, None]
        seg[-fout:] *= ramp
    return seg


def fit_length(data: np.ndarray, sr: int, need_seconds: float, xfade: float = 2.0) -> np.ndarray:
    """把素材适配成指定时长。

    比目标长 → 直接截取；比目标短 → 循环拼接，接缝交叉淡化（避免周期咔哒）。
    """
    need = int(round(need_seconds * sr))
    if need <= 0:
        return data[:0].copy()
    if len(data) >= need:
        return data[:need].copy()

    reps = need // len(data) + 1
    tiled = np.tile(data, (reps, 1))[:need].copy()

    xf = int(min(xfade * sr, len(data) // 4))
    if xf > 0:
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        for cut in range(len(data), need - xf, len(data)):
            a = tiled[cut - xf:cut].copy()
            b = tiled[cut:cut + xf].copy()
            tiled[cut - xf:cut] = a * (1.0 - ramp) + b * ramp
    return tiled


# ---------------------------------------------------------------- 响度
def peak(data: np.ndarray) -> float:
    return float(np.max(np.abs(data))) if np.size(data) else 0.0


def _k_weighting(sr: int):
    """K 加权两段双二阶系数（近似 ITU-R BS.1770）。返回 (b_hp, a_hp, b_sh, a_sh)。"""
    # 高频搁架 pre-filter：G=+4dB, Q≈0.707, fc≈1682Hz
    g_db, q_shelf, fc_shelf = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = np.tan(np.pi * fc_shelf / sr)
    vh = 10.0 ** (g_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q_shelf + k * k
    b_sh = np.array([vh + vb * k / q_shelf + k * k, 2.0 * (k * k - vh),
                     vh - vb * k / q_shelf + k * k]) / a0
    a_sh = np.array([1.0, 2.0 * (k * k - 1.0) / a0,
                     (1.0 - k / q_shelf + k * k) / a0])
    # 高通 RLB：Q≈0.5, fc≈38Hz
    q_hp, fc_hp = 0.5003270373238773, 38.13547087602444
    k = np.tan(np.pi * fc_hp / sr)
    a0 = 1.0 + k / q_hp + k * k
    b_hp = np.array([1.0, -2.0, 1.0]) / a0
    a_hp = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q_hp + k * k) / a0])
    return b_hp, a_hp, b_sh, a_sh


def loudness_lufs(data: np.ndarray, sr: int) -> float:
    """近似整合响度（LUFS）。

    简化实现：K 加权 + 0.4s 块 / 75% 重叠 + 绝对门限 -70 + 相对门限 -10。
    L/R 权重 1.0（按 BS.1770 对声道**求和**）。够做基准响度归一，不是认证级测量。
    """
    if data.ndim == 1:
        data = data[:, None]
    b_hp, a_hp, b_sh, a_sh = _k_weighting(sr)
    x = signal.lfilter(b_sh, a_sh, data.astype(np.float64), axis=0)
    x = signal.lfilter(b_hp, a_hp, x, axis=0)

    block = int(round(0.4 * sr))
    hop = max(1, int(round(0.1 * sr)))
    if len(x) < block:
        x = np.pad(x, ((0, block - len(x)), (0, 0)))

    n_blocks = 1 + (len(x) - block) // hop
    z = np.empty(n_blocks)
    for i in range(n_blocks):
        seg = x[i * hop:i * hop + block]
        z[i] = float(np.sum(np.mean(seg ** 2, axis=0)))

    loud = -0.691 + 10.0 * np.log10(z + 1e-12)
    mask = loud > -70.0
    if not np.any(mask):
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(np.mean(z[mask]) + 1e-12) - 10.0
    mask &= loud > relative
    if not np.any(mask):
        return float(loud[int(np.argmax(loud))])
    return float(-0.691 + 10.0 * np.log10(np.mean(z[mask]) + 1e-12))


def apply_gain_db(data: np.ndarray, gain_db: float) -> np.ndarray:
    return (data * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32, copy=False)


# ---------------------------------------------------------------- 削波保护
def soft_limit(data: np.ndarray, threshold: float = config.LIMITER_THRESHOLD,
               ceiling: float = config.LIMITER_CEILING) -> np.ndarray:
    """软限幅：|x| ≤ threshold 线性通过；超出部分 tanh 渐近到 ceiling。

    C1 连续（拐点斜率=1），不产生硬裁剪那种刺耳高频谐波。
    """
    ax = np.abs(data)
    span = max(ceiling - threshold, 1e-6)
    limited = threshold + span * np.tanh((ax - threshold) / span)
    out = np.where(ax <= threshold, data, np.sign(data) * limited)
    return out.astype(np.float32, copy=False)


def highpass(data: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    """高通滤波（默认二阶巴特沃斯），去掉低频隆隆声 / 直流。"""
    sos = signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    out = signal.sosfilt(sos, data, axis=0)
    return out.astype(np.float32, copy=False)

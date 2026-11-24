# -*- coding: utf-8 -*-
# 项目：Tingleloom
"""分析音频文件的真实编码质量。

用途：判断一个音频是不是被压缩过（比如 Pixabay 下载的 mp3 到底是多少 kbps）。

判断依据：有损编码会在高频留下“频谱截断”。比如 128kbps mp3
通常在 16kHz 附近就把高频砍掉了；320kbps mp3 能到 20kHz。

用法：
    python scripts/analyze_audio.py <file> [file2 ...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path):
    """返回 (data[frames, ch], samplerate, format, subtype) 或抛异常。"""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    info = sf.info(str(path))
    return data, sr, info.format, info.subtype


def spectrum_cutoff(data, sr, thresh_db=-80.0):
    """估算频谱截断频率：从高频往低扫，找到能量超过阈值的最高频率。"""
    mono = data.mean(axis=1)
    # 取最长可接受长度，太长则分段取平均
    n = len(mono)
    win = min(n, sr * 4)
    if n > win:
        # 多段平均，更稳定
        segs = [mono[i:i + win] for i in range(0, n - win, win)] or [mono[:win]]
        acc = np.zeros(win // 2 + 1)
        for s in segs:
            w = s * np.hanning(len(s))
            acc += np.abs(np.fft.rfft(w)) ** 2
        spec = acc / len(segs)
    else:
        w = mono * np.hanning(len(mono))
        spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
    spec_db = 10 * np.log10(spec + 1e-20)
    peak = spec_db.max()
    # 从高频往低找第一个超过阈值的位置
    thresh = peak + thresh_db
    idx = np.where(spec_db > thresh)[0]
    return float(freqs[idx[-1]]) if len(idx) else 0.0, freqs, spec_db


def analyze(path):
    p = Path(path)
    print("=" * 62)
    print("文件：%s" % p.name)
    if not p.exists():
        print("  （不存在）")
        return
    print("  大小：%.1f KB" % (p.stat().st_size / 1024))
    data, sr, fmt, subtype = load_audio(p)
    dur = len(data) / sr
    cutoff, freqs, spec_db = spectrum_cutoff(data, sr)
    peak = data.max()
    rms = float(np.sqrt((data ** 2).mean()))
    db = 20 * np.log10(rms + 1e-12)
    print("  格式：%s / %s" % (fmt, subtype))
    print("  采样率：%d Hz  |  声道：%d  |  时长：%.2f s" % (sr, data.shape[1], dur))
    print("  峰值：%.3f  |  RMS：%.5f  (%.1f dBFS)" % (peak, rms, db))
    print("  频谱截断（-80dB）：%.0f Hz" % cutoff)
    nyq = sr / 2
    if cutoff < nyq * 0.6:
        est = "疑似 128kbps 以下" if cutoff < 17000 else "疑似 192kbps 左右"
        print("  判断：高频被切（截断 %.0f / 奈奎斯特 %.0f Hz）→ %s" % (cutoff, nyq, est))
    else:
        print("  判断：高频完整（接近奈奎斯特）→ 疑似无损或高码率")
    # 打印几个关键频点的能量，供人工看
    for f in (8000, 12000, 16000, 18000, 20000, 22000):
        i = int(f / (sr / 2) * (len(spec_db) - 1))
        if i < len(spec_db):
            rel = spec_db[i] - spec_db.max()
            bar = "#" * max(0, int((rel + 80) / 4))
            print("    %5d Hz  %6.1f dBc  %s" % (f, rel, bar))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法：python analyze_audio.py <file> [file2 ...]")
    for a in sys.argv[1:]:
        analyze(a)

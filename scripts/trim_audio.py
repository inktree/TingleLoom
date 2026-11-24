# -*- coding: utf-8 -*-
# 项目：Tingleloom
"""音频剪切工具：从素材里取一段，带淡入淡出。

为什么需要：素材时长参差不齐（几秒到十几分钟），而台本里一个触发音
通常只要 1~5 秒。直接整段贴进去不行，得截。

两个硬性要求：
  1. 必须带淡入淡出——硬切会在切口处产生咔哒声（click），
     在安静的 ASMR 里格外明显。
  2. 同一素材要能用不同区间——否则重复用同一段会听出来，
     抽查不同 offset 能让同一素材听起来不重样。

用法：
    # 取 20.0s 起、3.0s 长的一段，带 50ms 淡入淡出
    python scripts/trim_audio.py in.mp3 -o out.wav --start 20.0 --dur 3.0 --fade 0.05

    # 床音：把素材适配成正好 10 分钟（不够长就循环拼接，接缝交叉淡化）
    python scripts/trim_audio.py in.mp3 -o bed.wav --fit 600 --xfade 2.0

    # 随机起点（同一素材重复用时，避免每次都取到同一段）
    python scripts/trim_audio.py in.mp3 -o out.wav --dur 5 --random

    # 自动选点：滑动窗找能量最高的 5 秒（点状素材用，保证截到有声音的地方）
    python scripts/trim_audio.py in.mp3 -o out.wav --dur 5 --auto

    # 不看文件，只查时长和格式
    python scripts/trim_audio.py in.mp3 --probe
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tingleloom import audio


def main():
    ap = argparse.ArgumentParser(description="音频剪切")
    ap.add_argument("src", help="源文件")
    ap.add_argument("-o", "--out", help="输出文件")
    ap.add_argument("--start", type=float, default=0.0, help="起点（秒）")
    ap.add_argument("--random", action="store_true",
                    help="随机选起点（同一素材重复使用时避免听出重复）")
    ap.add_argument("--auto", action="store_true",
                    help="自动选起点：滑动窗找能量最高的位置（适合点状素材）")
    ap.add_argument("--dur", type=float, default=None, help="时长（秒），不填则到结尾")
    ap.add_argument("--fit", type=float, default=None,
                    help="适配成正好这么多秒：长了截断，短了循环拼接（床音用）")
    ap.add_argument("--xfade", type=float, default=2.0,
                    help="--fit 循环时的接缝交叉淡化秒数（默认 2.0）")
    ap.add_argument("--fade", type=float, default=0.05, help="淡入淡出秒数（默认 0.05）")
    ap.add_argument("--fade-in", type=float, default=None, help="单独指定淡入")
    ap.add_argument("--fade-out", type=float, default=None, help="单独指定淡出")
    ap.add_argument("--probe", action="store_true", help="只报告时长/格式，不切")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit("找不到：%s" % src)

    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    total = len(data) / sr
    print("源文件：%s" % src.name)
    print("  采样率 %d Hz  声道 %d  总时长 %.3f s (%d 帧)" % (sr, data.shape[1], total, len(data)))
    if args.probe:
        return

    if not args.out:
        raise SystemExit("需要 -o 指定输出")

    fin = args.fade if args.fade_in is None else args.fade_in
    fout = args.fade if args.fade_out is None else args.fade_out

    if args.fit is not None:
        # 床音模式：适配成指定时长
        if args.start:
            a = int(round(args.start * sr))
            data = data[a:]
            total -= args.start
        seg = audio.fit_length(data, sr, args.fit, args.xfade)
        print("适配：素材 %.3f s → 目标 %.3f s" % (total, args.fit))
        if total < args.fit:
            print("  素材不够长，已循环拼接（接缝交叉淡化 %.0f ms）" % (args.xfade * 1000))
        else:
            print("  素材够长，直接截取前 %.3f s" % args.fit)
        # 床音首尾也要淡入淡出，避免段落切换时咔哒
        seg = audio.fade_edges(seg, sr, fin, fout)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), seg, sr, subtype="PCM_16")
        print("  输出：%s  (%.1f MB, %.3f s)"
              % (out, out.stat().st_size / 1048576, len(seg) / sr))
        print("  峰值 %.3f  RMS %.5f" % (np.abs(seg).max(), float(np.sqrt((seg ** 2).mean()))))
        return

    if args.auto and args.dur and args.dur < total:
        # 滑动窗找能量最高的一段，保证截到「有声音」的地方
        win = int(round(args.dur * sr))
        mono = data.mean(axis=1).astype(np.float64)
        c = np.concatenate(([0.0], np.cumsum(mono * mono)))
        hop = max(1, int(0.01 * sr))
        idx = np.arange(0, len(mono) - win + 1, hop)
        sums = c[idx + win] - c[idx]
        start = float(idx[int(np.argmax(sums))]) / sr
        print("自动选点（最高能量窗口）：%.3f s" % start)
    elif args.random and args.dur and args.dur < total:
        start = float(np.random.uniform(0.0, total - args.dur))
        print("随机起点：%.3f s" % start)
    else:
        start = max(0.0, args.start)
    if start >= total:
        raise SystemExit("起点 %.3f 超过总时长 %.3f" % (start, total))
    end = total if args.dur is None else min(total, start + args.dur)

    a = int(round(start * sr))
    b = int(round(end * sr))
    seg = data[a:b].copy()
    seg = audio.fade_edges(seg, sr, fin, fout)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), seg, sr, subtype="PCM_16")

    print("剪切：%.3f → %.3f s  (%.3f s / %d 帧)" % (start, end, len(seg) / sr, len(seg)))
    print("  淡入 %.0f ms / 淡出 %.0f ms" % (fin * 1000, fout * 1000))
    print("  输出：%s  (%.1f KB)" % (out, out.stat().st_size / 1024))
    # 简单质量检查
    print("  峰值 %.3f  RMS %.5f" % (np.abs(seg).max(), float(np.sqrt((seg ** 2).mean()))))


if __name__ == "__main__":
    main()

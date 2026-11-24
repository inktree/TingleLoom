# -*- coding: utf-8 -*-
"""全局配置与路径。

集中放「东西放在哪」「存成什么规格」两类常量，其他模块只引用，不再各写一份。

可用环境变量覆盖：
    TINGLELOOM_TTS_DIR         本地 ONNX TTS 部署包目录
    TINGLELOOM_TTS_PROVIDERS   cpu / cuda / hybrid
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PACKAGE_DIR / "assets"
SILERO_ENCODER_ONNX = ASSETS_DIR / "silero_encoder_v5.onnx"
SILERO_DECODER_ONNX = ASSETS_DIR / "silero_decoder_v5.onnx"
CLAP_AUDIO_ONNX = ASSETS_DIR / "clap_audio_quantized.onnx"
CLAP_PROMPTS_NPZ = ASSETS_DIR / "clap_asmr_prompts.npz"

SCRIPTS_DIR = ROOT / "scripts"
MATERIALS_DIR = ROOT / "materials"
CONTRIBUTIONS_DIR = ROOT / "contributions"
PROTOTYPES_NPZ = MATERIALS_DIR / "prototypes.npz"
SFX_DIR = MATERIALS_DIR / "sfx"
BED_DIR = MATERIALS_DIR / "bed"
RAW_DIR = MATERIALS_DIR / "raw"
PRESETS_DIR = ROOT / "presets"
BUILD_DIR = ROOT / "build"
CACHE_DIR = ROOT / "cache"
TTS_CACHE_DIR = CACHE_DIR / "tts"
OUTPUT_DIR = ROOT / "output"
SECRETS_DIR = ROOT / "secrets"
EXAMPLES_DIR = ROOT / "examples"

MATERIALS_INDEX = MATERIALS_DIR / "index.yaml"
BEDS_FILE = PRESETS_DIR / "beds.yaml"

# ---------------------------------------------------------------- 音频规格
SAMPLE_RATE = 44100              # 混音与成品采样率（依工具链文档）
OUTPUT_SUBTYPE = "PCM_16"        # 成品 16-bit PCM，兼容性最好
OUTPUT_MP3_BITRATE = "320k"      # 发布用 MP3 最高码率（320 kbps CBR）

# ---------------------------------------------------------------- 混音
TARGET_LUFS = -40.0              # 节目响度目标（按实测听感定的安静档）
LIMITER_THRESHOLD = 0.70         # 软限幅线性区上限
LIMITER_CEILING = 0.99           # 软限幅渐近上限
GAIN_LEVELS_DB = {"小": -6.0, "中": 0.0, "大": 4.0}   # 台本档位 → dB
SFX_REFERENCE_LUFS = -27.0       # 触发音统一到该基准（比人声低 3 dB），再叠档位
BED_LAYER_REFERENCE_LUFS = -35.0 # 床音各层之间先归一到该基准，仅用于配平
BED_OFFSET_DB = 22.0             # 床音整体比人声节目低这么多 dB（业界经验 20~25）
BED_HIGHPASS_HZ = 80.0           # 床音低切，去掉低频隆隆声
SFX_CUT_FADE = 0.05              # 单次触发音首尾淡入淡出（秒，防咔哒即可）
BED_FADE = 1.0                   # 床音段首尾淡入淡出（秒）
SINGLE_SFX_MAX = 6.0             # 单次触发音最长取用秒数，超出则取一段

# ---------------------------------------------------------------- 人声与成品
VOCAL_GAIN_DB = 0.0            # 人声额外增益（归一化之后再叠的微调）
VOCAL_REFERENCE_LUFS = -24.0   # 每段人声先归一化到该基准，保证人声/触发音配比稳定
AUTO_PAN_AMOUNT = 0.35   # @pan_swap 生效时，未显式指定声道的自动偏置量
PAN_SWAP_INTERVAL = 15.0  # LR 标记的默认段内左右轮换周期（秒）
PAN_SWAP_XFADE = 1.5      # LR 段内左右轮换的交叉淡化时长（秒）
TAIL_SECONDS = 1.0        # 末尾空白（秒）
OUTRO_FADE_SECONDS = 2.5  # 末尾淡出（秒）

# ---------------------------------------------------------------- TTS
TTS_ENGINE_ID = "onnx_tts"
SUPPORTED_PRESETS = ("normal_cn", "normal_ja", "whisper_cn", "whisper_ja")
PRESET_LANGUAGE = {
    "normal_cn": "zh",
    "whisper_cn": "zh",
    "normal_ja": "ja",
    "whisper_ja": "ja",
}
DEFAULT_PRESET = "whisper_cn"
DEFAULT_FRAGMENT_INTERVAL = 0.3
DEFAULT_MAX_FRAGMENT_CHARACTERS = 240

TTS_DIR_FILE = SECRETS_DIR / "tts_dir.txt"
DEFAULT_TTS_DIR = ROOT / "models" / "tts"

PIPELINE_VERSION = "0.1"


def tts_dir() -> Path:
    """依次从环境变量、本地配置文件、默认目录解析 ONNX TTS 部署包目录。"""
    value = os.environ.get("TINGLELOOM_TTS_DIR")
    if value and value.strip():
        return Path(value.strip()).expanduser().resolve()
    if TTS_DIR_FILE.is_file():
        for line in TTS_DIR_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return Path(line).expanduser().resolve()
    return DEFAULT_TTS_DIR.resolve()


def tts_providers():
    """返回 (providers, decoder_providers)。

    cpu    -> 全 CPU（可移植默认）
    cuda   -> 全 CUDA
    hybrid -> BERT/T2S 用 CUDA，SoVITS decoder 用 CPU（本机 RTX 推荐）
    """
    mode = (os.environ.get("TINGLELOOM_TTS_PROVIDERS") or "cpu").strip().lower()
    if mode == "cuda":
        return ("CUDAExecutionProvider",), ("CUDAExecutionProvider",)
    if mode == "hybrid":
        return ("CUDAExecutionProvider",), ("CPUExecutionProvider",)
    return ("CPUExecutionProvider",), ("CPUExecutionProvider",)


def ensure_dirs() -> None:
    for directory in (
        MATERIALS_DIR, SFX_DIR, BED_DIR, RAW_DIR, PRESETS_DIR,
        BUILD_DIR, CACHE_DIR, TTS_CACHE_DIR, OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

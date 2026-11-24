# -*- coding: utf-8 -*-
"""台本解析器：把 .asmr 文本解析成结构化事件。

格式：一行一句，用括号写指令。

    行类型
        # 标题                         节目标题
        @key value                     元信息（@preset/@seed/@bed/@title/@pan_swap）
        普通文本行                      一句台词，一行 = 一个 TTS 段
        (pause N)                      插入 N 秒静音
        (sfx 描述 ...)                  单次 / 持续触发音
        // 备注                        人类注释，忽略

    文本行内联触发音 / 声道：
        慢慢地，你会觉得身体越来越沉。(sfx page_turn @2.5)
        你好呀。(pan L70)

    (sfx 描述 [@秒] [时长] [档位] [声道])
        描述  自由词组（翻书 / page_turn）
        @秒   段内偏移，如 @2.5
        时长  90s / 3m / 1m30s（有则持续，无则单次）
        档位  小 / 中 / 大（默认 中）
        声道  C / L / R / L70 / R30 / LR / LR15（默认 auto，保留素材原声场）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from . import config

__all__ = [
    "ScriptError", "Pan", "AUTO_PAN", "SfxDirective", "TextSegment", "Pause", "SfxEvent",
    "BedChange", "Script", "parse_duration", "parse_pan", "parse_sfx",
    "parse_script", "load_script",
]


class ScriptError(Exception):
    """台本语法错误，带行号。"""

    def __init__(self, line_no: int, message: str):
        super().__init__("第 %d 行：%s" % (line_no, message))
        self.line_no = line_no
        self.message = message


# ---------------------------------------------------------------- 词法小工具
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_DURATION = re.compile(r"^(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$", re.IGNORECASE)
_AT = re.compile(r"^@(\d+(?:\.\d+)?)$")
_PAN = re.compile(r"^(?:C|LR(?:\d+(?:\.\d+)?)?|[LR]\d{0,3})$", re.IGNORECASE)
_LEVELS = ("小", "中", "大")

INLINE_SFX = re.compile(r"\(\s*sfx\s+([^)]*)\)")
INLINE_PAN = re.compile(r"\(\s*pan\s+([^)]*)\)")
_PAREN = re.compile(r"^\((.*)\)$")
_COMMAND = re.compile(r"^(\w+)\s*(.*)$")


def parse_duration(token: str) -> float:
    """`90` / `90s` / `3m` / `1m30s` → 秒。"""
    tok = token.strip()
    if _NUMBER.fullmatch(tok):
        return float(tok)
    m = _DURATION.fullmatch(tok)
    if not m or (m.group(1) is None and m.group(2) is None):
        raise ValueError("无法识别的时长：%r" % token)
    return float(m.group(1) or 0) * 60.0 + float(m.group(2) or 0)


@dataclass(frozen=True)
class Pan:
    raw: str
    value: float            # -1 全左 .. 0 居中 .. +1 全右
    explicit: bool = False  # 台本里是否显式写了声道标记
    swap_interval: Optional[float] = None  # LR 段内左右轮换周期（秒）

    def for_start(self, start: float, pan_swap: Optional[float]) -> "Pan":
        """未显式指定声道时，按 @pan_swap 的间隔左右交替（避免长时间单侧疲劳）。"""
        if self.explicit or not pan_swap:
            return self
        side = -1.0 if int(start // pan_swap) % 2 == 0 else 1.0
        return Pan("auto", side * config.AUTO_PAN_AMOUNT, False)


AUTO_PAN = Pan("auto", 0.0, False)


def parse_pan(token: str) -> Pan:
    tok = token.strip().upper()
    if tok == "C":
        return Pan("C", 0.0, True)
    m_lr = re.fullmatch(r"LR(\d+(?:\.\d+)?)?", tok)
    if m_lr:
        interval = float(m_lr.group(1)) if m_lr.group(1) else config.PAN_SWAP_INTERVAL
        if interval <= 0:
            raise ValueError("LR 轮换周期必须为正数：%r" % token)
        return Pan(tok, 0.0, True, swap_interval=interval)
    m = re.fullmatch(r"([LR])(\d{0,3})", tok)
    if not m:
        raise ValueError("无法识别的声道：%r" % token)
    sign = -1.0 if m.group(1) == "L" else 1.0
    amount = (int(m.group(2)) / 100.0) if m.group(2) else 0.7
    return Pan(tok, sign * min(amount, 1.0), True)


def _is_duration(token: str) -> bool:
    try:
        parse_duration(token)
        return True
    except ValueError:
        return False


def _looks_like_param(token: str) -> bool:
    # 以 @ 开头的一律视为参数（哪怕写错了），避免被当成描述吞掉
    if token.startswith("@") or token in _LEVELS or _PAN.fullmatch(token):
        return True
    return _is_duration(token)


def _is_bare_pan(token: str) -> bool:
    return bool(_PAN.fullmatch(token))


# ---------------------------------------------------------------- 指令
@dataclass
class SfxDirective:
    description: str
    at: Optional[float] = None       # 段内偏移（秒）
    duration: Optional[float] = None  # 持续（秒），None = 单次
    level: str = "中"
    pan: Pan = field(default_factory=lambda: AUTO_PAN)
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "at": self.at,
            "duration": self.duration,
            "level": self.level,
            "pan": self.pan.raw,
        }


def parse_sfx(args: str, line_no: int) -> SfxDirective:
    """解析 `sfx` 后的参数串：`描述 [@秒] [时长] [档位] [声道]`。"""
    tokens = args.split()
    if not tokens:
        raise ScriptError(line_no, "空的触发音指令")

    desc: List[str] = []
    i = 0
    while i < len(tokens) and not _looks_like_param(tokens[i]):
        desc.append(tokens[i])
        i += 1
    if not desc:
        raise ScriptError(line_no, "触发音缺少描述：%r" % args)

    at: Optional[float] = None
    duration: Optional[float] = None
    level = "中"
    pan = AUTO_PAN
    for tok in tokens[i:]:
        m = _AT.fullmatch(tok)
        if m:
            at = float(m.group(1))
            continue
        if tok in _LEVELS:
            level = tok
            continue
        if _PAN.fullmatch(tok):
            try:
                pan = parse_pan(tok)
            except ValueError as e:
                raise ScriptError(line_no, str(e))
            continue
        if _is_duration(tok):
            duration = parse_duration(tok)
            continue
        raise ScriptError(line_no, "无法识别的参数：%r（来自 %r）" % (tok, args))

    return SfxDirective(" ".join(desc), at, duration, level, pan, args)


# ---------------------------------------------------------------- 事件项
@dataclass
class TextSegment:
    text: str
    preset: str
    language: str
    sfx: List[SfxDirective]
    pan: Pan
    line_no: int


@dataclass
class Pause:
    seconds: float
    line_no: int


@dataclass
class SfxEvent:
    directive: SfxDirective
    line_no: int


@dataclass
class BedChange:
    name: str
    line_no: int


Item = Union[TextSegment, Pause, SfxEvent, BedChange]


@dataclass
class Script:
    path: Optional[Path]
    title: str
    seed: int
    default_preset: str
    pan_swap: Optional[float]
    items: List[Item]
    warnings: List[str] = field(default_factory=list)

    @property
    def text_segments(self) -> List[TextSegment]:
        return [it for it in self.items if isinstance(it, TextSegment)]

    @property
    def sfx_events(self) -> List[SfxEvent]:
        return [it for it in self.items if isinstance(it, SfxEvent)]

    def to_dict(self) -> dict:
        items = []
        for it in self.items:
            if isinstance(it, TextSegment):
                items.append({
                    "kind": "text", "line": it.line_no, "text": it.text,
                    "preset": it.preset, "language": it.language,
                    "pan": it.pan.raw, "sfx": [d.to_dict() for d in it.sfx],
                })
            elif isinstance(it, Pause):
                items.append({"kind": "pause", "line": it.line_no, "seconds": it.seconds})
            elif isinstance(it, SfxEvent):
                items.append({"kind": "sfx", "line": it.line_no, **it.directive.to_dict()})
            elif isinstance(it, BedChange):
                items.append({"kind": "bed", "line": it.line_no, "name": it.name})
        return {
            "path": str(self.path) if self.path else None,
            "title": self.title,
            "seed": self.seed,
            "default_preset": self.default_preset,
            "pan_swap": self.pan_swap,
            "warnings": self.warnings,
            "items": items,
        }


# ---------------------------------------------------------------- 解析入口
def parse_script(text: str, path: Optional[Path] = None) -> Script:
    title = ""
    seed = 1234
    default_preset = config.DEFAULT_PRESET
    pan_swap: Optional[float] = None
    items: List[Item] = []
    warnings: List[str] = []

    current_preset = config.DEFAULT_PRESET
    preset_seen = False

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            continue
        if line.startswith("@"):
            key, _, value = line[1:].partition(" ")
            key, value = key.strip(), value.strip()
            if key == "preset":
                if value not in config.SUPPORTED_PRESETS:
                    raise ScriptError(line_no, "未知 preset：%r（可选 %s）"
                                      % (value, "/".join(config.SUPPORTED_PRESETS)))
                current_preset = value
                if not preset_seen:
                    default_preset = value
                    preset_seen = True
            elif key == "seed":
                try:
                    seed = int(value)
                except ValueError:
                    raise ScriptError(line_no, "seed 必须是整数：%r" % value)
            elif key == "title":
                title = value
            elif key == "bed":
                if not value:
                    raise ScriptError(line_no, "@bed 缺少预设名")
                items.append(BedChange(value, line_no))
            elif key == "pan_swap":
                try:
                    pan_swap = float(value)
                except ValueError:
                    raise ScriptError(line_no, "pan_swap 必须是数字：%r" % value)
            else:
                raise ScriptError(line_no, "未知元信息：@%s" % key)
            continue

        # 独立的括号指令 (sfx ...) / (pause N)
        m = _PAREN.fullmatch(line)
        if m and "(" not in m.group(1):
            inner = m.group(1).strip()
            cmd = _COMMAND.fullmatch(inner)
            if cmd and cmd.group(1) == "sfx":
                items.append(SfxEvent(parse_sfx(cmd.group(2), line_no), line_no))
                continue
            if cmd and cmd.group(1) == "pause":
                items.append(Pause(_parse_pause(cmd.group(2), line_no), line_no))
                continue
            raise ScriptError(line_no, "无法识别的指令：%r" % line)

        # 文本行：剥离内联 (sfx ...) 与 (pan ...)
        inline = [parse_sfx(a, line_no) for a in INLINE_SFX.findall(line)]
        remaining = INLINE_SFX.sub("", line)

        voice_pan = AUTO_PAN
        pan_hits = INLINE_PAN.findall(remaining)
        if pan_hits:
            try:
                voice_pan = parse_pan(pan_hits[-1])
            except ValueError as e:
                raise ScriptError(line_no, str(e))
            remaining = INLINE_PAN.sub("", remaining)
        remaining = remaining.strip()

        if not pan_hits and remaining:
            head, _, tail = remaining.rpartition(" ")
            if head and _is_bare_pan(tail):
                try:
                    voice_pan = parse_pan(tail)
                except ValueError as e:
                    raise ScriptError(line_no, str(e))
                remaining = head.strip()

        if not remaining:
            for directive in inline:
                items.append(SfxEvent(directive, line_no))
            continue

        language = config.PRESET_LANGUAGE[current_preset]
        items.append(TextSegment(remaining, current_preset, language, inline, voice_pan, line_no))

    return Script(path, title, seed, default_preset, pan_swap, items, warnings)


def _parse_pause(arg: str, line_no: int) -> float:
    if not arg:
        raise ScriptError(line_no, "pause 缺少秒数")
    try:
        seconds = parse_duration(arg)
    except ValueError:
        raise ScriptError(line_no, "pause 秒数无法解析：%r" % arg)
    if seconds <= 0:
        raise ScriptError(line_no, "pause 必须为正数：%r" % arg)
    return seconds


def load_script(path: Union[str, Path]) -> Script:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_script(text, path)

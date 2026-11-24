# -*- coding: utf-8 -*-
"""素材清单（materials/index.yaml）：加载、保存、注册、检索匹配。

每个素材记一条，来源和授权必须留痕。注册时自动测量时长 / 采样率 /
声道 / 基准响度（base_lufs），供音量归一化用。
"""
from __future__ import annotations

import datetime
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from . import audio, config


CHANNEL_DESC = {"L": "左耳", "R": "右耳", "C": "双耳均衡", "LR": "左右交替", "RL": "右左交替"}


_ID_RE = re.compile(r"^(.+?)(?:_(\d{2}))?(?:_(L|R|C|LR|RL))?$")
_SOURCE_SPAN_RE = re.compile(r"^(.*?)\s*@\s*\S+\s*\((\d+)-(\d+)s\)$")


@dataclass
class Material:
    id: str
    file: str
    desc: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""
    source_url: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""
    attribution_required: bool = False
    ai_generated: bool = False
    commercial_ok: bool = True
    downloaded_at: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    base_lufs: Optional[float] = None

    @property
    def category(self) -> str:
        match = _ID_RE.fullmatch(self.id)
        return match[1] if match and (match[2] or match[3]) else self.id

    @property
    def is_pending(self) -> bool:
        return self.id.startswith("unknown_") or "待标注" in self.tags

    @property
    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else (config.MATERIALS_DIR / p)


BUILTIN_BEDS = (
    Material(
        id="room_tone", file="bed/room_tone.wav", desc="林间轻柔微风",
        tags=["环境音", "微风", "树叶"], source="procedural_dsp",
        license="CC0", commercial_ok=True, duration_sec=90.0,
        sample_rate=44100, channels=2, base_lufs=-43.29,
    ),
    Material(
        id="rain_hard", file="bed/rain_hard.wav", desc="屋檐夜雨（物理合成）",
        tags=["环境音", "雨滴", "夜雨"], source="procedural_dsp",
        license="CC0", commercial_ok=True, duration_sec=90.0,
        sample_rate=44100, channels=2, base_lufs=-33.86,
    ),
    Material(
        id="rain_soft", file="bed/818347_rain_soft.wav", desc="树荫下清晰柔和的春雨滴落声",
        tags=["环境音", "雨滴", "细雨"], source="freesound",
        source_url="https://freesound.org/people/newlocknew/sounds/818347/",
        author="newlocknew", license="CC0",
        license_url="http://creativecommons.org/publicdomain/zero/1.0/",
        commercial_ok=True, duration_sec=60.0, sample_rate=48000,
        channels=2, base_lufs=-38.12,
    ),
    Material(
        id="calm_waves", file="bed/520770_calm_waves.wav", desc="96kHz高清录制的平静湖岸微浪",
        tags=["环境音", "湖水", "微浪"], source="freesound",
        source_url="https://freesound.org/people/priesjensen/sounds/520770/",
        author="priesjensen", license="CC0",
        license_url="http://creativecommons.org/publicdomain/zero/1.0/",
        commercial_ok=True, duration_sec=40.5, sample_rate=96000,
        channels=2, base_lufs=-43.82,
    ),
    Material(
        id="leaves_rustle", file="bed/523389_leaves_rustle.wav", desc="微风吹拂树叶的轻柔沙沙声",
        tags=["环境音", "树叶", "微风"], source="freesound",
        source_url="https://freesound.org/s/523389/",
        author="Anya_Media", license="CC0",
        license_url="http://creativecommons.org/publicdomain/zero/1.0/",
        commercial_ok=True, duration_sec=60.0, sample_rate=48000,
        channels=2, base_lufs=-53.6,
    ),
    Material(
        id="campfire_ember", file="bed/729395_campfire_ember.wav", desc="温暖柔和的篝火炭火声",
        tags=["环境音", "篝火", "炭火"], source="freesound",
        source_url="https://freesound.org/s/729395/",
        author="HECKFRICKER", license="CC0",
        license_url="http://creativecommons.org/publicdomain/zero/1.0/",
        commercial_ok=True, duration_sec=60.0, sample_rate=44100,
        channels=2, base_lufs=-51.28,
    ),
)


def category_descriptions(prototypes_path: Path = config.PROTOTYPES_NPZ) -> dict[str, str]:
    with np.load(config.CLAP_PROMPTS_NPZ) as prompts:
        descriptions = dict(zip(prompts["slugs"].tolist(), prompts["descs"].tolist()))
    descriptions.update({
        "unknown_trigger": "待标注触发音",
        "ambient_bed": "稳态环境床音",
        "room_tone": "林间轻柔微风",
        "rain_soft": "细雨",
        "rain_hard": "大雨",
    })
    if prototypes_path.is_file():
        with np.load(prototypes_path) as prototypes:
            for slug, desc in zip(prototypes["proto_slugs"].tolist(), prototypes["proto_descs"].tolist()):
                descriptions.setdefault(str(slug), str(desc))
    return descriptions


def parse_source_span(source: str) -> Optional[tuple[str, int, int]]:
    match = _SOURCE_SPAN_RE.fullmatch(source)
    if match and match[1].strip() and int(match[2]) < int(match[3]):
        return match[1].strip(), int(match[2]), int(match[3])
    return None


def existing_source_spans(materials: List[Material], source_name: str) -> List[tuple[int, int]]:
    target = source_name.strip()
    spans = []
    for material in materials:
        if material.path.is_file():
            span = parse_source_span(material.source)
            if span is not None and span[0] == target:
                spans.append((span[1], span[2]))
    return spans


def load_index(path: Optional[Path] = None) -> List[Material]:
    is_default = path is None or Path(path) == config.MATERIALS_INDEX
    path = Path(path) if path is not None else config.MATERIALS_INDEX
    materials: List[Material] = []
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is not None:
            if not isinstance(raw, list):
                raise ValueError("index.yaml 顶层必须是列表：%s" % path)
            for item in raw:
                if not isinstance(item, dict) or "id" not in item or "file" not in item:
                    raise ValueError(f"index.yaml 条目缺少必填字段 id/file：{item!r}")
                materials.append(Material(**item))
    if is_default:
        existing_ids = {material.id for material in materials}
        for builtin in BUILTIN_BEDS:
            if builtin.id not in existing_ids and builtin.path.is_file():
                materials.append(replace(builtin, tags=list(builtin.tags)))
    return materials


def save_index(materials: List[Material], path: Optional[Path] = None) -> Path:
    path = Path(path) if path else config.MATERIALS_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(m) for m in materials]
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def prune_missing(index_path: Optional[Path] = None) -> List[Material]:
    """清理磁盘文件已缺失的索引记录，返回实际可用的素材。"""
    items = load_index(index_path)
    kept = [m for m in items if m.path.is_file()]
    if len(kept) != len(items):
        save_index(kept, index_path)
    return kept


def find_index(materials: List[Material], mat_id: str) -> Optional[Material]:
    for m in materials:
        if m.id == mat_id:
            return m
    return None


def next_available_id(
    prefix: str,
    ch_tag: str,
    materials: List[Material],
    out_dir: Path,
    start_seq: int = 1,
    exclude: Optional[Material] = None,
) -> tuple[str, int, Path]:
    used_ids = {m.id for m in materials if m is not exclude and m.path.is_file()}
    excluded_path = exclude.path if exclude is not None else None
    seq = start_seq
    while True:
        stem_head = f"{prefix}_{seq:02d}_"
        mat_id = f"{stem_head}{ch_tag}"
        dest = out_dir / f"{mat_id}.wav"
        id_taken = any(uid == mat_id or uid.startswith(stem_head) for uid in used_ids)
        file_taken = any(p != excluded_path for p in out_dir.glob(f"{stem_head}*.wav")) if out_dir.is_dir() else False
        if not id_taken and not file_taken:
            return mat_id, seq, dest
        seq += 1


def relabel_material(mat_id: str, new_prefix: Optional[str] = None,
                     desc: Optional[str] = None, tags: Optional[List[str]] = None,
                     index_path: Optional[Path] = None) -> Material:
    """重命名音频并更新标签；确认类别时同步默认描述并移除待标注标签。"""
    materials = load_index(index_path)
    mat = find_index(materials, mat_id)
    if mat is None:
        raise ValueError(f"未找到素材：{mat_id}，请先选择素材")
    if new_prefix is not None:
        new_prefix = new_prefix.strip()
        if not new_prefix or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", new_prefix) is None:
            raise ValueError("素材前缀只能包含英文字母、数字、下划线和连字符")
        old_path = mat.path
        match = re.fullmatch(r"(.+?)(?:_(\d{2}))?_(L|R|C|LR|RL)", old_path.stem)
        old_prefix, old_seq, ch_tag = (match[1], match[2] or "01", match[3]) if match else (old_path.stem, "01", "C")
        new_id, seq, dest = next_available_id(
            new_prefix, ch_tag, materials, old_path.parent, start_seq=1 if old_prefix == "unknown_trigger" else int(old_seq), exclude=mat,
        )
        with np.load(config.CLAP_PROMPTS_NPZ) as prompts:
            slugs = prompts["slugs"].tolist()
            tags_map = dict(zip(slugs, prompts["tags"].tolist()))
        descriptions = category_descriptions()
        suffix_re = r"\d+_(?:左耳|右耳|双耳均衡|左右交替|右左交替)$"
        if desc is None or desc == mat.desc:
            if re.search(suffix_re, mat.desc):
                new_base = descriptions.get(new_prefix, new_prefix)
                mat.desc = f"{new_base}{seq:02d}_{CHANNEL_DESC[ch_tag]}"
        elif re.search(suffix_re, desc):
            base = re.sub(suffix_re, "", desc)
            mat.desc = f"{base}{seq:02d}_{CHANNEL_DESC[ch_tag]}"
        else:
            mat.desc = desc
        if new_prefix != "unknown_trigger" and (tags is None or list(tags) == mat.tags):
            cleaned = [t for t in mat.tags if t != "待标注"]
            mat.tags = [t for t in tags_map[new_prefix].split(",") if t] if not cleaned and new_prefix in tags_map else cleaned
        elif tags is not None:
            mat.tags = list(tags)
        old_path.rename(dest)
        mat.id = new_id
        mat.file = str(Path(mat.file).with_name(dest.name)).replace("\\", "/")
    else:
        if desc is not None:
            mat.desc = desc
        if tags is not None:
            mat.tags = list(tags)
    if new_prefix is not None and new_prefix != "unknown_trigger" and mat.path.is_file():
        from . import mine
        base_desc = re.sub(r"\d+_(?:左耳|右耳|双耳均衡|左右交替|右左交替)$", "", mat.desc)
        mine.record_prototype(mat.path, new_prefix, base_desc, mat.tags)
    save_index(materials, index_path)
    return mat


# ---------------------------------------------------------------- 注册
def register(
    audio_path,
    materials: List[Material],
    mat_id: Optional[str] = None,
    desc: str = "",
    tags: Optional[List[str]] = None,
    source: str = "",
    source_url: str = "",
    author: str = "",
    license: str = "",
    license_url: str = "",
    attribution_required: bool = False,
    ai_generated: bool = False,
    commercial_ok: bool = True,
) -> Material:
    """把素材登记进清单，自动测量属性。同 id 覆盖。"""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError("素材不存在：%s" % audio_path)

    info = audio.probe(audio_path)
    data, sr = audio.read_audio(audio_path)
    lufs = audio.loudness_lufs(data, sr)

    try:
        rel = audio_path.resolve().relative_to(config.MATERIALS_DIR.resolve())
        file_field = str(rel).replace("\\", "/")
    except ValueError:
        file_field = str(audio_path)

    mat_id = mat_id or audio_path.stem
    material = Material(
        id=mat_id,
        file=file_field,
        desc=desc,
        tags=list(tags or []),
        source=source,
        source_url=source_url,
        author=author,
        license=license,
        license_url=license_url,
        attribution_required=attribution_required,
        ai_generated=ai_generated,
        commercial_ok=commercial_ok,
        downloaded_at=datetime.date.today().isoformat(),
        duration_sec=round(len(data) / sr, 3),
        sample_rate=int(sr),
        channels=int(info.channels),
        base_lufs=round(float(lufs), 2) if np.isfinite(lufs) else None,
    )
    for i, existing in enumerate(materials):
        if existing.id == mat_id:
            materials[i] = material
            break
    else:
        materials.append(material)
    return material


def register_and_save(audio_path, index_path=None, **kwargs) -> Material:
    materials = load_index(index_path)
    material = register(audio_path, materials, **kwargs)
    save_index(materials, index_path)
    return material


# ---------------------------------------------------------------- 检索匹配
def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text).strip().lower())


def score_material(material: Material, query: str) -> float:
    q = _norm(query)
    if not q:
        return 0.0
    score = 0.0
    mid = _norm(material.id)
    if mid == q:
        score += 5.0
    elif q in mid:
        score += 2.0
    if material.desc:
        nd = _norm(material.desc)
        if nd == q:
            score += 3.0
        elif q in nd or nd in q:
            score += 1.5
    for tag in material.tags:
        nt = _norm(tag)
        if not nt:
            continue
        if nt == q:
            score += 3.0
        elif nt in q or q in nt:
            score += 1.0
    return score


def find_all(materials: List[Material], query: str, limit: int = 5) -> List[Material]:
    scored = [(score_material(m, query), m) for m in materials]
    scored = [pair for pair in scored if pair[0] > 0 and pair[1].path.is_file()]
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [m for _, m in scored[:limit]]


def find_best(materials: List[Material], query: str) -> Optional[Material]:
    hits = find_all(materials, query, limit=1)
    return hits[0] if hits else None

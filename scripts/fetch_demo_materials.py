# -*- coding: utf-8 -*-
"""抓取 demo 用的少量 CC0 素材并登记进 materials/index.yaml。

只用 Freesound 的 CC0 素材（可商用、免署名）。默认下预览版（mp3），
需要原始质量时用 --original（要求 secrets/freesound_oauth.json 已授权）。

    python scripts/fetch_demo_materials.py
    python scripts/fetch_demo_materials.py --original
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import freesound_search as fs  # noqa: E402

from tingleloom import config  # noqa: E402
from tingleloom import materials as materials_mod  # noqa: E402

# id 已知的用 id 精确定位；否则用 query 取第一条 CC0 结果
SPECS = [
    {"key": "page_turn_01", "id": 614080, "desc": "翻书", "subdir": "sfx",
     "tags": ["翻页", "纸质", "page turn", "paper"]},
    {"key": "tapping_01", "query": "finger tapping", "desc": "轻敲", "subdir": "sfx",
     "tags": ["敲击", "轻敲", "tapping", "tap"]},
    {"key": "rain_soft", "id": 440488, "desc": "小雨声", "subdir": "bed",
     "tags": ["雨", "小雨", "雨声", "rain"]},
    {"key": "room_tone", "query": "room tone", "desc": "房间底噪", "subdir": "bed",
     "tags": ["底噪", "房间", "room tone", "ambience"]},
]


def _pick_sound(token, spec):
    if spec.get("id"):
        return fs.get_sound(token, spec["id"])
    results = fs.search(token, spec["query"], cc0=True, n=5)
    return results[0] if results else None


def _download(sound, dest_dir: Path, original: bool) -> Path:
    files = []
    if original:
        try:
            files = fs.download_original([sound], str(dest_dir))
        except (urllib.error.URLError, TimeoutError, RuntimeError, SystemExit) as exc:
            print("  原始下载失败，回退预览版：%s" % exc)
    if not files:
        files = fs.download_previews([sound], str(dest_dir))
    if not files:
        raise RuntimeError("该素材没有可用下载地址")
    return files[0][1]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true", help="尝试下载原始质量")
    args = ap.parse_args()

    config.ensure_dirs()
    token = fs.load_token()

    for spec in SPECS:
        print("→ %s（%s）" % (spec["key"], spec.get("id") or spec["query"]))
        try:
            sound = _pick_sound(token, spec)
        except SystemExit as exc:
            print("  检索失败：%s" % exc)
            continue
        if not sound:
            print("  没有命中，跳过")
            continue
        subdir = config.MATERIALS_DIR / spec["subdir"]
        try:
            path = _download(sound, subdir, args.original)
        except Exception as exc:
            print("  下载失败：%s" % exc)
            continue
        material = materials_mod.register_and_save(
            path, mat_id=spec["key"], desc=spec["desc"], tags=spec["tags"],
            source="freesound",
            source_url="https://freesound.org/s/%s/" % sound["id"],
            author=sound.get("username", ""),
            license=fs.license_label(sound.get("license", "")),
            commercial_ok=True, attribution_required=False,
        )
        lufs_str = "%.1f LUFS" % material.base_lufs if material.base_lufs is not None else "- LUFS"
        print("  已登记 %s  (%.2fs, %d Hz, %s) ← %s"
              % (material.id, material.duration_sec, material.sample_rate,
                 lufs_str, path.name))

    print("\n当前素材库：")
    for m in materials_mod.load_index():
        print("  %-14s %-5s %6.1fs  %s" % (m.id, m.license, m.duration_sec, m.desc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Freesound 检索工具（可复用）。

Freesound 不支持中文检索 —— 用英文（或日文罗马字）关键词。
本工具接上了 Freesound 真正有用的筛选/排序：
    --tags              按 tag 过滤（Freesound 的核心检索维度）
    --min/max-duration  时长范围（秒）
    --min-rating        最低评分（0–5）
    --min-downloads     最低下载量
    --sort              score / rating_desc / downloads_desc / duration_desc / duration_asc ...
    --cc0               只要 CC0

用法：
    python scripts/freesound_search.py "rain" --cc0 --min-duration 120 --sort downloads_desc --n 8
    python scripts/freesound_search.py "page turn" --cc0 --min-duration 2 --max-duration 10 --sort rating_desc --n 8
    python scripts/freesound_search.py "page turn" --cc0 --download materials/sfx
"""
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / "secrets" / "freesound_token.txt"
BASE = "https://freesound.org/apiv2"
FIELDS = ("id,name,tags,license,username,duration,samplerate,channels,type,"
          "previews,avg_rating,num_downloads")
SORTS = ("score", "rating_desc", "rating_asc", "downloads_desc", "downloads_asc",
         "created_desc", "duration_desc", "duration_asc")


def load_token() -> str:
    tok = os.environ.get("FREESOUND_TOKEN", "").strip()
    if tok:
        return tok
    if not TOKEN_FILE.exists():
        raise SystemExit("token not found: %s (or set FREESOUND_TOKEN)" % TOKEN_FILE)
    for line in TOKEN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit("token file is empty: %s" % TOKEN_FILE)


def api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "asmr-sound-collector"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read(300).decode("utf-8", "replace")
        raise SystemExit("HTTP %s: %s" % (e.code, body))


def build_filter(cc0=False, tags=None, min_duration=None, max_duration=None,
                 min_rating=None, min_downloads=None, user="") -> str:
    parts = []
    if cc0:
        parts.append('license:"Creative Commons 0"')
    for tag in tags or []:
        parts.append("tag:%s" % tag)
    if min_duration or max_duration:
        lo = int(min_duration) if min_duration else 0
        hi = int(max_duration) if max_duration else 36000
        parts.append("duration:[%d TO %d]" % (lo, hi))
    if min_rating:
        parts.append("avg_rating:[%s TO 5]" % min_rating)
    if min_downloads:
        parts.append("num_downloads:[%d TO 1000000000]" % int(min_downloads))
    if user:
        parts.append('username:"%s"' % user)
    return " ".join(parts)


def license_label(url: str) -> str:
    normalized = url.lower()
    if "zero" in normalized:
        return "CC0"
    if "/by-nc/" in normalized:
        return "CC-BY-NC"
    if "/by/" in normalized:
        return "CC-BY"
    return url.rstrip("/").rsplit("/", 1)[-1] or "unknown"


def get_sound(token: str, sound_id: int) -> dict:
    params = urllib.parse.urlencode({"token": token, "fields": FIELDS})
    return api_get(f"{BASE}/sounds/{sound_id}/?{params}")


def show_one(index: int, s: dict) -> None:
    lic = license_label(s.get("license", ""))
    rating = s.get("avg_rating")
    dls = s.get("num_downloads")
    print("%2d. [%s] %s" % (index, lic, s.get("name")))
    print("    id=%s user=%s dur=%.1fs sr=%s ch=%s rating=%s dls=%s"
          % (s.get("id"), s.get("username"), s.get("duration") or 0,
             s.get("samplerate"), s.get("channels") or 0,
             ("%.2f" % rating) if rating else "-", dls if dls is not None else "-"))
    tags = s.get("tags") or []
    if tags:
        print("    tags: %s" % ", ".join(tags[:14]))
    print("    https://freesound.org/s/%s/" % s.get("id"))


def search(token: str, query: str, cc0: bool = False, n: int = 10, user: str = "",
           tags=None, min_duration=None, max_duration=None, min_rating=None,
           min_downloads=None, sort: str = "score") -> list:
    params = {"query": query, "fields": FIELDS, "page_size": str(n), "token": token}
    filter_str = build_filter(cc0, tags, min_duration, max_duration,
                              min_rating, min_downloads, user)
    if filter_str:
        params["filter"] = filter_str
    if sort and sort != "score":
        params["sort"] = sort
    data = api_get("%s/search/text/?%s" % (BASE, urllib.parse.urlencode(params)))
    results = data.get("results", [])
    print("count=%s  filter=%s\n" % (data.get("count"), filter_str or "-"))
    for i, s in enumerate(results, 1):
        show_one(i, s)
    return results


def _safe_name(s: dict) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in (s.get("name") or "sound"))[:60]


def download_previews(results: list, out_dir: str) -> list[tuple[dict, Path]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for s in results:
        pv = (s.get("previews") or {}).get("preview-hq-mp3")
        if not pv:
            continue
        dest = out / ("%s_%s.mp3" % (s.get("id"), _safe_name(s)))
        if dest.exists():
            print("  exists, skip: %s" % dest.name)
            downloaded.append((s, dest))
            continue
        req = urllib.request.Request(pv, headers={"User-Agent": "asmr-sound-collector"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print("  saved: %s (%d bytes)" % (dest.name, dest.stat().st_size))
        downloaded.append((s, dest))
    return downloaded


def download_original(results: list, out_dir: str) -> list[tuple[dict, Path]]:
    import freesound_oauth

    token = freesound_oauth.ensure_access_token()
    headers = {"User-Agent": "asmr-sound-collector", "Authorization": "Bearer " + token}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for s in results:
        sid = s.get("id")
        req = urllib.request.Request("%s/sounds/%s/" % (BASE, sid), headers=headers)
        with urllib.request.urlopen(req, timeout=40) as r:
            info = json.loads(r.read().decode("utf-8"))
        dl_uri = info.get("download")
        if not dl_uri:
            print("  skip %s: no download URI" % sid)
            continue
        ext = (info.get("type") or "wav").lstrip(".")
        dest = out / ("%s_%s.%s" % (sid, _safe_name(s), ext))
        if dest.exists():
            print("  exists, skip: %s" % dest.name)
            downloaded.append((s, dest))
            continue
        req = urllib.request.Request(dl_uri, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        print("  saved(original): %s  %s %s/%sch" % (dest.name, ext, info.get("samplerate"), info.get("channels")))
        downloaded.append((s, dest))
    return downloaded


def main():
    ap = argparse.ArgumentParser(description="Freesound search tool")
    ap.add_argument("query", help="search query (English; Freesound does not support Chinese)")
    ap.add_argument("--cc0", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tags", default="", help="comma-separated tag filter, e.g. field-recording,rain")
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--max-duration", type=float, default=None)
    ap.add_argument("--min-rating", type=float, default=None)
    ap.add_argument("--min-downloads", type=int, default=None)
    ap.add_argument("--sort", choices=SORTS, default="score")
    ap.add_argument("--user", default="")
    ap.add_argument("--download", metavar="DIR")
    ap.add_argument("--original", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    token = load_token()
    filter_str = build_filter(args.cc0, tags, args.min_duration, args.max_duration,
                              args.min_rating, args.min_downloads, args.user)

    if args.json:
        params = {"query": args.query, "fields": FIELDS,
                  "page_size": str(args.n), "token": token}
        if filter_str:
            params["filter"] = filter_str
        if args.sort != "score":
            params["sort"] = args.sort
        print(json.dumps(api_get("%s/search/text/?%s" % (BASE, urllib.parse.urlencode(params))),
                         ensure_ascii=False, indent=2))
        return

    results = search(token, args.query, args.cc0, args.n, args.user, tags,
                     args.min_duration, args.max_duration, args.min_rating,
                     args.min_downloads, args.sort)
    if args.download:
        if args.original:
            print("\ndownload ORIGINAL -> %s" % args.download)
            download_original(results, args.download)
        else:
            print("\ndownload PREVIEW -> %s" % args.download)
            download_previews(results, args.download)


if __name__ == "__main__":
    main()

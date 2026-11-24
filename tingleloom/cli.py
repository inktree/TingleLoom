# -*- coding: utf-8 -*-
"""统一命令行入口。

    python -m tingleloom run    examples/demo.asmr
    python -m tingleloom parse  examples/demo.asmr
    python -m tingleloom tts    examples/demo.asmr
    python -m tingleloom match  examples/demo.asmr
    python -m tingleloom material register materials/sfx/x.wav --desc 翻书 --tags 翻页,纸质
    python -m tingleloom material list
    python -m tingleloom material fetch "page turn" --cc0 --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import sys
from pathlib import Path

from . import config
from . import materials as materials_mod
from .script import BedChange, Pause, ScriptError, SfxEvent, TextSegment, load_script
from .sfx import resolve_sfx
from .tts import OnnxDeployEngine, SynthOptions, TTSCache, provider_signature, tts_fingerprint, synthesize_all


def _setup_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------- parse
def cmd_parse(args) -> int:
    script = load_script(args.script)
    if args.json:
        print(json.dumps(script.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print("标题：%s" % (script.title or "(无)"))
    print("默认 preset：%s   seed：%d   pan_swap：%s"
          % (script.default_preset, script.seed, script.pan_swap))
    for item in script.items:
        if isinstance(item, TextSegment):
            extra = ""
            if item.sfx:
                extra = "  sfx=" + ",".join(d.description for d in item.sfx)
            if item.pan.explicit:
                extra += "  pan=%s" % item.pan.raw
            print("  L%-3d text [%s] %s%s" % (item.line_no, item.preset, item.text, extra))
        elif isinstance(item, Pause):
            print("  L%-3d pause %.1fs" % (item.line_no, item.seconds))
        elif isinstance(item, SfxEvent):
            d = item.directive
            print("  L%-3d sfx   %s%s%s" % (
                item.line_no, d.description,
                "" if d.duration is None else "  %.1fs" % d.duration,
                "" if d.at is None else "  @%.1f" % d.at))
        elif isinstance(item, BedChange):
            print("  L%-3d bed   %s" % (item.line_no, item.name))
    print("文本段 %d，独立触发音 %d" % (len(script.text_segments), len(script.sfx_events)))
    for w in script.warnings:
        print("警告：%s" % w)
    return 0


# ---------------------------------------------------------------- tts
def cmd_tts(args) -> int:
    config.ensure_dirs()
    script = load_script(args.script)
    fingerprint = tts_fingerprint(args.tts_dir)
    providers = provider_signature()
    cache = TTSCache()

    def make_engine():
        return OnnxDeployEngine(deploy=args.tts_dir, output_dir=config.BUILD_DIR / "tts_tmp")

    def progress(done, total, seg):
        print("  [%d/%d] 合成：%s" % (done, total, seg.text[:40]))

    audios = synthesize_all(
        script.text_segments, SynthOptions(), script.seed, cache,
        fingerprint["engine_id"], fingerprint["sample_rate"], providers,
        make_engine, progress=progress,
    )
    print("共 %d 段：" % len(audios))
    for sa in audios:
        print("  %6.2fs  %-7s  %s" % (
            sa.duration, "[缓存]" if sa.from_cache else "[新合成]", sa.segment.text))
    return 0


# ---------------------------------------------------------------- match
def cmd_match(args) -> int:
    script = load_script(args.script)
    materials = materials_mod.load_index()
    resolved, gaps = resolve_sfx(script, materials)
    print("触发音匹配（库内共 %d 条素材）：" % len(materials))
    for r in resolved:
        d = r.occurrence.directive
        target = d.description
        hit = r.material.id if r.material else "— 缺 —"
        print("  L%-3d %-16s → %s%s" % (
            r.occurrence.line_no, target, hit,
            "" if d.duration is None else "  (%.1fs)" % d.duration))
    if gaps:
        print("\n缺口清单（需采购 / 生成）：")
        for g in gaps:
            print("  - %s" % g)
    else:
        print("\n全部命中，无缺口。")
    return 0


# ---------------------------------------------------------------- run
def cmd_run(args) -> int:
    if args.provider:
        os.environ["TINGLELOOM_TTS_PROVIDERS"] = args.provider
    from .pipeline import run_pipeline

    def progress(done, total, seg):
        print("  [%d/%d] 合成：%s" % (done, total, seg.text[:40]))

    result = run_pipeline(
        args.script, out_path=args.out, seed=args.seed, target_lufs=args.target_lufs,
        tts_dir=args.tts_dir, strict=args.strict, progress=progress,
    )
    report = result.mix_report
    print("\n成品 WAV：%s\n成品 MP3：%s" % (report.output_path, report.mp3_path))
    print("  时长 %.1f s  采样率 %d Hz  峰值 %.3f  整体响度 %.1f LUFS"
          % (report.duration_seconds, report.sample_rate, report.peak, report.loudness_lufs))
    if report.beds:
        print("  节目 %.1f LUFS（增益 %+.1f dB）  床音 %.1f LUFS（低于节目 %.0f dB）"
              % (report.program_lufs, report.program_gain_db, report.bed_lufs,
                 report.bed_offset_db))
    else:
        print("  节目 %.1f LUFS（增益 %+.1f dB）  无床音"
              % (report.program_lufs, report.program_gain_db))
    print("  人声 %d 段，触发音 %d 个，床音段 %d" % (report.vocals, report.sfx, report.beds))
    if result.gaps:
        print("  缺口(%d)：%s" % (len(result.gaps), ", ".join(result.gaps)))
    for w in result.warnings:
        print("  警告：%s" % w)
    return 0


# ---------------------------------------------------------------- material
def cmd_material_register(args) -> int:
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    material = materials_mod.register_and_save(
        args.file, mat_id=args.id, desc=args.desc or "", tags=tags,
        source=args.source or "", source_url=args.source_url or "",
        author=args.author or "", license=args.license or "",
        commercial_ok=not args.no_commercial,
    )
    lufs_str = "%.1f LUFS" % material.base_lufs if material.base_lufs is not None else "- LUFS"
    print("已登记：%s  (%.2fs, %d Hz, %d ch, %s)"
          % (material.id, material.duration_sec, material.sample_rate,
             material.channels, lufs_str))
    return 0


def cmd_material_synth_bed(args) -> int:
    from . import presets as presets_mod

    bed_ids = [x.strip() for x in args.ids.split(",") if x.strip()] or None
    created = presets_mod.synthesize_beds(
        bed_ids=bed_ids, duration_sec=args.dur, seed=args.seed, overwrite=args.overwrite,
    )
    if not created:
        print("基础床音已齐全（如需重新生成请加 --overwrite）。")
        return 0
    for m in created:
        print("已合成床音：%s  (%.1fs, %d Hz, %d ch, %.1f LUFS) -> %s"
              % (m.id, m.duration_sec, m.sample_rate, m.channels, m.base_lufs, m.file))
    return 0


def cmd_material_proto_export(args) -> int:
    from . import mine

    pkg_dir = mine.export_prototype(
        args.slug, out_dir=Path(args.out_dir) if args.out_dir is not None else config.CONTRIBUTIONS_DIR,
        proof_audio=Path(args.audio) if args.audio is not None else None, proof_seconds=args.dur,
    )
    print(f"已导出原型贡献包：{pkg_dir}")
    return 0


def cmd_material_proto_verify(args) -> int:
    from . import mine

    report = mine.verify_prototype(Path(args.pkg_dir))
    print(f"原型 {report.slug}：{'通过' if report.passed else '未通过'}；"
          f"repro_cosine={report.repro_cosine:.4f}，vad_ratio={report.vad_ratio:.2%}，"
          f"vad_max_run_sec={report.vad_max_run_sec:.2f}，voice_score={report.voice_score:.3f}，"
          f"clip_samples={report.clip_samples}，top1={report.top1_slug}，margin={report.margin:.4f}")
    if not report.passed:
        raise RuntimeError("原型校验未通过：" + "；".join(report.reasons))
    return 0


def cmd_material_proto_merge(args) -> int:
    from . import mine

    report = mine.merge_prototype(Path(args.pkg_dir), remove_proof=not args.keep_proof)
    print(f"已合入原型 {report.slug}：repro_cosine={report.repro_cosine:.4f}，margin={report.margin:.4f}")
    return 0


def cmd_material_mine(args) -> int:
    from .mine import format_span, mine_and_ingest

    hints_path = Path(args.hints) if args.hints and "\n" not in args.hints else None
    hints_text = hints_path.read_text(encoding="utf-8") if hints_path is not None and hints_path.is_file() else args.hints
    report, results = mine_and_ingest(
        Path(args.file), prefix=args.prefix, profile=args.profile, hints_text=hints_text,
        slice_seconds=args.dur, min_slice_seconds=args.min_dur,
        step_seconds=args.step, top_n=args.top,
        max_vad_ratio=args.max_vad, max_vad_run_sec=args.max_vad_run,
        min_clap_score=args.min_clap_score,
        max_clap_voice=args.max_clap_voice, diversity_max_sim=args.diversity_sim,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        desc_prefix=args.desc_prefix,
        tags=tuple(t.strip() for t in args.tags.split(",") if t.strip()),
        source=args.source, license=args.license,
        commercial_ok=not args.no_commercial, probe_only=args.probe_only,
    )
    print(f"整轨体检：{report.duration_sec:.2f}s  {report.sample_rate} Hz  {report.channels} ch")
    print(f"  底噪 P10 {report.noise_floor_db:.1f} dBFS  活跃 P90 {report.active_db:.1f} dBFS"
          f"  SNR {report.snr_db:.1f} dB  高频截断 {report.cutoff_hz:.0f} Hz")
    print(f"  削波样本 {report.total_clip_samples}  双耳相关 {report.mean_lr_corr:.3f}")
    print("序号  时间区间(s)  声道  RMS(dBFS)  VAD占比  最长连续VAD(s)  CLAP标签(得分/耳语分)  5秒声道轨迹  落盘ID / LUFS")
    for rank, (cand, material) in enumerate(results, 1):
        saved = "—（仅探测）" if material is None else f"{material.id} / {material.base_lufs} LUFS"
        print(f"{rank:02d}  {format_span(cand.start_sec, cand.end_sec)}  {cand.channel_tag:<2}"
              f"  {cand.mean_rms_db:7.1f}  {cand.vad_ratio:.2%}  {cand.vad_max_run_sec:.3f}"
              f"  {cand.clap_desc}({cand.clap_score:.2f}/{cand.clap_voice_score:.2f})"
              f"  {cand.trajectory_5s}  {saved}")
    print(f"入选 {len(results)} 条" + ("（未写入文件）" if args.probe_only else ""))
    return 0


def cmd_material_label(args) -> int:
    if args.id is None:
        for mat in materials_mod.prune_missing():
            if mat.is_pending:
                print(f"{mat.id}  {mat.desc}  {mat.path}")
        return 0
    tags = None if args.tags is None else [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    mat = materials_mod.relabel_material(args.id, new_prefix=args.prefix,
                                         desc=args.desc, tags=tags)
    print(f"{mat.id}  {mat.desc}  {mat.path}")
    return 0


def cmd_material_list(args) -> int:
    materials = materials_mod.prune_missing()
    if not materials:
        print("素材库为空。")
        return 0
    print("%-18s %-8s %6s %5s %8s  %s" % ("id", "license", "秒", "声道", "LUFS", "desc"))
    for m in materials:
        lufs_str = "%8.1f" % m.base_lufs if m.base_lufs is not None else "%8s" % "-"
        print("%-18s %-8s %6.1f %5d %s  %s"
              % (m.id, m.license or "-", m.duration_sec, m.channels, lufs_str, m.desc))
    return 0


def cmd_material_fetch(args) -> int:
    sys.path.insert(0, str(config.SCRIPTS_DIR))
    import freesound_search as fs

    if not args.query and args.id is None:
        raise ValueError("请提供搜索关键词或 --id")
    token = fs.load_token()
    if args.id is not None:
        sound = fs.get_sound(token, args.id)
        if args.cc0 and fs.license_label(sound.get("license", "")) != "CC0":
            raise ValueError(f"Freesound {args.id} 不是 CC0 素材")
        results = [sound]
    else:
        results = fs.search(token, args.query, cc0=args.cc0, n=args.n,
                            min_duration=args.min_duration, max_duration=args.max_duration,
                            sort=args.sort)
    dest_dir = Path(args.dir) if args.dir else {
        "sfx": config.SFX_DIR, "bed": config.BED_DIR, "raw": config.RAW_DIR,
    }[args.type]
    downloaded = []
    for sound in results:
        files = []
        if args.original:
            try:
                files = fs.download_original([sound], str(dest_dir))
            except (urllib.error.URLError, TimeoutError, RuntimeError, SystemExit) as exc:
                print(f"  原始下载失败，回退高质预览：{exc}")
        if not files:
            files = fs.download_previews([sound], str(dest_dir))
        downloaded.extend(files)
    if not downloaded:
        raise RuntimeError("没有下载到可用素材")
    if not args.no_register:
        materials = materials_mod.load_index()
        for i, (sound, path) in enumerate(downloaded, 1):
            mat_id = (args.mat_id if len(downloaded) == 1 else f"{args.mat_id}_{i:02d}") if args.mat_id else path.stem
            tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else sound.get("tags", [])[:6]
            if args.type == "bed" and "环境音" not in tags:
                tags.append("环境音")
            lic = fs.license_label(sound.get("license", ""))
            material = materials_mod.register(
                path, materials, mat_id=mat_id, desc=args.desc or sound.get("name", ""),
                tags=tags, source="freesound", source_url=f"https://freesound.org/s/{sound['id']}/",
                author=sound.get("username", ""), license=lic,
                license_url=sound.get("license", ""), commercial_ok=lic in ("CC0", "CC-BY"),
                attribution_required=(lic == "CC-BY"),
            )
            lufs = f"{material.base_lufs:.1f} LUFS" if material.base_lufs is not None else "未测得 LUFS"
            print(f"  已登记 {material.id} ({material.duration_sec:.2f}s, {material.sample_rate} Hz, {lufs}) ← {path.name}")
        materials_mod.save_index(materials)
    return 0


def cmd_web(args) -> int:
    from .webui import launch_webui

    if args.tts_dir:
        os.environ["TINGLELOOM_TTS_DIR"] = str(args.tts_dir)
    launch_webui(host=args.host, port=args.port, share=args.share, inbrowser=not args.no_browser)
    return 0


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tingleloom", description="ASMR 音频制作流水线")
    parser.add_argument("--tts-dir", default=None, help="本地 ONNX TTS 部署包目录")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("web", help="启动 Web 界面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("run", help="跑完整流水线出成品")
    p.add_argument("script")
    p.add_argument("-o", "--out", default=None, help="成品输出路径")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target-lufs", type=float, default=None)
    p.add_argument("--provider", choices=("cpu", "cuda", "hybrid"), default=None)
    p.add_argument("--strict", action="store_true", help="有缺口就报错")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("parse", help="只解析台本并打印结构")
    p.add_argument("script")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("tts", help="只跑 TTS 并落缓存，打印实测时长")
    p.add_argument("script")
    p.set_defaults(func=cmd_tts)

    p = sub.add_parser("match", help="只做素材匹配，打印命中与缺口")
    p.add_argument("script")
    p.set_defaults(func=cmd_match)

    material = sub.add_parser("material", help="素材库管理")
    msub = material.add_subparsers(dest="material_command", required=True)

    p = msub.add_parser("register", help="登记一个素材（自动测属性）")
    p.add_argument("file")
    p.add_argument("--id", default=None)
    p.add_argument("--desc", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--source", default="")
    p.add_argument("--source-url", default="")
    p.add_argument("--author", default="")
    p.add_argument("--license", default="")
    p.add_argument("--no-commercial", action="store_true")
    p.set_defaults(func=cmd_material_register)

    p = msub.add_parser("proto-export", help="导出原型 PR 贡献包")
    p.add_argument("slug")
    p.add_argument("--audio", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dur", type=float, default=10.0)
    p.set_defaults(func=cmd_material_proto_export)

    p = msub.add_parser("proto-verify", help="核验原型贡献包")
    p.add_argument("pkg_dir")
    p.set_defaults(func=cmd_material_proto_verify)

    p = msub.add_parser("proto-merge", help="核验并归并原型至内置词表")
    p.add_argument("pkg_dir")
    p.add_argument("--keep-proof", action="store_true")
    p.set_defaults(func=cmd_material_proto_merge)

    p = msub.add_parser("mine", help="声学体检、滤除人声并自动切分入库")
    p.add_argument("file")
    p.add_argument("--prefix", default="auto")
    p.add_argument("--hints", default="", help="人类时间轴文本或 UTF-8 时间轴文件路径")
    p.add_argument("--profile", choices=("high", "low", "balanced", "bed"), default="high")
    p.add_argument("--dur", type=int, default=35)
    p.add_argument("--min-dur", type=int, default=None)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--max-vad", type=float, default=0.015)
    p.add_argument("--max-vad-run", type=float, default=0.25)
    p.add_argument("--min-clap-score", type=float, default=0.04)
    p.add_argument("--max-clap-voice", type=float, default=0.42)
    p.add_argument("--diversity-sim", type=float, default=0.90)
    p.add_argument("--out-dir", default=None, help="输出目录（默认触发音写入 materials/sfx，bed 模式写入 materials/bed）")
    p.add_argument("--desc-prefix", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--source", default="")
    p.add_argument("--license", default="")
    p.add_argument("--no-commercial", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    p.set_defaults(func=cmd_material_mine)

    p = msub.add_parser("synth-bed", help="程序化合成零版权无缝基础床音（room_tone / rain_soft / rain_hard）")
    p.add_argument("--ids", default="", help="逗号分隔的床音 ID（留空合成全部内置基础床音）")
    p.add_argument("--dur", type=float, default=90.0, help="合成时长（秒，默认 90）")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名床音")
    p.set_defaults(func=cmd_material_synth_bed)

    p = msub.add_parser("label", help="列出待标注素材，或重命名并更新标签")
    p.add_argument("--id")
    p.add_argument("--prefix")
    p.add_argument("--desc")
    p.add_argument("--tags")
    p.set_defaults(func=cmd_material_label)

    p = msub.add_parser("list", help="列出素材库")
    p.set_defaults(func=cmd_material_list)

    p = msub.add_parser("fetch", help="从 Freesound 检索下载并自动测量登记素材")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--id", type=int, default=None, help="按 Freesound Sound ID 精确拉取")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--type", choices=("sfx", "bed", "raw"), default="sfx")
    p.add_argument("--dir", default=None, help="覆盖素材分类的默认目录")
    p.add_argument("--mat-id", default=None)
    p.add_argument("--desc", default="")
    p.add_argument("--tags", default="", help="逗号分隔的素材标签")
    p.add_argument("--min-duration", type=float, default=None)
    p.add_argument("--max-duration", type=float, default=None)
    p.add_argument("--sort", choices=("score", "rating_desc", "rating_asc", "downloads_desc", "downloads_asc", "created_desc", "duration_desc", "duration_asc"), default="downloads_desc")
    p.add_argument("--cc0", action="store_true", help="仅下载免署名可商用的 CC0 素材")
    p.add_argument("--original", action="store_true", help="OAuth2 原始质量下载，失败回退高质 MP3")
    p.add_argument("--no-register", action="store_true", help="仅下载，不登记 index.yaml")
    p.set_defaults(func=cmd_material_fetch)

    return parser


def main(argv=None) -> int:
    _setup_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ScriptError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

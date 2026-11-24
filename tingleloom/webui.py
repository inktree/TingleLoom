"""Gradio 四面板工作台；业务处理器可脱离界面直接调用。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import gradio_client.utils as _gradio_client_utils

from . import config, llm, materials, mine, pipeline


# Gradio Client 1.3.0 尚未处理 Pydantic 生成的布尔 JSON Schema。
_ORIG_SCHEMA_TO_TYPE = _gradio_client_utils._json_schema_to_python_type


def _json_schema_to_python_type(schema, defs=None) -> str:
    if isinstance(schema, bool):
        return "Any" if schema else "Never"
    return _ORIG_SCHEMA_TO_TYPE(schema, defs)


_gradio_client_utils._json_schema_to_python_type = _json_schema_to_python_type


def run_mine_task(audio_file: str, profile: str, slice_seconds: int,
                  min_slice_seconds: int, top_n: int, prefix: str,
                  license_name: str, probe_only: bool, hints_text: str = ""
                  ) -> Tuple[str, List[List[object]], Optional[str]]:
    if not audio_file:
        raise ValueError("请提供源音频路径")
    report, results = mine.mine_and_ingest(
        Path(audio_file), prefix=prefix or "auto", profile=profile,
        slice_seconds=int(slice_seconds), min_slice_seconds=int(min_slice_seconds),
        top_n=int(top_n), license=license_name, probe_only=bool(probe_only), hints_text=hints_text,
    )
    target_dir = config.BED_DIR if profile == "bed" else config.SFX_DIR
    status_line = (
        f"- 仅探测模式（未入库）：筛出 {len(results)} 条候选（取消勾选“仅探测候选”后再次点击即可切分入库）"
        if probe_only else
        f"- 已自动切分写入 `{target_dir}` 并登记入库 `{config.MATERIALS_INDEX}`：{len(results)} 条"
        "（点击下方表格任意行可直接试听；切换到上方「2. 人工试听打标」可改名/鉴别，或切到「3. AI 台本生成」直接使用）"
    )
    summary = (
        f"### 源音频体检\n"
        f"- 时长：{report.duration_sec:.1f}s | 采样率：{report.sample_rate}Hz | 声道：{report.channels}\n"
        f"- 底噪：{report.noise_floor_db:.1f} dB | 中位电平：{report.median_db:.1f} dB | "
        f"活跃电平：{report.active_db:.1f} dB | 信噪比：{report.snr_db:.1f} dB\n"
        f"- 峰值：{report.max_peak_db:.1f} dB | 削波采样：{report.total_clip_samples} | "
        f"截止频率：{report.cutoff_hz:.0f}Hz | 左右相关性：{report.mean_lr_corr:.2f}\n"
        f"{status_line}"
    )
    if hints_text.strip():
        positive = sum(not h.exclude for h in report.timeline_hints)
        skipped = sum(h.exclude for h in report.timeline_hints)
        if probe_only:
            learned = "仅探测，未更新原型库"
        elif any(mat is not None and cand.timeline_hint is not None and not mat.is_pending for cand, mat in results):
            learned = "已更新本地声学原型库"
        elif any(mat is not None and cand.timeline_hint is not None for cand, mat in results):
            learned = "已入库为待标注素材，确认英文前缀后更新原型库"
        else:
            learned = "无正向素材入库，未更新原型库"
        summary += f"\n- 监督提示已生效：解析出 {positive} 个正向章节、{skipped} 个跳过区间（{learned}）"
    rows = [
        [rank, mine.format_span(cand.start_sec, cand.end_sec), cand.channel_tag,
         f"{cand.mean_rms_db:.1f}", f"{cand.vad_ratio:.2%} / {cand.vad_max_run_sec:.2f}s",
         f"{cand.clap_desc} ({cand.clap_score:.2f} / 耳语{cand.clap_voice_score:.2f})",
         cand.trajectory_5s, material.id if material else cand.clap_slug]
        for rank, (cand, material) in enumerate(results, 1)
    ]
    first_audio = next((str(mat.path) for _, mat in results if mat is not None), None)
    return summary, rows, first_audio


def run_synthesize_beds(overwrite: bool = False, index_path: Path = config.MATERIALS_INDEX,
                        out_dir: Path = config.BED_DIR) -> str:
    from . import presets as presets_mod

    created = presets_mod.synthesize_beds(out_dir=out_dir, index_path=index_path, overwrite=bool(overwrite))
    if created:
        return f"已合成并入库 {len(created)} 条基础床音：{', '.join(m.id for m in created)}"
    return "基础床音已齐全（room_tone, rain_soft, rain_hard），无需重复生成"


def preview_mined_row(rows: List[List[object]], row_index: int,
                      index_path: Path = config.MATERIALS_INDEX) -> Optional[str]:
    mat = materials.find_index(materials.load_index(index_path), str(rows[row_index][-1]))
    return str(mat.path) if mat is not None and mat.path.is_file() else None


def list_material_categories(only_pending: bool = False,
                             index_path: Path = config.MATERIALS_INDEX) -> List[Tuple[str, str]]:
    items = materials.prune_missing(index_path)
    if only_pending:
        items = [m for m in items if m.is_pending]
    counts = {}
    for material in items:
        counts[material.category] = counts.get(material.category, 0) + 1
    cat_descs = materials.category_descriptions()
    return [(f"全部类别 ({len(items)})", "all")] + [
        (f"{cat_descs.get(cat, cat)} · {cat} ({counts[cat]})", cat) for cat in sorted(counts)
    ]


def list_material_rows(only_pending: bool = False,
                       index_path: Path = config.MATERIALS_INDEX, category: str = "all"
                       ) -> Tuple[List[List[object]], List[str]]:
    items = materials.prune_missing(index_path)
    if only_pending:
        items = [m for m in items if m.is_pending]
    if category != "all":
        items = [m for m in items if m.category == category]
    return (
        [[m.id, m.desc, ",".join(m.tags), m.source, m.duration_sec, m.base_lufs, m.file] for m in items],
        [m.id for m in items],
    )


MATERIAL_PAGE_SIZE = 10


def paginate_material_rows(rows: List[List[object]], page: int = 1,
                           page_size: int = MATERIAL_PAGE_SIZE
                           ) -> Tuple[List[List[object]], int, int, str]:
    total = len(rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    current_page = min(max(1, int(page)), total_pages)
    start = (current_page - 1) * page_size
    return (rows[start:start + page_size], current_page, total_pages,
            f"第 {current_page} / {total_pages} 页（共 {total} 条，每页 {page_size} 条）")


def _require_material(mat_id: str, index_path: Path) -> materials.Material:
    material = materials.find_index(materials.load_index(index_path), mat_id)
    if material is None:
        raise ValueError(f"未找到素材：{mat_id}，请先选择素材")
    return material


def inspect_material(mat_id: str, index_path: Path = config.MATERIALS_INDEX
                     ) -> Tuple[Optional[str], str, str, str, str]:
    m = _require_material(mat_id, index_path)
    return (
        str(m.path) if m.path.is_file() else None, m.id, m.desc, ",".join(m.tags),
        f"时长 {m.duration_sec}s | 采样率 {m.sample_rate}Hz | 声道 {m.channels} | 基准响度 {m.base_lufs} LUFS"
        f" | 来源位置：{m.source or '未记录'}",
    )


def run_ai_assist_identify(mat_id: str, acoustic_hint: str, send_audio: bool,
                           index_path: Path = config.MATERIALS_INDEX,
                           llm_config_path: Path = llm.LLM_CONFIG_PATH
                           ) -> Tuple[str, str, str, str]:
    m = _require_material(mat_id, index_path)
    full_hint = (
        f"id: {m.id}\ndesc: {m.desc}\ntags: {','.join(m.tags)}\n"
        f"source: {m.source}\n"
        f"duration_sec: {m.duration_sec}\nbase_lufs: {m.base_lufs}\n"
        f"用户附加提示：{acoustic_hint}"
    )
    res = llm.assist_identify_material(
        m.path, acoustic_hint=full_hint, send_audio=send_audio,
        cfg=llm.load_llm_config(llm_config_path),
    )
    return res["prefix"], res["desc"], ",".join(res["tags"]), res["reason"]


def apply_material_relabel(mat_id: str, new_prefix: str, new_desc: str, new_tags: str,
                           only_pending: bool = False,
                           index_path: Path = config.MATERIALS_INDEX, category: str = "all"
                           ) -> Tuple[str, List[List[object]], List[str]]:
    status, rows, choices, _, _ = _relabel_material_view(
        mat_id, new_prefix, new_desc, new_tags, only_pending, index_path, category,
    )
    return status, rows, choices


def _relabel_material_view(mat_id: str, new_prefix: str, new_desc: str, new_tags: str,
                           only_pending: bool, index_path: Path, category: str
                           ) -> Tuple[str, List[List[object]], List[str], List[Tuple[str, str]], str]:
    updated = materials.relabel_material(
        mat_id, new_prefix=new_prefix.strip() or None, desc=new_desc.strip() or None,
        tags=[t.strip() for t in new_tags.split(",") if t.strip()] if new_tags.strip() else None,
        index_path=index_path,
    )
    category_choices = list_material_categories(only_pending, index_path)
    category = category if category in {value for _, value in category_choices} else "all"
    rows, choices = list_material_rows(only_pending, index_path, category)
    status = f"已更新：{updated.id} ({updated.desc}) -> {updated.file}"
    return status, rows, choices, category_choices, category


def run_batch_pending_relabel(only_pending: bool = False, index_path: Path = config.MATERIALS_INDEX,
                              category: str = "all", llm_config_path: Path = llm.LLM_CONFIG_PATH
                              ) -> Tuple[str, List[List[object]], List[str], List[Tuple[str, str]], str]:
    updated = llm.batch_relabel_pending_materials(index_path=index_path, llm_config_path=llm_config_path)
    category_choices = list_material_categories(only_pending, index_path)
    category = category if category in {value for _, value in category_choices} else "all"
    rows, choices = list_material_rows(only_pending, index_path, category)
    return f"已批量翻译归类 {len(updated)} 条待标注素材", rows, choices, category_choices, category


def run_script_generation(user_prompt: str, template_id: str,
                           index_path: Path = config.MATERIALS_INDEX,
                           llm_config_path: Path = llm.LLM_CONFIG_PATH,
                           templates_path: Path = llm.SCRIPT_TEMPLATES_PATH) -> str:
    return llm.generate_script(
        user_prompt, materials.load_index(index_path), template_id=template_id,
        cfg=llm.load_llm_config(llm_config_path), templates_path=templates_path,
    )


def save_and_render_script(script_text: str, script_name: str = "webui_custom.asmr",
                           provider: str = "cpu") -> Tuple[str, str, str]:
    script_path = config.EXAMPLES_DIR / Path(script_name).with_suffix(".asmr").name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    os.environ["TINGLELOOM_TTS_PROVIDERS"] = provider
    result = pipeline.run_pipeline(script_path)
    report = result.mix_report
    summary = (
        f"成品 WAV：{report.output_path}\n"
        f"成品 MP3：{report.mp3_path}\n"
        f"时长 {report.duration_seconds:.1f}s | "
        f"采样率 {report.sample_rate}Hz | 响度 {report.loudness_lufs:.1f} LUFS | 峰值 {report.peak:.3f}\n"
        f"素材缺口：{', '.join(result.gaps) or '无'}\n"
        f"警告：{'; '.join(result.warnings) or '无'}"
    )
    return str(report.output_path), str(report.mp3_path), summary


def _api_key_placeholder(cfg: llm.LLMConfig) -> str:
    redacted = llm.redact_api_key(cfg.api_key)
    return (f"已配置 {redacted}（留空保持不变，输入新值则覆盖）" if redacted else
            f"未配置（输入密钥或使用环境变量 {llm.PROVIDER_PRESETS[cfg.provider]['env_prefix']}_API_KEY）")


def inspect_llm_provider(provider: str, path: Path = llm.LLM_CONFIG_PATH
                         ) -> Tuple[str, str, str, str, str, str, float]:
    cfg = llm.load_llm_config(path, provider=provider)
    return cfg.provider, cfg.base_url, "", _api_key_placeholder(cfg), cfg.model, cfg.audio_model, cfg.temperature


def update_llm_settings(base_url: str, api_key: str, model: str, audio_model: str,
                        temperature: float, path: Path = llm.LLM_CONFIG_PATH,
                        provider: str = "openai") -> str:
    saved = llm.save_llm_config(llm.LLMConfig(base_url, api_key, model, audio_model, temperature, provider), path)
    loaded = llm.load_llm_config(path, provider=provider)
    masked = llm.redact_api_key(loaded.api_key) or "未配置"
    channel = "" if loaded.provider == "openai" else f" [{loaded.provider}]"
    return f"API 配置已保存{channel}：{saved}（密钥状态：{masked}）"


def update_template_item(template_id: str, name: str, description: str,
                         system_prompt: str, user_prompt_template: str,
                         path: Path = llm.SCRIPT_TEMPLATES_PATH) -> str:
    template_id = template_id.strip()
    name = name.strip()
    if not template_id or not name:
        raise ValueError("模板 ID 和模板名称不能为空")
    templates = llm.load_templates(path)
    updated = llm.ScriptTemplate(template_id, name, description, system_prompt, user_prompt_template)
    for i, template in enumerate(templates):
        if template.id == template_id:
            templates[i] = updated
            break
    else:
        templates.append(updated)
    saved = llm.save_templates(templates, path)
    return f"模板 {template_id} 已保存：{saved}"


def build_webui() -> gr.Blocks:
    settings = llm.load_llm_config()
    templates = llm.load_templates()
    template_choices = [(t.name, t.id) for t in templates]
    first_template = templates[0]
    rows, choices = list_material_rows()
    initial_rows, initial_page, _, initial_page_label = paginate_material_rows(rows)

    # 事件适配仅转换 Gradio 组件更新；业务读写仍走上面的处理器。
    def material_view_updates(current_rows, current_choices, category_choices, category, page):
        page_rows, current_page, _, page_label = paginate_material_rows(current_rows, page)
        return (page_rows, gr.Dropdown(choices=current_choices, value=None), None, "", "", "", "", "", "",
                gr.Dropdown(choices=category_choices, value=category), current_page, page_label)

    def refresh_materials(only_pending, category="all", page=1):
        category_choices = list_material_categories(only_pending)
        category = category if category in {value for _, value in category_choices} else "all"
        current_rows, current_choices = list_material_rows(only_pending, category=category)
        return material_view_updates(current_rows, current_choices, category_choices, category, page)

    def select_material(mat_id):
        return (*inspect_material(mat_id), "", "")

    def on_material_row_select(table_rows, evt: gr.SelectData):
        idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else int(evt.index)
        mat_id = str(table_rows[idx][0])
        return (mat_id, *select_material(mat_id))

    def change_material_page(only_pending, category, page, direction):
        all_rows, _ = list_material_rows(only_pending, category=category)
        page_rows, current_page, _, page_label = paginate_material_rows(all_rows, page + direction)
        return page_rows, current_page, page_label

    def relabel_and_refresh(mat_id, new_prefix, new_desc, new_tags, only_pending, category="all", page=1):
        status, current_rows, current_choices, category_choices, category = _relabel_material_view(
            mat_id, new_prefix, new_desc, new_tags, only_pending, config.MATERIALS_INDEX, category,
        )
        return (status, *material_view_updates(current_rows, current_choices, category_choices, category, page))

    def batch_relabel_and_refresh(only_pending, category="all", page=1):
        status, current_rows, current_choices, category_choices, category = run_batch_pending_relabel(
            only_pending, category=category,
        )
        return (status, *material_view_updates(current_rows, current_choices, category_choices, category, page))

    def synth_beds_and_refresh(only_pending, category="all", page=1):
        status = run_synthesize_beds()
        return (status, *refresh_materials(only_pending, category, page))

    def select_template(template_id):
        for template in llm.load_templates():
            if template.id == template_id:
                return (template.id, template.name, template.description,
                        template.system_prompt, template.user_prompt_template)
        raise ValueError(f"未知台本模板：{template_id}")

    def save_template_and_refresh(template_id, name, description, system_prompt, user_prompt_template):
        status = update_template_item(template_id, name, description, system_prompt, user_prompt_template)
        return status, gr.Dropdown(
            choices=[(t.name, t.id) for t in llm.load_templates()], value=template_id.strip(),
        )

    with gr.Blocks(title="Tingleloom ASMR 工作台") as app:
        gr.Markdown("# Tingleloom ASMR 工作台\n从原音频提取、人工确认到 AI 台本与成品合成。")
        with gr.Tab("1. 素材自动提取与导入"):
            source_path = gr.Textbox(
                label="源音频 / 视频路径",
                placeholder="输入本地音频或视频路径（如 .mp4 / .m4a / .mp3 / .wav），或在下方上传",
            )
            upload = gr.File(
                label="上传源音频 / 视频",
                type="filepath",
                file_types=["audio", "video"],
            )
            upload.upload(lambda path: path, upload, source_path)
            with gr.Row():
                profile = gr.Dropdown(["high", "low", "balanced", "bed"], value="high", label="声学 profile")
                slice_seconds = gr.Number(value=35, precision=0, minimum=1, label="切片时长（秒）")
                min_slice_seconds = gr.Number(value=10, precision=0, minimum=1, label="最短捡漏时长（秒）")
                top_n = gr.Number(value=15, precision=0, minimum=1, label="候选数量上限（低命中率自动淘汰）")
            with gr.Row():
                prefix = gr.Textbox(value="auto", label="英文前缀（auto 自动识别）")
                license_name = gr.Textbox(label="来源授权 / License")
                probe = gr.Checkbox(value=False, label="仅探测候选（不写入素材库）")
            hints_box = gr.Textbox(
                label="人类时间戳 / 章节监督提示（可选，任意格式，用于引导提取与增量学习）", lines=4,
                placeholder="可直接粘贴 B站/YouTube 时间轴或自由输入，例如：\n00:00-01:40 预览说话（跳过）\n01:40-14:21 耳边吹气\n14:22--18:37 长火柴燃烧\n35分-42分 气泡水喷淋",
            )
            mine_button = gr.Button("开始提取并自动切分入库", variant="primary")
            mine_summary = gr.Markdown()
            mine_table = gr.Dataframe(
                headers=["排名", "时间段", "声道", "RMS dB", "VAD 比例 / 连续时长", "CLAP 判断", "5 秒轨迹", "素材 ID"],
                interactive=False, type="array",
            )
            mine_audio = gr.Audio(label="候选素材试听（点击上方表格任意行切换试听）", type="filepath")

            def on_mine_row_select(rows, evt: gr.SelectData):
                idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else int(evt.index)
                return preview_mined_row(rows, idx)

            mine_table.select(on_mine_row_select, [mine_table], [mine_audio])

        with gr.Tab("2. 人工试听打标 & 强力模型辅助鉴别") as materials_tab:
            with gr.Row():
                only_pending = gr.Checkbox(value=False, label="仅显示待标注 (unknown_trigger)")
                category_filter = gr.Dropdown(choices=list_material_categories(), value="all", label="按类别筛选")
            with gr.Row():
                refresh_button = gr.Button("刷新素材库")
                synth_bed_button = gr.Button("一键合成基础床音 (room_tone / rain)")
                batch_relabel_button = gr.Button("一键 AI 批量翻译归类待标注素材", variant="primary")
            material_table = gr.Dataframe(
                value=initial_rows, type="array", headers=["ID", "描述", "标签", "来源与原片位置", "时长（秒）", "基准 LUFS", "文件"], interactive=False,
            )
            with gr.Row():
                prev_page_button = gr.Button("上一页")
                page_status = gr.Markdown(initial_page_label)
                next_page_button = gr.Button("下一页")
            page_state = gr.State(initial_page)
            selected = gr.Dropdown(choices=choices, label="选择素材")
            preview = gr.Audio(label="素材试听", type="filepath")
            current_id = gr.Textbox(label="当前素材 ID", interactive=False)
            metadata = gr.Textbox(label="声学信息", interactive=False)
            hint = gr.Textbox(label="附加声学 / CLAP 提示", lines=2)
            send_audio = gr.Checkbox(value=True, label="发送原音频给模型（截取中间最多 10 秒；取消则仅发送提示）")
            assist_button = gr.Button("召唤强力模型听音辅助鉴别")
            reason = gr.Textbox(label="AI 鉴别理由", lines=3, interactive=False)
            with gr.Row():
                new_prefix = gr.Textbox(label="新英文前缀（留空保持）")
                new_desc = gr.Textbox(label="新中文描述")
                new_tags = gr.Textbox(label="标签（英文逗号分隔）")
            relabel_button = gr.Button("确认改名并更新素材库", variant="primary")
            relabel_status = gr.Textbox(label="更新结果", interactive=False)
            selected.input(select_material, selected,
                           [preview, current_id, new_desc, new_tags, metadata, new_prefix, reason])
            material_table.select(on_material_row_select, material_table,
                                  [selected, preview, current_id, new_desc, new_tags, metadata, new_prefix, reason])
            for button, direction in ((prev_page_button, -1), (next_page_button, 1)):
                button.click(change_material_page, [only_pending, category_filter, page_state, gr.State(direction)],
                             [material_table, page_state, page_status])
            for event in (refresh_button.click, only_pending.change, category_filter.change, materials_tab.select):
                event(refresh_materials, [only_pending, category_filter, page_state],
                      [material_table, selected, preview, current_id, new_desc, new_tags, metadata, new_prefix, reason, category_filter, page_state, page_status])
            assist_button.click(run_ai_assist_identify, [selected, hint, send_audio],
                                [new_prefix, new_desc, new_tags, reason])
            synth_bed_button.click(synth_beds_and_refresh, [only_pending, category_filter, page_state],
                                   [relabel_status, material_table, selected, preview, current_id,
                                    new_desc, new_tags, metadata, new_prefix, reason, category_filter, page_state, page_status])
            batch_relabel_button.click(batch_relabel_and_refresh, [only_pending, category_filter, page_state],
                                       [relabel_status, material_table, selected, preview, current_id,
                                        new_desc, new_tags, metadata, new_prefix, reason, category_filter, page_state, page_status])
            relabel_button.click(relabel_and_refresh,
                                 [selected, new_prefix, new_desc, new_tags, only_pending, category_filter, page_state],
                                  [relabel_status, material_table, selected, preview, current_id,
                                   new_desc, new_tags, metadata, new_prefix, reason, category_filter, page_state, page_status])

        def mine_and_refresh(audio_file, profile, slice_seconds, min_slice_seconds,
                             top_n, prefix, license_name, probe_only, only_pending_flag, hints_text="", category="all", page=1):
            summary, mined_rows, first_audio = run_mine_task(
                audio_file, profile, slice_seconds, min_slice_seconds,
                top_n, prefix, license_name, probe_only, hints_text,
            )
            refreshed = refresh_materials(only_pending_flag, category, page)
            return summary, mined_rows, first_audio, refreshed[0], refreshed[1], refreshed[9], refreshed[10], refreshed[11]

        mine_button.click(
            mine_and_refresh,
            [source_path, profile, slice_seconds, min_slice_seconds, top_n, prefix, license_name, probe, only_pending, hints_box, category_filter, page_state],
            [mine_summary, mine_table, mine_audio, material_table, selected, category_filter, page_state, page_status],
        )

        with gr.Tab("3. AI 台本生成、提示词模板 & 一键合成"):
            template_selector = gr.Dropdown(template_choices, value=first_template.id, label="选择台本模板")
            user_prompt = gr.Textbox(label="用户提示词", lines=4, placeholder="节目主题、时长、触发音偏好……")
            generate_button = gr.Button("AI 生成 .asmr 台本", variant="primary")
            script_text = gr.Code(label="可编辑 .asmr 台本", language=None, lines=20)
            generate_button.click(run_script_generation, [user_prompt, template_selector], script_text)
            with gr.Accordion("提示词模板管理", open=False):
                gr.Markdown("### 提示词模板\n选择上方模板以编辑；填写新的模板 ID 可新增。占位符：`{{user_prompt}}` / `{{materials_catalog}}`。")
                with gr.Row():
                    template_id = gr.Textbox(value=first_template.id, label="模板 ID")
                    template_name = gr.Textbox(value=first_template.name, label="模板名称")
                description = gr.Textbox(value=first_template.description, label="模板说明")
                system_prompt = gr.Textbox(value=first_template.system_prompt, label="system_prompt", lines=6)
                user_prompt_template = gr.Textbox(value=first_template.user_prompt_template,
                                                 label="user_prompt_template", lines=6)
                save_template = gr.Button("保存到 script_templates.yaml")
                template_status = gr.Textbox(label="模板保存结果", interactive=False)
                template_selector.change(select_template, template_selector,
                                         [template_id, template_name, description, system_prompt, user_prompt_template])
                save_template.click(save_template_and_refresh,
                                    [template_id, template_name, description, system_prompt, user_prompt_template],
                                    [template_status, template_selector])
            with gr.Row():
                script_name = gr.Textbox(value="webui_custom.asmr", label="保存台本文件名（examples 目录）")
                provider = gr.Dropdown(["cpu", "cuda", "hybrid"], value="cpu", label="TTS 执行后端")
            render_button = gr.Button("一键合成成品（WAV + 320k MP3）", variant="primary")
            with gr.Row():
                output_audio = gr.Audio(label="成品 WAV（无损 PCM_16）", type="filepath")
                output_mp3 = gr.Audio(label="成品 MP3（320kbps 双耳立体声）", type="filepath")
            render_summary = gr.Textbox(label="合成报告", lines=4, interactive=False)
            render_button.click(save_and_render_script, [script_text, script_name, provider],
                                [output_audio, output_mp3, render_summary], concurrency_limit=1)
        with gr.Tab("4. API 与多协议模型配置"):
            provider_selector = gr.Dropdown(
                [(info["label"], key) for key, info in llm.PROVIDER_PRESETS.items()],
                value=settings.provider, label="API 协议格式 / Provider",
            )
            base_url = gr.Textbox(value=settings.base_url, label="base_url")
            api_key = gr.Textbox(
                value="", placeholder=_api_key_placeholder(settings),
                label="api_key（脱敏保护，留空保持原配置）",
                type="password",
            )
            with gr.Row():
                model = gr.Textbox(value=settings.model, label="model（文本生成）")
                audio_model = gr.Textbox(value=settings.audio_model, label="audio_model（听音鉴别）")
                temperature = gr.Slider(0, 2, value=settings.temperature, step=0.05, label="temperature")
            save_settings = gr.Button("保存 API 配置")
            settings_status = gr.Textbox(label="API 保存结果", interactive=False)

            def inspect_provider(provider):
                provider, base, key, placeholder, text_model, listen_model, temp = inspect_llm_provider(provider)
                return provider, base, gr.Textbox(value=key, placeholder=placeholder), text_model, listen_model, temp

            def save_provider_settings(base, key, text_model, listen_model, temp, provider):
                status = update_llm_settings(base, key, text_model, listen_model, temp, provider=provider)
                _, _, key, placeholder, _, _, _ = inspect_llm_provider(provider)
                return status, gr.Textbox(value=key, placeholder=placeholder)

            provider_selector.change(inspect_provider, provider_selector,
                                     [provider_selector, base_url, api_key, model, audio_model, temperature])
            save_settings.click(save_provider_settings,
                                [base_url, api_key, model, audio_model, temperature, provider_selector],
                                [settings_status, api_key])
    return app


def launch_webui(host: str = "127.0.0.1", port: int = 7860,
                 share: bool = False, inbrowser: bool = True):
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for local in ("127.0.0.1", "localhost"):
            if local not in existing:
                existing.append(local)
        os.environ[key] = ",".join(existing)
    return build_webui().queue().launch(
        server_name=host, server_port=port, share=share, inbrowser=inbrowser,
    )

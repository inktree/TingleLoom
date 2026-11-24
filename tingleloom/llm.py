"""多协议 API 配置、台本模板与生成、素材听音辅助鉴别。"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import yaml

from . import audio, config, materials as materials_mod, presets as presets_mod

LLM_CONFIG_PATH = config.SECRETS_DIR / "llm.yaml"
SCRIPT_TEMPLATES_PATH = config.PRESETS_DIR / "script_templates.yaml"


PROVIDER_PRESETS = {
    "openai": dict(label="OpenAI Chat Completions 兼容 (/chat/completions)", base_url="https://api.openai.com/v1", model="gpt-4o-mini", audio_model="gpt-4o-audio-preview", env_prefix="OPENAI", supports_audio=True),
    "openai-responses": dict(label="OpenAI Responses API (/responses)", base_url="https://api.openai.com/v1", model="gpt-4o-mini", audio_model="gpt-4o-audio-preview", env_prefix="OPENAI", supports_audio=True),
    "deepseek": dict(label="DeepSeek Chat Completions (/chat/completions)", base_url="https://api.deepseek.com/v1", model="deepseek-chat", audio_model="deepseek-chat", env_prefix="DEEPSEEK", supports_audio=False),
    "anthropic": dict(label="Anthropic Claude Messages (/v1/messages)", base_url="https://api.anthropic.com", model="claude-3-5-sonnet-latest", audio_model="claude-3-5-sonnet-latest", env_prefix="ANTHROPIC", supports_audio=False),
    "gemini": dict(label="Google Gemini Native (:generateContent)", base_url="https://generativelanguage.googleapis.com", model="gemini-2.5-flash", audio_model="gemini-2.5-flash", env_prefix="GEMINI", supports_audio=True),
}


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    audio_model: str = "gpt-4o-audio-preview"
    temperature: float = 0.7
    provider: str = "openai"


def _provider_block(raw: dict, provider: str) -> dict:
    return raw.get("providers", {}).get(provider, raw.get(provider, {}))


def load_llm_config(path: Path = LLM_CONFIG_PATH, provider: Optional[str] = None) -> LLMConfig:
    """加载独立渠道；旧版扁平配置仍作为当前渠道的回退基线。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("LLM 配置 YAML 顶层必须是映射对象")
    active = (provider or raw.get("provider") or "openai").strip().lower()
    if active not in PROVIDER_PRESETS:
        raise ValueError(f"不支持的 LLM 协议格式：{active}")
    preset = PROVIDER_PRESETS[active]
    block = _provider_block(raw, active)
    use_flat = provider is None or active == (raw.get("provider") or "openai")
    values = {}
    for key in ("base_url", "api_key", "model", "audio_model", "temperature"):
        value = block.get(key)
        if value in (None, "") and use_flat:
            value = raw.get(key)
        if isinstance(value, str):
            value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: os.environ.get(match[1], ""), value)
        if key == "temperature":
            values[key] = 0.7 if value in (None, "") else float(value)
        else:
            env_suffix = key.upper()
            values[key] = value or os.environ.get(f"{preset['env_prefix']}_{env_suffix}", "") or os.environ.get(f"OPENAI_{env_suffix}", "") or preset.get(key, "")
    return LLMConfig(**values, provider=active)


def redact_api_key(api_key: str) -> str:
    k = api_key.strip()
    if not k:
        return ""
    if k.startswith("${") and k.endswith("}") and len(k) > 3:
        return f"<env:{k[2:-1].strip()}>"
    return "****" if len(k) <= 4 else f"****{k[-4:]}"


def save_llm_config(cfg: LLMConfig, path: Path = LLM_CONFIG_PATH) -> Path:
    provider = cfg.provider.strip().lower() or "openai"
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"不支持的 LLM 协议格式：{provider}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(raw, dict):
        raw = {}
    key = cfg.api_key.strip()
    if not key or key.startswith("****") or key.startswith("<env:"):
        block = _provider_block(raw, provider)
        key = block.get("api_key", "")
        if not key and provider == (raw.get("provider") or "openai"):
            key = raw.get("api_key", "")
    payload = {"base_url": cfg.base_url.strip(), "api_key": key,
               "model": cfg.model.strip(), "audio_model": cfg.audio_model.strip(),
               "temperature": float(cfg.temperature)}
    providers = raw.setdefault("providers", {})
    previous_provider = (raw.get("provider") or "openai").strip().lower()
    if previous_provider != provider and previous_provider not in providers and any(field in raw for field in payload):
        previous = _provider_block(raw, previous_provider)
        providers[previous_provider] = {field: raw[field] for field in payload if field in raw}
        providers[previous_provider].update({field: value for field, value in previous.items() if value not in (None, "")})
    providers[provider] = payload
    raw.update(payload, provider=provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@dataclass
class ScriptTemplate:
    id: str
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str


def load_templates(path: Path = SCRIPT_TEMPLATES_PATH) -> List[ScriptTemplate]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("台本模板 YAML 顶层必须是列表")
    return [ScriptTemplate(**item) for item in raw]


def save_templates(templates: List[ScriptTemplate], path: Path = SCRIPT_TEMPLATES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump([asdict(template) for template in templates], allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


DEFAULT_MAX_CATALOG_ITEMS = 24
DEFAULT_MAX_PER_CATEGORY = 3


def _score_for_script(material: materials_mod.Material, query: str, category_desc: str) -> float:
    if not query.strip():
        return 0.0
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query)
    score = materials_mod.score_material(material, query)
    for tok in tokens:
        score += materials_mod.score_material(material, tok)
        if len(tok) >= 4 and re.fullmatch(r"[\u4e00-\u9fff]+", tok):
            score += sum(materials_mod.score_material(material, tok[i:i + 2]) for i in range(len(tok) - 1))
    query_norm = materials_mod._norm(query)
    for name in (category_desc, material.category):
        norm = materials_mod._norm(name)
        if norm and norm in query_norm:
            score += 3.0
    return score


def _is_bed_material(material: materials_mod.Material) -> bool:
    return (material.file.replace("\\", "/").startswith("bed/")
            or material.path.parent == config.BED_DIR
            or material.id in presets_mod.SYNTH_BED_SPECS
            or "环境音" in material.tags or "床音" in material.tags)


def retrieve_materials_for_script(
    materials: List[materials_mod.Material], query: str = "",
    max_items: int = DEFAULT_MAX_CATALOG_ITEMS,
    max_per_category: int = DEFAULT_MAX_PER_CATEGORY,
) -> List[materials_mod.Material]:
    ready = [m for m in materials if not m.is_pending and m.path.is_file() and not _is_bed_material(m)]
    cat_descs = materials_mod.category_descriptions()
    return _select_ready_materials(ready, cat_descs, query, max_items, max_per_category)


def _select_ready_materials(
    ready: List[materials_mod.Material], cat_descs: dict[str, str], query: str,
    max_items: int, max_per_category: int,
) -> List[materials_mod.Material]:
    if not ready or max_items <= 0 or max_per_category <= 0:
        return []
    groups = {}
    for material in ready:
        groups.setdefault(material.category, []).append(material)
    queues = {}
    scores = {}
    for cat, group in groups.items():
        ranked = sorted(
            ((_score_for_script(m, query, cat_descs.get(cat, cat)), m) for m in group),
            key=lambda pair: (-pair[0], -pair[1].duration_sec, pair[1].id),
        )
        scores[cat] = ranked[0][0]
        seen_channels = set()
        representatives, remaining = [], []
        for _, material in ranked:
            suffix = material.id.rsplit("_", 1)[-1]
            channel = suffix if suffix in materials_mod.CHANNEL_DESC else "C"
            if channel not in seen_channels:
                representatives.append(material)
                seen_channels.add(channel)
            else:
                remaining.append(material)
        queues[cat] = (representatives + remaining)[:max_per_category]
    categories = sorted(queues, key=lambda cat: (-scores[cat], cat))
    selected = []
    for cat in categories:
        if scores[cat] > 0:
            selected.extend(queues[cat][:max_items - len(selected)])
            if len(selected) == max_items:
                return selected
    unmatched = [cat for cat in categories if scores[cat] == 0]
    for rank in range(max_per_category):
        for cat in unmatched:
            if rank < len(queues[cat]):
                selected.append(queues[cat][rank])
                if len(selected) == max_items:
                    return selected
    return selected


def format_available_beds(
    materials: List[materials_mod.Material], beds_path: Optional[Path] = None,
) -> str:
    presets = presets_mod.load_beds(beds_path)
    ready_ids = {m.id for m in materials if not m.is_pending and m.path.is_file()}
    lines = []
    for preset in presets.values():
        if preset.layers and all(layer.material in ready_ids for layer in preset.layers):
            layer_ids = ", ".join(layer.material for layer in preset.layers)
            offset = preset.offset_db if preset.offset_db is not None else config.BED_OFFSET_DB
            lines.append(f"- @bed {preset.name}（层素材: {layer_ids}, 低于节目 {offset:g}dB）")
    return "\n".join(lines)


def format_materials_catalog(
    materials: List[materials_mod.Material], query: str = "",
    max_items: int = DEFAULT_MAX_CATALOG_ITEMS,
    max_per_category: int = DEFAULT_MAX_PER_CATEGORY,
) -> str:
    ready = [m for m in materials if not m.is_pending and m.path.is_file() and not _is_bed_material(m)]
    cat_descs = materials_mod.category_descriptions()
    selected = _select_ready_materials(ready, cat_descs, query, max_items, max_per_category)
    groups = {}
    for material in selected:
        groups.setdefault(material.category, []).append(material)
    lines = []
    if len(selected) < len(ready):
        lines.append(f"# 已从素材库 {len({m.category for m in ready})} 类 {len(ready)} 条可用素材中按主题检索出 {len(selected)} 条代表素材")
    for cat, group in groups.items():
        lines.append(f"[{cat_descs.get(cat, cat)} ({cat})]")
        lines.extend(
            f"- id: {material.id} | desc: {material.desc} | "
            f"duration_sec: {material.duration_sec:g} | tags: {', '.join(material.tags)}"
            for material in group
        )
    catalog = "\n".join(lines) if selected else "（当前素材库为空）"
    beds_text = format_available_beds(materials)
    if beds_text:
        catalog += "\n\n[可用环境床音预设 (@bed，仅允许使用以下预设名，不写进 sfx)]\n" + beds_text
    return catalog


def chat_completion(messages: list, model: str, cfg: LLMConfig) -> str:
    if not cfg.api_key:
        raise ValueError("未配置 LLM API Key（请在前端 API 配置面板、secrets/llm.yaml 或环境变量 OPENAI_API_KEY 中设置）")
    base = cfg.base_url.rstrip("/")
    provider = cfg.provider
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"不支持的 LLM 协议格式：{provider}")
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    if provider in ("openai", "deepseek"):
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        payload = {"messages": messages, "model": model, "temperature": cfg.temperature}
    else:
        system = "\n".join(msg["content"] for msg in messages if msg["role"] == "system")
        conversation = [msg for msg in messages if msg["role"] != "system"]
        if provider == "openai-responses":
            url = base if base.endswith("/responses") else base + "/responses"
            input_items = []
            for msg in conversation:
                content = msg["content"]
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                converted = []
                for part in parts:
                    if part["type"] == "text":
                        converted.append({"type": "input_text", "text": part["text"]})
                    elif part["type"] == "input_audio":
                        converted.append({"type": "input_audio", **part["input_audio"]})
                input_items.append({"type": "message", "role": msg["role"], "content": converted})
            payload = {"model": model, "input": input_items, "temperature": cfg.temperature}
            if system:
                payload["instructions"] = system
        elif provider == "anthropic":
            url = base if base.endswith("/messages") else base + ("/messages" if base.endswith("/v1") else "/v1/messages")
            headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            converted = []
            for msg in conversation:
                content = msg["content"]
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                converted.append({"role": msg["role"], "content": [{"type": "text", "text": part["text"]} for part in parts if part["type"] == "text"]})
            payload = {"model": model, "max_tokens": 4096, "temperature": cfg.temperature, "messages": converted}
            if system:
                payload["system"] = system
        else:  # Gemini 原生多模态协议
            prefix = base if base.endswith("/v1beta") else base + "/v1beta"
            url = f"{prefix}/models/{model}:generateContent"
            headers = {"x-goog-api-key": cfg.api_key, "Content-Type": "application/json"}
            contents = []
            for msg in conversation:
                content = msg["content"]
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                converted = []
                for part in parts:
                    if part["type"] == "text":
                        converted.append({"text": part["text"]})
                    elif part["type"] == "input_audio":
                        data = part["input_audio"]
                        converted.append({"inlineData": {"mimeType": f"audio/{data['format']}", "data": data["data"]}})
                contents.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": converted})
            payload = {"contents": contents, "generationConfig": {"temperature": cfg.temperature}}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败 ({exc.code})：{detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"LLM 请求失败：{exc}") from exc
    if provider in ("openai", "deepseek"):
        return result["choices"][0]["message"]["content"]
    if provider == "openai-responses":
        if isinstance(result.get("output_text"), str) and result["output_text"]:
            return result["output_text"]
        return "".join(part.get("text", "") for item in result.get("output", []) if item.get("type") == "message" for part in item.get("content", []))
    if provider == "anthropic":
        return "".join(block["text"] for block in result["content"] if block.get("type") == "text")
    return "".join(part.get("text", "") for part in result["candidates"][0]["content"]["parts"] if part.get("thought") is not True)


def _strip_fence(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```[^\n]*\n(.*?)\n```", text, flags=re.DOTALL)
    return fence[1].strip() if fence else text


def generate_script(
    user_prompt: str,
    materials: List[materials_mod.Material],
    template_id: str = "sleep_coaching",
    cfg: Optional[LLMConfig] = None,
    templates_path: Path = SCRIPT_TEMPLATES_PATH,
) -> str:
    templates = load_templates(templates_path)
    for template in templates:
        if template.id == template_id:
            break
    else:
        raise ValueError(f"未知台本模板：{template_id}")
    catalog = format_materials_catalog(materials, query=f"{user_prompt}\n{template.name}\n{template.description}")
    user_content = template.user_prompt_template.replace("{{user_prompt}}", user_prompt).replace("{{materials_catalog}}", catalog)
    cfg = load_llm_config() if cfg is None else cfg
    return _strip_fence(chat_completion([
        {"role": "system", "content": template.system_prompt},
        {"role": "user", "content": user_content},
    ], cfg.model, cfg))


def assist_identify_material(
    audio_path: Path,
    acoustic_hint: str = "",
    send_audio: bool = True,
    cfg: Optional[LLMConfig] = None,
) -> dict:
    """发送中间最多十秒的音频，或仅据本地声学/CLAP 提示提出标签建议。"""
    cfg = load_llm_config() if cfg is None else cfg
    effective_send_audio = bool(send_audio and PROVIDER_PRESETS[cfg.provider]["supports_audio"])
    prompt_text = (
        "请辅助鉴别 ASMR 触发音素材的道具与动作，不把候选提示当成确定事实。"
        "只返回 JSON 对象，不加解释："
        '{"prefix":"英文小写字母、数字与下划线前缀","desc":"中文简短道具描述",'
        '"tags":["标签1","标签2"],"reason":"听感与声学判断依据"}。'
        "prefix 仅使用英文小写字母、数字与下划线，desc 和 reason 非空。"
        + ("根据上传的音频与提示综合判断。" if effective_send_audio else "本次未上传音频，只据提示推断并在 reason 中说明不确定性。")
        + f"\n声学特征与 CLAP 候选提示：\n{acoustic_hint}"
    )
    if effective_send_audio:
        data, sr = audio.read_audio(audio_path, target_sr=16000)
        length = min(len(data), 10 * sr)
        if length == 0:
            raise ValueError("无法鉴别空音频")
        start = (len(data) - length) // 2
        segment = data[start:start + length]
        channel = np.argmax(np.mean(segment ** 2, axis=0))
        wav = io.BytesIO()
        sf.write(wav, segment[:, channel], sr, format="WAV", subtype="PCM_16")
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav.getvalue()).decode("ascii"), "format": "wav",
            }},
        ]
        model = cfg.audio_model or cfg.model
    else:
        content = prompt_text
        model = cfg.model
    response = chat_completion([{"role": "user", "content": content}], model, cfg)
    try:
        result = json.loads(_strip_fence(response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"素材鉴别返回了非法 JSON：{exc}") from exc
    required = {"prefix", "desc", "tags", "reason"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("素材鉴别结果必须是含 prefix、desc、tags、reason 的 JSON 对象")
    if not isinstance(result["prefix"], str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", result["prefix"]):
        raise ValueError("素材鉴别 prefix 必须是英文小写字母、数字与下划线前缀")
    if any(not isinstance(result[key], str) or not result[key].strip() for key in ("desc", "reason")):
        raise ValueError("素材鉴别 desc 和 reason 必须是非空字符串")
    if not isinstance(result["tags"], list) or any(not isinstance(tag, str) for tag in result["tags"]):
        raise ValueError("素材鉴别 tags 必须是字符串列表")
    return result


def classify_timeline_labels(labels: List[str], source_hint: str = "",
                             cfg: Optional[LLMConfig] = None) -> dict[str, dict]:
    """结合原片主题，将未知章节批量翻译为声学类别；未配密钥时保留本地结果。"""
    unique_labels = list(dict.fromkeys(label.strip() for label in labels if label.strip()))
    if not unique_labels:
        return {}
    cfg = load_llm_config() if cfg is None else cfg
    if not cfg.api_key:
        return {}
    categories = {slug: desc for slug, desc in materials_mod.category_descriptions().items() if slug != "unknown_trigger"}
    response = chat_completion([
        {"role": "system", "content": (
            "你是 ASMR 道具声学分类助手。结合原片主题，将章节标签翻译并按道具的物理声学本质归类，而非按食品外观命名。"
            "例如用木头和剃须膏制作零食：木製マカロン、木製フィナンシェ属于木块敲击摩擦；"
            "コルクデコレーションケーキ、ミニコルク生クリームケーキ属于软木/泡沫刮擦；"
            "五色のクリーム作り属于剃须膏搅拌挤压；絵の具アイスチョコ属于颜料刷涂。"
            "优先复用贴合声学本质的已知 slug，否则创建英文小写数字下划线 slug，不使用 unknown_trigger。"
            "desc 必须是简短规范中文道具动作描述，不含序号和声道；tags 为中文标签字符串列表。"
            "说话、片头、预览章节设 exclude 为 true。仅返回 JSON 对象，键必须是原始标签："
            '{"原始标签":{"slug":"wood_tapping","desc":"木块敲击摩擦","tags":["木块","轻敲"],"exclude":false}}'
        )},
        {"role": "user", "content": json.dumps({"source_hint": source_hint, "labels": unique_labels, "known_categories": categories}, ensure_ascii=False)},
    ], cfg.model, cfg)
    try:
        result = json.loads(_strip_fence(response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"章节归类返回了非法 JSON：{exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("章节归类结果必须是以原始标签为键的 JSON 对象")
    classified = {}
    for label in unique_labels:
        item = result.get(label)
        if not isinstance(item, dict):
            continue
        slug, desc, tags = item.get("slug"), item.get("desc"), item.get("tags")
        exclude = bool(item.get("exclude", False))
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", slug) or (not exclude and slug == "unknown_trigger"):
            continue
        if not isinstance(desc, str) or not desc.strip() or not re.search(r"[\u4e00-\u9fff]", desc):
            continue
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            continue
        classified[label] = {"slug": slug, "desc": desc.strip(), "tags": tags, "exclude": exclude}
    return classified


def batch_relabel_pending_materials(index_path: Path = config.MATERIALS_INDEX,
                                    llm_config_path: Path = LLM_CONFIG_PATH
                                    ) -> List[materials_mod.Material]:
    """一次请求翻译待标注线索，沿已有改名与原型学习路径转正有效项。"""
    items = materials_mod.prune_missing(index_path)
    pending = [m for m in items if m.is_pending]
    if not pending:
        return []
    cfg = load_llm_config(llm_config_path)
    if not cfg.api_key:
        raise ValueError("未配置 LLM API Key，无法执行批量翻译归类")
    labels = {}
    sources = {}
    for m in pending:
        label = re.sub(r"\d+_(?:左耳|右耳|双耳均衡|左右交替|右左交替)$", "", m.desc).strip()
        if not label or label == "待标注触发音":
            label = "、".join(tag for tag in m.tags if tag != "待标注") or m.id
        labels[m.id] = label
        sources.setdefault(label, []).append(m.source.split(" @ ", 1)[0])
    source_hint = json.dumps({label: list(dict.fromkeys(names)) for label, names in sources.items()}, ensure_ascii=False)
    classified = classify_timeline_labels(list(labels.values()), source_hint=source_hint, cfg=cfg)
    updated = []
    for i, m in enumerate(items):
        if not m.is_pending:
            continue
        item = classified.get(labels[m.id])
        if item is None or item["exclude"]:
            continue
        merged_tags = list(dict.fromkeys([*item["tags"], *(t for t in m.tags if t != "待标注")])) or None
        renamed = materials_mod.relabel_material(m.id, new_prefix=item["slug"], desc=f"{item['desc']}01_双耳均衡",
                                                  tags=merged_tags, index_path=index_path)
        items[i] = renamed
        updated.append(renamed)
    return updated


def parse_timeline_hints_text(raw_text: str, duration_sec: float, cfg: Optional[LLMConfig] = None) -> str:
    cfg = cfg if cfg is not None else load_llm_config()
    if not cfg.api_key:
        raise ValueError("未能从时间戳提示中解析出有效时间区间，请使用 MM:SS-MM:SS 描述 格式或配置 LLM API Key")
    return chat_completion([
        {"role": "system", "content": "将人类时间轴描述转换为每行 MM:SS-MM:SS 标签 的标准区间文本（超过一小时可用 HH:MM:SS）。只输出时间轴，保留跳过、说话、预览等排除标签，不编造未提及的章节。"},
        {"role": "user", "content": f"音视频总时长：{duration_sec} 秒。\n{raw_text}"},
    ], cfg.model, cfg)

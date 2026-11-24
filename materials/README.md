# 素材库（`materials/`）

存放本地挖掘、下载与整备好的触发音（SFX）、环境床音（Bed）、素材索引清单及本地增量学习的声学原型库。

> **隐私与版权隔离**：`materials/bed/*.wav` 下经 CC0 授权核验的 6 条内置环境床音随仓库分发，开箱即用。用户本地挖掘的触发音 `materials/sfx/`、原始素材 `materials/raw/`、本地索引 `materials/index.yaml` 与原型库 `materials/prototypes.npz` 均在 `.gitignore` 中忽略，不纳入 Git 追踪，防止误传有版权或私有的触发音素材。新增 `bed/*.wav` 也会进入 Git 候选文件，上传前请多加注意。

---

## 目录结构

```text
materials/
├── sfx/             触发音切片（统一命名：{prefix}_{序号:02d}_{声道:L|R|C|LR|RL}.wav）
├── bed/             随仓分发的 6 条 CC0 环境床音（仅通过 @bed 预设调度）
│   ├── 523389_leaves_rustle.wav    微风树叶，48 kHz
│   ├── 729395_campfire_ember.wav   篝火炭火，44.1 kHz
│   ├── 818347_rain_soft.wav        树荫细雨，48 kHz
│   ├── 520770_calm_waves.wav       湖岸微浪，96 kHz
│   ├── room_tone.wav              合成林间微风，44.1 kHz
│   └── rain_hard.wav              合成屋檐夜雨，44.1 kHz
├── raw/             外部下载或未裁切的原始素材
├── index.yaml       本地素材注册清单（自动测量响度与时长，留存来源与原片时间戳）
└── prototypes.npz   本地 Few-Shot 声学原型库（512 维 L2 归一化 CLAP 向量，随人工确认与时间轴监督增量更新）
```

---

## 内置 CC0 环境床音与预设

`tingleloom/materials.py` 的 `BUILTIN_BEDS` 保存内置床音的 ID、文件路径、来源、作者、CC0 授权与实测声学元数据。默认 `load_index()` 会自动补入文件存在、尚未在本地索引中登记的内置床音；没有 `index.yaml` 的新克隆仓库也能直接使用全部 8 个 `@bed` 预设，无需下载或手工登记。本地已登记的同 ID 条目优先。

| 素材 ID | 来源 / 作者 | 原片选段与随仓时长 | 采样率与授权 |
| --- | --- | --- | --- |
| `leaves_rustle` | Freesound [#523389](https://freesound.org/s/523389/) / Anya_Media | 微风吹拂树叶，60 秒 | 48 kHz，CC0 |
| `campfire_ember` | Freesound [#729395](https://freesound.org/s/729395/) / HECKFRICKER | 原片 `33–93s`，60 秒 | 44.1 kHz，CC0 |
| `rain_soft` | Freesound [#818347](https://freesound.org/s/818347/) / newlocknew | 原片 `15–75s`，60 秒 | 48 kHz，CC0 |
| `calm_waves` | Freesound [#520770](https://freesound.org/s/520770/) / priesjensen | 平静湖岸微浪，40.5 秒 | 96 kHz，CC0 |
| `room_tone` | 本地物理合成（`procedural_dsp`） | 林间轻柔微风，90 秒 | 44.1 kHz，CC0 |
| `rain_hard` | 本地物理合成（`procedural_dsp`） | 屋檐夜雨，90 秒 | 44.1 kHz，CC0 |

4 条录音经 OAuth2 获取上传者原始无损音频，保留原始采样率与双声道，选取最多 60 秒，应用 80 Hz 高通与 1.5 秒首尾淡化；篝火额外使用温和软限幅以抑制扎耳火星，其余录音未作低通或软削峰。2 条本地合成床音使用树叶纹理/慢速阵风与雨滴表面微冲击模型使用双耳去相关与循环交叉淡化。以上 6 条均按 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 分发，可商用且无需署名，可放心使用。

`presets/beds.yaml` 将这些床音组合为以下 8 个预设。`offset_db` 为相对节目主干的压低量，实际使用中需要根据情况进行调整：

| `@bed` 预设 | 组成素材（层增益） | `offset_db` |
| --- | --- | --- |
| `room_quiet` | `leaves_rustle` | 22 dB |
| `rain_light` | `rain_soft` | 15 dB |
| `rain_heavy` | `rain_hard` | 14 dB |
| `library` | `campfire_ember`（-3 dB）+ `rain_soft`（-6 dB） | 17 dB |
| `cafe` | `campfire_ember`（-2 dB）+ `room_tone`（-6 dB） | 19 dB |
| `night_outdoor` | `leaves_rustle` + `campfire_ember`（-6 dB） | 18 dB |
| `campfire` | `campfire_ember` | 16 dB |
| `lake_waves` | `calm_waves` | 16 dB |

床音属于远景背景层，只通过 `@bed <预设名>` 调度，不写入 `(sfx ...)`。`llm.format_materials_catalog` 从触发音候选中隔离 `materials/bed/`，末尾由 `format_available_beds` 注入当前所有层均就绪的床音预设。

---

## 素材清单 `index.yaml` 真实契约

每条素材对应 `tingleloom.materials.Material` 数据结构，注册或自动提取入库时自动测量声学属性：

```yaml
- id: water_spray_01_R
  file: sfx/water_spray_01_R.wav
  desc: 气泡水喷淋01_右耳
  tags:
    - 气泡
    - 水声
    - 喷雾
  source: 3dio_session.mp4 @ 35:40-36:15 (2140-2175s)
  source_url: ""
  author: ""
  license: CC0
  license_url: ""
  attribution_required: false
  ai_generated: false
  commercial_ok: true
  downloaded_at: "2026-11-22"
  duration_sec: 35.0
  sample_rate: 44100
  channels: 2
  base_lufs: -38.42
```

### 核心字段与单一事实源机制

| 字段 / 属性 | 说明与全链路职责 |
|---|---|
| `id` 与派生属性 `category` | 命名遵循 `{prefix}_{seq:02d}_{ch_tag}`（如 `water_spray_01_R`）。`Material.category` 直接从 `id` 派生英文分类前缀（`water_spray`），联动内置 26 类 `clap_asmr_prompts.npz` 与 `prototypes.npz` 映射中文类别名，不在 YAML 中重复存储分类字段。 |
| `source`（原片时间戳与同源去重） | 自动提取入库时统一记为 `"{原片名} @ MM:SS-MM:SS (start-ends)"`（超 1 小时为 `H:MM:SS`）。对同一长音视频二次提取时，系统自动解析同源已入库且磁盘文件存在的 `(start, end)` 区间并在选片阶段前置避让，顺延挖掘尚未入库的新片段。 |
| `is_pending`（待标注判定） | 当 `id` 以 `unknown_` 开头或 `tags` 含 `"待标注"` 时为待标注素材。待标注素材可在 WebUI Tab 2 试听、使用多模态模型辅助鉴别并改名转正；**未转正前不会注入 AI 台本生成上下文，请多加注意**。 |
| `base_lufs` | 注册时实测的积分响度（LUFS，静音为 `None`），混音阶段据此将触发音归一到 `SFX_REFERENCE_LUFS = -27.0 LUFS`，再叠加三档相对增益（`小 -6 dB` / `中 0 dB` / `大 +4 dB`）；人声按 `VOCAL_REFERENCE_LUFS = -24.0 LUFS` 配平，节目主干整体平移至 `TARGET_LUFS = -40.0 LUFS`。床音单独按预设相对节目配平，不参与节目响度归一化。 |
| `tags` / `desc` | 供 `.asmr` 台本 `(sfx <描述或ID> ...)` 匹配检索，以及 AI 台本生成时的“分类配额 + 多样性 Top-K 检索”（默认每类最多选 3 条声道互补代表、总计最多 24 条注入 Prompt，防止素材库规模化后撑爆上下文）。 |
| `license` / `commercial_ok` / `attribution_required` | 记录授权协议、商用许可与署名要求，确保合规溯源。 |

---

## 常用工作流与命令

### 1. 从长音视频自动声学提取与时间轴监督入库

可通过 Gradio 工作台（双击 `start_webui.bat` 或运行 `python main.py`）在 **Tab 1** 操作，或使用 CLI。`high` 为高频清脆触发音，`low` 为低频贴耳触发音，`balanced` 为均衡选片，`bed` 为稳态环境床音：

```bash
# 无监督全自动声学提取（Silero VAD v5 每 5 秒局部 RMS 归一化并重置 LSTM 隐状态 + CLAP 去 Hubness 打标）
python -m tingleloom material mine "path/to/video.mp4" --profile high --dur 35 --min-dur 10 --top 15

# 稳态环境床音：rms_std <= 2.5、crest <= 14.0、双耳居中、0 人声，自动落盘 materials/bed/
python -m tingleloom material mine "path/to/field_recording.wav" --profile bed --dur 60 --top 5

# 带人类时间轴章节监督提示提取（自动跳过说话/预览区间，按章节均衡选片并增量学习声学原型）
python -m tingleloom material mine "path/to/video.mp4" --hints "00:00-01:40 预览说话跳过
01:40-14:21 耳边吹气
14:22-18:37 火柴燃烧"
```

### 2. 待标注素材人工确认与 Few-Shot 原型增量学习

在 WebUI **Tab 2** 按类别或勾选「仅显示待标注」筛选，试听后确认英文前缀并点击「确认改名并更新素材库」，或通过 CLI。`prune_missing` 自动清理磁盘已删文件的僵尸索引，释放序号；`relabel_material` 统一负责改名、移除“待标注”标签与沉淀 CLAP 原型：

```bash
# 列出当前所有待标注素材 (unknown_trigger / 待标注)
python -m tingleloom material label

# 确认类别前缀并重命名音频文件（自动移除“待标注”标签，并将 512 维声学向量记入 materials/prototypes.npz）
python -m tingleloom material label --id unknown_trigger_01_R --prefix water_spray --desc "气泡水喷淋01_右耳" --tags "气泡,水声"
```

配置 `secrets/llm.yaml` 后，可通过 `llm.batch_relabel_pending_materials()` 一次请求翻译归类待标注线索，并沿同一转正路径处理有效项；章节监督也支持外语/未知章节的 AI 翻译归类，沉淀于 `materials/prototypes.npz` 的 Few-Shot 原型会用于后续提取：

```python
from tingleloom import llm

updated = llm.batch_relabel_pending_materials()
print(f"已转正 {len(updated)} 条素材")
```

确认并积累本地原型后，可通过 `proto-export`、`proto-verify` 与 `proto-merge` 导出 PR 贡献包、执行三道防投毒核验（向量复现性、Silero VAD + CLAP 无人声与无削波纯净度、类间区分度 Margin）并归并至内置 `tingleloom/assets/clap_asmr_prompts.npz`：

```bash
python -m tingleloom material proto-export water_spray
python -m tingleloom material proto-verify contributions/water_spray
python -m tingleloom material proto-merge  contributions/water_spray
```

### 3. 拉取与维护环境床音

```bash
# Freesound 英文关键词检索，限定 CC0，下载后自动测量并登记
python -m tingleloom material fetch "gentle rain" --type bed --cc0 --original --n 1 --min-duration 60

# 已选定的 Freesound ID 精确拉取；不加 --no-register 时自动入库
python -m tingleloom material fetch --id 523389 --type bed --cc0 --original --mat-id leaves_rustle --desc "微风树叶"

# 维护两条本地合成床音，保留随仓 rain_soft 真实录音
python -m tingleloom material synth-bed --ids room_tone,rain_hard --dur 90 --overwrite
```

`fetch --original` 需要 Freesound OAuth2，原始下载失败会回退高质 MP3 预览，不能仅凭文件扩展名断言无损；凭据配置见 [secrets/README.md](../secrets/README.md)。拉取不会复现随仓录音的选段与声学处理，也不自动保证无人声，新录音仍需质检、裁剪并在 `presets/beds.yaml` 配置。

`synth-bed` 默认提供 `room_tone`、`rain_soft`、`rain_hard` 三种合成器，不覆盖已有文件；`--overwrite` 才会刷新。随仓 `rain_soft` 是真实录音，维护时用 `--ids room_tone,rain_hard` 避免替换。扩充床音也可用上面的 `material mine --profile bed`，从有使用授权的长录音中筛选。

### 4. 手工登记单条外部素材与查看清单

```bash
# 手工登记单条 WAV 素材
python -m tingleloom material register materials/sfx/page_turn_01_C.wav --desc "翻书01_双耳均衡" --tags "翻页,纸张" --license CC0

# 查看已登记素材清单
python -m tingleloom material list
```

---

## 双格式成品与 Agent 技能

素材最终经三轨流水线渲染，默认同时导出无损 PCM_16 WAV 与 320kbps CBR True Stereo MP3。MP3 直接从 float32 混音总线通过 `libmp3lame` 编码（`compression_level=0`、`joint_stereo=0`），左右耳独立编码。`output/`、`build/` 和 `cache/` 均为本地产物。

面向 AI Agent 的完整流程见 [tingleloom-asmr 技能手册](../.agents/skills/tingleloom-asmr/SKILL.md)，覆盖素材获取、自动提取、转正、非阻塞 `.asmr` 编排、双语耳语 TTS 与渲染交付。

---

## 外部开源素材授权与回溯技巧

- **商用底线**：本仓库内置床音为 CC0；其他素材是否可商用、是否需署名或允许再分发，应以各自授权条款为准。
- **Pixabay Content License**：免署名可商用，但禁止原样再分发或批量打包。
- **Freesound**：注意区分 **CC0**（可商用免署名）、**CC-BY**（需署名）与 **CC-BY-NC**（非商用）。
- **无损回溯**：若 Pixabay 详情页标注了原作者与 `(Freesound)`，可通过脚本直接从 Freesound 检索下载原始无损文件：

```bash
python scripts/freesound_search.py "page turn" --cc0 --original --download materials/raw
```

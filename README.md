# Tingleloom

Tingleloom 是一个面向 ASMR 节目制作的本地音频流水线与素材工作台。项目将长音视频触发音自动提取、素材管理、AI 台本生成、中日双语耳语 TTS 合成和多轨混音拆分为独立模块，支持通过 Gradio Web 界面或 CLI 命令行运行。

## 架构概览

Tingleloom 由四部分组成：

- **本地声学提取引擎 (`tingleloom/mine.py`)**：通过 PyAV 流式解复用音视频，结合整轨频谱体检、Silero VAD v5 与 LAION-CLAP，在 `high`、`low`、`balanced`、`bed` 四种声学 profile 下筛选无人声触发音或稳态环境床音；支持人类时间轴章节监督、未明确标记章节 AI 翻译归类、本地 Few-Shot 原型增量学习及同源时间区间去重避让。
- **素材库与索引 (`tingleloom/materials.py`)**：以 `materials/index.yaml` 管理本地素材元数据，登记时实测时长、采样率、声道与基准响度（`base_lufs`）。随仓分发的 6 条 CC0 环境床音可在没有 `index.yaml` 时自动发现，开箱即用；`prune_missing` 清理磁盘已删除的素材文件的僵尸索引并释放序号，`relabel_material` 转正时自动移除“待标注”标签并沉淀 CLAP 原型向量。
- **三轨混音流水线 (`tingleloom/script.py` / `tts/` / `sfx.py` / `timeline.py` / `mix.py`)**：解析 `.asmr` 领域台本，调用TTS引擎合成人声（支持内容寻址缓存与断点续跑），将触发音、人声、环境床音分别渲染为独立音轨后按响度基准同时导出无损 `PCM_16` `.wav` 与发布用 `320kbps` True Stereo `.mp3` 成品。MP3 由 float32 混音总线直接编码，使用 `libmp3lame` 最高算法精度（`compression_level=0`）。
- **多协议 LLM 接入与 Web 工作台 (`tingleloom/llm.py` / `tingleloom/webui.py`)**：支持 `openai`（`/chat/completions`）、`openai-responses`（`/responses`）、`deepseek`、`anthropic`（`/v1/messages`）、`gemini`（`:generateContent`）5 种 API 协议与多渠道独立凭据存储，支持基于分类配额检索生成 `.asmr` 台本、截取音频片段调用多模态模型辅助鉴别道具，并提供四面板可视化操作界面。

## 成品试听示例

仓库内置了一段由 Tingleloom 全链路自动合成的 320kbps 双耳独立立体声（True Stereo，`joint_stereo=0`）成品样带，可直接点击下方播放器在线试听或下载原文件（建议佩戴耳机）：

https://github.com/user-attachments/assets/054cf6fc-1112-4416-9f9f-0377bae34fba

- **[下载 320kbps 双耳立体声原文件：`examples/demo_sample.mp3`](examples/demo_sample.mp3)**（时长 `3分15秒` · 44.1 kHz · 320kbps CBR True Stereo · 综合响度 `-40.12 LUFS`）
  - **环境床音**：`@bed room_quiet`（微风树叶环境音，锚定在节目主干下方 `24.0 dB`）
  - **人声与触发音编排**：7 段中文耳语（`whisper_cn`）配合 4 段双耳空间触发音——右耳 `R70` 木块轻敲（`12s–47s`）→ 左耳 `L70` 绵密泡沫（`58s–93s`）→ `LR12` 双耳 12 秒周期平滑往返刷麦（`105s–140s`）→ 双耳捂耳收尾（`144s–180s`）。

## 核心特性

### 长音视频触发音自动提取与零样本分类

`tingleloom/mine.py` 直接读取 `.mp4`、`.mkv`、`.m4a`、`.mp3`、`.wav`、`.flac` 等常见音视频容器，无需预先将视频转码为庞大的临时 WAV 文件：

- **自适应整轨体检 (`scan_audio`)**：按秒统计 RMS 分布（底噪 P10、中位电平 P50、活跃电平 P90）、信噪比、高频截断频率、削波采样数与左右耳相关性，根据当前录音的动态范围自适应计算静音门限与活跃窗口阈值。
- **分块归一化人声剔除 (`SileroVad`)**：内置 Silero VAD v5 ONNX 模型（`tingleloom/assets/silero_{encoder,decoder}_v5.onnx`）。采用 5 秒分块局部 RMS 归一化，并在分块边界重置 LSTM 隐状态，避免高振幅贴麦触发音掩盖轻声说话或长时宽频摩擦致盲后续人声检测；结合 1D 布尔游程去毛刺、最长单段连续人声与总人声占比双阈值过滤人声。
- **去基线校准零样本打标 (`ClapTagger`)**：内置 INT8 量化 LAION-CLAP 音频编码器与 26 类物理声学特征向量（`tingleloom/assets/clap_*`）。对长于 10 秒的切片执行首尾全覆盖多窗口推理，人声/耳语负向得分采用最大值池化一票否决；触发音分类通过整轨活跃锚点均值执行 Anti-Hubness 去基线校准，淘汰低命中率片段，并按 512 维嵌入余弦相似度做音色去重与多尺度短窗捡漏。

- **四种声学 profile**：`high` 偏向高频清脆触发音，`low` 偏向低频贴耳触发音，`balanced` 均衡选片；`bed` 专选稳态环境床音，要求 `rms_std <= 2.5`、`crest <= 14.0`、双耳居中且 0 人声，自动落盘 `materials/bed/`，不混入触发音目录。
- **章节监督与本地学习**：通过 `--hints` 输入人类时间轴章节，跳过说话/预览区间并按章节选片；配置 LLM 后可自动翻译归类未明确标记的章节。人工转正与章节监督将 512 维 CLAP 原型按累计样本数（`proto_counts`）加权更新写入 `materials/prototypes.npz`，后续提取复用本地 Few-Shot 声学经验；同源已入库且文件仍存在的时间区间自动避让，重复运行可继续挖掘新片段。
- **声学原型众包导出、防投毒核验与归并 (`export_prototype` / `verify_prototype` / `merge_prototype`)**：
  - `material proto-export <slug>` 从本地原型库与代表音频中截取 `5~10s` 无削波高能量片段，输出 `contributions/<slug>/prototype.json`（类别元数据、累计样本数 `sample_count` 与 512 维 L2 归一化向量）及验证凭据 `proof_10s.wav`。
  - `material proto-verify <pkg_dir>` 基于固定 ONNX 编码器与不加载本地原型的纯净基线执行三道防投毒核验：① 向量真伪复现（单样本 $\cos \ge 0.999$，多样本均值 $\cos \ge 0.85$）；② 纯净度与无人声（`SileroVad` 人声占比与连续时长为 `0`、CLAP 负向耳语分 `< 0.35`、削波采样 `0`）；③ 类间区分度（新增类别 Top-1 命中自身且 `margin > 0.03`，既有类别按样本数加权融合后仍命中自身）。
  - `material proto-merge <pkg_dir>` 在核验通过后将原型按样本计数加权融合（或作为新类别追加）入内置词表 `tingleloom/assets/clap_asmr_prompts.npz`，并默认移除临时验证音频 `proof_10s.wav`，使仓库主干仅保留不可逆特征向量。

### 三轨独立混音与双耳声像控制

`tingleloom/mix.py` 与 `tingleloom/sfx.py` 针对 ASMR 节目听感设计了独立的电平与声像处理：

- **触发音与人声分层配平**：每条触发音按实测 `base_lufs` 归一化至触发音基准（`SFX_REFERENCE_LUFS = -27.0 LUFS`）并叠加台本档位（`小 -6 dB` / `中 0 dB` / `大 +4 dB`）；每段人声归一化至人声基准（`VOCAL_REFERENCE_LUFS = -24.0 LUFS`）；两者合成节目主干后整体平移至目标响度（`TARGET_LUFS = -40.0 LUFS`）。
- **相对偏移环境床音**：环境床音（`presets/beds.yaml`）各层按素材响度配平叠加、统一经 80 Hz 高通低切后，始终锚定在节目实测响度下方 `BED_OFFSET_DB`（默认 `22.0 dB`），不参与节目响度归一化，防止安静段落底噪被拉高。
- **能量守恒声像重定位与段内周期轮换**：未指定声道或指定 `C` 时保留素材原生立体声；指定 `L`、`R`、`L70`、`R30` 等固定偏置时，先按左右声道 RMS 能量权重汇聚为单声道再投射至目标方向，避免单侧录制素材跨声道移动时被静音；指定 `LR` 或 `LR<秒数>`（如 `LR8`）时，在单段触发音内部按指定周期与 `1.5s` 交叉淡化在左右耳之间平滑往返轮换。

### 内容寻址 TTS 缓存与断点续跑

`tingleloom/tts/` 封装了本地 GPT-SoVITS ONNX 部署包调用（需要用户自行准备onnx模型或调用其他tts模型）：

- 台本中每一行普通文本对应一个独立 TTS 合成段，支持在台本内通过 `@preset` 随时切换 `whisper_cn`、`whisper_ja`、`normal_cn`、`normal_ja`。（前提是使用的tts引擎支持对应语言）
- 合成产物按请求内容哈希存储在 `cache/tts/<hash>.wav`，相同台本二次运行直接复用缓存；修改单行台本仅重新合成对应段落。

### 多协议 AI 台本生成与听音辅助鉴别

`tingleloom/llm.py` 提供多协议 LLM 调用、素材分类配额检索与凭据脱敏管理：

- **多协议与多渠道独立存储**：支持 `openai`（Chat Completions）、`openai-responses`（Responses API）、`deepseek`、`anthropic`（Claude Messages API）与 `gemini`（Google Gemini 原生 `:generateContent`）5 种协议格式，在 `secrets/llm.yaml` 中按 `providers.<channel>` 独立保存，并在「API 与多协议模型配置」面板中随时切换。
- **分类配额 Top-K 检索台本生成**：内置 4 套提示词模板（`presets/script_templates.yaml`），生成时自动剔除待标注与缺失文件，按主题检索并按类别配额（默认每类最多 3 条声道互补代表、总计最多 24 条）注入 `{{materials_catalog}}`，防止素材库规模化后撑爆上下文；末尾由 `format_available_beds` 自动附上所有层素材均已就绪的 `@bed` 环境床音预设。
- **待标注素材批量转正**：`batch_relabel_pending_materials` 一次请求翻译、归类待标注素材线索，沿 `relabel_material` 的统一改名/去待标注标签/CLAP 原型学习路径转正有效项。
- **多模态听音辅助鉴别**：针对待确认素材，自动截取中间最多 10 秒音频编码为 16 kHz 单声道 WAV Base64（OpenAI `input_audio` 或 Gemini `inlineData`），结合本地声学与 CLAP 候选信息交由多模态模型给出前缀、描述与标签建议（在 `deepseek` / `anthropic` 纯文本渠道下自动降级为声学特征文本鉴别）。
- **凭据脱敏保护**：`secrets/llm.yaml` 支持 `${OPENAI_API_KEY}`、`${ANTHROPIC_API_KEY}`、`${GEMINI_API_KEY}`、`${DEEPSEEK_API_KEY}` 等环境变量占位符展开；Web 界面对已配置密钥统一按 `****后4位` 或 `<env:VAR>` 脱敏显示，前端不回显明文，留空保存配置时自动保留该渠道原有密钥及环境变量引用。

## 目录结构

```text
Tingleloom_ASMR/
├── main.py                     # 根目录启动入口（默认启动 WebUI，支持转发 CLI 子命令）
├── start_webui.bat             # Windows 双击启动脚本
├── requirements.txt            # Python 依赖清单
├── tingleloom/                 # 核心源码包
│   ├── config.py               # 路径约定、音频规格与混音电平常量
│   ├── audio.py                # 音频读写、重采样、LUFS 测量、淡入淡出与软限幅
│   ├── script.py               # .asmr 台本解析器
│   ├── materials.py            # 素材索引读写、自动测量登记、重命名打标与检索匹配
│   ├── mine.py                 # 长音视频声学体检、Silero VAD 滤人声与 CLAP 零样本打标
│   ├── llm.py                  # 多协议 LLM 接口调用、凭据脱敏、分类配额检索台本生成与听音鉴别
│   ├── webui.py                # Gradio 四面板可视化工作台（自动提取、固定分页打标、台本合成、多协议 API 配置）
│   ├── sfx.py                  # 触发音匹配、切片循环预处理与声像变换
│   ├── presets.py              # 环境床音预设加载与多轨渲染
│   ├── timeline.py             # 游标时间轴编排
│   ├── mix.py                  # 三轨混音器与总线软限幅输出
│   ├── pipeline.py             # 端到端流水线编排与构建快照输出
│   ├── cli.py                  # CLI 命令行实现
│   ├── tts/                    # 本地 ONNX TTS 引擎适配、内容寻址缓存与批处理
│   └── assets/                 # 内置 Silero VAD v5 与 LAION-CLAP INT8 ONNX 模型资产
├── presets/                    # 预设配置
│   ├── beds.yaml               # 环境床音预设定义
│   └── script_templates.yaml   # AI 台本生成提示词模板
├── scripts/                    # 独立辅助脚本 (trim_audio, analyze_audio, freesound_*)
├── examples/                   # 示例台本 (demo.asmr) 与 320kbps 双耳立体声成品样带 (demo_sample.mp3)
├── .agents/skills/tingleloom-asmr/SKILL.md  # AI Agent 全链路 ASMR 制作技能手册
├── materials/                  # 素材库：随仓 bed/*.wav 共 6 条 CC0 床音；本地 sfx/、raw/、index.yaml、prototypes.npz 由 Git 忽略
├── secrets/                    # 本地凭据目录 (llm.yaml, freesound 凭据；Git 忽略)
└── tests/                      # pytest 测试套件
```

## 快速开始

### 1. 准备环境

需要 Python 3.10 或更高版本：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

如需运行人声 TTS 合成，需准备本地 GPT-SoVITS ONNX 部署包，并通过环境变量 `TINGLELOOM_TTS_DIR`、`secrets/tts_dir.txt` 或命令行 `--tts-dir` 指定目录（默认查找 `models/tts`）。

### 2. 配置凭据与内置 CC0 环境床音

AI 台本生成、听音辅助鉴别与批量翻译归类可通过 `secrets/llm.yaml` 或环境变量配置。推荐按协议渠道独立保存；当前源码用顶层 `provider` 选择活动渠道：

```yaml
provider: openai
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-mini"
    audio_model: "gpt-4o-audio-preview"
    temperature: 0.7
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
    audio_model: "deepseek-chat"
    temperature: 0.7
```

`providers` 还支持 `openai-responses`、`anthropic`、`gemini` 渠道。切换渠道不会覆盖其他渠道的配置。

Freesound 检索与下载凭据配置方式见 [`secrets/README.md`](secrets/README.md)。`secrets/` 与 `.env` 已配置在 `.gitignore` 中。

#### Freesound 自动拉取与入库

`material fetch` 默认完成检索、下载、测量并登记 `materials/index.yaml`（时长、采样率、声道、`base_lufs` 及授权来源），无需手工执行 `material register`：

```powershell
# 触发音默认保存到 materials/sfx/，只要 CC0，按下载量选取 3 条
python -m tingleloom material fetch "page turn" --cc0 --n 3

# 环境床音保存到 materials/bed/，自动加上“环境音”标签
python -m tingleloom material fetch "gentle rain" --type bed --cc0 --n 1 --min-duration 60 --max-duration 240 --original --mat-id gentle_rain --desc "柔和雨声"

# 已选定的 Freesound ID 可直接精确拉取，免填搜索词
python -m tingleloom material fetch --id 523389 --type bed --cc0 --original --mat-id leaves_rustle --desc "微风吹拂树叶的轻柔沙沙声"
```

- `--original` 使用 OAuth2 下载上传者的原始音频（WAV / FLAC / AIFF 等），自动刷新即将过期的 token；单条原始下载失败时会提示原因并回退到高质 MP3 预览。首次需先完成 `scripts/freesound_oauth.py authorize` 与 `exchange`。
- `--type sfx|bed|raw` 选择默认输出目录；`--dir` 显式覆盖目录。`--no-register` 仅下载，不写索引。
- `--mat-id` 设置入库 ID，多条时自动加 `_01` 等后缀；省略时使用文件 stem。`--desc`、逗号分隔的 `--tags` 可覆盖名称及前 6 个 Freesound 标签。
- `--sort` 支持 `score`、`rating_desc`、`downloads_desc`（默认）、`duration_desc`、`created_desc`；`--cc0` 限定免署名可商用授权（按 ID 拉取时也检查）。
- 环境录音仍需筛选人声、突发声与场景适配。床音应只用于 `@bed`，不要写成 `(sfx ...)`。

#### 随仓分发的 6 条 CC0 环境床音

以下文件已经内置于 `materials/bed/`，均为 **CC0 免署名可商用**床音；4 条 Freesound 录音来自 OAuth2 原始无损音频，另外 2 条为本地零纯音物理合成。**拉取后无需额外下载，也无需创建 `materials/index.yaml`，即可直接使用全部 8 个 `@bed` 预设**。

| 素材 ID | 随仓 WAV 文件 | 来源与选段 | 采样率 |
| --- | --- | --- | --- |
| `leaves_rustle` | `523389_leaves_rustle.wav` | Freesound [#523389](https://freesound.org/s/523389/)，Anya_Media；微风树叶，60 秒 | 48 kHz |
| `campfire_ember` | `729395_campfire_ember.wav` | Freesound [#729395](https://freesound.org/s/729395/)，HECKFRICKER；篝火炭火，原片 `33–93s` | 44.1 kHz |
| `rain_soft` | `818347_rain_soft.wav` | Freesound [#818347](https://freesound.org/s/818347/)，newlocknew；树荫细雨，原片 `15–75s` | 48 kHz |
| `calm_waves` | `520770_calm_waves.wav` | Freesound [#520770](https://freesound.org/s/520770/)，priesjensen；平静湖岸微浪，40.5 秒 | 96 kHz |
| `room_tone` | `room_tone.wav` | 本地物理合成；林间轻柔微风，90 秒 | 44.1 kHz |
| `rain_hard` | `rain_hard.wav` | 本地物理合成；屋檐夜雨，90 秒 | 44.1 kHz |

4 条录音保留原始采样率，选取最多 60 秒，应用 80 Hz 高通和 1.5 秒首尾淡化；篝火床音额外使用温和软限幅以消除扎耳火星，其余录音不作低通或软削峰。CC0 授权与来源说明详见 [素材库说明](materials/README.md)。

#### 8 个环境床音预设

| 预设 | 声景 | 相对节目压低 |
| --- | --- | --- |
| `room_quiet` | 轻柔的窗外树叶微动 | 22 dB |
| `rain_light` | 树荫春雨滴落 | 15 dB |
| `rain_heavy` | 零纯音表面微冲击合成屋檐夜雨 | 14 dB |
| `library` | 温暖炭火（-3 dB）与春雨（-6 dB）的雨夜书房 | 17 dB |
| `cafe` | 柔和篝火（-2 dB）与轻柔微风（-6 dB），不含交谈 | 19 dB |
| `night_outdoor` | 林间树叶与远景篝火（-6 dB） | 18 dB |
| `campfire` | 纯粹柔和篝火 | 16 dB |
| `lake_waves` | 96kHz 平静湖岸微浪 | 16 dB |

本地合成 `room_tone` 使用双耳树叶纹理与慢速阵风；合成雨声使用远景微雨珠和短促滤波噪声轻拍波包。维护合成床音可运行 `python -m tingleloom material synth-bed --ids room_tone,rain_hard --overwrite`；扩充真实录音可用 `material fetch --type bed --cc0 --original`，随后质检并在 `presets/beds.yaml` 配置预设。`material synth-bed` 默认提供 `room_tone`、`rain_soft`、`rain_hard` 三种合成器，不加 `--overwrite` 会保留已有文件；使用 `--ids` 可避免刷新时把真实 `rain_soft` 录音替换为合成雨声。`output/` 为本地运行输出目录。

### 3. 启动 Web 工作台

Windows 下可直接双击根目录的 `start_webui.bat`，或在终端执行：

```bash
python main.py
```

服务默认监听 `http://127.0.0.1:7860` 并自动打开浏览器。可通过参数调整监听端口或指定 TTS 部署包路径：

```bash
python main.py --port 7861 --no-browser
python main.py --tts-dir /path/to/onnx_tts_deploy
```

### 4. 命令行使用

`python main.py` 与 `python -m tingleloom` 支持相同的 CLI 子命令：

```bash
# 从长音视频自动筛选、切分触发音并登记入库（默认最多提取 15 段，自动淘汰低命中率片段）
python -m tingleloom material mine /path/to/video_or_audio.mp4 --profile high --top 15

# 从长音视频挖掘稳态、双耳居中且 0 人声的环境床音，自动落盘 materials/bed/
python -m tingleloom material mine /path/to/field_recording.wav --profile bed --dur 60 --top 5

# 维护基础合成床音；只刷新这两个 ID，保留随仓 rain_soft 原声
python -m tingleloom material synth-bed --ids room_tone,rain_hard --dur 90 --overwrite

# 从 Freesound 扩充床音（需配置凭据；新录音应另行质检并配置床音预设）
python -m tingleloom material fetch "gentle rain" --type bed --cc0 --original --n 1 --min-duration 60

# 仅执行声学体检与候选探测，不写入素材库
python -m tingleloom material mine /path/to/video_or_audio.mp4 --probe-only

# 列出待标注素材，或对指定素材重命名前缀并更新描述/标签
python -m tingleloom material label
python -m tingleloom material label --id unknown_trigger_01_L --prefix wood_tapping --desc "敲击木块01_左耳" --tags "敲击,木块"

# 导出本地学习的声学原型贡献包 / 执行三道防投毒声学校验 / 归并入内置词表并清理验证音频
python -m tingleloom material proto-export wood_tapping
python -m tingleloom material proto-verify contributions/wood_tapping
python -m tingleloom material proto-merge  contributions/wood_tapping

# 查看素材库清单 / 手动登记单条素材
python -m tingleloom material list
python -m tingleloom material register materials/sfx/page_turn.wav --desc "翻书" --tags "翻页,纸质"

# 运行端到端台本合成流水线（默认同时输出 output/<台本名>.wav 与 .mp3）
python -m tingleloom run examples/demo.asmr --provider cpu

# 分步调试：解析台本 / 仅跑 TTS 缓存 / 检查素材匹配缺口
python -m tingleloom parse examples/demo.asmr
python -m tingleloom tts   examples/demo.asmr
python -m tingleloom match examples/demo.asmr
```

### 5. 运行测试

```bash
pytest -q
```

### 环境变量参考

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `TINGLELOOM_TTS_DIR` | 本地 ONNX TTS 部署包目录 | `secrets/tts_dir.txt` 或 `models/tts` |
| `TINGLELOOM_TTS_PROVIDERS` | ONNX 推理后端（`cpu` / `cuda` / `hybrid`） | `cpu` |
| `OPENAI_BASE_URL` | OpenAI 兼容 API 端点 | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | OpenAI 兼容 API 密钥 | 无 |
| `OPENAI_MODEL` | 台本生成默认文本模型 | `gpt-4o-mini` |
| `OPENAI_AUDIO_MODEL` | 听音辅助鉴别多模态音频模型 | `gpt-4o-audio-preview` |
| `FREESOUND_TOKEN` | Freesound API Token | 无（也可存于 `secrets/freesound_token.txt`） |

## 台本语法与混音规范

`.asmr` 台本以行为单位编排人声与触发音事件。`# 标题` 等价于 `@title 标题`，普通注释使用 `//`。`(sfx ...)` 非阻塞放置声音、不推进时间轴游标，需用 `(pause ...)` 或台词实测时长留出播放空间：

```text
@title 深夜音疗工坊
@preset whisper_cn
@seed 20261006
@bed room_quiet
@pan_swap 15

今晚先不用说话，把眼睛轻轻闭上。
(pause 4)
(sfx 气泡水02_左耳 25s 中 L)
(pause 25)

换到右边，再听一会儿。
(pause 3)
(sfx 气泡水01_右耳 25s 中 R)
(pause 25)

最后让木块声在左右耳之间慢慢交替。
(pause 3)
(sfx 小木块02_左右交替密集 40s 中 LR8)
(pause 40)
```

| 指令 | 说明 |
| --- | --- |
| `@title <标题>` | 设置节目标题 |
| `@preset <预设名>` | 切换 TTS 音色预设（`whisper_cn` / `whisper_ja` / `normal_cn` / `normal_ja`） |
| `@bed <床音名>` | 切换 `presets/beds.yaml` 中定义的环境床音（传 `none` 关闭） |
| `@pan_swap <秒数>` | 未显式指定声像的事件按该秒数自动左右交替偏置 |
| `@seed <整数>` | 设置固定随机种子，保证随机切片起点与合成结果可复现 |
| `(pause <秒数>)` | 时间轴游标前进指定秒数 |
| `(sfx <查询词> [@偏移] [时长s] [小\|中\|大] [声道])` | 放置触发音；带时长为持续铺底（游标不推进），不带时长为单次触发；声道支持 `C`、`L`、`R`、`L70`、`R30`、`LR`、`LR<秒数>` |

## 相关说明文档

- [素材库与内置 CC0 床音说明](materials/README.md)
- [AI Agent 全链路 ASMR 制作技能](.agents/skills/tingleloom-asmr/SKILL.md)：覆盖素材获取、声学提取、批量转正、非阻塞台本编排、本地 ONNX TTS、WAV + 320kbps True Stereo MP3 渲染与交付
- [凭据与脱敏配置说明](secrets/README.md)

---
name: tingleloom-asmr
description: "驱动 Tingleloom ASMR 音频制作全流程：Freesound CC0 自动化检索拉取（含 OAuth2 原始无损质量）、长音视频声学提取与 Silero VAD 人声剔除、CLAP 零样本打标与 Few-Shot 原型转正、环境床音采集与程序化合成、符合非阻塞游标与去 AI 腔原则的 .asmr 台本编排、本地耳语 TTS 合成，以及无损 WAV 与 320kbps 双耳独立立体声 MP3 成品渲染。当用户要求用 AI 制作 ASMR 音频、编写 .asmr 台本、从 Freesound 或音视频获取触发音/床音、标注整理素材库或执行 Tingleloom 混音流水线时使用。"
---

# Tingleloom ASMR 音频工程与导演手册

本技能是 **Tingleloom** 的完整声学与节目制作指南。在 Tingleloom 中，你同时扮演两个角色：
1. **音频工程师**：管理触发音（`materials/sfx/`）与环境床音（`materials/bed/`），利用 Freesound API 自动化拉取 CC0 原声、从长音视频自动提取触发音、校验高频截断与零人声污染，并驱动三轨独立响度校准混音器。
2. **ASMR 声音导演**：运用**非阻塞时间轴游标**与**双耳声场变换**编排 `.asmr` 台本，让触发音占据 75%～85% 的呼吸空间，配合自然、克制、零 AI 腔的耳语人声与稳态床音，合成睡眠级成品。

执行任何 CLI 命令或辅助脚本前，确认当前工作目录为 `Tingleloom_ASMR` 仓库根目录。

---

## 一、全局架构与三条红线

动手前先建立对数据流与边界的准确认知：

1. **触发音（SFX）与环境床音（Bed）物理隔离**：
   - **触发音（`materials/sfx/`）**：近场、贴耳、有纹理变化的道具声（木块敲击、气泡水、刷麦、翻页、掏耳等），在台本中用 `(sfx ...)` 调度，混音时按实测 `base_lufs` 归一到 **`-27.0 LUFS`** 再叠 `小(-6dB)/中(0dB)/大(+4dB)` 档位。
   - **环境床音（`materials/bed/`）**：远场、柔和、不抢触发音戏的自然环境音（微风吹拂树叶、柔和篝火炭火、树荫微雨、平静湖岸微浪），由 `presets/beds.yaml` 组合成预设，在台本中**只通过 `@bed <预设名>` 调度**。混音时过 80Hz 高通低切，再锚定在节目主干响度下方 `offset_db`（14～22 dB），**绝不写入 `(sfx ...)`**。
2. **未转正素材（`is_pending`）不进正式台本**：
   - `id` 以 `unknown_` 开头或 `tags` 含 `"待标注"` 的素材属于待确认切片。入本前应先试听/鉴别并执行 `material label` 转正；若直接写进台本，你根本不知道那段声音究竟是什么道具，胡乱使用会导致严重的后果。
3. **写台本前必须先查库，渲染前必须过 `parse` + `match` 静态门禁**：
   - 严禁凭空捏造库里不存在的素材 ID 或 `@bed` 预设名。

---

## 二、素材获取与治理（四条供给通道）

当用户需要新床音、新触发音，或现有素材库不足以支撑主题时，按场景选择以下四条通道（可组合使用）：

### 通道 A：Freesound CC0 自动化检索、拉取与入库（联网首选）

仓库内置了完整的 Freesound APIv2 + OAuth2 工具链（凭据位于 `secrets/freesound_token.txt` 与 `secrets/freesound_oauth.json`，OAuth2 支持自动刷新 `access_token` 以下载上传者原始无损 WAV/FLAC）。
**注意**：Freesound 仅支持**英文关键词**检索。

1. **一键检索、下载并自动测量 LUFS 登记入库（`material fetch`）**：
   ```powershell
   # 拉取 CC0 环境床音（自动落盘 materials/bed/ 并带 "环境音" 标签写入 materials/index.yaml）
   python -m tingleloom material fetch --id 523389 --type bed --cc0 --original --mat-id leaves_rustle --desc "微风吹拂树叶的轻柔沙沙声" --tags "环境音,树叶,微风"

   # 按已知 Freesound Sound ID 精确拉取单条原始质量素材
   python -m tingleloom material fetch --id 614080 --type sfx --cc0 --original --mat-id page_turn_01 --desc "翻书页" --tags "翻页,纸质"

   # 批量拉取 CC0 触发音到 materials/sfx/
   python -m tingleloom material fetch "binaural tapping" --type sfx --cc0 --original --min-duration 5 --max-duration 120 --n 3
   ```
   - `--type`：`sfx`（写入 `materials/sfx/`）、`bed`（写入 `materials/bed/`）、`raw`（写入 `materials/raw/`）。
   - `--original`：自动检查并刷新 `secrets/freesound_oauth.json` 中的 OAuth2 token 下载原始无损格式（WAV/FLAC）；若无 OAuth2 或单条失败则自动回退 `preview-hq-mp3`。
   - 默认自动调用 `materials.register` 测量时长、采样率、声道与 `base_lufs` 并保存 `materials/index.yaml`（记录 `source="freesound"`、`source_url`、`author`、`license="CC0"`）；若只需下载到临时目录先做筛选，加 `--no-register`。

2. **先广泛搜索对比，再挑精品入库（`scripts/freesound_search.py`）**：
   当不确定哪条声音最干净时，先只搜索不下载，查看时长、采样率、声道数、评分与下载量：
   ```powershell
   python scripts/freesound_search.py "gentle rain loop" --cc0 --min-duration 60 --sort downloads_desc --n 10
   ```
   看准某条 `id` 后，再用 `python -m tingleloom material fetch --id <ID> --type bed --cc0 --original ...` 精准拉取入库。
   - **床音拉取质检**：从外部拉取的床音必须确保**无人声交谈**（可用 `SileroVad` 检测 `vad_ratio == 0`）且电平平稳。若配置了新床音 ID，记得将其加入或组合进 `presets/beds.yaml`。

3. **OAuth2 凭据状态检查与续期（`scripts/freesound_oauth.py`）**：
   ```powershell
   python scripts/freesound_oauth.py status
   python scripts/freesound_oauth.py token   # 若过期会自动使用 refresh_token 续期
   ```

---

### 通道 B：本地长音视频声学提取（`material mine`）

从本地 `.mp4` / `.mkv` / `.wav` / `.flac` / `.mp3` 等长音视频中流式解复用，自动执行整轨频谱体检、**Silero VAD v5**（每 5 秒局部 RMS 归一化 + 定期重置 LSTM 状态防宽频摩擦致盲）剔除人声、**LAION-CLAP INT8** 26 类去 Hubness 零样本打标 + 本地 `materials/prototypes.npz` Few-Shot 原型匹配，并自动跳过同一原片已入库的重叠时间段：

```powershell
# 挖掘高频清脆触发音（敲击/水雾/木块/硬物），默认切 35s，最多取 Top 15
python -m tingleloom material mine "<音视频路径>" --profile high --dur 35 --top 15

# 挖掘低频贴耳触发音（捂耳/心跳/低沉刷麦/软木），可附带人类时间轴文本或文件辅助监督
python -m tingleloom material mine "<音视频路径>" --profile low --dur 35 --top 15 --hints "<时间轴文本或文件路径>"

# 从长音视频或环境录音中挖掘稳态环境床音（要求 rms_std<=2.5、无瞬态毛刺、双耳居中、0人声，自动写入 materials/bed/）
python -m tingleloom material mine "<音视频路径>" --profile bed --dur 60 --top 5

# 仅做整轨声学体检与候选探测，不切分落盘
python -m tingleloom material mine "<音视频路径>" --probe-only
```

---

### 通道 C：程序化零版权基础床音合成（`material synth-bed`）

当本地缺少基础床音或需要完全可复现的双耳去相关无缝循环环境音时，一键合成 `room_tone`（林间轻柔微风）、`rain_soft`（树叶微雨 · 零正弦纯音表面微冲击模型）、`rain_hard`（屋檐夜雨 · 零正弦纯音表面微冲击模型）并登记入库：

```powershell
python -m tingleloom material synth-bed
# 如需覆盖已有同名床音或调整时长（默认 90s，首尾 2.0s 环形无缝交叉淡化）：
python -m tingleloom material synth-bed --dur 90 --overwrite
```

---

### 通道 D：音频质检、精裁与手动登记（`scripts/` + `material register/label`）

1. **频谱截断与真假无损体检（`scripts/analyze_audio.py`）**：
   检查外部下载的音频是否被低码率有损压缩截断高频（128kbps MP3 通常在 16kHz 截断，而 ASMR 气音与沙沙声极度依赖 16kHz～20kHz 频段）：
   ```powershell
   python scripts/analyze_audio.py materials/sfx/<文件名>
   ```
2. **带防咔哒淡入淡出的精裁与循环延展（`scripts/trim_audio.py`）**：
   ```powershell
   # 截取高能量 5 秒片段，带 50ms 淡入淡出
   python scripts/trim_audio.py in.wav -o materials/sfx/out.wav --dur 5 --auto --fade 0.05
   # 将短床音无缝循环拼接适配为 600 秒（接缝 2.0s 交叉淡化）
   python scripts/trim_audio.py in.wav -o materials/bed/out.wav --fit 600 --xfade 2.0
   ```
3. **手动登记已有音频（`material register`）**：
   ```powershell
   python -m tingleloom material register materials/sfx/custom.wav --id custom_01_C --desc "自定义触发音" --tags "标签1,标签2" --license CC0
   ```
4. **待标注素材（`unknown_*`）鉴别与转正（`material label`）**：
   - 列出所有待标注素材（同时自动清理磁盘已删除的僵尸索引）：
     ```powershell
     python -m tingleloom material label
     ```
   - 单条转正（自动重命名音频文件、移除 `"待标注"` 标签，并将该音频的 CLAP 512 维特征沉淀入 `materials/prototypes.npz`，让后续声学识别越用越准）：
     ```powershell
     python -m tingleloom material label --id unknown_trigger_01_C --prefix wood_tapping --desc "木块清脆敲击01_双耳均衡" --tags "木头,敲击"
     ```
   - 若待标注素材带有外语/非标准时间轴描述且已配置 `secrets/llm.yaml`，可调用 `llm.batch_relabel_pending_materials()` 批量 AI 翻译归类转正。
5. **声学原型众包导出、三道防投毒校验与内置词表归并（`material proto-*`）**：
   - 将本地已确认的某类触发音导出为 `contributions/<slug>/{prototype.json,proof_10s.wav}`，经复现性、零人声零削波纯净度与类间区分度三道校验后合入内置 `tingleloom/assets/clap_asmr_prompts.npz`（合入后默认自动清理临时验证音频 `proof_10s.wav`）：
     ```powershell
     python -m tingleloom material proto-export wood_tapping
     python -m tingleloom material proto-verify contributions/wood_tapping
     python -m tingleloom material proto-merge  contributions/wood_tapping
     ```

---

## 三、写台本前的素材与床音盘点

**每次构思或修改 `.asmr` 台本前，必须先盘点当前库内的真实可用资产**：

1. **查看全库素材清单**：
   ```powershell
   python -m tingleloom material list
   ```
2. **查看面向台本的就绪触发音与可用 `@bed` 预设清单**：
   `llm.format_materials_catalog` 会自动滤除 `is_pending` 待标注素材、缺失文件以及 `materials/bed/` 床音，按类别配额（每类最多 3 条声道互补代表、总计最多 24 条）选出最佳触发音，并在末尾附上**当前所有层素材均已就绪的 `@bed` 预设**：
   ```powershell
   @'
   from tingleloom import llm, materials
   print(llm.format_materials_catalog(materials.prune_missing(), query="你想创作的主题或材质关键词"))
   '@ | python -
   ```
   - **只从输出的触发音中选择 `(sfx ...)` 目标**；
   - **只从输出的 `[可用环境床音预设 (@bed ...)]` 中选择 `@bed` 预设名**（或根据已入库的床音先在 `presets/beds.yaml` 中新增/调整预设）。

---

## 四、`.asmr` 台本工程规范与导演法则

优先由你（Agent）直接根据盘点出的真实素材撰写 `.asmr` 台本并保存至 `examples/<name>.asmr`；只有当用户明确要求测试项目内置的 LLM 台本生成接口时，才调用 `tingleloom.llm.generate_script`。

### 1. 语法结构与指令全表（与 `tingleloom/script.py` 严格对齐）

```text
# 深夜木工坊与微雨
@preset whisper_cn
@seed 20261006
@bed rain_light
@pan_swap 15

// 注释行以 // 开头；顶部的 # 标题 等价于 @title 标题
先找个舒服的姿势靠好。(pan L70)
(pause 3)

// 持续型触发音：非阻塞启动 35s，游标 t 不动！紧跟的 pause/台词累计必须 >= 38s
(sfx wood_tapping_01_L 35s 中 L70)
(pause 14)
这边的木头声音比较脆。(pan L70)
(pause 24)

// 动态切换环境床音（切段处混音器自动按预设 xfade 交叉淡化）
@bed room_quiet
@preset whisper_ja
次は、右側でゆっくり聞いてみて。(pan R70)
(pause 3)

// 单次触发音：不写时长，播放素材自身长度（最长截 6s），也可内联在台词行尾 @偏移秒
慢慢把呼吸放轻。(sfx water_spray_01_R @1.2 小 R70)
(pause 6)
```

| 语法元素 | 精确语义与物理行为 |
| --- | --- |
| `# 标题` 或 `@title 标题` | 设置节目标题（注意：`#` 在 `.asmr` 中会改写 `script.title`，**普通注释请一律用 `//`**，不要在正文中间用 `#` 写注释！） |
| `@preset <预设>` | 切换 TTS 音色预设：`whisper_cn`（中文耳语，默认）、`whisper_ja`（日文耳语）、`normal_cn`（中文常语）、`normal_ja`（日文常语）。可在台本任意行单独占一行随时切换。（注意此功能仅在支持多语言 TTS 的版本中可用） |
| `@seed <整数>` | 固定全局随机种子，保证 TTS 采样与长触发音随机切片起点 100% 可复现。 |
| `@bed <预设名\|none>` | 在当前游标 `t` 处启用或切换 `presets/beds.yaml` 中的环境床音预设；传 `@bed none`（或在无预设段）关闭。多段 `@bed` 切换时，混音器会在交界处按预设 `xfade`（2～4s）自动做平滑交叉淡化。 |
| `@pan_swap <秒数>` | 可选。对**未显式指定声道**的人声与触发音，按 `int(start // pan_swap) % 2` 每隔指定秒数在左右（`±0.35`）之间自动交替，防止单侧听觉疲劳。 |
| `普通台词行` | 从游标 `t` 起播人声，**播完后游标 `t` 前进该句 TTS 实测时长**（约 2～4s）。行尾可附带 `(pan L70)` 或直接空格写 `L70`，也可内联 `(sfx 描述 @1.5 小 R)`（在 `t + 1.5s` 处触发）。 |
| `(pause <时长>)` | **游标 `t` 前进指定时长**。支持 `4`、`25s`、`1m30s`。**空行不产生任何停顿**，一切留白与触发音独奏等待都必须显式写 `(pause ...)`。 |
| `(sfx <描述/ID> [@秒] [时长] [档位] [声道])` | 在 `t + (@秒 或 0)` 处**非阻塞放置触发音，游标 `t` 完全不动**（各参数语义见表下列表）。 |

- **描述/ID**：优先写素材精确 `id`（如 `water_spray_01_R`），也支持中文描述或 tag 匹配。
- **时长**：写 `30s` / `1m` 为**持续型**（自动循环/裁剪适配至指定秒数，首尾 1.0s 淡入淡出，循环接缝 2.0s 交叉淡化）；省略时长为**单次触发**（最长取 `6.0s`，随机起点防重样，首尾 50ms 防咔哒淡化）。
- **档位**：`小`（`-6 dB`）、`中`（`0 dB`，默认）、`大`（`+4 dB`）。
- **声道**：省略或 `C`（保留素材原生双耳立体声场）；`L` / `R`（默认偏置 70%）或 `L70` / `R30` / `L100`（先按左右声道 RMS 能量权重汇聚单声道再定向投射，防止单耳素材跨耳变静音）；`LR` 或 `LR10` / `LR15`（在单段触发音内部按指定秒数周期与 1.5s 交叉淡化在左右耳间平滑往返，深度 0.75 保留对侧 25% 自然串音）。

### 2. 非阻塞时间轴编排的黄金数学法则

新手写 `.asmr` 最容易犯的致命错误，是忘记 **`(sfx ...)` 不会推进游标 `t`**。
当你写下一条持续 $T$ 秒的 `(sfx foo 40s 中 L70)` 时，声音从当前游标 $t_0$ 开始播放到 $t_0 + 40$ 秒，但游标依然停在 $t_0$！

- **单轨独奏 + 呼吸留白（最常用）**：
  启动时长为 $T$ 的 `(sfx ...)` 后，紧随其后的 `(pause ...)` 与穿插短台词的累计时间必须满足：
  $$\sum t_{\text{pause}} + \sum t_{\text{vocal}} \ge T + (3\sim 5\text{s})$$
  例如铺一条 `40s` 的触发音：可以先 `(pause 15)` 让听众安静听 15 秒 → 插一句 2.5 秒的耳语 → 再 `(pause 26)` 让后半段触发音播完并留出约 3.5 秒纯净床音呼吸期，然后再进下一段道具。
- **双层同源/互补叠音（进阶技法，谨慎使用）**：
  只有在**刻意设计双耳包裹或质感叠加**时（例如左耳 `L70` 轻柔刷麦 `30s` 铺出 5 秒后，右耳 `R70` 叠入同材质或互补的细腻敲击 `25s`），才允许两段 `(sfx ...)` 在时间上重叠，且两层中至少一层应设为 `小` 档位（`-6 dB`），绝不能把两种杂乱的异类道具同时以 `中/大` 档位撞在一起。

### 3. 去 AI 腔与声音导演铁律

1. **彻底根除客服腔、朗诵腔与冥想 App 套话**：
   严禁出现“亲爱的朋友”“在这个静谧/温柔的夜晚”“卸下一天的疲惫与烦恼”“感受内心的宁静”“开启一段奇妙的放松之旅”等一切书面套话。真实的人头麦耳语就像在耳边随口说话，干净、具体、不煽情。
2. **禁止“报菜名”式预告**：
   绝不写“接下来我要在你的左耳敲击木块三十五秒了”。自然做法是：**先从左边听听看**（比如先铺 `(sfx ...)` + `(pause 12)`），在声音进行中顺口搭一句“这边的节奏会稍微慢一点。”，或者在换道具的空隙轻声说“换个软一点的试试。”
3. **极短句、零长难句**：
   每一行台词都会作为独立请求送入本地 TTS 引擎，**每行严格控制在 2～40 个汉字（或对应长度其他语言短句）**，句中最多两个逗号，句尾记得使用标点。绝不写排比句或散文长句。
4. **中日双语自然接续，绝不翻译复读**：
   使用 `@preset whisper_cn` 与 `@preset whisper_ja` 交替时，严禁上一句说中文、下一句翻译成同义日文。两句应当是顺着动作自然往下接的耳语（如中文说“左边差不多了。”，切 `whisper_ja` 后接“次は右耳の近くいくね。”）。当然，如果用户没有要求的话，**不必刻意切换日文**，中文耳语就够了。
5. **75%～85% 纯触发音占比与睡眠弧线（Sleep Arc）**：
   - **前段（唤醒与定位）**：耳语稍多一点，利用 `L70` ↔ `R70` 左右交替建立近场空间感，触发音档位以 `中` 为主；
   - **中段（深度沉浸）**：拉长单段触发音至 `35s～60s`，配合 `LR12` / `LR15` 慢速环绕往返，每 30～45 秒最多插几句耳语；
   - **尾段（入睡收束）**：台词逐渐归零，触发音档位全部退到 `小`（`-6 dB`），留长段纯触发音与环境床音，最后用一句“晚安。”或“おやすみ。”加长停顿自然淡出。

---

## 五、静态门禁、渲染合成与交付核验

台本写好保存到 `examples/<name>.asmr` 后，按顺序执行以下两步完成渲染交付：

### 1. 静态语法解析与素材匹配门禁（必跑，0 耗时）

```powershell
python -m tingleloom parse examples/<name>.asmr
python -m tingleloom match examples/<name>.asmr
```
- `parse` 必须无 `ScriptError`；
- `match` 必须输出 **“全部命中，无缺口。”**。只要出现任何 `— 缺 —`，立即核对 `python -m tingleloom material list` 修正台本或用 `material fetch` / `material mine` 补齐素材，严禁带缺口强跑。

### 2. 全链路合成渲染（WAV + 320kbps True Stereo MP3 同步导出）

```powershell
python -m tingleloom run examples/<name>.asmr -o output/<name>.wav --strict
```
- **TTS 引擎与加速说明**：
  - 默认按 `--tts-dir <路径>` → 环境变量 `TINGLELOOM_TTS_DIR` → `secrets/tts_dir.txt` → `models/tts` 的顺序解析本地 TTS 部署包目录。
  - 可通过 `--provider hybrid`（推荐，BERT/T2S 走 CUDA、SoVITS 解码走 CPU，省显存且超实时）或 `--provider cpu` / `--provider cuda` 选择推理后端。（前提是使用的 TTS 模型支持该推理方案）
  - 所有合成过的人声行都会按全量参数 SHA-1 内容寻址缓存在 `cache/tts/<hash>.wav`，修改台本时间轴、停顿或触发音后重跑 `run`，已合成台词 **0 秒瞬时命中缓存**。
- **三轨混音与双格式直接落盘**：
  1. **人声轨**：每段实测 LUFS 归一化到 `VOCAL_REFERENCE_LUFS (-24.0 LUFS)` + 声像投射；
  2. **触发音轨**：每段按素材 `base_lufs` 归一化到 `SFX_REFERENCE_LUFS (-27.0 LUFS)` + 档位增益（`小 -6dB` / `中 0dB` / `大 +4dB`）+ 单次 `apply_pan` 声场投射；
  3. **节目主干与床音叠加**：人声与触发音合成为节目主干后整体平移至目标睡眠响度 `TARGET_LUFS (-40.0 LUFS)`；床音轨过 80Hz 高通低切后，按各预设独立的 `offset_db`（如 `room_quiet 22dB`、`cafe 19dB`、`night_outdoor 18dB`、`library 17dB`、`campfire 16dB`、`lake_waves 16dB`、`rain_light 15dB`、`rain_heavy 14dB`）精确垫在节目主干下方，叠加后经 `soft_limit` 软限幅、末尾 `2.5s` 淡出与 `1.0s` 留白收尾；
  4. **双成品同步写出**：同一 `float32` 总线缓冲同时写出无损 `output/<name>.wav`（44.1kHz, 16-bit PCM）与 `output/<name>.mp3`（PyAV `libmp3lame` 直接从 `fltp` 编码，**320kbps CBR**, `compression_level=0`, **`joint_stereo=0` 双耳独立立体声 True Stereo**，完整保留双耳相位差与 3D 定位感），并生成 `build/timeline.json`。

### 3. 向用户汇报交付清单

渲染完成后，清晰汇报：
- **成品路径**：无损 `.wav` 与 320kbps True Stereo `.mp3` 路径；
- **声学指标**：总时长、整体响度（LUFS）、节目响度与床音压低偏移（dB）、峰值（Peak）、人声段数、触发音个数与床音段数；
- **编排结构**：各章节使用的触发音与 `@bed` 预设、左右耳空间轨迹设计以及纯触发音留白占比。

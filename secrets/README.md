# 凭据目录

放 API 凭据。**不要提交到 git。**

---

## Freesound

### 1. 普通 token（下载预览版用）

写入 `freesound_token.txt`，单行就是 token 本身。

或者用环境变量：`FREESOUND_TOKEN`。

### 2. OAuth2（下载原始质量用）

先跑一次授权，凭据会存到 `freesound_oauth.json`：

```bash
# 1. 存应用凭据（在 freesound.org/apiv2/apply 创建后获得）
python scripts/freesound_oauth.py setup --client-id "你的ID" --client-secret "你的密钥"

# 2. 生成授权链接
python scripts/freesound_oauth.py authorize

# 3. 浏览器打开链接 → 登录 → Authorize → 复制 code

# 4. 用 code 换 token
python scripts/freesound_oauth.py exchange "授权码"

# 5. 查看状态 / 取当前 token
python scripts/freesound_oauth.py status
python scripts/freesound_oauth.py token
```

access token 24 小时过期，**refresh token 会自动续期**，跑一次就不用管了。

---

## 本地 TTS 部署包路径（可选）

可在 `secrets/tts_dir.txt` 中单行写入本地 ONNX TTS 部署包目录的绝对路径，避免在代码中写死本机路径。也可设置环境变量 `TINGLELOOM_TTS_DIR`，或在命令行传入 `--tts-dir <路径>`。

目录解析优先级为：`--tts-dir` → `TINGLELOOM_TTS_DIR` → `secrets/tts_dir.txt` → `models/tts`。

---

## 多协议 LLM 配置（台本生成、时间轴解析与多模态听音鉴别）

`secrets/llm.yaml` 默认不纳入 Git 追踪。支持 **`openai`（`/chat/completions`）**、**`openai-responses`（`/responses`）**、**`deepseek`**、**`anthropic`（`/v1/messages`）**、**`gemini`（`:generateContent`）** 5 种 API 协议格式与各渠道独立凭据存储（同时向下兼容旧版单渠道扁平 YAML）：

```yaml
provider: openai
base_url: "https://api.openai.com/v1"
api_key: "${OPENAI_API_KEY}"
model: "gpt-4o-mini"
audio_model: "gpt-4o-audio-preview"
temperature: 0.7
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-6.2-sol"
    audio_model: "gpt-realtime-2.1"
    temperature: 0.7
  anthropic:
    base_url: "https://api.anthropic.com"
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-opus-5-5"
    audio_model: "claude-opus-5-5"
    temperature: 0.7
  gemini:
    base_url: "https://generativelanguage.googleapis.com"
    api_key: "${GEMINI_API_KEY}"
    model: "gemini-4-argon"
    audio_model: "gemini-4-argon"
    temperature: 0.7
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-flash"
    audio_model: "deepseek-flash"
    temperature: 0.7
```

启动前可在进程环境中设置对应渠道的环境变量（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`DEEPSEEK_API_KEY` 及对应的 `*_BASE_URL` / `*_MODEL`）。`.env` 与 `.env.*` 已忽略（`.env.example` 除外）。

WebUI 独立的「**4. API 与多协议模型配置**」面板中：
- 切换 `API 协议格式 / Provider` 时自动回显该渠道独立保存的 `base_url`、`model`、`audio_model` 与脱敏密钥占位提示；
- `api_key` 输入框初始留空，绝不向浏览器下发明文密钥；占位提示与保存状态统一使用 `****后4位`（`<=4` 位显示 `****`）或 `<env:VAR>` 脱敏；
- 留空或提交脱敏掩码会保留磁盘上该渠道的原密钥（包括 `${OPENAI_API_KEY}` 等环境变量引用）；输入新密钥则覆盖该渠道配置并设为当前激活协议。

---

## 安全提醒

- `freesound_oauth.json` 里含 client_secret 和 refresh token，等同于账号权限，**注意安全，不要随意上传**
- 如果曾在公开场合贴过 client_secret，建议去 Freesound 后台轮换
- 本目录已加入 `.gitignore`，仅保留本说明

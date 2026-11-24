# -*- coding: utf-8 -*-
# 项目：Tingleloom
"""Freesound OAuth2 助手（下载原始质量音频必需）。

背景
----
普通 token 只能下载预览版（mp3 ~128kbps / ogg ~192kbps）。
要下载上传时的原始质量（wav/flac/aiff），必须完成 OAuth2 授权码流程。

三步流程
--------
1. authorize —— 生成一个授权链接
2. 浏览器打开它，登录 Freesound，点「Authorize」
3. Freesound 跳转到凭据里登记的 Callback URL，URL 里带 code
   把 code 复制出来，用 exchange 换成 token

用法
----
  # 1. 保存应用凭据（Client ID / Client secret，创建凭据时获得）
  python scripts/freesound_oauth.py setup \
      --client-id "XXX" --client-secret "YYY"

  # 2. 生成授权链接
  python scripts/freesound_oauth.py authorize

  # 3. 授权后，把拿到的 code 换 token
  python scripts/freesound_oauth.py exchange "CODE"

  # 4. 查看状态 / 打印当前可用 token（过期自动刷新）
  python scripts/freesound_oauth.py status
  python scripts/freesound_oauth.py token
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# 脚本位于 <root>/scripts/ 下，parent.parent 才是工作区根目录
ROOT = Path(__file__).resolve().parent.parent
SECRET_FILE = ROOT / "secrets" / "freesound_oauth.json"

AUTHORIZE_URL = "https://freesound.org/apiv2/oauth2/authorize/"
TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"


def load() -> dict:
    if not SECRET_FILE.exists():
        return {}
    return json.loads(SECRET_FILE.read_text(encoding="utf-8"))


def save(data: dict) -> None:
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRET_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        SECRET_FILE.chmod(0o600)
    except OSError:
        pass


def post_form(url: str, payload: dict) -> dict:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": "asmr-sound-collector",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read(400).decode("utf-8", "replace")
        raise SystemExit("HTTP %s: %s" % (e.code, detail))


def cmd_setup(args) -> None:
    data = load()
    if args.client_id:
        data["client_id"] = args.client_id.strip()
    if args.client_secret:
        data["client_secret"] = args.client_secret.strip()
    save(data)
    print("凭据已保存到 %s" % SECRET_FILE)
    print("  client_id     : %s" % (data.get("client_id") or "(未设置)"))
    print("  client_secret : %s" % ("(已设置)" if data.get("client_secret") else "(未设置)"))


def cmd_authorize(args) -> None:
    data = load()
    cid = data.get("client_id")
    if not cid:
        raise SystemExit("还没设置 client_id，请先跑 setup 命令")
    params = {"client_id": cid, "response_type": "code", "state": "asmr-collector"}
    url = AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)
    print("在浏览器打开下面这个链接：\n")
    print(url)
    print("\n登录 Freesound 并点「Authorize」后，浏览器会跳转到一个页面。")
    print("把跳转后 URL 里的 code 参数复制出来（整条 URL 也行），然后执行：")
    print('  python scripts/freesound_oauth.py exchange "粘贴的code"')


def extract_code(text: str) -> str:
    """同时兼容「纯 code」和「完整回调 URL」。"""
    text = text.strip()
    if "code=" in text:
        q = urllib.parse.urlparse(text).query or text.split("?", 1)[-1]
        vals = urllib.parse.parse_qs(q)
        if "code" in vals:
            return vals["code"][0]
    return text


def cmd_exchange(args) -> None:
    data = load()
    cid, secret = data.get("client_id"), data.get("client_secret")
    if not (cid and secret):
        raise SystemExit("还没设置 client_id / client_secret，请先跑 setup 命令")
    resp = post_form(TOKEN_URL, {
        "client_id": cid,
        "client_secret": secret,
        "grant_type": "authorization_code",
        "code": extract_code(args.code),
    })
    data["access_token"] = resp.get("access_token")
    data["refresh_token"] = resp.get("refresh_token")
    data["expires_at"] = int(time.time()) + int(resp.get("expires_in", 86400))
    data["token_type"] = resp.get("token_type", "Bearer")
    save(data)
    print("OAuth2 授权成功！access token 有效期 %s 秒" % resp.get("expires_in"))


def _refresh(data: dict) -> dict:
    if not data.get("refresh_token"):
        raise SystemExit("没有 refresh_token，需要重新走一次 authorize 流程")
    resp = post_form(TOKEN_URL, {
        "client_id": data["client_id"],
        "client_secret": data["client_secret"],
        "grant_type": "refresh_token",
        "refresh_token": data["refresh_token"],
    })
    data["access_token"] = resp.get("access_token")
    data["refresh_token"] = resp.get("refresh_token") or data.get("refresh_token")
    data["expires_at"] = int(time.time()) + int(resp.get("expires_in", 86400))
    save(data)
    return data


def ensure_access_token() -> str:
    """返回有效的 OAuth2 token，过期前自动刷新并保存。"""
    data = load()
    if not data.get("access_token"):
        raise RuntimeError("还没有 access token，请先跑 authorize + exchange")
    if time.time() >= data.get("expires_at", 0) - 60:
        data = _refresh(data)
    return data["access_token"]


def cmd_token(args) -> None:
    print(ensure_access_token())


def cmd_status(args) -> None:
    data = load()
    if not data:
        print("尚无凭据。")
        return
    print("client_id     : %s" % (data.get("client_id") or "(未设置)"))
    print("client_secret : %s" % ("(已设置)" if data.get("client_secret") else "(未设置)"))
    print("access_token  : %s" % ("(已设置)" if data.get("access_token") else "(未设置)"))
    print("refresh_token : %s" % ("(已设置)" if data.get("refresh_token") else "(未设置)"))
    if data.get("access_token"):
        remain = int(data.get("expires_at", 0) - time.time())
        print("剩余有效期    : %d 秒 (%.1f 小时)" % (remain, remain / 3600))


def main():
    ap = argparse.ArgumentParser(description="Freesound OAuth2 助手")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="保存 client_id / client_secret")
    p.add_argument("--client-id")
    p.add_argument("--client-secret")
    p.set_defaults(func=cmd_setup)

    sub.add_parser("authorize", help="生成授权链接").set_defaults(func=cmd_authorize)

    p = sub.add_parser("exchange", help="用授权码换 token")
    p.add_argument("code")
    p.set_defaults(func=cmd_exchange)

    sub.add_parser("token", help="打印可用 access token（自动刷新）").set_defaults(func=cmd_token)
    sub.add_parser("status", help="查看状态").set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

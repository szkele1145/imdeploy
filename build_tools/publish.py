"""把 imdeploy 发布到 GitHub：建仓库、推代码、发 Release、传安装包。"""
import json
import os
import subprocess
import urllib.error
import urllib.request

TOK = os.environ.get("GITHUB_TOKEN", "")
if not TOK:
    raise SystemExit("请先设置 GITHUB_TOKEN 环境变量")
USER = "szkele1145"
REPO = "imdeploy"
ROOT = r"F:\vibe coding工程\imdeploy"
SETUP = ROOT + r"\dist\installer\imdeploy-setup-0.1.0.exe"
PROXY = "http://127.0.0.1:65532"

op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}))
H = {"Authorization": f"token {TOK}", "Accept": "application/vnd.github+json",
     "User-Agent": "dsh", "X-GitHub-Api-Version": "2022-11-28"}


def call(method, url, data=None, hdr=None, want_json=True):
    h = dict(H)
    h.update(hdr or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with op.open(req, timeout=300) as resp:
            raw = resp.read()
            if not want_json or not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"  [HTTP {e.code}] {method} {url}\n  {body[:300]}")
        raise


def api(method, path, payload=None):
    data = None
    hdr = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        hdr["Content-Type"] = "application/json; charset=utf-8"
    return call(method, f"https://api.github.com{path}", data, hdr)


def git(*args):
    r = subprocess.run(["git", "-c", f"http.proxy={PROXY}", *args],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("  git", " ".join(args), "->", (r.stderr or r.stdout).strip()[:400])
    return r.returncode == 0


def main():
    # 1) 建仓库
    print("[1] 检查仓库")
    try:
        info = api("GET", f"/repos/{USER}/{REPO}")
        print(f"  仓库已存在: {info['html_url']}")
    except urllib.error.HTTPError:
        print("  创建仓库…")
        info = api("POST", "/user/repos", {
            "name": REPO,
            "description": "本地大模型部署器 —— 按硬件推荐并下载 GGUF 模型，提供 OpenAI 兼容 API",
            "private": False,
            "has_issues": True,
            "has_wiki": False,
        })
        print(f"  已创建: {info['html_url']}")

    clone = info["clone_url"]
    auth = clone.replace("https://", f"https://{USER}:{TOK}@")

    # 2) 初始化 git
    print("[2] 提交并推送")
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        git("init", "-b", "main")
    git("add", "-A")
    git("-c", "user.email=bot@imdeploy.local", "-c", "user.name=imdeploy",
        "commit", "-m", "ImDeploy 0.1.0：本地模型部署器（Windows 版）")
    git("remote", "remove", "origin")
    git("remote", "add", "origin", auth)
    ok = git("push", "-u", "origin", "main", "--force")
    print("  推送", "成功" if ok else "失败")

    # 3) Release
    print("[3] 创建 Release")
    try:
        rel = api("GET", f"/repos/{USER}/{REPO}/releases/tags/v0.1.0")
        print(f"  Release 已存在: {rel['html_url']}")
    except urllib.error.HTTPError:
        body = (
            "## ImDeploy 0.1.0 —— 本地大模型部署器（Windows）\n\n"
            "按本机硬件自动推荐并下载 GGUF 模型，提供 OpenAI 兼容 API。\n\n"
            "### 功能\n\n"
            "- 自动探测 CPU / 内存 / GPU，推荐 0-2B 优先、长上下文友好的模型\n"
            "- 从 ModelScope / HuggingFace 抓取模型目录，断点续传下载\n"
            "- OpenAI 兼容端点：`/v1/chat/completions`、`/v1/models`\n"
            "- 多模型别名路径：`/api/qwen`、`/api/llama`、`/api/gemma` …\n"
            "- 中文桌面面板：总览 / 推荐 / 模型库 / 在线目录 / 设置\n"
            "- 后端支持 llama.cpp（llama-server）与 Ollama\n\n"
            "### 安装\n\n"
            "下载 `imdeploy-setup-0.1.0.exe` 双击安装。\n\n"
            "### 说明\n\n"
            "模型推理需要 `llama-server.exe`（llama.cpp Windows 构建），"
            "放在 PATH 或设置里指定路径。"
        )
        rel = api("POST", f"/repos/{USER}/{REPO}/releases", {
            "tag_name": "v0.1.0",
            "name": "ImDeploy v0.1.0（Windows）",
            "body": body,
            "draft": False,
            "prerelease": False,
        })
        print(f"  已创建: {rel['html_url']}")

    # 4) 上传安装包
    print("[4] 上传安装包")
    rid = rel["id"]
    name = os.path.basename(SETUP)
    existing = {a["name"]: a for a in rel.get("assets", [])}
    if name in existing:
        call("DELETE", existing[name]["url"], want_json=False)
        print("  已删除同名旧资产")

    size = os.path.getsize(SETUP)
    with open(SETUP, "rb") as f:
        data = f.read()
    res = call("POST",
               f"https://uploads.github.com/repos/{USER}/{REPO}/releases/{rid}/assets?name={name}",
               data, {"Content-Type": "application/octet-stream",
                      "Content-Length": str(size)})
    print(f"  已上传: {res['name']} {res['size'] / 1024 / 1024:.1f} MB")

    # 5) 汇总
    final = api("GET", f"/repos/{USER}/{REPO}/releases/{rid}")
    print("\n=== 最终结果 ===")
    print("仓库:", final["html_url"].split("/releases")[0])
    print("Release:", final["html_url"])
    for a in final["assets"]:
        print(f"  {a['name']:36s} {a['size'] / 1024 / 1024:7.2f} MB")


if __name__ == "__main__":
    main()

"""更新 imdeploy Release 资产与说明。"""
import json
import os
import urllib.error
import urllib.request

TOK = os.environ["GITHUB_TOKEN"]
REPO = "szkele1145/imdeploy"
RID = int(os.environ.get("RID", "0")) or None

#: 要上传的资产：(本地路径, Release 上的文件名)
ASSETS: list[tuple[str, str]] = [
    (r"F:\vibe coding工程\imdeploy\dist\installer\imdeploy-setup-0.1.0.exe",
     "imdeploy-setup-0.1.0.exe"),
    (r"F:\vibe coding工程\imdeploy\dist\imdeploy-linux-x86_64.tar.gz",
     "imdeploy-linux-x86_64.tar.gz"),
    (r"F:\vibe coding工程\imdeploy\dist\imdeploy-android.apk",
     "imdeploy-android.apk"),
]

op = urllib.request.build_opener(
    urllib.request.ProxyHandler({"https": "http://127.0.0.1:65532"}))
H = {"Authorization": f"token {TOK}", "Accept": "application/vnd.github+json",
     "User-Agent": "dsh"}


def call(method, url, data=None, hdr=None, want_json=True):
    h = dict(H)
    h.update(hdr or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with op.open(req, timeout=300) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if (want_json and raw) else None


def main():
    if RID:
        rel = call("GET", f"https://api.github.com/repos/{REPO}/releases/{RID}")
    else:
        rel = call("GET", f"https://api.github.com/repos/{REPO}/releases/tags/v0.1.0")
    rid = rel["id"]
    print("release:", rel["tag_name"], rel["html_url"])

    body = (
        "## ImDeploy 0.1.0 —— 本地大模型部署器（Windows）\n\n"
        "按本机硬件自动推荐并下载 GGUF 模型，提供 OpenAI 兼容 API。\n\n"
        "### 懒人版：零配置\n\n"
        "安装后打开面板，在「总览 → 推理运行库」点 **一键安装推理运行库**，"
        "程序会自动检测你的显卡、从 llama.cpp 官方下载对应版本并解压到位，"
        "自动写进配置。不需要手动下载、解压、配 PATH。\n\n"
        "### 功能\n\n"
        "- 自动探测 CPU / 内存 / GPU，推荐 0-2B 优先、长上下文友好的模型\n"
        "- 从 ModelScope / HuggingFace 抓取模型目录，断点续传下载\n"
        "- OpenAI 兼容端点：`/v1/chat/completions`、`/v1/models`\n"
        "- 多模型别名路径：`/api/qwen`、`/api/llama`、`/api/gemma` …\n"
        "- 中文安装程序 + 中文桌面面板\n\n"
        "### 安装\n\n"
        "**Windows**：下载 `imdeploy-setup-0.1.0.exe` 双击安装，全程中文。\n\n"
        "**Linux**：下载 `imdeploy-linux-x86_64.tar.gz` 后执行：\n\n"
        "```bash\n"
        "tar -xzf imdeploy-linux-x86_64.tar.gz\n"
        "cd imdeploy-linux-x86_64\n"
        "bash install.sh\n"
        "```\n\n"
        "装完运行 `~/.local/share/imdeploy/run.sh` 启动面板，"
        "`run-api.sh` 只起 API 服务。\n\n"
        "**Android**：下载 `imdeploy-android.apk` 安装（Android 10+），"
        "内置 Termux 环境，首次启动会引导安装 Termux 并准备 Python 环境。"
    )
    call("PATCH", f"https://api.github.com/repos/{REPO}/releases/{rid}",
         json.dumps({"body": body}, ensure_ascii=False).encode("utf-8"),
         {"Content-Type": "application/json; charset=utf-8"}, want_json=False)
    print("说明已更新")

    rel = call("GET", f"https://api.github.com/repos/{REPO}/releases/{rid}")
    wanted = {name for _, name in ASSETS}
    for a in rel.get("assets", []):
        if a["name"] in wanted:
            call("DELETE", a["url"], want_json=False)
            print("已删除旧资产:", a["name"])

    for path, name in ASSETS:
        if not os.path.exists(path):
            print(f"[跳过] {name}（本地不存在: {path}）")
            continue
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            data = f.read()
        print(f"上传 {name}（{size / 1048576:.1f} MB）")
        res = call("POST",
                   f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets?name={name}",
                   data, {"Content-Type": "application/octet-stream",
                          "Content-Length": str(size)})
        print("完成:", res["name"], f"{res['size'] / 1048576:.2f} MB")

    final = call("GET", f"https://api.github.com/repos/{REPO}/releases/{rid}")
    print("\n=== 最终 ===")
    print(final["html_url"])
    for a in final["assets"]:
        print(f"  {a['name']:34s} {a['size'] / 1048576:7.2f} MB")


if __name__ == "__main__":
    main()
